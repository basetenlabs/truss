import asyncio
import hashlib
import hmac
import logging
import time

import bcp47
import data_types
import helpers
import httpx
import numpy as np
import tenacity
import truss_chains as chains

_IMAGE = docker_image = chains.DockerImage(
    apt_requirements=["ffmpeg"], pip_requirements=["pandas", "bcp47"]
)

_WHISPER_URL = "https://model-5woz91z3.api.baseten.co/production/predict"
_WEBHOOK_URL = "https://model-4q99pkxq.api.baseten.co/development/predict"


class DeployedWhisper(chains.StubBase):
    """Transcribes b64_encoded wave snippets.

    Treat the whisper model like an external third party tool."""

    async def run(self, audio_b64: str) -> data_types.WhisperOutput:
        resp = await self._remote.predict_async(json_payload={"audio": audio_b64})
        # TODO: Get Whisper model with langauge, ideally also timestamps.
        language = "dummy language"
        bcp47_key = bcp47.languages.get(language.capitalize(), "default")
        return data_types.WhisperOutput(
            text=resp["text"], language=language, bcp47_key=bcp47_key
        )


class MacroChunkWorker(chains.ChainletBase):
    """Downloads and transcribes larger chunks of the total file (~300s)."""

    remote_config = chains.RemoteConfig(
        docker_image=_IMAGE, compute=chains.Compute(cpu_count=16, memory="32G")
    )
    _whisper: DeployedWhisper

    def __init__(
        self,
        context: chains.DeploymentContext = chains.provide_context(),
    ) -> None:
        super().__init__(context)
        self._whisper = DeployedWhisper.from_url(
            _WHISPER_URL,
            context,
            options=chains.RPCOptions(retries=3),
        )

    async def run(
        self,
        media_url: str,
        params: data_types.TranscribeParams,
        macro_chunk_index: int,
        start_time: str,
        duration_sec: int,
    ) -> data_types.TranscribeOutput:
        t0 = time.time()
        tasks = []
        seg_infos = []
        logging.debug(f"Macro-chunk [{macro_chunk_index:03}]: Starting.")
        async with helpers.DownloadSubprocess(
            media_url, start_time, duration_sec, params.wav_sampling_rate_hz
        ) as wav_stream:
            chunk_stream = helpers.wav_chunker(params, wav_stream, macro_chunk_index)
            async for seg_info, audio_b64 in chunk_stream:
                tasks.append(asyncio.ensure_future(self._whisper.run(audio_b64)))
                seg_infos.append(seg_info)

        results: list[data_types.WhisperOutput] = await asyncio.gather(*tasks)
        segments = []
        for transcription, seg_info in zip(results, seg_infos):
            segments.append(
                data_types.Segment(transcription=transcription, segment_info=seg_info)
            )
        logging.debug(f"Chunk [{macro_chunk_index:03}]: Complete.")
        t1 = time.time()
        processing_duration_sec = t1 - t0
        return data_types.TranscribeOutput(
            segments=segments,
            input_duration_sec=duration_sec,
            processing_duration_sec=processing_duration_sec,
            speedup=duration_sec / processing_duration_sec,
        )


class Transcribe(chains.ChainletBase):
    """Transcribes one file end-to-end and sends results to webhook."""

    remote_config = chains.RemoteConfig(
        docker_image=_IMAGE,
        compute=chains.Compute(cpu_count=16, memory="32G"),
        assets=chains.Assets(secret_keys=["dummy_webhook_key"]),
    )
    _video_worker: MacroChunkWorker
    _async_http: httpx.AsyncClient

    def __init__(
        self,
        context: chains.DeploymentContext = chains.provide_context(),
        video_worker: MacroChunkWorker = chains.provide(MacroChunkWorker, retries=3),
    ) -> None:
        super().__init__(context)
        self._video_worker = video_worker
        self._async_http = httpx.AsyncClient()
        logging.getLogger("httpx").setLevel(logging.WARNING)

    async def _assert_media_supports_range_downloads(self, media_url: str) -> None:
        ok = False
        try:
            head_response = await self._async_http.head(media_url)
            if "bytes" in head_response.headers.get("Accept-Ranges", ""):
                ok = True
            # Check by making a test range request to see if '206' is returned.
            range_header = {"Range": "bytes=0-0"}
            range_response = await self._async_http.get(media_url, headers=range_header)
            ok = range_response.status_code == 206
        except httpx.HTTPError as e:
            logging.error(f"Error checking URL: {e}")

        if not ok:
            raise NotImplementedError(f"Range downloads unsupported for `{media_url}`.")

    async def run(
        self,
        media_url: str,
        params: data_types.TranscribeParams,
        job_descr: data_types.JobDescriptor,
    ) -> data_types.TranscribeOutput:
        t0 = time.time()
        await self._assert_media_supports_range_downloads(media_url)
        duration_secs = await helpers.query_video_length_secs(media_url)
        video_chunks = helpers.generate_time_chunks(
            int(np.ceil(duration_secs)), params.macro_chunk_size_sec
        )
        tasks = []
        for i, chunk_limits in enumerate(video_chunks):
            logging.info(f"Starting macro-chunk [{i+1:03}/{len(video_chunks):03}].")
            tasks.append(
                asyncio.ensure_future(
                    self._video_worker.run(media_url, params, i, *chunk_limits)
                )
            )

        results: list[data_types.TranscribeOutput] = await asyncio.gather(*tasks)
        t1 = time.time()
        processing_time = t1 - t0
        result = data_types.TranscribeOutput(
            segments=[seg for part in results for seg in part.segments],
            input_duration_sec=duration_secs,
            processing_duration_sec=processing_time,
            speedup=duration_secs / processing_time,
        )

        external_result = data_types.TranscriptionExternal(
            media_url=job_descr.media_url,
            media_id=job_descr.media_id,
            job_uuid=job_descr.job_uuid,
            status=data_types.JobStatus.SUCCEEDED,
            text=[
                data_types.TranscriptionSegmentExternal(
                    start=seg.segment_info.start_time_sec,
                    end=seg.segment_info.end_time_sec,
                    text=seg.transcription.text,
                    language=seg.transcription.language,
                    bcp47_key=seg.transcription.bcp47_key,
                )
                for part in results
                for seg in part.segments
            ],
        )
        await self._call_webhook(params.result_webhook_url, external_result)
        return result

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=2, min=1, max=10),
        reraise=True,
    )
    async def _call_webhook(
        self, webhook_url: str, result: data_types.TranscriptionExternal
    ):
        # TODO: change secret key. Remove baseten auth.
        result_json = result.json()
        payload_signature = hmac.new(
            self._context.secrets["dummy_webhook_key"].encode("utf-8"),
            result_json.encode("utf-8"),
            hashlib.sha1,
        ).hexdigest()
        headers = {
            "X-Baseten-Signature": payload_signature,
            "Authorization": f"Api-Key {self._context.get_baseten_api_key()}",
        }
        # Extra key `transcription` is needed for test webhook.
        resp = await self._async_http.post(
            webhook_url, json={"transcription": result.dict()}, headers=headers
        )
        if resp.status_code == 200:
            return
        else:
            raise Exception(f"Could not call results webhook: {resp.content.decode()}.")


# Shims for external APIs ##############################################################


class BatchTranscribe(chains.ChainletBase):
    """Accepts a request with multiple transcription jobs and starts the sub-jobs."""

    remote_config = chains.RemoteConfig(
        docker_image=_IMAGE, compute=chains.Compute(cpu_count=16, memory="32G")
    )

    def __init__(
        self,
        context: chains.DeploymentContext = chains.provide_context(),
        transcribe: Transcribe = chains.provide(Transcribe),
    ):
        super().__init__(context)
        self._transcribe = transcribe

    async def run(self, worklet_input: data_types.WorkletInput) -> list[float]:
        logging.info(worklet_input)
        logging.info(f"Got `{len(worklet_input.media_for_transcription)}` tasks.")

        params = data_types.TranscribeParams(  # type: ignore[call-arg]
            result_webhook_url=_WEBHOOK_URL, micro_chunk_size_sec=30
        )
        tasks = []
        for job in worklet_input.media_for_transcription:
            tasks.append(
                asyncio.ensure_future(self._transcribe.run(job.media_url, params, job))
            )
        results = []
        for completed_task in asyncio.as_completed(tasks):
            # TODO: report errors with webhook.
            result: data_types.TranscribeOutput = await completed_task
            logging.info(
                f"Finished `{result.input_duration_sec}` with {result.speedup} x."
            )
            results.append(result)

        return [result.speedup for result in results]


class Webhook(chains.ChainletBase):
    """Receives results for debugging."""

    remote_config = chains.RemoteConfig(
        docker_image=_IMAGE, compute=chains.Compute(cpu_count=16, memory="32G")
    )

    async def run(self, transcription: data_types.TranscriptionExternal) -> None:
        logging.info(transcription.json(indent=4))
