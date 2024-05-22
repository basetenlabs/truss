import asyncio
import logging
import time

import bcp47
import data_types
import helpers
import httpx
import numpy as np
import truss_chains as chains

IMAGE = chains.DockerImage(
    apt_requirements=["ffmpeg"],
    pip_requirements=["pandas", "bcp47", "google-auth", "google-cloud-bigquery"],
)
# Whisper is deployed as a normal truss model from examples/library.
_WHISPER_URL = "https://model-5woz91z3.api.baseten.co/production/predict"
NO_OP_SHADOW = True  # If on, just tests task creation and skips actual transcriptions.


class DeployedWhisper(chains.StubBase):
    """Transcribes b64_encoded wave snippets.

    Treat the whisper model like an external third party tool."""

    async def run_remote(self, audio_b64: str) -> data_types.WhisperOutput:
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
        docker_image=IMAGE,
        compute=chains.Compute(
            cpu_count=8, memory="16G", predict_concurrency="cpu_count"
        ),
    )
    _whisper: DeployedWhisper

    def __init__(
        self,
        context: chains.DeploymentContext = chains.depends_context(),
    ) -> None:
        self._whisper = DeployedWhisper.from_url(
            _WHISPER_URL,
            context,
            options=chains.RPCOptions(retries=3),
        )
        logging.getLogger("httpx").setLevel(logging.WARNING)

    async def run_remote(
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
                tasks.append(asyncio.ensure_future(self._whisper.run_remote(audio_b64)))
                seg_infos.append(seg_info)

        results: list[data_types.WhisperOutput] = await asyncio.gather(*tasks)
        segments = []
        for transcription, seg_info in zip(results, seg_infos):
            segments.append(
                data_types.SegmentInternal(
                    transcription=transcription, segment_info=seg_info
                )
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


@chains.entrypoint
class Transcribe(chains.ChainletBase):
    """Transcribes one file end-to-end and sends results to webhook."""

    remote_config = chains.RemoteConfig(
        docker_image=IMAGE,
        compute=chains.Compute(
            cpu_count=8, memory="16G", predict_concurrency="cpu_count"
        ),
        assets=chains.Assets(secret_keys=["dummy_webhook_key"]),
    )
    _context: chains.DeploymentContext
    _video_worker: MacroChunkWorker
    _async_http: httpx.AsyncClient

    def __init__(
        self,
        video_worker: MacroChunkWorker = chains.depends(MacroChunkWorker, retries=3),
        context: chains.DeploymentContext = chains.depends_context(),
    ) -> None:
        self._context = context
        self._video_worker = video_worker
        self._async_http = httpx.AsyncClient()
        logging.getLogger("httpx").setLevel(logging.WARNING)
        if NO_OP_SHADOW:
            logging.info("No-Op-Shadow on: will not create chunk jobs.")
        else:
            logging.info("No-Op-Shadow off.")

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

    async def run_remote(
        self, job_descr: data_types.JobDescriptor, params: data_types.TranscribeParams
    ) -> data_types.Result:
        t0 = time.time()
        media_url = job_descr.media_url
        await self._assert_media_supports_range_downloads(media_url)
        duration_secs = await helpers.query_video_length_secs(media_url)
        logging.info(f"Transcribe request for `{duration_secs:.1f}` seconds.")
        if NO_OP_SHADOW:
            return data_types.Result(
                job_descr=job_descr.copy(
                    update={"status": data_types.JobStatus.DEBUG_RESULT}
                ),
                segments=[],
                input_duration_sec=duration_secs,
                processing_duration_sec=0,
                speedup=0,
            )

        video_chunks = helpers.generate_time_chunks(
            int(np.ceil(duration_secs)), params.macro_chunk_size_sec
        )
        tasks = []
        for i, chunk_limits in enumerate(video_chunks):
            logging.debug(f"Starting macro-chunk [{i + 1:03}/{len(video_chunks):03}].")
            tasks.append(
                asyncio.ensure_future(
                    self._video_worker.run_remote(media_url, params, i, *chunk_limits)
                )
            )

        results: list[data_types.TranscribeOutput] = await asyncio.gather(*tasks)
        t1 = time.time()
        processing_time = t1 - t0
        logging.info(
            data_types.TranscribeOutput(
                segments=[],
                input_duration_sec=duration_secs,
                processing_duration_sec=processing_time,
                speedup=duration_secs / processing_time,
            )
        )
        segments = [
            data_types.Segment(
                start=seg.segment_info.start_time_sec,
                end=seg.segment_info.end_time_sec,
                text=seg.transcription.text,
                language=seg.transcription.language,
                bcp47_key=seg.transcription.bcp47_key,
            )
            for part in results
            for seg in part.segments
        ]
        result = data_types.Result(
            job_descr=job_descr.copy(update={"status": data_types.JobStatus.SUCCEEDED}),
            segments=segments,
            input_duration_sec=duration_secs,
            processing_duration_sec=processing_time,
            speedup=duration_secs / processing_time,
        )
        return result


if __name__ == "__main__":
    import os

    url_ = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4"
    job_ = data_types.JobDescriptor(job_uuid="job_uuid_0", media_id=0, media_url=url_)
    params_ = data_types.TranscribeParams(micro_chunk_size_sec=30)

    with chains.run_local(
        secrets={"baseten_chain_api_key": os.environ["BASETEN_API_KEY"]}
    ):
        transcribe_job = Transcribe()

        result_ = asyncio.run(transcribe_job.run_remote(job_, params_))
        print(result_)
