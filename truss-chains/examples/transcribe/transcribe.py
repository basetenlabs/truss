import asyncio
import logging
import time

import data_types
import helpers
import httpx
import truss_chains as chains
import vad

IMAGE = chains.DockerImage(
    apt_requirements=["ffmpeg"],
    pip_requirements=["google-auth", "google-cloud-bigquery"],
)
# Whisper is deployed as a normal truss model from examples/library.
# See `whisper-truss/model/model.py`.
_WHISPER_URL = "https://model-v31y4243.api.baseten.co/development/predict"


class DeployedWhisper(chains.StubBase):
    """Transcribes b64_encoded wave snippets.

    Treat the whisper model like an external third party tool."""

    async def run_remote(self, audio_b64: str) -> data_types.WhisperResult:
        resp = await self._remote.predict_async(json_payload={"audio_b64": audio_b64})
        return data_types.WhisperResult.parse_obj(resp)


class MacroChunkWorker(chains.ChainletBase):
    """Downloads and transcribes larger chunks of the total file (~300s)."""

    _cpu_count = 8
    remote_config = chains.RemoteConfig(
        docker_image=IMAGE,
        compute=chains.Compute(
            cpu_count=_cpu_count, memory="16G", predict_concurrency=_cpu_count * 3
        ),
    )
    _whisper: DeployedWhisper

    def __init__(
        self,
        context: chains.DeploymentContext = chains.depends_context(),
    ) -> None:
        self._vad_model = vad.VAD()
        self._whisper = DeployedWhisper.from_url(
            _WHISPER_URL,
            context,
            options=chains.RPCOptions(retries=2),
        )
        logging.getLogger("httpx").setLevel(logging.WARNING)

    async def run_remote(
        self,
        media_url: str,
        params: data_types.TranscribeParams,
        macro_chunk: data_types.ChunkInfo,
    ) -> data_types.SegmentList:
        t0 = time.time()
        tasks = []
        micro_chunks = []
        logging.debug(f"Macro-chunk [{macro_chunk.macro_chunk:03}]: Starting.")
        async with helpers.DownloadSubprocess(
            media_url, macro_chunk, params.wav_sampling_rate_hz
        ) as wav_stream:
            chunk_stream = helpers.wav_chunker(params, wav_stream, self._vad_model)
            async for micro_chunk, audio_b64 in chunk_stream:
                tasks.append(asyncio.ensure_future(self._whisper.run_remote(audio_b64)))
                micro_chunks.append(micro_chunk)

        whisper_results = await helpers.gather(tasks)
        segments: list[data_types.Segment] = []
        whisper_result: data_types.WhisperResult
        for whisper_result, micro_chunk in zip(whisper_results, micro_chunks):
            segments.extend(
                helpers.convert_whisper_segments(whisper_result, micro_chunk)
            )

        logging.debug(f"Chunk [{macro_chunk.macro_chunk:03}]: Complete.")
        t1 = time.time()
        return data_types.SegmentList(
            segments=segments,
            chunk_info=macro_chunk.copy(update={"processing_duration": t1 - t0}),
        )


@chains.mark_entrypoint
class Transcribe(chains.ChainletBase):
    """Transcribes one file end-to-end and sends results to webhook."""

    remote_config = chains.RemoteConfig(
        docker_image=IMAGE,
        compute=chains.Compute(cpu_count=8, memory="16G", predict_concurrency=128),
        assets=chains.Assets(secret_keys=["dummy_webhook_key"]),
    )
    _context: chains.DeploymentContext
    _macro_chunk_worker: MacroChunkWorker
    _async_http: httpx.AsyncClient

    def __init__(
        self,
        macro_chunk_worker: MacroChunkWorker = chains.depends(
            MacroChunkWorker, retries=3
        ),
        context: chains.DeploymentContext = chains.depends_context(),
    ) -> None:
        self._context = context
        self._macro_chunk_worker = macro_chunk_worker
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

    async def run_remote(
        self, job_descr: data_types.JobDescriptor, params: data_types.TranscribeParams
    ) -> data_types.TranscribeOutput:
        t0 = time.time()
        media_url = job_descr.media_url
        await self._assert_media_supports_range_downloads(media_url)
        duration_secs = await helpers.query_source_length_secs(media_url)
        logging.info(f"Transcribe request for `{duration_secs:.1f}` seconds.")
        # TODO: use silence-aware time chunking.
        macro_chunks = helpers.generate_time_chunks(
            duration_secs,
            params.macro_chunk_size_sec,
            params.macro_chunk_overlap_sec,
        )
        # for c in macro_chunks:
        #     print(c.model_dump_json(indent=4))
        tasks = []
        for i, macro_chunk in enumerate(macro_chunks):
            logging.debug(f"Starting macro-chunk [{i + 1:03}/{len(macro_chunks):03}].")
            tasks.append(
                asyncio.ensure_future(
                    self._macro_chunk_worker.run_remote(media_url, params, macro_chunk)
                )
            )
        segments = [
            seg
            for segment_list in await helpers.gather(tasks)
            for seg in segment_list.segments
        ]
        processing_time = time.time() - t0
        result = data_types.TranscribeOutput(
            job_descr=job_descr.copy(update={"status": data_types.JobStatus.SUCCEEDED}),
            segments=segments,
            input_duration_sec=duration_secs,
            processing_duration_sec=processing_time,
            speedup=duration_secs / processing_time,
        )
        logging.info(result.model_dump_json(indent=4, exclude={"segments"}))
        return result


if __name__ == "__main__":
    import os

    url_ = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4"
    # url_ = "https://file-examples.com/storage/fe15076da466528199d9c5a/2017/11/file_example_MP3_5MG.mp3"
    job_ = data_types.JobDescriptor(job_uuid="job_uuid_0", media_id=0, media_url=url_)
    params_ = data_types.TranscribeParams(micro_chunk_size_sec=25)

    with chains.run_local(
        secrets={"baseten_chain_api_key": os.environ["BASETEN_API_KEY"]}
    ):
        transcribe_job = Transcribe()

        result_ = asyncio.get_event_loop().run_until_complete(
            transcribe_job.run_remote(job_, params_)
        )
        print(result_.model_dump_json(indent=4))
