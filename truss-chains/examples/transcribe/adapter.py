import asyncio
import datetime
import json
import logging
import re
import time
from typing import Optional

import data_types
import helpers
import httpx
import numpy as np
import pydantic
import tenacity
import transcribe as transcribe_base
import truss_chains as chains
from google.cloud import bigquery
from google.oauth2 import service_account
from truss_chains import stub

NO_OP_SHADOW = False  # If on, just tests task creation and skips actual transcriptions.
FILE_EXT_RE = re.compile(r"\.([a-zA-Z0-9]+)\?")


class EnqueueInput(pydantic.BaseModel):
    # This typo `media_for_transciption` is for backwards compatibility.
    # Also, this would ideally be `list[JobDescriptor]` instead of string...
    media_for_transciption: str


class EnqueueOutput(pydantic.BaseModel):
    # Do not change, taken from existing App.
    success: bool
    jobs: list[data_types.JobDescriptor]
    error_message: Optional[str] = None


class TranscribeInput(pydantic.BaseModel):
    # Do not change, taken from existing App.
    job_descr: data_types.JobDescriptor
    params: data_types.TranscribeParams


class RetryConfig(pydantic.BaseModel):
    # Do not change, matches `async_predict` data-models.
    max_attempts: int = 1
    initial_delay_ms: int = 0
    max_delay_ms: int = 5000


class AsyncTranscribeRequest(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(protected_namespaces=())
    # Do not change, matches `async_predict` data-models.
    model_input: TranscribeInput
    webhook_endpoint: Optional[str]
    inference_retry_config: RetryConfig


class TranscribeResult(pydantic.BaseModel):
    # Do not change, taken from existing App.
    media_url: str
    media_id: int  # Seems to be just 0 or 1.
    job_uuid: str
    status: data_types.JobStatus
    # TODO: this is not a great name.
    text: Optional[list[data_types.Segment]] = None
    failure_reason: Optional[str] = None

    @classmethod
    def from_job_descr(
        cls, job_descr: data_types.JobDescriptor, status: data_types.JobStatus
    ):
        return TranscribeResult(
            media_url=job_descr.media_url,
            media_id=job_descr.media_id,
            job_uuid=job_descr.job_uuid,
            status=status,
        )


# Chainlets ############################################################################


class InternalWebhook(chains.ChainletBase):
    """Receives results for debugging."""

    remote_config = chains.RemoteConfig(
        docker_image=transcribe_base.IMAGE,
        compute=chains.Compute(cpu_count=1, memory="2G", predict_concurrency=128),
        assets=chains.Assets(secret_keys=["big_query_service_account"]),
    )
    _async_http: httpx.AsyncClient

    def __init__(self, context: chains.DeploymentContext = chains.depends_context()):
        key_dict = json.loads(context.secrets["big_query_service_account"])
        credentials = service_account.Credentials.from_service_account_info(key_dict)
        self._async_http = httpx.AsyncClient()
        self._bigquery_client = bigquery.Client(
            credentials=credentials, project=credentials.project_id
        )
        self._table_id = "staging-workload-plane-1.Transcription_DBG.Segments"
        logging.getLogger("httpx").setLevel(logging.WARNING)

    async def run_remote(self, result: TranscribeResult) -> None:
        row_data = {
            "timestamp": datetime.datetime.utcnow().isoformat("T") + "Z",
            "job_uuid": result.job_uuid,
            "result_json": result.json(),
            "error_msg": result.failure_reason,
        }
        if result.failure_reason:
            logging.error(result.failure_reason)

        errors = self._bigquery_client.insert_rows_json(self._table_id, [row_data])
        if not errors:
            logging.info("Result uploaded to bigquery.")
        else:
            logging.error(f"Result upload to bigquery failed: {errors}.")

        # This would call external webhook next.
        # payload_str = json.dumps(result)
        # payload_signature = hmac.new(
        #     self._context.get_secret("***_webhook_key").encode("utf-8"),
        #     payload_str.encode("utf-8"),
        #     hashlib.sha1,
        # ).hexdigest()
        # headers = {"X-Baseten-Signature": payload_signature}
        # # Extra key `transcription` is needed for test webhook.
        # resp = await self._async_http.post(
        #   webhook_url, data=payload_str, headers=headers)
        # if resp.status_code == 200:
        #     return
        # else:
        #     raise Exception(f"Could not call webhook: {resp.content.decode()}.")


class TranscribeWithWebhook(chains.ChainletBase):
    """Transcribes one file end-to-end and sends results to webhook."""

    # TODO: this would be a subclass of `transcribe_base.Transcribe`, but chains
    #  does not support subclasses currently. Keep code in sync manually!

    remote_config = chains.RemoteConfig(
        docker_image=transcribe_base.IMAGE,
        compute=chains.Compute(cpu_count=8, memory="16G", predict_concurrency=128),
        assets=chains.Assets(secret_keys=["dummy_webhook_key"]),
    )
    _context: chains.DeploymentContext
    _video_worker: transcribe_base.MacroChunkWorker
    _async_http: httpx.AsyncClient

    def __init__(
        self,
        video_worker: transcribe_base.MacroChunkWorker = chains.depends(
            transcribe_base.MacroChunkWorker, retries=3
        ),
        internal_webhook: InternalWebhook = chains.depends(InternalWebhook, retries=2),
        context: chains.DeploymentContext = chains.depends_context(),
    ) -> None:
        self._context = context
        self._video_worker = video_worker
        self._internal_webhook = internal_webhook
        self._async_http = httpx.AsyncClient()
        logging.getLogger("httpx").setLevel(logging.WARNING)
        if NO_OP_SHADOW:
            logging.info("No-Op-Shadow on: will no create chunk jobs.")
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

    async def _transcribe(
        self, job_descr: data_types.JobDescriptor, params: data_types.TranscribeParams
    ) -> TranscribeResult:
        t0 = time.time()
        media_url = job_descr.media_url
        file_ext = "<unknown>"
        try:
            match = re.search(FILE_EXT_RE, media_url)
            if match:
                file_ext = match.group(1)
        except Exception:
            pass

        await self._assert_media_supports_range_downloads(media_url)
        duration_secs = await helpers.query_video_length_secs(media_url)
        logging.info(
            f"Transcribe request for `{duration_secs:.1f}` seconds, "
            f"file_ext: `{file_ext}`."
        )
        if NO_OP_SHADOW:
            return TranscribeResult.from_job_descr(
                job_descr,
                status=data_types.JobStatus.DEBUG_RESULT,  # type: ignore[arg-type]
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
        # Ignored type issue seems to be a mypy bug.
        result = TranscribeResult.from_job_descr(
            job_descr, status=data_types.JobStatus.SUCCEEDED  # type: ignore[arg-type]
        )
        result.text = [
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
        return result

    async def run_remote(
        self, job_descr: data_types.JobDescriptor, params: data_types.TranscribeParams
    ) -> TranscribeResult:
        # When `Transcribe.run` is called with `async_predict` the async framework will
        # invoke the webhook, but it expects that the webhook does not need
        # authentication (instead will validate via payload signature).
        # We can't run a chainlet (i.e. truss model) without requiring authentication,
        # and we don't want to stand up and manage another service - so we basically
        # ignore async's webhook integration and "manually" call a webhook from here
        # where we can explicitly add the authentication.
        # This is a bit suboptimal, because it does not wrap around catching all errors
        # (there could be errors raised from the generated truss model if the input
        # payload is malformed).
        try:
            retrying = tenacity.AsyncRetrying(
                stop=tenacity.stop_after_attempt(3),
                retry=tenacity.retry_if_exception_type(Exception),
                reraise=True,
            )
            async for attempt in retrying:
                with attempt:
                    if (num := attempt.retry_state.attempt_number) > 1:
                        logging.info(
                            f"Retrying `{self._transcribe.__name__}`, attempt {num}"
                        )

                    result = await self._transcribe(job_descr, params)

        except Exception as e:
            logging.exception("Transcription failed, still invoking webhook.")
            error_msg = f"{type(e)}: {str(e)}"
            # Ignored type issue seems to be a mypy bug.
            failure_result = TranscribeResult.from_job_descr(
                job_descr,
                status=data_types.JobStatus.PERMAFAILED,  # type: ignore[arg-type]
            )
            failure_result.failure_reason = error_msg
            await self._internal_webhook.run_remote(result=failure_result)
            raise  # Make this considered failed to caller / async processor.

        await self._internal_webhook.run_remote(result=result)
        return result


@chains.mark_entrypoint
class EnqueueBatch(chains.ChainletBase):
    """Accepts a request with multiple transcription jobs and starts the sub-jobs."""

    remote_config = chains.RemoteConfig(
        docker_image=transcribe_base.IMAGE,
        compute=chains.Compute(cpu_count=8, memory="16G", predict_concurrency=128),
    )

    def __init__(
        self,
        transcribe: TranscribeWithWebhook = chains.depends(TranscribeWithWebhook),
        context=chains.depends_context(),
    ):
        logging.info(
            f"Replacing {transcribe.__class__.__name__} with `async_predict` version."
        )
        transcribe_service = context.get_service_descriptor(
            TranscribeWithWebhook.__name__
        )
        predict_url = transcribe_service.predict_url.replace("predict", "async_predict")
        async_transcribe = transcribe_service.copy(update={"predict_url": predict_url})
        self._async_transcribe = stub.BasetenSession(
            async_transcribe, context.get_baseten_api_key()
        )
        logging.getLogger("httpx").setLevel(logging.WARNING)

    def _enqueue(
        self, job: data_types.JobDescriptor, params: data_types.TranscribeParams
    ):
        # While we can't use the async framework's webhooks, we have to do the retry
        # logic manually inside `Transcribe`, not use async's retry.
        # The internal webhook endpoint will return 401 because it requires
        # a baseten API key for authentication.
        request = AsyncTranscribeRequest(
            model_input=TranscribeInput(
                job_descr=job,
                params=params,
            ),
            webhook_endpoint=None,
            inference_retry_config=RetryConfig(max_attempts=1),
        )
        return self._async_transcribe.predict_async(request.model_dump())

    async def run_remote(self, worklet_input: EnqueueInput) -> EnqueueOutput:
        # `worklet_input` must not be renamed, it matches the expected payload.
        media_for_transcription = json.loads(worklet_input.media_for_transciption)
        jobs = [
            data_types.JobDescriptor.model_validate(x) for x in media_for_transcription
        ]
        logging.info(f"Got `{len(jobs)}` jobs in batch.")
        params = data_types.TranscribeParams()
        tasks = [asyncio.ensure_future(self._enqueue(job, params)) for job in jobs]
        queued_jobs = []
        for i, val in enumerate(await asyncio.gather(*tasks, return_exceptions=True)):
            job = jobs[i]
            if isinstance(val, Exception):
                logging.exception(
                    f"Could not enqueue `{job.json()}`. {val}", stack_info=True
                )
                job = job.copy(update={"status": data_types.JobStatus.PERMAFAILED})
            else:
                logging.info(f"Async call response: {val}")
                job = job.copy(update={"status": data_types.JobStatus.QUEUED})

            queued_jobs.append(job)
        output = EnqueueOutput(success=True, jobs=queued_jobs)
        return output


if __name__ == "__main__":
    import os

    from truss_chains import definitions

    url_ = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4"
    job_ = data_types.JobDescriptor(job_uuid="job_uuid_0", media_id=0, media_url=url_)
    worklet_input_ = EnqueueInput(
        media_for_transciption=json.dumps([job_.model_dump()])
    )

    with open(
        "/home/marius-baseten/Downloads/staging-workload-plane-1-79e9a36615ed.json", "r"
    ) as f:
        account_json = f.read()

    with chains.run_local(
        secrets={
            "baseten_chain_api_key": os.environ["BASETEN_API_KEY"],
            "big_query_service_account": account_json,
        },
        chainlet_to_service={
            "TranscribeWithWebhook": definitions.ServiceDescriptor(
                name="TranscribeWithWebhook",
                predict_url="https://model-5qexxkp3.api.baseten.co/development/predict",
                options=chains.RPCOptions(),
            )
        },
    ):
        transcribe_chainlet = EnqueueBatch()
        result_ = asyncio.run(transcribe_chainlet.run_remote(worklet_input_))
        print(result_)
