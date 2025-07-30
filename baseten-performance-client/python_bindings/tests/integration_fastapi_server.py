# Based on ther performance client, this script implements a FastAPI server that can be used to test the performance of the Python bindings.
# The fastapi server implements the types for embeddings/rerank and classify - all strings are pydantic json string models, with metadata as a dict.

import asyncio
import random
import threading
import time
from typing import Optional

import fastapi
from baseten_performance_client import PerformanceClient
from pydantic import BaseModel, Field


class HijackPayload(BaseModel):
    """
    The stringified payload that is part of every batched request.
    """

    number_of_requests: int
    random_payload: str
    uuid_int: int
    max_batch_size: int
    max_chars_per_request: Optional[int] = None
    send_429_until_time: Optional[float] = None
    stall_x_many_requests: Optional[int] = None
    stall_for_seconds: Optional[float] = None
    internal_server_error_no_stall: bool = False

    def to_string(self) -> str:
        """
        Convert the HijackPayload to a string.
        """
        return self.model_dump_json()

    @classmethod
    def from_string(cls, string: str) -> "HijackPayload":
        """
        Convert a string to a HijackPayload.
        """
        return cls.model_validate_json(string)


class OpenAIEmbeddingRequest(BaseModel):
    """
    OpenAI embedding request model.
    """

    model: str
    input: list[str]


class EmbeddingData(BaseModel):
    """
    Data model for embedding data.
    """

    embedding: list[float]
    index: int
    object: str = "list"


class OpenAIEmbeddingResponse(BaseModel):
    """
    OpenAI embedding response model.
    """

    model: str
    object: str = "list"
    data: list[EmbeddingData]
    usage: dict = Field(default_factory=lambda: {"prompt_tokens": 0, "total_tokens": 0})


def build_server():
    """
    Build the FastAPI server.
    """
    app = fastapi.FastAPI()
    # add storage to the app state
    app.state.storage = {"processed_requests": 0, "successful_requests": 0}
    app.state.async_lock = asyncio.Lock()

    def validate_hijack_payload(inputs: list[str]) -> list[HijackPayload]:
        """
        Validate the hijack payload.
        """
        if not len(inputs):
            raise fastapi.HTTPException(
                status_code=400,
                detail=r"Hijack payload must contain at least one input.",
            )
        try:
            hijack_payloads = [HijackPayload.from_string(i) for i in inputs]
        except Exception as e:
            raise fastapi.HTTPException(
                status_code=400, detail=f"Invalid hijack payload: {str(e)}"
            )
        hijack_payload = hijack_payloads[0]
        if len(inputs) > hijack_payload.max_batch_size:
            raise fastapi.HTTPException(
                status_code=400, detail=r"Batch size exceeds maximum allowed size."
            )
        if hijack_payload.max_chars_per_request is not None:
            total_chars = sum(len(input_str) for input_str in inputs)
            if total_chars > hijack_payload.max_chars_per_request:
                raise fastapi.HTTPException(
                    status_code=400, detail=r"Total characters exceed maximum allowed."
                )
        return hijack_payloads

    async def action_hijack_payload(
        hijack_payload: HijackPayload, fapi: fastapi.Request
    ):
        """
        Perform an action based on the hijack payload.
        """
        # Here you would implement the logic to handle the hijack payload
        # For now, we just return a dummy response

        # sleep for the specified stall time if provided
        # 429 responses
        async with app.state.async_lock:
            app.state.storage["processed_requests"] += 1
            processed_requests = app.state.storage.get("processed_requests")
        if hijack_payload.send_429_until_time is not None:
            t = time.time() - hijack_payload.send_429_until_time
            # stall until the specified time
            if t < 0:
                raise fastapi.HTTPException(
                    status_code=429, detail="Too many requests, please try again later."
                )
        elif hijack_payload.stall_x_many_requests:
            if hijack_payload.stall_x_many_requests >= processed_requests:
                if hijack_payload.internal_server_error_no_stall:
                    raise fastapi.HTTPException(
                        status_code=500,
                        detail="Internal server error, no stall, please retry another replica.",
                    )
                elif hijack_payload.stall_for_seconds is not None:
                    await asyncio.sleep(hijack_payload.stall_for_seconds)
        if await fapi.is_disconnected():
            raise fastapi.HTTPException(
                status_code=499, detail="client disconnected, code will not matter."
            )
        async with app.state.async_lock:
            app.state.storage["successful_requests"] += 1

    @app.post("/v1/embeddings", response_model=OpenAIEmbeddingResponse)
    async def embeddings(request: OpenAIEmbeddingRequest, fapi: fastapi.Request):
        """
        Handle OpenAI embedding requests.
        """
        # Here you would implement the logic to handle the embedding request
        # For now, we return a dummy response
        if not request.input:
            raise fastapi.HTTPException(
                status_code=400, detail="Input cannot be empty."
            )
        # Simulate a hijack payload
        hijack_payloads = validate_hijack_payload(request.input)
        await action_hijack_payload(hijack_payloads[0], fapi)
        return OpenAIEmbeddingResponse(
            model=request.model,
            # uuid_int is used as embedding
            data=[
                EmbeddingData(embedding=[load.uuid_int], index=i)
                for i, load in enumerate(hijack_payloads)
            ],
            object="list",
        )

    @app.post("/reset")
    async def reset():
        """
        Reset the server state.
        """
        # Here you would implement the logic to reset the server state
        # For now, we just return a success message
        async with app.state.async_lock:
            processed_requests = app.state.storage.get("processed_requests")
            successful_requests = app.state.storage.get("successful_requests")
            app.state.storage["processed_requests"] = 0
            app.state.storage["successful_requests"] = 0
            return {
                "processed_requests": processed_requests,
                "successful_requests": successful_requests,
            }

    return app


def run_server():
    """
    Run the FastAPI server.
    """
    import uvicorn

    app = build_server()
    uvicorn.run(app, host="0.0.0.0", port=8000)


def run_client():
    """
    Run the Baseten performance client.
    """
    client = PerformanceClient(
        base_url="http://0.0.0.0:8000/",
        api_key="dummy_api_key",  # Replace with your actual API key
    )

    def prepare_hijack_payloads(
        number_of_requests: int,
        max_batch_size: int,
        max_chars_per_request: Optional[int] = None,
        send_429_until_time: Optional[float] = None,
        stall_x_many_requests: Optional[int] = None,
        stall_for_seconds: Optional[float] = None,
        internal_server_error_no_stall: bool = False,
    ):
        """
        Prepare
        hijack payloads for the client.
        """
        hijack_payloads = []
        for i in range(number_of_requests):
            hijack_payloads.append(
                HijackPayload(
                    number_of_requests=number_of_requests,
                    random_payload=f"random_payload_{i}" * random.randint(1, 20),
                    uuid_int=i,
                    max_batch_size=max_batch_size,
                    max_chars_per_request=max_chars_per_request,
                    send_429_until_time=send_429_until_time,
                    stall_x_many_requests=stall_x_many_requests,
                    stall_for_seconds=stall_for_seconds,
                    internal_server_error_no_stall=internal_server_error_no_stall,
                )
            )
        return hijack_payloads

    # run scenario, regular embedding
    def scenario_regular_embedding(
        number_of_requests, max_chars_per_request=None, add_429_seconds=None
    ):
        resp_429_until_time = (
            (time.time() + add_429_seconds) if add_429_seconds else None
        )
        hijack_payloads = prepare_hijack_payloads(
            number_of_requests=number_of_requests,
            max_batch_size=4,
            max_chars_per_request=max_chars_per_request,
            send_429_until_time=resp_429_until_time,
        )
        response = client.embed(
            model="text-embedding-ada-002",
            input=[hp.to_string() for hp in hijack_payloads],
            batch_size=4,
            max_concurrent_requests=64,
            max_chars_per_request=max_chars_per_request,
        )
        assert response is not None, "Response should not be None"
        assert len(response.data) == number_of_requests, (
            "Response should contain `number_of_requests` embeddings"
        )
        assert all(len(embedding.embedding) == 1 for embedding in response.data), (
            "Each embedding should be a list with one element"
        )
        indexes = [embedding.index for embedding in response.data]
        assert indexes == list(range(number_of_requests)), (
            "Indexes should match the range of number_of_requests"
        )
        # UUID match
        for i, embedding in enumerate(response.data):
            assert embedding.embedding[0] == hijack_payloads[i].uuid_int, (
                f"Embedding {i} should match UUID {hijack_payloads[i].uuid_int}"
            )
        reset_message = client.batch_post("/reset", [{}]).data[0]
        if not max_chars_per_request:
            assert reset_message["successful_requests"] == number_of_requests // 4, (
                "Successful requests should match the number of requests"
            )
        else:
            assert reset_message["successful_requests"] >= number_of_requests // 4, (
                "Successful requests should match the number of requests divided by batch size"
            )
            assert reset_message["successful_requests"] <= number_of_requests, (
                "Successful requests should not exceed the number of requests"
            )
        print(f"Scenario regular embedding with {number_of_requests} requests passed.")

    scenario_regular_embedding(12)
    scenario_regular_embedding(1000)
    scenario_regular_embedding(12, max_chars_per_request=1000)
    scenario_regular_embedding(1000, max_chars_per_request=1000)
    scenario_regular_embedding(12, add_429_seconds=0.5)
    scenario_regular_embedding(200, add_429_seconds=2)

    def scenario_stalled(
        number_of_requests: int,
        stall_x_many_requests: int,
        stall_for_seconds: float,
        internal_server_error_no_stall: bool = False,
        hedge_delay: Optional[float] = None,
        timeout: float = 360.0,
    ):
        """
        Run a scenario where the server stalls for a specified number of requests.
        """
        hijack_payloads = prepare_hijack_payloads(
            number_of_requests=number_of_requests,
            max_batch_size=1,
            stall_x_many_requests=stall_x_many_requests,
            stall_for_seconds=stall_for_seconds,
            internal_server_error_no_stall=internal_server_error_no_stall,
        )
        response = client.embed(
            model="text-embedding-ada-002",
            input=[hp.to_string() for hp in hijack_payloads],
            batch_size=1,
            max_concurrent_requests=512,
            hedge_delay=hedge_delay,
            timeout_s=timeout,
        )
        assert response is not None, "Response should not be None"
        assert len(response.data) == number_of_requests, (
            "Response should contain `number_of_requests` embeddings"
        )
        indexes = [embedding.index for embedding in response.data]
        assert indexes == list(range(number_of_requests)), (
            "Indexes should match the range of number_of_requests"
        )
        reset_message = client.batch_post("/reset", [{}]).data[0]
        assert (
            reset_message["processed_requests"]
            == number_of_requests + stall_x_many_requests
        ), "Processed requests should match the number of requests + stalled requests"
        assert reset_message["successful_requests"] == number_of_requests, (
            "Successful requests should match the number of requests"
        )
        print(f"Scenario stalled with {number_of_requests} requests passed.")

    scenario_stalled(
        100,
        stall_x_many_requests=4,
        stall_for_seconds=-1,
        internal_server_error_no_stall=True,
    )
    t = time.time()
    scenario_stalled(100, stall_x_many_requests=5, stall_for_seconds=3, hedge_delay=0.5)
    total = time.time() - t
    assert total < 1, f"Stall should be mitigated by hedge delay but took {total}s"
    time.sleep(2.5)
    scenario_stalled(
        100,
        stall_x_many_requests=3,
        stall_for_seconds=5,
        internal_server_error_no_stall=False,
        timeout=2,
    )


if __name__ == "__main__":
    # Run the server in a separate thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Keep the main thread alive
    time.sleep(0.5)  # Give the server some time to start
    run_client()
