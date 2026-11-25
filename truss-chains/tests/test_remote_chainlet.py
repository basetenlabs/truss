import asyncio
import contextlib
import logging
import re
import sys
import threading
import time

import pytest
from aiohttp import ClientPayloadError, web

import truss_chains as chains
from truss_chains.remote_chainlet import stub as stub_module


@pytest.fixture
def stub_session(event_loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(event_loop)
    stub = chains.StubBase(
        service_descriptor=chains.DeployedServiceDescriptor(
            name="TestChainlet",
            internal_url=None,
            predict_url="dummy-URL",
            options=chains.RPCOptions(concurrency_limit=2),
            display_name="TestChainlet",
        ),
        api_key="dummy-API-key",
    )
    stub._sync_semaphore_wrapper._log_interval_sec = 0.1
    stub._async_semaphore_wrapper._log_interval_sec = 0.1
    return stub


def test_waiting_logging_sync(stub_session, caplog):
    def use_sync_client():
        with stub_session._client_sync():
            time.sleep(0.2)

    threads = [threading.Thread(target=use_sync_client) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    time.sleep(0.3)
    with stub_session._client_sync():
        pass

    logs = [r.message for r in caplog.records]
    print("Captured logs:", logs)

    pattern = re.compile(
        r"Queueing calls to `(?P<name>[^`]+)` Chainlet\. "
        r"Momentarily there are (?P<ongoing>\d+) ongoing requests and (?P<waiting_now>\d+) waiting requests\.\n"
        r"Wait stats: p50=(?P<p50>0\.\d{3})s, p90=(?P<p90>0\.\d{3})s\.\n"
        r"Of the last (?P<total>\d+) requests, (?P<waiting>\d+) had to wait\."
    )

    matching_logs = [pattern.search(m) for m in logs if "Queueing calls to" in m]
    matching_logs = [m for m in matching_logs if m]

    assert matching_logs, "Expected logs in expected format. Logs:\n" + "\n".join(logs)

    for match in matching_logs:
        ongoing = int(match.group("ongoing"))
        waiting_now = int(match.group("waiting_now"))
        waiting_total = int(match.group("waiting"))
        total = int(match.group("total"))
        p50 = float(match.group("p50"))
        p90 = float(match.group("p90"))

        assert waiting_total >= 1, (
            f"Expected at least one waiting request, got {waiting_total}"
        )
        assert total >= 3, f"Expected at least 3 requests sampled, got {total}"
        assert 0.0 <= p50 <= p90, f"Invalid p50/p90 values: p50={p50}, p90={p90}"
        assert (
            ongoing + waiting_now
            <= stub_session._sync_semaphore_wrapper._concurrency_limit
            + (total - waiting_total)
        ), (
            f"Inconsistent live state: {ongoing} running + {waiting_now} waiting_now "
            f"vs {waiting_total} waiting of {total} requests"
        )


@pytest.mark.asyncio
async def test_waiting_logging_async(stub_session, caplog):
    async def use_async_client():
        async with stub_session._client_async():
            await asyncio.sleep(0.2)

    await asyncio.gather(*(use_async_client() for _ in range(4)))
    await asyncio.sleep(0.3)
    async with stub_session._client_async():
        pass

    logs = [r.message for r in caplog.records]
    print("Captured logs (async):", logs)

    pattern = re.compile(
        r"Queueing calls to `(?P<name>[^`]+)` Chainlet\. "
        r"Momentarily there are (?P<ongoing>\d+) ongoing requests and (?P<waiting_now>\d+) waiting requests\.\n"
        r"Wait stats: p50=(?P<p50>0\.\d{3})s, p90=(?P<p90>0\.\d{3})s\.\n"
        r"Of the last (?P<total>\d+) requests, (?P<waiting>\d+) had to wait\."
    )

    matching_logs = [pattern.search(m) for m in logs if "Queueing calls to" in m]
    matching_logs = [m for m in matching_logs if m]

    assert matching_logs, "Expected logs in expected format. Logs:\n" + "\n".join(logs)

    for match in matching_logs:
        ongoing = int(match.group("ongoing"))
        waiting_now = int(match.group("waiting_now"))
        waiting_total = int(match.group("waiting"))
        total = int(match.group("total"))
        p50 = float(match.group("p50"))
        p90 = float(match.group("p90"))

        assert waiting_total >= 1, (
            f"Expected at least one waiting request, got {waiting_total}"
        )
        assert total >= 3, f"Expected at least 3 requests sampled, got {total}"
        assert 0.0 <= p50 <= p90, f"Invalid p50/p90 values: p50={p50}, p90={p90}"
        assert (
            ongoing + waiting_now
            <= stub_session._sync_semaphore_wrapper._concurrency_limit
            + (total - waiting_total)
        ), (
            f"Inconsistent live state: {ongoing} running + {waiting_now} waiting_now "
            f"vs {waiting_total} waiting of {total} requests"
        )


def test_no_waiting_logging_sync(stub_session, caplog):
    caplog.set_level(logging.DEBUG)

    for _ in range(2):
        with stub_session._client_sync():
            time.sleep(0.01)

    time.sleep(0.2)
    with stub_session._client_sync():
        pass

    logs = [r.message for r in caplog.records]
    assert any("No queueing" in m for m in logs), logs


@pytest.mark.asyncio
async def test_no_waiting_logging_async(stub_session, caplog):
    caplog.set_level(logging.DEBUG)

    async def use():
        async with stub_session._client_async():
            await asyncio.sleep(0.01)

    await asyncio.gather(use(), use())
    await asyncio.sleep(0.2)
    async with stub_session._client_async():
        pass

    logs = [r.message for r in caplog.records]

    assert any("No queueing" in m for m in logs), logs


@contextlib.asynccontextmanager
async def _serve_stub(handler, **options):
    app = web.Application()
    app.router.add_post("/", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    stub = chains.StubBase.from_url(
        f"http://127.0.0.1:{runner.addresses[0][1]}/",
        "dummy-API-key",
        chains.RPCOptions(**options),
    )
    try:
        yield stub
    finally:
        if stub._cached_async_client:
            await stub._cached_async_client[0].close()
        await stub.shut_down()
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("rollover", [False, True], ids=["no-rollover", "rollover"])
async def test_predictions_survive_session_rollover(rollover):
    inflight = 2
    body_started = asyncio.Event()
    finish_body = asyncio.Event()
    requests = 0
    bodies_started = 0

    async def predict(request):
        nonlocal requests, bodies_started
        requests += 1
        first_wave = requests <= inflight
        if first_wave:
            response = web.StreamResponse(headers={"Content-Type": "application/json"})
            await response.prepare(request)
            await response.write(b"tr")
            bodies_started += 1
            if bodies_started == inflight:
                body_started.set()
            await finish_body.wait()
            await response.write(b"ue")
            return response
        return web.json_response(True)

    async with _serve_stub(
        predict, retries=1, timeout_sec=5, concurrency_limit=2 * inflight
    ) as stub:
        first = [asyncio.create_task(stub.predict_async({})) for _ in range(inflight)]
        try:
            await asyncio.wait_for(body_started.wait(), timeout=5)
            cached = stub._cached_async_client
            old_client = cached[0]
            if rollover:
                expired = (old_client, cached[1] - stub._client_cycle_time_sec - 1)
                stub._cached_async_client = expired
            second = await asyncio.gather(
                *(stub.predict_async({}) for _ in range(inflight)),
                return_exceptions=True,
            )
            finish_body.set()
            results = await asyncio.gather(*first, return_exceptions=True)
            assert results + second == [True] * (2 * inflight)
            await stub.shut_down()
            assert old_client.closed == rollover
        finally:
            finish_body.set()
            await asyncio.gather(*first, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_prediction_releases_retired_session():
    body_started = asyncio.Event()
    finish_body = asyncio.Event()

    async def predict(request):
        if (await request.json()).get("hold"):
            response = web.StreamResponse(headers={"Content-Type": "application/json"})
            await response.prepare(request)
            await response.write(b"tr")
            body_started.set()
            await finish_body.wait()
            return response
        return web.json_response(True)

    async with _serve_stub(predict, retries=1, timeout_sec=5) as stub:
        task = asyncio.create_task(stub.predict_async({"hold": True}))
        try:
            await asyncio.wait_for(body_started.wait(), timeout=5)
            old_client = stub._cached_async_client[0]
            stub._client_cycle_time_sec = -1
            assert await stub.predict_async({}) is True
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await stub.shut_down()
            assert old_client.closed
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            finish_body.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_queued", [False, True], ids=["complete", "cancel"])
async def test_queued_predictions_survive_session_rollover(cancel_queued):
    body_started = asyncio.Event()
    finish_body = asyncio.Event()

    async def predict(request):
        if (await request.json()).get("hold"):
            response = web.StreamResponse(headers={"Content-Type": "application/json"})
            await response.prepare(request)
            await response.write(b"tr")
            body_started.set()
            await finish_body.wait()
            await response.write(b"ue")
            return response
        return web.json_response(True)

    async with _serve_stub(
        predict, retries=1, timeout_sec=5, concurrency_limit=1
    ) as stub:

        async def wait_for_queue(count):
            while stub._async_semaphore_wrapper.queued_requests < count:
                await asyncio.sleep(0)

        first = asyncio.create_task(stub.predict_async({"hold": True}))
        tasks = [first]
        try:
            await asyncio.wait_for(body_started.wait(), timeout=5)
            queued = asyncio.create_task(stub.predict_async({}))
            tasks.append(queued)
            await asyncio.wait_for(wait_for_queue(1), timeout=5)
            old_client, timestamp = stub._cached_async_client
            stub._cached_async_client = (old_client, timestamp - 3601)
            third = asyncio.create_task(stub.predict_async({}))
            tasks.append(third)
            await asyncio.wait_for(wait_for_queue(2), timeout=5)
            if cancel_queued:
                queued.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await queued
            await stub.shut_down()
            assert not old_client.closed
            finish_body.set()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            expected = [True, True, True]
            if cancel_queued:
                assert isinstance(results[1], asyncio.CancelledError)
                expected[1] = results[1]
            assert results == expected
            await stub.shut_down()
            assert old_client.closed
        finally:
            finish_body.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_stream_remains_readable_after_session_rollover():
    finish_body = asyncio.Event()

    async def predict(request):
        if (await request.json()).get("stream"):
            response = web.StreamResponse()
            await response.prepare(request)
            await response.write(b"first")
            await finish_body.wait()
            await response.write(b"last")
            return response
        return web.json_response(True)

    async with _serve_stub(predict) as stub:
        stream = await stub.predict_async_stream({"stream": True})
        old_client, timestamp = stub._cached_async_client
        try:
            stub._cached_async_client = (old_client, timestamp - 3601)
            await stub.predict_async({})
            finish_body.set()
            assert b"".join([chunk async for chunk in stream]) == b"firstlast"
            await stub.shut_down()
            assert old_client.closed
        finally:
            finish_body.set()
            await old_client.close()


@pytest.mark.asyncio
async def test_failed_stream_releases_retired_session():
    finish_body = asyncio.Event()
    frames = []

    async def predict(request):
        if (await request.json()).get("stream"):
            response = web.StreamResponse(headers={"Content-Length": "5"})
            await response.prepare(request)
            await response.write(b"abc")
            await finish_body.wait()
            request.transport.close()
            return response
        return web.json_response(True)

    def retain_rpc_frame(frame, event, value):
        if (
            event == "return"
            and frame.f_code.co_filename == stub_module.__file__
            and frame.f_code.co_name == "_rpc"
            and hasattr(value, "__anext__")
        ):
            frames.append(frame)
        return retain_rpc_frame

    async with _serve_stub(predict) as stub:
        previous_trace = sys.gettrace()
        try:
            # Profilers can retain the response after the RPC has returned.
            sys.settrace(retain_rpc_frame)
            try:
                stream = await stub.predict_async_stream({"stream": True})
            finally:
                sys.settrace(previous_trace)
            assert len(frames) == 1
            assert await stream.__anext__() == b"abc"
            old_client, timestamp = stub._cached_async_client
            stub._cached_async_client = (old_client, timestamp - 3601)
            assert await stub.predict_async({}) is True
            finish_body.set()
            with pytest.raises(ClientPayloadError):
                await stream.__anext__()
            await stub.shut_down()
            assert old_client.closed
        finally:
            finish_body.set()
            frames.clear()
