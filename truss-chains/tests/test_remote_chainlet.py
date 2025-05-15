import asyncio
import logging
import re
import threading
import time

import pytest

import truss_chains as chains


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
