import json
import logging
import signal
import subprocess
import sys
import tarfile
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from bin import build_platform, generate_base_images
from truss.base.constants import SUPPORTED_PYTHON_VERSIONS

API_URL = "https://build.example.invalid/api/internal"
REPOSITORY = "baseten/truss-server-base"
TAG = "3.13-gpu-v0.1.0"
DIGEST = "sha256:" + "a" * 64
ECR_REF = f"000000000000.dkr.ecr.us-east-1.amazonaws.com/{REPOSITORY}:{TAG}"
RESULT = {
    "status": "succeeded",
    "digest": DIGEST,
    "final_image_refs": [f"docker.io/{REPOSITORY}:{TAG}", ECR_REF],
}


@pytest.fixture
def platform_api(monkeypatch):
    calls = []
    uploads = []
    statuses = [{"status": "running"}, RESULT]

    def handle(request):
        calls.append(request)
        if request.url.host == "oidc.example.invalid":
            assert request.headers["Authorization"] == "Bearer request-token"
            assert request.url.params["audience"] == "build-control"
            return httpx.Response(200, json={"value": "oidc-token"})
        if request.url.host == "upload.example.invalid":
            assert "Authorization" not in request.headers
            uploads.append(request.content)
            return httpx.Response(200)
        assert request.headers["Authorization"] == "Bearer oidc-token"
        if request.url.path.endswith("/image-build-contexts"):
            return httpx.Response(
                200,
                json={
                    "context_id": "00000000-0000-4000-8000-000000000001",
                    "upload_url": "https://upload.example.invalid/context?signature=private",
                },
            )
        if request.url.path.endswith("/image-builds"):
            return httpx.Response(
                202, json={"workflow_id": "build-1", "status": "running"}
            )
        if request.url.path.endswith("/wait"):
            return httpx.Response(200, json=statuses.pop(0))
        if request.url.path.endswith("/cancel"):
            return httpx.Response(202, json={"status": "cancellation_requested"})
        raise AssertionError(f"Unexpected API path: {request.url.path}")

    client = httpx.Client(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(build_platform.httpx, "request", client.request)
    monkeypatch.setattr(build_platform.httpx, "put", client.put)
    monkeypatch.setenv(
        "ACTIONS_ID_TOKEN_REQUEST_URL", "https://oidc.example.invalid/token?existing=1"
    )
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "request-token")
    yield calls, uploads, statuses
    client.close()


def test_publish_context_and_dual_exports(
    tmp_path, platform_api, capsys, caplog, monkeypatch
):
    calls, uploads, _ = platform_api
    records = tmp_path / "workflows"
    monkeypatch.setenv("BUILD_PLATFORM_WORKFLOW_DIR", str(records))
    caplog.set_level(logging.DEBUG, logger="httpx")
    caplog.set_level(logging.DEBUG, logger="httpcore")
    summary_path = tmp_path / "summary.md"
    summary_path.write_text("Existing summary\n")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_path))
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    dockerfile = b"FROM python:3.13\r\n"
    (context_dir / "Dockerfile").write_bytes(dockerfile)
    (context_dir / "requirements.txt").write_bytes(b"example==1.0\n")
    build_platform.build_and_push(API_URL, context_dir, REPOSITORY, TAG)
    archive_path = tmp_path / "uploaded.tar.gz"
    archive_path.write_bytes(uploads[0])
    with tarfile.open(archive_path) as archive:
        assert set(archive.getnames()) == {"Dockerfile", "requirements.txt"}
        assert archive.extractfile("Dockerfile").read() == dockerfile
        assert archive.extractfile("requirements.txt").read() == b"example==1.0\n"
    assert (context_dir / "Dockerfile").read_bytes() == dockerfile
    request = next(call for call in calls if call.url.path.endswith("/image-builds"))
    assert json.loads(request.content) == {
        "context_id": "00000000-0000-4000-8000-000000000001",
        "dockerfile_path": "Dockerfile",
        "platforms": ["linux/arm64", "linux/amd64"],
        "image_format": "estargz",
        "builder_affinity_key": REPOSITORY,
        "enable_proxy_for_build_runs": False,
        "exports": [
            {"destination": "ecr", "repository": REPOSITORY, "tags": [TAG]},
            {"destination": "dockerhub", "repository": REPOSITORY, "tags": [TAG]},
        ],
    }
    assert not any(call.url.path.endswith("/cancel") for call in calls)
    assert list(records.iterdir()) == []
    output = capsys.readouterr().out
    assert "published baseten/truss-server-base" in output
    assert API_URL not in output
    assert ECR_REF not in output
    assert "signature=private" not in output
    logs_url = next(
        line.removeprefix("Internal build logs (Grafana): ")
        for line in output.splitlines()
        if line.startswith("Internal build logs (Grafana): ")
    )
    parsed = urlsplit(logs_url)
    assert parsed.netloc == "grafana.baseten.co"
    assert parsed.path == "/d/build-platform-workflow-logs"
    query = parse_qs(parsed.query)
    assert query["var-workflow_id"] == ["build-1"]
    assert query["to"] == ["now"]
    assert query["refresh"] == ["5s"]
    assert int(query["from"][0]) > 0
    assert (
        summary_path.read_text()
        == f"Existing summary\n[Internal build logs (Grafana) for {REPOSITORY}:{TAG}]({logs_url})\n\n"
    )
    assert not any(
        record.name.startswith(("httpx", "httpcore")) for record in caplog.records
    )
    assert logging.getLogger("httpx").level == logging.DEBUG
    assert logging.getLogger("httpcore").level == logging.DEBUG


@pytest.mark.parametrize(
    "status", ["failed", "canceled", "terminated", "timed_out", "unexpected"]
)
def test_failed_publication(tmp_path, platform_api, status):
    calls, _, statuses = platform_api
    statuses[:] = [{"status": status}]
    (tmp_path / "Dockerfile").write_text("FROM python:3.13\n")
    with pytest.raises(build_platform.BuildPlatformError):
        build_platform.build_and_push(API_URL, tmp_path, REPOSITORY, TAG)
    assert any(call.url.path.endswith("/cancel") for call in calls) == (
        status == "unexpected"
    )


def test_timeout_cancels_build(tmp_path, platform_api):
    calls, _, _ = platform_api
    (tmp_path / "Dockerfile").write_text("FROM python:3.13\n")
    with pytest.raises(build_platform.BuildPlatformError, match="Timed out"):
        build_platform.build_and_push(API_URL, tmp_path, REPOSITORY, TAG, timeout_sec=0)
    assert any(call.url.path.endswith("/cancel") for call in calls)


def test_upload_failure_redacted(tmp_path, platform_api, monkeypatch):
    (tmp_path / "Dockerfile").write_text("FROM python:3.13\n")

    def fail_upload(url, **kwargs):
        request = httpx.Request("PUT", url)
        raise httpx.HTTPStatusError(
            "private upload details",
            request=request,
            response=httpx.Response(403, request=request),
        )

    monkeypatch.setattr(build_platform.httpx, "put", fail_upload)
    with pytest.raises(build_platform.BuildPlatformError) as exc:
        build_platform.build_and_push(API_URL, tmp_path, REPOSITORY, TAG)
    assert str(exc.value) == "Build Platform context upload failed"
    assert not tmp_path.with_suffix(".tar.gz").exists()


def test_missing_oidc_credentials(monkeypatch):
    monkeypatch.delenv("ACTIONS_ID_TOKEN_REQUEST_URL", raising=False)
    monkeypatch.delenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", raising=False)
    with pytest.raises(build_platform.BuildPlatformError, match="id-token: write"):
        build_platform._authorization_header()


@pytest.mark.parametrize(
    "result",
    [
        {**RESULT, "digest": None},
        {**RESULT, "final_image_refs": [ECR_REF]},
        {**RESULT, "final_image_refs": [f"{REPOSITORY}:{TAG}"]},
        {**RESULT, "final_image_refs": [f"{REPOSITORY}:other", ECR_REF]},
    ],
)
def test_incomplete_publication_rejected(result):
    with pytest.raises(build_platform.BuildPlatformError):
        build_platform._validate_result(result, REPOSITORY, TAG)


@pytest.mark.parametrize("method,attempts", [("GET", 4), ("POST", 1)])
def test_request_failures_redacted(monkeypatch, method, attempts):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(503, text="private response details")

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(build_platform.httpx, "request", client.request)
        monkeypatch.setattr(build_platform.time, "sleep", lambda _: None)
        with pytest.raises(build_platform.BuildPlatformError) as exc:
            build_platform._request_json(API_URL, method=method)
    assert len(calls) == attempts
    assert str(exc.value) == "Build Platform request failed (HTTP 503)"


@pytest.mark.parametrize("python_version", SUPPORTED_PYTHON_VERSIONS)
@pytest.mark.parametrize("use_gpu", [True, False])
def test_generated_context_preserves_variants(monkeypatch, python_version, use_gpu):
    seen = []

    def publish(api_url, context_dir, repository, tag):
        seen.append(tag)
        assert api_url == API_URL
        assert repository == REPOSITORY
        assert (
            context_dir / "Dockerfile"
        ).read_text() == generate_base_images._render_dockerfile(
            python_version, use_gpu
        )
        assert (context_dir / "requirements.txt").read_bytes() == (
            generate_base_images.templates_path / "server" / "requirements.txt"
        ).read_bytes()
        assert (context_dir / "control").is_dir()

    monkeypatch.setattr(generate_base_images, "build_and_push", publish)
    monkeypatch.setattr(
        generate_base_images.subprocess,
        "run",
        lambda *a, **kw: pytest.fail("Unexpected Docker invocation"),
    )
    generate_base_images._build(
        python_version,
        use_gpu,
        push=True,
        version_tag="v0.1.0",
        dry_run=False,
        build_platform_api_url=API_URL,
    )
    assert seen == [f"{python_version}{'-gpu' if use_gpu else ''}-v0.1.0"]


def test_local_build_unchanged(monkeypatch):
    calls = []
    monkeypatch.setattr(
        generate_base_images.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd)
    )
    generate_base_images._build("3.13", push=True, dry_run=False)
    assert calls[0] == ["docker", "buildx", "create", "--use"]
    assert calls[1][:5] == [
        "docker",
        "buildx",
        "build",
        "--platform=linux/arm64,linux/amd64",
        ".",
    ]
    assert "compression=estargz" in calls[1][-1]
    assert calls[1][-1].endswith(",push=true")


def test_dry_run_does_not_publish(monkeypatch):
    monkeypatch.setattr(
        generate_base_images,
        "build_and_push",
        lambda *a: pytest.fail("Unexpected publication"),
    )
    generate_base_images._build_all(
        dry_run=True, push=True, build_platform_api_url=API_URL
    )


def test_cli_requires_push():
    result = subprocess.run(
        [sys.executable, str(Path(generate_base_images.__file__)), "--build-platform"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--build-platform requires --push" in result.stderr


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("cancel_fails", [False, True])
def test_signal_cancels_build_and_restores_handlers(
    tmp_path, platform_api, monkeypatch, signum, cancel_fails, capsys
):
    calls, _, _ = platform_api
    records = tmp_path / "workflows"
    monkeypatch.setenv("BUILD_PLATFORM_WORKFLOW_DIR", str(records))
    (tmp_path / "Dockerfile").write_text("FROM python:3.13\n")
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    original_request = build_platform.httpx.request
    cancellation_timeouts = []

    def request(method, url, **kwargs):
        if url.endswith("/wait"):
            assert json.loads((records / "build-1.json").read_text()) == {
                "workflow_id": "build-1"
            }
            signal.raise_signal(signum)
            pytest.fail("Signal did not interrupt the wait")
        if url.endswith("/cancel"):
            cancellation_timeouts.append(kwargs["timeout"])
            if cancel_fails:
                return httpx.Response(503, request=httpx.Request(method, url))
        return original_request(method, url, **kwargs)

    monkeypatch.setattr(build_platform.httpx, "request", request)
    with pytest.raises(SystemExit) as exc:
        build_platform.build_and_push(API_URL, tmp_path, REPOSITORY, TAG)
    assert exc.value.code == 128 + signum
    assert cancellation_timeouts == [3]
    assert {sig: signal.getsignal(sig) for sig in previous} == previous
    assert (records / "build-1.json").exists()
    assert any(call.url.path.endswith("/cancel") for call in calls) == (
        not cancel_fails
    )
    assert API_URL not in capsys.readouterr().out


def test_cancellation_uses_short_auth_timeout(platform_api, monkeypatch):
    original_request = build_platform.httpx.request
    timeouts = []

    def request(method, url, **kwargs):
        timeouts.append(kwargs["timeout"])
        return original_request(method, url, **kwargs)

    monkeypatch.setattr(build_platform.httpx, "request", request)
    build_platform.cancel_build(API_URL, "build-1")
    assert timeouts == [2, 3]


def test_cancellation_auth_failure_is_redacted(monkeypatch, capsys):
    attempts = []

    def request(method, url, **kwargs):
        attempts.append(kwargs["timeout"])
        raise httpx.ConnectError("private connection details")

    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_URL", API_URL)
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "request-token")
    monkeypatch.setattr(build_platform.httpx, "request", request)
    build_platform.cancel_build(API_URL, "build-1")
    assert attempts == [2]
    output = capsys.readouterr().out
    assert "could not cancel" in output
    assert "private" not in output
    assert API_URL not in output


@pytest.mark.parametrize("damaged", ["{", "[]", "{}", '{"workflow_id": null}'])
def test_cleanup_skips_damaged_records(tmp_path, platform_api, damaged):
    calls, _, _ = platform_api
    (tmp_path / "00-broken.json").write_text(damaged)
    (tmp_path / "build.json").write_text('{"workflow_id": "workflow/id"}')
    build_platform.cancel_pending_builds(API_URL, tmp_path)
    cancellations = [call for call in calls if call.url.path.endswith("/cancel")]
    assert len(cancellations) == 1
    assert cancellations[0].url.raw_path.endswith(b"/workflow%2Fid/cancel")


def test_cleanup_before_submission_is_noop(tmp_path, platform_api):
    calls, _, _ = platform_api
    build_platform.cancel_pending_builds(API_URL, tmp_path / "missing")
    assert calls == []


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("phase", ["submission", "response", "record"])
def test_submission_interruption_records_and_cancels_build(
    tmp_path, platform_api, monkeypatch, signum, phase
):
    calls, _, _ = platform_api
    records = tmp_path / "workflows"
    monkeypatch.setenv("BUILD_PLATFORM_WORKFLOW_DIR", str(records))
    (tmp_path / "Dockerfile").write_text("FROM python:3.13\n")
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    original_request = build_platform.httpx.request
    original_json = build_platform._request_json
    original_write = Path.write_text

    def request(method, url, **kwargs):
        response = original_request(method, url, **kwargs)
        if url.endswith("/image-builds"):
            assert kwargs["timeout"] == 3
            if phase == "submission":
                signal.raise_signal(signum)
        return response

    def request_json(url, **kwargs):
        response = original_json(url, **kwargs)
        if url.endswith("/image-builds") and phase == "response":
            signal.raise_signal(signum)
        return response

    def write_text(path, *args, **kwargs):
        if path.name == "build-1.json" and phase == "record":
            signal.raise_signal(signum)
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(build_platform.httpx, "request", request)
    monkeypatch.setattr(build_platform, "_request_json", request_json)
    monkeypatch.setattr(Path, "write_text", write_text)
    with pytest.raises(SystemExit) as exc:
        build_platform.build_and_push(API_URL, tmp_path, REPOSITORY, TAG)
    assert exc.value.code == 128 + signum
    assert json.loads((records / "build-1.json").read_text()) == {
        "workflow_id": "build-1"
    }
    assert sum(call.url.path.endswith("/image-builds") for call in calls) == 1
    assert sum(call.url.path.endswith("/cancel") for call in calls) == 1
    assert not any(call.url.path.endswith("/wait") for call in calls)
    assert {sig: signal.getsignal(sig) for sig in previous} == previous


def test_polling_failure_retains_record_for_cleanup_retry(
    tmp_path, platform_api, monkeypatch
):
    records = tmp_path / "workflows"
    monkeypatch.setenv("BUILD_PLATFORM_WORKFLOW_DIR", str(records))
    (tmp_path / "Dockerfile").write_text("FROM python:3.13\n")
    original_request = build_platform.httpx.request
    cancellations = []

    def request(method, url, **kwargs):
        if url.endswith("/wait"):
            raise httpx.ConnectError("private connection details")
        if url.endswith("/cancel"):
            cancellations.append(url)
            if len(cancellations) == 1:
                return httpx.Response(503, request=httpx.Request(method, url))
        return original_request(method, url, **kwargs)

    monkeypatch.setattr(build_platform.httpx, "request", request)
    monkeypatch.setattr(build_platform.time, "sleep", lambda _: None)
    with pytest.raises(build_platform.BuildPlatformError, match="connection error"):
        build_platform.build_and_push(API_URL, tmp_path, REPOSITORY, TAG)
    assert (records / "build-1.json").exists()
    build_platform.cancel_pending_builds(API_URL, records)
    assert len(cancellations) == 2


@pytest.mark.parametrize("workflow_id", [None, "", 123])
def test_invalid_submission_id_does_not_cancel(
    tmp_path, platform_api, monkeypatch, workflow_id
):
    calls, _, _ = platform_api
    (tmp_path / "Dockerfile").write_text("FROM python:3.13\n")
    original_request = build_platform.httpx.request

    def request(method, url, **kwargs):
        if url.endswith("/image-builds"):
            return httpx.Response(
                202,
                json={"workflow_id": workflow_id},
                request=httpx.Request(method, url),
            )
        return original_request(method, url, **kwargs)

    monkeypatch.setattr(build_platform.httpx, "request", request)
    with pytest.raises(build_platform.BuildPlatformError, match="workflow ID"):
        build_platform.build_and_push(API_URL, tmp_path, REPOSITORY, TAG)
    assert not any(call.url.path.endswith("/cancel") for call in calls)
