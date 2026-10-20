import json
import subprocess
import sys
import tarfile
from pathlib import Path

import httpx
import pytest

from bin import generate_base_images
from truss.base.constants import SUPPORTED_PYTHON_VERSIONS
from truss.contexts.image_builder import build_platform

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
        if request.url.path.endswith("/dockerfiles/prepare"):
            data = json.loads(request.content)
            return httpx.Response(
                200, json={**data, "dockerfile": "FROM prepared:latest\n"}
            )
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


def test_publish_context_and_dual_exports(tmp_path, platform_api, capsys):
    calls, uploads, _ = platform_api
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "Dockerfile").write_text("FROM python:3.13\n")
    (context_dir / "requirements.txt").write_text("example==1.0\n")
    build_platform.build_and_push(API_URL, context_dir, REPOSITORY, TAG)
    archive_path = tmp_path / "uploaded.tar.gz"
    archive_path.write_bytes(uploads[0])
    with tarfile.open(archive_path) as archive:
        assert set(archive.getnames()) == {"Dockerfile", "requirements.txt"}
        assert archive.extractfile("Dockerfile").read() == b"FROM prepared:latest\n"
        assert archive.extractfile("requirements.txt").read() == b"example==1.0\n"
    request = next(call for call in calls if call.url.path.endswith("/image-builds"))
    assert json.loads(request.content) == {
        "context_id": "00000000-0000-4000-8000-000000000001",
        "dockerfile_path": "Dockerfile",
        "platforms": ["linux/arm64", "linux/amd64"],
        "image_format": "estargz",
        "enable_proxy_for_build_runs": True,
        "exports": [
            {"destination": "ecr", "repository": REPOSITORY, "tags": [TAG]},
            {"destination": "dockerhub", "repository": REPOSITORY, "tags": [TAG]},
        ],
    }
    assert not any(call.url.path.endswith("/cancel") for call in calls)
    output = capsys.readouterr().out
    assert "published baseten/truss-server-base" in output
    assert API_URL not in output
    assert ECR_REF not in output
    assert "signature=private" not in output


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
