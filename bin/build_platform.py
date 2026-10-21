import argparse
import json
import logging
import os
import re
import signal
import tarfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional
from urllib.parse import quote, urlencode

import httpx


class BuildPlatformError(RuntimeError):
    pass


@contextmanager
def _quiet_http_logs() -> Iterator[None]:
    loggers = [logging.getLogger(name) for name in ("httpx", "httpcore")]
    levels = [logger.level for logger in loggers]
    try:
        for logger in loggers:
            logger.setLevel(logging.CRITICAL + 1)
        yield
    finally:
        for logger, level in zip(loggers, levels):
            logger.setLevel(level)


def _request_json(
    url: str,
    *,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    payload: Optional[Dict[str, Any]] = None,
    timeout_sec: float = 60,
    max_attempts: int = 4,
) -> Dict[str, Any]:
    for attempt in range(max_attempts):
        try:
            with _quiet_http_logs():
                response = httpx.request(
                    method, url, headers=headers, json=payload, timeout=timeout_sec
                )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = f"HTTP {exc.response.status_code}"
            retryable = (
                exc.response.status_code == 429 or exc.response.status_code >= 500
            )
        except httpx.RequestError:
            detail = "connection error"
            retryable = True
        except httpx.InvalidURL:
            raise BuildPlatformError("Invalid Build Platform request URL") from None
        else:
            try:
                result = response.json()
            except ValueError:
                raise BuildPlatformError("Invalid JSON response") from None
            if not isinstance(result, dict):
                raise BuildPlatformError("Expected a JSON object")
            return result
        if method != "GET" or not retryable or attempt == max_attempts - 1:
            raise BuildPlatformError(
                f"Build Platform request failed ({detail})"
            ) from None
        time.sleep(2**attempt)
    raise AssertionError("Request retry loop exhausted")


def _authorization_header(
    *, timeout_sec: float = 60, max_attempts: int = 4
) -> Dict[str, str]:
    request_url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL")
    request_token = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    if not request_url or not request_token:
        raise BuildPlatformError(
            "Build Platform requires GitHub Actions id-token: write"
        )
    separator = "&" if "?" in request_url else "?"
    result = _request_json(
        f"{request_url}{separator}audience=build-control",
        headers={"Authorization": f"Bearer {request_token}"},
        timeout_sec=timeout_sec,
        max_attempts=max_attempts,
    )
    token = result.get("value")
    if not isinstance(token, str) or not token or "\n" in token or "\r" in token:
        raise BuildPlatformError("GitHub returned an invalid OIDC token")
    print(f"::add-mask::{token}", flush=True)
    return {"Authorization": f"Bearer {token}"}


def _api_request(
    api_base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return _request_json(
        f"{api_base_url.rstrip('/')}/{path}",
        method=method,
        headers=_authorization_header(),
        payload=payload,
    )


def cancel_build(api_base_url: str, workflow_id: str) -> None:
    print("Build Platform: requesting build cancellation", flush=True)
    try:
        _request_json(
            f"{api_base_url.rstrip('/')}/image-builds/{quote(workflow_id, safe='')}/cancel",
            method="POST",
            headers=_authorization_header(timeout_sec=2, max_attempts=1),
            payload={},
            timeout_sec=3,
            max_attempts=1,
        )
    except BuildPlatformError:
        print("Build Platform: could not cancel the build", flush=True)
    else:
        print("Build Platform: build cancellation requested", flush=True)


def cancel_pending_builds(api_base_url: str, directory: Path) -> None:
    for record in sorted(directory.glob("*.json")):
        try:
            data = json.loads(record.read_text())
            workflow_id = data["workflow_id"]
            if not isinstance(workflow_id, str) or not workflow_id:
                raise ValueError("Invalid workflow ID")
        except (OSError, ValueError, TypeError, KeyError):
            print("Build Platform: could not read a build record", flush=True)
            continue
        cancel_build(api_base_url, workflow_id)


def _validate_result(result: Dict[str, Any], repository: str, tag: str) -> None:
    digest = result.get("digest")
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise BuildPlatformError("Build Platform did not return an image digest")
    refs = result.get("final_image_refs")
    if not isinstance(refs, list) or not all(isinstance(ref, str) for ref in refs):
        raise BuildPlatformError("Build Platform did not return image references")
    image = f"{repository}:{tag}"
    if not any(
        ref in refs
        for ref in (
            image,
            f"docker.io/{image}",
            f"index.docker.io/{image}",
            f"registry-1.docker.io/{image}",
        )
    ):
        raise BuildPlatformError("Build Platform did not return the Docker Hub export")
    if not any(
        ref.endswith(f"/{image}") and ".dkr.ecr." in ref.split("/", 1)[0]
        for ref in refs
    ):
        raise BuildPlatformError("Build Platform did not return the ECR export")


def build_and_push(
    api_base_url: str,
    context_dir: Path,
    repository: str,
    tag: str,
    *,
    timeout_sec: float = 7200,
) -> None:
    started = time.monotonic()
    print(f"Build Platform: publishing {repository}:{tag}", flush=True)
    prepared = _api_request(
        api_base_url,
        "dockerfiles/prepare",
        method="POST",
        payload={
            "dockerfile": (context_dir / "Dockerfile").read_text(),
            "enable_proxy_for_build_runs": False,
        },
    )
    if (
        not isinstance(prepared.get("dockerfile"), str)
        or prepared.get("enable_proxy_for_build_runs") is not False
    ):
        raise BuildPlatformError("Invalid Dockerfile preparation response")
    (context_dir / "Dockerfile").write_bytes(prepared["dockerfile"].encode())
    archive_path = context_dir.parent / f"{context_dir.name}.tar.gz"
    try:
        with tarfile.open(archive_path, "w:gz") as archive:
            for child in sorted(context_dir.iterdir()):
                archive.add(child, arcname=child.name)
        context = _api_request(
            api_base_url, "image-build-contexts", method="POST", payload={}
        )
        if (
            not isinstance(context.get("context_id"), str)
            or not context["context_id"]
            or not isinstance(context.get("upload_url"), str)
            or not context["upload_url"].startswith("https://")
        ):
            raise BuildPlatformError("Invalid context upload response")
        try:
            with _quiet_http_logs():
                response = httpx.put(
                    context["upload_url"],
                    content=archive_path.read_bytes(),
                    timeout=300,
                )
            response.raise_for_status()
        except (httpx.HTTPError, httpx.InvalidURL):
            raise BuildPlatformError("Build Platform context upload failed") from None
    finally:
        archive_path.unlink(missing_ok=True)

    terminal = False
    workflow_id = None
    workflow_record = None
    previous_handlers = {}
    tracking_submission = False
    pending_signal = None

    def interrupted(signum, frame):
        nonlocal pending_signal
        if tracking_submission:
            pending_signal = signum
        else:
            raise SystemExit(128 + signum)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, interrupted)
        headers = _authorization_header(timeout_sec=2, max_attempts=1)
        tracking_submission = True
        build = _request_json(
            f"{api_base_url.rstrip('/')}/image-builds",
            method="POST",
            headers=headers,
            timeout_sec=3,
            max_attempts=1,
            payload={
                "context_id": context["context_id"],
                "dockerfile_path": "Dockerfile",
                "platforms": ["linux/arm64", "linux/amd64"],
                "image_format": "estargz",
                "builder_affinity_key": repository,
                "enable_proxy_for_build_runs": False,
                "exports": [
                    {
                        "destination": destination,
                        "repository": repository,
                        "tags": [tag],
                    }
                    for destination in ("ecr", "dockerhub")
                ],
            },
        )
        submitted_id = build.get("workflow_id")
        if not isinstance(submitted_id, str) or not submitted_id:
            raise BuildPlatformError("Build Platform did not return a workflow ID")
        workflow_id = submitted_id
        if directory := os.environ.get("BUILD_PLATFORM_WORKFLOW_DIR"):
            workflow_record = Path(directory) / f"{quote(workflow_id, safe='')}.json"
            workflow_record.parent.mkdir(parents=True, exist_ok=True)
            workflow_record.write_text(json.dumps({"workflow_id": workflow_id}))
        tracking_submission = False
        if pending_signal is not None:
            raise SystemExit(128 + pending_signal)
        logs_url = (
            "https://grafana.baseten.co/d/build-platform-workflow-logs?"
            + urlencode(
                {
                    "var-workflow_id": workflow_id,
                    "from": int((time.time() - 60) * 1000),
                    "to": "now",
                    "refresh": "5s",
                }
            )
        )
        print(f"Internal build logs (Grafana): {logs_url}", flush=True)
        path = f"image-builds/{quote(workflow_id, safe='')}"
        deadline = time.monotonic() + timeout_sec
        if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
            with Path(summary_path).open("a") as summary:
                summary.write(
                    f"[Internal build logs (Grafana) for {repository}:{tag}]({logs_url})\n\n"
                )
        while time.monotonic() < deadline:
            result = _api_request(api_base_url, f"{path}/wait")
            status = result.get("status")
            if status == "running":
                continue
            if status == "succeeded":
                terminal = True
                _validate_result(result, repository, tag)
                print(
                    f"Build Platform: published {repository}:{tag} in {time.monotonic() - started:.1f}s",
                    flush=True,
                )
                return
            if status in ("failed", "canceled", "terminated", "timed_out"):
                terminal = True
                raise BuildPlatformError(f"Build Platform build {status}")
            raise BuildPlatformError("Build Platform returned an unknown build status")
        raise BuildPlatformError("Timed out waiting for Build Platform publication")
    finally:
        try:
            if terminal:
                if workflow_record is not None:
                    workflow_record.unlink(missing_ok=True)
            elif workflow_id is not None:
                for signum in previous_handlers:
                    signal.signal(signum, signal.SIG_IGN)
                cancel_build(api_base_url, workflow_id)
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancel", action="store_true", required=True)
    parser.parse_args()
    cancel_pending_builds(
        os.environ["BUILD_PLATFORM_API_BASE_URL"],
        Path(os.environ["BUILD_PLATFORM_WORKFLOW_DIR"]),
    )
