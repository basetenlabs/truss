from __future__ import annotations

import asyncio
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel


def _parse_int_env(name: str, default: int, minimum: int) -> int:
    raw_value = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _parse_window_spec(spec: str, upper_bound: int) -> tuple[range, ...]:
    normalized = spec.strip()
    if normalized in {"", "*"}:
        return (range(0, upper_bound),)

    ranges: list[range] = []
    for part in normalized.split(","):
        piece = part.strip()
        if not piece:
            continue
        if "-" in piece:
            start_raw, end_raw = piece.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if not 0 <= start < upper_bound:
                raise ValueError(
                    f"window start {start} out of bounds 0..{upper_bound - 1}"
                )
            if not 0 < end <= upper_bound:
                raise ValueError(f"window end {end} out of bounds 1..{upper_bound}")
            if start >= end:
                raise ValueError(f"window range {piece!r} must satisfy start < end")
            ranges.append(range(start, end))
        else:
            value = int(piece)
            if not 0 <= value < upper_bound:
                raise ValueError(
                    f"window value {value} out of bounds 0..{upper_bound - 1}"
                )
            ranges.append(range(value, value + 1))

    if not ranges:
        raise ValueError("window spec must not be empty")
    return tuple(ranges)


def _contains(ranges: Iterable[range], value: int) -> bool:
    return any(value in rng for rng in ranges)


@dataclass(frozen=True)
class WindowConfig:
    active_seconds_into_hour: frozenset[int]

    @classmethod
    def from_env_specs(cls, minute_spec: str, second_spec: str) -> "WindowConfig":
        minute_ranges = _parse_window_spec(minute_spec, 60)
        second_ranges = _parse_window_spec(second_spec, 60)
        return cls(
            active_seconds_into_hour=frozenset(
                minute * 60 + second
                for minute in range(60)
                if _contains(minute_ranges, minute)
                for second in range(60)
                if _contains(second_ranges, second)
            )
        )

    @staticmethod
    def _seconds_into_hour(minute_utc: int, second_utc: int) -> int:
        return minute_utc * 60 + second_utc

    def is_active(self, minute_utc: int, second_utc: int) -> bool:
        return (
            self._seconds_into_hour(minute_utc, second_utc)
            in self.active_seconds_into_hour
        )

    def is_active_with_grace(
        self, minute_utc: int, second_utc: int, grace_seconds: int
    ) -> bool:
        if grace_seconds <= 0:
            return self.is_active(minute_utc, second_utc)

        seconds_into_hour = self._seconds_into_hour(minute_utc, second_utc)
        return any(
            (seconds_into_hour - delta) % 3600 in self.active_seconds_into_hour
            for delta in range(grace_seconds + 1)
        )


@dataclass(frozen=True)
class AppConfig:
    server_name: str
    serve_window: WindowConfig
    embedding_dim: int
    response_delay_ms: int
    serve_grace_period_s: int


class EmbeddingsRequest(BaseModel):
    input: list[str]
    model: str
    encoding_format: str | None = None
    dimensions: int | None = None
    user: str | None = None


def _load_config() -> AppConfig:
    server_name = os.environ.get(
        "INTEGRATION_SERVER_NAME", "integration-truss-server"
    ).strip()
    if not server_name:
        raise ValueError("INTEGRATION_SERVER_NAME must not be empty")

    embedding_dim = _parse_int_env("EMBEDDING_DIM", default=8, minimum=1)
    response_delay_ms = _parse_int_env("RESPONSE_DELAY_MS", default=0, minimum=0)

    return AppConfig(
        server_name=server_name,
        serve_window=WindowConfig.from_env_specs(
            os.environ.get("SERVE_MINUTES_UTC", "*"),
            os.environ.get("SERVE_SECONDS_UTC", "*"),
        ),
        embedding_dim=embedding_dim,
        response_delay_ms=response_delay_ms,
        serve_grace_period_s=_parse_int_env(
            "SERVE_GRACE_PERIOD_S", default=3, minimum=0
        ),
    )


def _embedding_for_text(text: str, dim: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    for index in range(dim):
        byte_value = digest[index % len(digest)]
        values.append(round(byte_value / 255.0, 6))
    return values


def _now_utc() -> tuple[int, int]:
    now = datetime.now(timezone.utc)
    return now.minute, now.second


def _apply_debug_headers(
    *,
    config: AppConfig,
    minute_utc: int,
    second_utc: int,
    health_active: bool,
    serve_active: bool,
    serve_accepting: bool,
) -> dict[str, str]:
    return {
        "x-integration-server-name": config.server_name,
        "x-integration-server-minute-utc": str(minute_utc),
        "x-integration-server-second-utc": str(second_utc),
        "x-integration-health-active": str(health_active).lower(),
        "x-integration-serve-active": str(serve_active).lower(),
        "x-integration-serve-accepting": str(serve_accepting).lower(),
    }


CONFIG = _load_config()
app = FastAPI(title="Integration Truss Server")


@app.get("/always_healthy")
async def always_healthy() -> dict[str, object]:
    return {"status": "healthy", "server_name": CONFIG.server_name}


@app.get("/health")
@app.get("/true_health")
async def health(response: Response) -> dict[str, object]:
    minute_utc, second_utc = _now_utc()
    serve_active = CONFIG.serve_window.is_active(minute_utc, second_utc)
    health_active = serve_active
    serve_accepting = CONFIG.serve_window.is_active_with_grace(
        minute_utc, second_utc, CONFIG.serve_grace_period_s
    )

    response.headers.update(
        _apply_debug_headers(
            config=CONFIG,
            minute_utc=minute_utc,
            second_utc=second_utc,
            health_active=health_active,
            serve_active=serve_active,
            serve_accepting=serve_accepting,
        )
    )

    if not health_active:
        response.status_code = 503
        return {
            "status": "unhealthy",
            "server_name": CONFIG.server_name,
            "minute_utc": minute_utc,
            "second_utc": second_utc,
            "serve_active": serve_active,
            "serve_accepting": serve_accepting,
        }

    return {
        "status": "healthy",
        "server_name": CONFIG.server_name,
        "minute_utc": minute_utc,
        "second_utc": second_utc,
        "serve_active": serve_active,
        "serve_accepting": serve_accepting,
    }


@app.post("/v1/embeddings", response_model=None)
async def embeddings(
    request: EmbeddingsRequest, response: Response
) -> dict[str, object] | JSONResponse:
    minute_utc, second_utc = _now_utc()
    serve_active = CONFIG.serve_window.is_active(minute_utc, second_utc)
    health_active = serve_active
    serve_accepting = CONFIG.serve_window.is_active_with_grace(
        minute_utc, second_utc, CONFIG.serve_grace_period_s
    )

    debug_headers = _apply_debug_headers(
        config=CONFIG,
        minute_utc=minute_utc,
        second_utc=second_utc,
        health_active=health_active,
        serve_active=serve_active,
        serve_accepting=serve_accepting,
    )
    response.headers.update(debug_headers)

    if CONFIG.response_delay_ms > 0:
        await asyncio.sleep(CONFIG.response_delay_ms / 1000.0)

    if not serve_accepting:
        return JSONResponse(
            status_code=400,
            headers=debug_headers,
            content={
                "detail": {
                    "message": "You have violated the health check protocol.",
                    "server_name": CONFIG.server_name,
                    "minute_utc": minute_utc,
                    "second_utc": second_utc,
                    "health_active": health_active,
                    "serve_active": serve_active,
                    "serve_accepting": serve_accepting,
                }
            },
        )

    dimensions = request.dimensions or CONFIG.embedding_dim
    data = []
    for index, text in enumerate(request.input):
        data.append(
            {
                "object": "embedding",
                "embedding": _embedding_for_text(text, dimensions),
                "index": index,
            }
        )

    return {
        "object": "list",
        "data": data,
        "model": request.model,
        "usage": {
            "prompt_tokens": len(request.input),
            "total_tokens": len(request.input),
        },
    }
