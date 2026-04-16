"""Shared test helpers for trainers-server tests."""

import socket

import pytest
import torch


MODEL_PATH = "/mnt/user/Qwen3-0.6B"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(s.getsockname()[1])


def gpu_count() -> int:
    try:
        return torch.cuda.device_count()
    except Exception:
        return 0


def skip_if_no_gpu(n: int = 1):
    return pytest.mark.skipif(
        gpu_count() < n,
        reason=f"requires {n} CUDA GPU(s), found {gpu_count()}",
    )
