"""Shared pytest configuration for trainers-server tests."""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: marks tests that require CUDA GPUs (skip with -m 'not gpu')",
    )
