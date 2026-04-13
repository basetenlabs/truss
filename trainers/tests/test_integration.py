"""Smoke test: verify we can reach the live training request manager."""

import os

import pytest

from trainers.queue_client import QueueClient


BASE_URL = os.environ.get("TRM_BASE_URL", "https://trm.demo.api.baseten.co")
API_KEY = os.environ.get("TRM_API_KEY", "")


@pytest.mark.integration
def test_health():
    client = QueueClient(BASE_URL, api_key=API_KEY)
    client.health()  # raises on non-200
    client.close()
