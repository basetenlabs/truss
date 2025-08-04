#! /bin/bash
uv sync --all-extras --group dev --group dev-server
uv run pre-commit install
git lfs install
