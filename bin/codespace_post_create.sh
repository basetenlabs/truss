#! /bin/bash
uv sync
uv run pre-commit install
git lfs install
