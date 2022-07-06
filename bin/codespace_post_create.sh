#! /bin/bash
poetry install
poetry run pre-commit install
git lfs install
