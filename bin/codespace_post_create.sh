#! /bin/bash
poetry install --with=dev,dev-server --extras=all
poetry run pre-commit install
git lfs install
