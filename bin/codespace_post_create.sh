#! /bin/bash
poetry install --with=dev --extras=all
poetry run pre-commit install
git lfs install
