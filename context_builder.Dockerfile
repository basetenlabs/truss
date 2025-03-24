# Builds baseten/truss-context-builder, a light-weight image that can be used
# for creating docker build context out of a Truss.
# Build that image as:
# docker buildx build . -f context_builder.Dockerfile --platform=linux/amd64 -t baseten/truss-context-builder
FROM python:3.9-slim AS builder

RUN apt-get update \
    && apt-get install --yes --no-install-recommends curl build-essential golang-go \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* /tmp/library-scripts/

RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="/root/.local/bin:${PATH}"

# NB(nikhil): We need to explicitly install Rust and golang-go on arm64 architectures
# for certain truss dependencies (blake3, dockerfile) that don't have prebuilt wheels.
# This is a multi stage image, so the extra tools don't bloat the final image.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup update stable && rustup default stable

# Will only exist for arm64 builds, but is a no-op otherwise.
ENV PATH="/root/.cargo/bin:${PATH}"

# Chains/train source code is not actually needed (and deps from chains group won't be
# installed when using `--only builder`). But nonetheless poetry fails the install
# if this directory is not present/empty - so we copy it.
WORKDIR /app
COPY ./truss ./truss
COPY ./truss-chains ./truss-chains
COPY ./truss-train ./truss-train
COPY ./pyproject.toml ./pyproject.toml
COPY ./poetry.lock ./poetry.lock
COPY ./README.md ./README.md

# https://python-poetry.org/docs/configuration/#virtualenvsin-project
# to write to project root .venv file to be used for context builder test
RUN poetry config virtualenvs.in-project true && poetry install --extras=all

FROM python:3.9-slim

WORKDIR /app
COPY --from=builder /app /app

# Copy `poetry` and required files
COPY --from=builder /root/.local /root/.local

ENV PATH="/root/.local/bin:${PATH}"
ENV ENGINE_BUILDER_TRUSS_RUNTIME_MIGRATION="True"

RUN apt-get update && apt-get install --yes --no-install-recommends curl tar
