# Builds baseten/truss-context-builder, a light-weight image that can be used
# for creating docker build context out of a Truss.
# Build that image as:
# docker buildx build . -f context_builder.Dockerfile --platform=linux/amd64 -t baseten/truss-context-builder
FROM python:3.9-slim AS builder

RUN apt-get update \
    && apt-get install --yes --no-install-recommends curl \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* /tmp/library-scripts/

RUN curl -sSLf --retry 5 --retry-delay 5 https://astral.sh/uv/0.7.12/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Chains/train source code is not actually needed (and deps from chains group won't be
# installed when using `--only builder`). But nonetheless `uv` fails the install
# if this directory is not present/empty - so we copy it.
WORKDIR /app
COPY ./truss ./truss
COPY ./truss-chains ./truss-chains
COPY ./truss-train ./truss-train
COPY ./pyproject.toml ./pyproject.toml
COPY ./uv.lock ./uv.lock
COPY ./README.md ./README.md

RUN uv sync

FROM python:3.9-slim

WORKDIR /app
COPY --from=builder /app /app

# Copy `uv` and required files
COPY --from=builder /root/.local /root/.local

ENV PATH="/root/.local/bin:${PATH}"
ENV ENGINE_BUILDER_TRUSS_RUNTIME_MIGRATION="True"

RUN apt-get update && apt-get install --yes --no-install-recommends curl tar
