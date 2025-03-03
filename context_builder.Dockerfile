# Builds baseten/truss-context-builder, a light-weight image that can be used
# for creating docker build context out of a Truss.
# Build that image as:
# docker buildx build . -f context_builder.Dockerfile --platform=linux/amd64 -t baseten/truss-context-builder
FROM python:3.9-slim

RUN apt-get update \
    && apt-get install --yes --no-install-recommends curl tar \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* /tmp/library-scripts/

RUN curl -sSL https://install.python-poetry.org | python -

ENV PATH="/root/.local/bin:${PATH}"
COPY ./truss ./truss
# Chains source code is not actually needed (and deps from chains group won't be
# installed when using `--only builder`). But nonetheless poetry fails the install
# if this directory is not present/empty - so we copy it.
COPY ./truss-chains ./truss-chains
COPY ./pyproject.toml ./pyproject.toml
COPY ./poetry.lock ./poetry.lock
COPY ./README.md ./README.md

# Declare the build argument; Docker sets TARGETARCH automatically.
ARG TARGETARCH

# Add rustup's cargo to the PATH unconditionally.
ENV PATH="/root/.cargo/bin:${PATH}"

# Conditionally install rustup, update stable, and install golang for arm64
RUN if [ "$TARGETARCH" = "arm64" ]; then \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    rustup update stable && \
    apt-get update && apt-get install -y golang; \
fi

# https://python-poetry.org/docs/configuration/#virtualenvsin-project
# to write to project root .venv file to be used for context builder test
RUN poetry config virtualenvs.in-project true && poetry install --extras=all
