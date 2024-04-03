# Builds baseten/truss-context-builder, a light-weight image that can be used
# for creating docker build context out of a Truss.
# Build that image as:
# docker buildx build . -f context_builder.Dockerfile --platform=linux/amd64,linux/arm64 -t baseten/truss-context-builder
FROM python:3.9-slim

RUN apt-get update \
    && apt-get install --yes --no-install-recommends curl tar \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/* /tmp/library-scripts/

RUN curl -sSL https://install.python-poetry.org | python - \
    && curl https://sh.rustup.rs -o rustup && sh ./rustup -y && rm -fr rustup
ENV PATH="/root/.local/bin:/root/.cargo/bin:${PATH}"
COPY ./truss ./truss
COPY ./pyproject.toml ./pyproject.toml
COPY ./poetry.lock ./poetry.lock
COPY ./README.md ./README.md

# https://python-poetry.org/docs/configuration/#virtualenvsin-project
# to write to project root .venv file to be used for context builder test
RUN poetry config virtualenvs.in-project true \
    && poetry install --only builder \
    && rm -fr /root/.cargo /root/.bashrc \
    && apt autoremove build-essential -y 
