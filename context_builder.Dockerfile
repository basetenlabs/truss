# Builds baseten/truss-context-builder, a light-weight image that can be used
# for creating docker build context out of a Truss.
# In order to use this Dockerfile, the poetry.lock file needs to be generated if pyproject.toml or build step changed
####
# docker run -it -v `pwd`/truss:/truss -v `pwd`/pyproject.toml:/pyproject.toml -v `pwd`/README.md:/README.md -v `pwd`/lock:/lock python:3.9-slim bash
# follow the steps to install packages in the container
# If everything succeeded, run poetry lock to generate the lock, copy it to lock
# exit container, copy content of lock to poetry.lock then try the build below
# Build that image as:
# docker buildx build . -f context_builder.Dockerfile --platform=linux/amd64,linux/arm64 -t baseten/truss-context-builder
FROM python:3.9-slim

RUN apt-get update \
    && apt-get install --yes --no-install-recommends curl tar \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/* /tmp/library-scripts/

# both build-essential and rustup are needed to build some python
# packages on arm64 while amd64 versions are pre-built
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
