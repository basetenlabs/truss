# Builds baseten/truss-context-builder, a light-weight image that can be used
# for creating docker build context out of a Truss.
# Build that image as:
# docker buildx build . -f context_builder.Dockerfile --platform=linux/amd64 -t baseten/truss-context-builder

# This step is to install blake3
FROM python:3.9-alpine
RUN apk add curl cargo
ENV PATH="/root/.local/bin:${PATH}"
RUN curl -sSL https://install.python-poetry.org | python -
COPY . .
RUN poetry add blake3 --group=builder --lock
RUN poetry install --only builder

FROM python:3.9-alpine

# libgcc is for blake3
RUN apk add curl bash tar libgcc --no-cache
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="/root/.local/bin:${PATH}"
COPY . .

COPY --from=0 /root/.cache/pypoetry/virtualenvs/truss-il7asoJj-py3.9 /root/.cache/pypoetry/virtualenvs/truss-il7asoJj-py3.9
COPY --from=0 /pyproject.toml /pyproject.toml
COPY --from=0 /poetry.lock /poetry.lock
