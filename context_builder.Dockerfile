# Builds baseten/truss-context-builder, a light-weight image that can be used
# for creating docker build context out of a Truss.
# Build that image as:
# docker buildx build . -f context_builder.Dockerfile --platform=linux/amd64 -t baseten/truss-context-builder
FROM python:3.9-alpine

RUN apk add curl bash --no-cache

RUN curl -sSL https://install.python-poetry.org | python -

ENV PATH="/root/.local/bin:${PATH}"
COPY . .

RUN poetry install --only builder
