# Builds baseten/truss-context-builder, a light-weight image that can be used
# for creating docker build context out of a Truss.
# Build that image as:
# docker buildx build . -f context_builder.Dockerfile --platform=linux/amd64 -t baseten/truss-context-builder

# This step is to install blake3
FROM python:3.9-alpine
RUN apk add cargo && pip install blake3

FROM python:3.9-alpine

# libgcc is for blake3
RUN apk add curl bash tar libgcc --no-cache
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="/root/.local/bin:${PATH}"
COPY . .

# Copy over blake3 from previous stage
COPY --from=0 /usr/local/lib/python3.9/site-packages/blake3 /usr/local/lib/python3.9/site-packages/blake3
RUN poetry install --only builder
