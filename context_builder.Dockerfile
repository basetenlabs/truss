FROM python:3.9-alpine

RUN apk add curl bash --no-cache

RUN curl -sSL https://install.python-poetry.org | python -

ENV PATH="/root/.local/bin:${PATH}"
COPY . .

RUN poetry install --only builder
