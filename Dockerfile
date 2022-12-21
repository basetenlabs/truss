FROM python:3.9

RUN curl -sSL https://install.python-poetry.org | python -

ENV PATH="/root/.local/bin:${PATH}"
COPY . .

RUN poetry install --only main
