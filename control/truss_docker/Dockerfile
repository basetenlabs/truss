ARG PYVERSION=py39



FROM baseten/baseten-server-base-$PYVERSION:latest



COPY ./system_packages.txt system_packages.txt
RUN apt-get update && apt-get install --yes --no-install-recommends $(cat system_packages.txt) \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY ./server_requirements.txt server_requirements.txt
RUN pip install -r server_requirements.txt

COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt

ENV PORT 8080

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY ./server /app



COPY ./model /app/model

COPY ./data /app/data


COPY ./packages /packages

COPY ./config.yaml /app/config.yaml




CMD exec python3 /app/inference_server.py
