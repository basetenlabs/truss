ARG PYVERSION=py39
ARG HOME
FROM baseten/truss-server-base:3.9-v0.4.3 AS truss_server
ENV PYTHON_EXECUTABLE="/usr/local/bin/python3"
ENV HOME=${HOME:-/root}
ENV APP_HOME=/app
RUN mkdir -p ${APP_HOME} /control
RUN useradd -u 60000 -ms /bin/bash app
ENV DEBIAN_FRONTEND=noninteractive
RUN grep -w 'ID=debian\|ID_LIKE=debian' /etc/os-release || { echo "ERROR: Supplied base image is not a debian image"; exit 1; }
RUN /usr/local/bin/python3 -c "import sys; \
    sys.exit(0) \
    if sys.version_info.major == 3 \
    and sys.version_info.minor >= 8 \
    and sys.version_info.minor <= 13 \
    else sys.exit(1)" \
    || { echo "ERROR: Supplied base image does not have 3.8 <= python <= 3.13"; exit 1; }
RUN if ! command -v uv >/dev/null 2>&1; then \
    command -v curl >/dev/null 2>&1 || (apt update && apt install -y curl) && \
    curl -LsSf --retry 5 --retry-delay 5 https://astral.sh/uv/0.8.22/install.sh | sh && \
    test -x ${HOME}/.local/bin/uv; \
fi
ENV PATH=${PATH}:${HOME}/.local/bin
ENV PYTHONUNBUFFERED="True"
RUN apt update && \
    apt install -y bash build-essential git curl ca-certificates \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
COPY --chown= ./base_server_requirements.txt base_server_requirements.txt
RUN UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-300} uv pip install --index-strategy unsafe-best-match --python /usr/local/bin/python3 -r base_server_requirements.txt --no-cache-dir
COPY --chown= ./requirements.txt requirements.txt
RUN UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-300} uv pip install --index-strategy unsafe-best-match --python /usr/local/bin/python3 -r requirements.txt --no-cache-dir
WORKDIR $APP_HOME
COPY --chown= ./data ${APP_HOME}/data
COPY --chown= ./server ${APP_HOME}
COPY --chown= ./config.yaml ${APP_HOME}/config.yaml
COPY --chown= ./model ${APP_HOME}/model
COPY --chown= ./packages /packages
ENV INFERENCE_SERVER_PORT="8080"
ENV SERVER_START_CMD="/usr/local/bin/python3 /app/main.py"
ENTRYPOINT ["/usr/local/bin/python3", "/app/main.py"]
