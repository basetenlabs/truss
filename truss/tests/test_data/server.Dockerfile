ARG PYVERSION=py39
FROM baseten/truss-server-base:3.9-v0.4.3 AS truss_server
ENV PYTHON_EXECUTABLE="/usr/local/bin/python3"
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
    curl -LsSf https://astral.sh/uv/0.7.19/install.sh | sh >/dev/null 2>&1; \
fi
ENV PATH="/root/.local/bin:$PATH"
ENV PYTHONUNBUFFERED="True"
ENV DEBIAN_FRONTEND="noninteractive"
RUN apt update && \
    apt install -y bash \
                build-essential \
                git \
                curl \
                ca-certificates \
                software-properties-common \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
COPY ./base_server_requirements.txt base_server_requirements.txt
RUN uv pip install --python /usr/local/bin/python3 -r base_server_requirements.txt --no-cache-dir
COPY ./requirements.txt requirements.txt
RUN uv pip install --python /usr/local/bin/python3 -r requirements.txt --no-cache-dir
ENV APP_HOME="/app"
WORKDIR $APP_HOME
COPY ./data /app/data
COPY ./server /app
COPY ./config.yaml /app/config.yaml
COPY ./model /app/model
COPY ./packages /packages
ENV INFERENCE_SERVER_PORT="8080"
ENV SERVER_START_CMD="/usr/local/bin/python3 /app/main.py"
ENTRYPOINT ["/usr/local/bin/python3", "/app/main.py"]
