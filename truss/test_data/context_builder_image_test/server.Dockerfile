ARG PYVERSION=py39
FROM baseten/truss-server-base:3.9-v0.4.3

RUN grep -w 'ID=debian\|ID_LIKE=debian' /etc/os-release || { echo "ERROR: Supplied base image is not a debian image"; exit 1; }
RUN pythonVersion=$(echo $(command -v python >/dev/null 2>&1 && python --version || python3 --version) | cut -d" " -f2 | cut -d"." -f1,2) \
    && python -c "import sys; sys.exit(0) if sys.version_info.major == 3 and sys.version_info.minor >=8 and sys.version_info.minor <=11 else sys.exit(1)" \
    || python3 -c "import sys; sys.exit(0) if sys.version_info.major == 3 and sys.version_info.minor >=8 and sys.version_info.minor <=11 else sys.exit(1)" \
    && apt-get update && ( apt-get install -y --no-install-recommends python$pythonVersion-venv || apt-get install -y --no-install-recommends python3-venv )  \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* \
    || { echo "ERROR: Supplied base image does not have 3.8 <= python <= 3.11"; exit 1; }

RUN pip install --upgrade pip --no-cache-dir \
    && rm -rf /root/.cache/pip

# If user base image is supplied in config, apply build commands from truss base image
ENV PYTHONUNBUFFERED True
ENV DEBIAN_FRONTEND=noninteractive

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
RUN pip install -r base_server_requirements.txt --no-cache-dir && rm -rf /root/.cache/pip

RUN mkdir -p /app/bin \
    && curl https://baseten-public.s3.us-west-2.amazonaws.com/bin/b10cp-0.0.2-linux-amd64 -o /app/bin/b10cp \
    && chmod +x /app/bin/b10cp

ENV APP_HOME /app
WORKDIR $APP_HOME

COPY ./server /app
COPY ./model /app/model
COPY ./config.yaml /app/config.yaml
COPY ./data /app/data

COPY ./packages /packages

ENV INFERENCE_SERVER_PORT 8080
CMD exec python3 /app/inference_server.py
