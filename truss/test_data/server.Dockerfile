ARG PYVERSION=py39
FROM baseten/truss-server-base:3.9-v0.4.3 as truss_server

ENV PYTHON_EXECUTABLE /usr/local/bin/python3
ENV JSON_LOG True

RUN grep -w 'ID=debian\|ID_LIKE=debian' /etc/os-release || { echo "ERROR: Supplied base image is not a debian image"; exit 1; }
RUN $PYTHON_EXECUTABLE -c "import sys; sys.exit(0) if sys.version_info.major == 3 and sys.version_info.minor >=8 and sys.version_info.minor <=11 else sys.exit(1)" \
    || { echo "ERROR: Supplied base image does not have 3.8 <= python <= 3.11"; exit 1; }

RUN pip install --upgrade pip --no-cache-dir \
    && rm -rf /root/.cache/pip


# Always install the truss package
COPY ./truss/ /lib/truss_pkg/truss
COPY ./pyproject.toml /lib/truss_pkg/
COPY ./README.md /lib/truss_pkg/
RUN pip install /lib/truss_pkg --no-cache-dir && rm -rf /root/.cache/pip


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

ENV APP_HOME /app
WORKDIR $APP_HOME

# Copy data before code for better caching
COPY ./data /app/data
COPY ./model /app/model
COPY ./config.yaml /app/config.yaml

COPY ./packages /app/packages

ENV INFERENCE_SERVER_PORT 8080
ENV SERVER_START_CMD="/usr/local/bin/python3 -m truss.server.inference_server"
ENTRYPOINT ["/usr/local/bin/python3", "-m", "truss.server.inference_server"]
