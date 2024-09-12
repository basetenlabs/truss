

ARG PYVERSION=py39
FROM vllm/vllm-openai:v0.5.5 as truss_server

ENV PYTHON_EXECUTABLE python3


RUN grep -w 'ID=debian\|ID_LIKE=debian' /etc/os-release || { echo "ERROR: Supplied base image is not a debian image"; exit 1; }
RUN $PYTHON_EXECUTABLE -c "import sys; sys.exit(0) if sys.version_info.major == 3 and sys.version_info.minor >=8 and sys.version_info.minor <=11 else sys.exit(1)" \
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















COPY ./requirements.txt requirements.txt
RUN cat requirements.txt
RUN pip install -r requirements.txt --no-cache-dir && rm -rf /root/.cache/pip






ENV APP_HOME /app
WORKDIR $APP_HOME




# Copy data before code for better caching
COPY ./data /app/data

COPY ./server /app
COPY ./model /app/model
COPY ./config.yaml /app/config.yaml








ENV HF_TOKEN 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

ENV VLLM_LOGGING_LEVEL DEBUG



COPY ./supervisor_checks /app/supervisor_checks
COPY ./proxy.conf /etc/nginx/conf.d/proxy.conf
RUN mkdir -p /var/log/supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
# TODO: move supervisor_checks to better place
RUN pip3 install /app/supervisor_checks
ENV SUPERVISOR_SERVER_URL http://localhost:8080
ENV SERVER_START_CMD /usr/bin/supervisord
ENTRYPOINT ["/usr/bin/supervisord"]
