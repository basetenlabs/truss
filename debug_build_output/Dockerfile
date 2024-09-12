FROM vllm/vllm-openai:v0.5.5

RUN grep -w 'ID=debian\|ID_LIKE=debian' /etc/os-release || { echo "ERROR: Supplied base image is not a debian image"; exit 1; }

EXPOSE 8080-9000

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl nginx supervisor python3-pip && \
        rm -rf /var/lib/apt/lists/*

COPY ./model /app/model

COPY ./data /app/data

COPY ./supervisor_checks /app/supervisor_checks

# Install supervisor_checks using pip
RUN pip3 install /app/supervisor_checks

RUN pip3 install boto3

ENV SUPERVISOR_SERVER_URL http://localhost:8080

COPY ./proxy.conf /etc/nginx/conf.d/proxy.conf

RUN mkdir -p /var/log/supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf


ENV HF_TOKEN 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

ENV VLLM_LOGGING_LEVEL DEBUG


ENV SERVER_START_CMD /usr/bin/supervisord

ENTRYPOINT ["/usr/bin/supervisord"]