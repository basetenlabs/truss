FROM nvcr.io/nvidia/tritonserver:23.07-py3

RUN pip install --upgrade pip --no-cache-dir \
    && rm -rf /root/.cache/pip

ENV PYTHONUNBUFFERED True
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y bash \
                build-essential \
                git \
                curl \
                ca-certificates \
                software-properties-common \
                nginx \
                supervisor \
                curl \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

COPY ./proxy.conf /etc/nginx/conf.d/proxy.conf
COPY ./supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY ./requirements.txt /app/requirements.txt
COPY ./system_packages.txt /app/system_packages.txt
COPY ./config.pbtxt.jinja /app/config.pbtxt.jinja

# Pydantic, PyYAML, Jinja2 are required for Triton <> Truss integration
RUN pip install pydantic pyyaml jinja2 --no-cache-dir && rm -rf /root/.cache/pip

ENV APP_HOME /app
WORKDIR $APP_HOME

# We copy into directory '1' as Triton expects the model code to be nested underneath a version directory
RUN mkdir -p /app/model/1
COPY ./model /app/model/1

RUN pip install -r requirements.txt --no-cache-dir && rm -rf /root/.cache/pip

RUN apt-get update && apt-get install --yes --no-install-recommends $(cat system_packages.txt) \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Generate config.pbtxt
COPY ./generate_config.py /app/generate_config.py
RUN python3 /app/generate_config.py

ENV SERVER_START_CMD /usr/bin/supervisord
ENTRYPOINT ["/usr/bin/supervisord"]
