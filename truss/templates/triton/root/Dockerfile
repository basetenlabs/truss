FROM nvcr.io/nvidia/tritonserver:23.07-py3

ENV PYTHONUNBUFFERED True
ENV DEBIAN_FRONTEND=noninteractive
ENV APP_HOME /app

WORKDIR $APP_HOME

RUN pip install --upgrade pip --no-cache-dir \
    && pip install pydantic pyyaml jinja2 --no-cache-dir \
    && rm -rf /root/.cache/pip

# Install fixed system dependencies
RUN apt update && \
    apt install -y bash \
                build-essential \
                git \
                curl \
                ca-certificates \
                software-properties-common \
                nginx \
                supervisor \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Copy system_packages.txt and install them
COPY ./system_packages.txt /app/system_packages.txt
RUN apt-get update && apt-get install --yes --no-install-recommends $(cat system_packages.txt) \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Copy model and other static files
COPY ./model /app/model/1
COPY ./generate_config.py /app/generate_config.py
COPY ./proxy.conf /etc/nginx/conf.d/proxy.conf
COPY ./supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY ./config.pbtxt.jinja /app/config.pbtxt.jinja

# Copy requirements.txt and install Python packages
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt --no-cache-dir && rm -rf /root/.cache/pip

# Generate config.pbtxt
RUN python3 /app/generate_config.py

ENV SERVER_START_CMD /usr/bin/supervisord
ENTRYPOINT ["/usr/bin/supervisord"]
