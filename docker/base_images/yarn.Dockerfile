# Use an official CUDA runtime as a parent image
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

# Update, upgrade, install packages and clean up
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt install -y --no-install-recommends git wget curl bash software-properties-common nginx && \
    apt install python3.10-dev python3.10-venv -y --no-install-recommends && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen


# Set up Python and pip
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py

# Install vllm and pandas (dependency of Ray)
RUN pip install --upgrade --no-cache-dir pip
RUN git clone https://github.com/jquesnelle/yarn && cd yarn && pip install -e .
RUN MAX_JOBS=4 pip install flash-attn --no-build-isolation --no-cache-dir
