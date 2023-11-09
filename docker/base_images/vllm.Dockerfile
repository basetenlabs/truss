# Use an official CUDA runtime as a parent image
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

# Update, upgrade, install packages and clean up
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt install --yes --no-install-recommends git wget curl bash software-properties-common nginx && \
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

RUN pip install --upgrade --no-cache-dir pip && rm -rf /root/.cache/pip
RUN pip install --upgrade --no-cache-dir torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118 && rm -rf /root/.cache/pip
RUN pip install --upgrade --no-cache-dir vllm==0.2.1.post1 pandas>=1.3 && rm -rf /root/.cache/pip
