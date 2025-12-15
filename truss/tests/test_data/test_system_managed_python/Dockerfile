FROM debian:trixie-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        build-essential \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*
