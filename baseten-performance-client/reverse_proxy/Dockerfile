# syntax=docker/dockerfile:1
FROM --platform=$TARGETPLATFORM debian:bookworm-slim AS chef

# Install Rust and build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    make \
    musl-tools \
    gcc \
    g++ \
    clang \
    llvm \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install cargo-chef
RUN cargo install cargo-chef

WORKDIR /app

# Planner stage - analyze dependencies
FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# Builder stage - build dependencies and app
FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json

# Build dependencies - this layer is cached unless dependencies change
RUN cargo chef cook --release --package baseten-performance-reverse-proxy --recipe-path recipe.json

# Copy source code and build the binary
COPY . .
RUN cargo build --release --package baseten-performance-reverse-proxy

# Runtime stage using Debian Bookworm
FROM --platform=$TARGETPLATFORM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    libc6 \
    python3 \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates
# Python3 is not needed, but baseten likes to have it inside the docker image.
# Create non-root user
RUN useradd -m -u 1000 proxyuser

# Set working directory
WORKDIR /app

# Copy the binary from builder stage
RUN mkdir -p /usr/local/bin
COPY --from=builder /app/target/release/baseten-performance-proxy /usr/local/bin/baseten-performance-proxy
RUN chmod +x /usr/local/bin/baseten-performance-proxy

# Change ownership to non-root user
RUN chown -R proxyuser:proxyuser /app /usr/local/bin/baseten-performance-proxy

# Switch to non-root user
USER proxyuser

# Expose port
EXPOSE 8081

# Environment variables (can be overridden at runtime)
ENV PORT=8081
ENV HTTP_VERSION=2
ENV PERFORMANCE_CLIENT_LOG_LEVEL=info

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8081/health_internal || exit 1

# Set entrypoint for the container
ENTRYPOINT ["baseten-performance-proxy"]

# Default command (can be overridden at runtime)
CMD ["--port", "8081"]

# Volume mounts for API key files (optional)
# VOLUME ["/etc/baseten", "/app/secrets"]
