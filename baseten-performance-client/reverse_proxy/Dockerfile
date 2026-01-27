# syntax=docker/dockerfile:1
FROM --platform=$BUILDPLATFORM rust:1.91-slim AS builder

# Install build dependencies for both x86_64 and aarch64
# These are needed for ring/crypto/reqwest compilation
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    perl \
    make \
    musl-tools \
    gcc \
    g++ \
    clang \
    llvm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Cargo files
COPY Cargo.toml Cargo.lock ./
COPY core/Cargo.toml ./core/
COPY reverse_proxy/Cargo.toml ./reverse_proxy/
COPY python_bindings/Cargo.toml ./python_bindings/
COPY node_bindings/Cargo.toml ./node_bindings/

# Copy the actual source code
COPY core/src ./core/src/
COPY reverse_proxy/src ./reverse_proxy/src/
COPY python_bindings/src ./python_bindings/src/
COPY node_bindings/src ./node_bindings/src/

# Set target architecture (defaults to build platform architecture)
ARG TARGETARCH

# Determine the Rust target triple based on target architecture
RUN case "$TARGETARCH" in \
    "amd64") echo "x86_64-unknown-linux-musl" > /tmp/rust_target ;; \
    "arm64") echo "aarch64-unknown-linux-musl" > /tmp/rust_target ;; \
    *) echo "Unsupported architecture: $TARGETARCH" && exit 1 ;; \
    esac

# Build with platform-appropriate musl target
# Using musl ensures compatibility with manylinux_2_28
RUN RUST_TARGET=$(cat /tmp/rust_target) && \
    rustup target add $RUST_TARGET && \
    cargo build --release --package baseten-performance-reverse-proxy --target $RUST_TARGET

# Runtime stage using manylinux_2_28 compatible base (Debian Bookworm = glibc 2.36)
FROM --platform=$TARGETPLATFORM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    libssl3 \
    libc6 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 proxyuser

# Set working directory
WORKDIR /app

# Copy the binary from builder stage (platform-aware)
ARG TARGETARCH
RUN case "$TARGETARCH" in \
    "amd64") echo "x86_64-unknown-linux-musl" > /tmp/rust_target ;; \
    "arm64") echo "aarch64-unknown-linux-musl" > /tmp/rust_target ;; \
    esac

RUN mkdir -p /usr/local/bin
COPY --from=builder /app/target/*/release/baseten-reverse-proxy /usr/local/bin/baseten-reverse-proxy
RUN chmod +x /usr/local/bin/baseten-reverse-proxy

# Change ownership to non-root user
RUN chown -R proxyuser:proxyuser /app /usr/local/bin/baseten-reverse-proxy

# Switch to non-root user
USER proxyuser

# Expose port
EXPOSE 80

# Environment variables (can be overridden at runtime)
ENV PORT=80
ENV LOG_LEVEL=info
ENV HTTP_VERSION=2

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health_internal || exit 1

# Default command (target URL is optional, can be provided per request)
CMD ["baseten-reverse-proxy", "--port", "80"]

# Volume mounts for API key files (optional)
# VOLUME ["/etc/baseten", "/app/secrets"]
