# syntax=docker/dockerfile:1
FROM --platform=$BUILDPLATFORM debian:bookworm-slim AS builder

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

# Build the binary
RUN cargo build --release --package baseten-performance-reverse-proxy

# Runtime stage using Debian Bookworm
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    libc6 \
    && rm -rf /var/lib/apt/lists/*

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
