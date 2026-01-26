# syntax=docker/dockerfile:1
FROM rustlang/rust:nightly-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    perl \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Cargo files
COPY Cargo.toml Cargo.lock ./
COPY core/Cargo.toml ./core/
COPY reverse_proxy/Cargo.toml ./reverse_proxy/
COPY python_bindings/Cargo.toml ./python_bindings/
COPY node_bindings/Cargo.toml ./node_bindings/

# Create dummy source files to cache dependencies
RUN mkdir -p src core/src reverse_proxy/src python_bindings/src node_bindings/src
RUN echo "fn main() {}" > src/main.rs
RUN echo "fn main() {}" > core/src/lib.rs
RUN echo "fn main() {}" > reverse_proxy/src/main.rs
RUN echo "fn main() {}" > python_bindings/src/lib.rs
RUN echo "fn main() {}" > node_bindings/src/lib.rs

# Build dependencies
RUN cargo build --release --package baseten-performance-reverse-proxy
RUN rm -rf src core/src reverse_proxy/src python_bindings/src node_bindings/src

# Copy the actual source code
COPY core/src ./core/src/
COPY reverse_proxy/src ./reverse_proxy/src/
COPY python_bindings/src ./python_bindings/src/
COPY node_bindings/src ./node_bindings/src/

# Build the reverse proxy binary
RUN cargo build --release --package baseten-performance-reverse-proxy

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 proxyuser

# Set working directory
WORKDIR /app

# Copy the binary from builder stage
COPY --from=builder /app/target/release/baseten-reverse-proxy /usr/local/bin/

# Change ownership to non-root user
RUN chown -R proxyuser:proxyuser /app

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
