# Build stage
FROM rust:1.75-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Cargo files
COPY Cargo.toml Cargo.lock ./
COPY core/Cargo.toml ./core/
COPY reverse_proxy/Cargo.toml ./reverse_proxy/

# Create dummy source files to cache dependencies
RUN mkdir -p src core/src reverse_proxy/src
RUN echo "fn main() {}" > src/main.rs
RUN echo "fn main() {}" > core/src/lib.rs
RUN echo "fn main() {}" > reverse_proxy/src/main.rs

# Build dependencies
RUN cargo build --release
RUN rm -rf src core/src reverse_proxy/src

# Copy the actual source code
COPY . .

# Build the reverse proxy binary
RUN cargo build --release --bin baseten-reverse-proxy

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
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["baseten-reverse-proxy", "--port", "8080", "--target-url", "https://api.baseten.co"]
