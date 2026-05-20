FROM rust:1.78-slim AS build
WORKDIR /src
COPY Cargo.toml ./
COPY src ./src
RUN cargo build --release --bin assigner

FROM debian:bookworm-slim
RUN apt-get update -y && apt-get install -y --no-install-recommends \
        python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /src/target/release/assigner /app/assigner
EXPOSE 8000
ENTRYPOINT ["/app/assigner"]
