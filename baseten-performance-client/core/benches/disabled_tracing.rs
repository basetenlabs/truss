//! Per-call cost with client span export off, the default: sequential embed calls against a
//! local server, so tracing overhead would show up next to a loopback round trip.
//!
//! Run with `cargo bench -p baseten_performance_client_core --bench disabled_tracing`. Keep
//! `BASETEN_PERFORMANCE_CLIENT_OTLP_ENDPOINT` unset; the benchmark refuses to run otherwise.

use axum::{routing::post, Json, Router};
use baseten_performance_client_core::*;
use serde_json::{json, Value};
use std::time::{Duration, Instant};

const WARMUP_CALLS: usize = 500;
const MEASURED_CALLS: usize = 5_000;
const PARENT: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

async fn embeddings(Json(request): Json<CoreOpenAIEmbeddingsRequest>) -> Json<Value> {
    let data: Vec<Value> = request
        .input
        .iter()
        .enumerate()
        .map(|(index, _)| json!({"object": "embedding", "embedding": [0.1], "index": index}))
        .collect();
    Json(json!({
        "object": "list",
        "data": data,
        "model": request.model,
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }))
}

async fn measure(client: &PerformanceClientCore, preference: &RequestProcessingPreference) {
    let call = || async {
        client
            .process_embeddings_requests(
                vec!["hello".to_string()],
                "bench-model".to_string(),
                None,
                None,
                None,
                preference,
            )
            .await
            .expect("embed succeeds");
    };
    for _ in 0..WARMUP_CALLS {
        call().await;
    }
    let mut samples: Vec<Duration> = Vec::with_capacity(MEASURED_CALLS);
    for _ in 0..MEASURED_CALLS {
        let start = Instant::now();
        call().await;
        samples.push(start.elapsed());
    }
    samples.sort();
    let micros = |d: Duration| d.as_secs_f64() * 1e6;
    let mean = samples.iter().sum::<Duration>() / MEASURED_CALLS as u32;
    println!(
        "  mean {:7.1} µs   p50 {:7.1} µs   p99 {:7.1} µs",
        micros(mean),
        micros(samples[MEASURED_CALLS / 2]),
        micros(samples[MEASURED_CALLS * 99 / 100]),
    );
}

fn main() {
    assert!(
        std::env::var(OTLP_ENDPOINT_ENV_VAR).is_err(),
        "unset {OTLP_ENDPOINT_ENV_VAR}: this benchmark measures the export-off path"
    );
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    runtime.block_on(async {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener binds");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            let app = Router::new().route("/v1/embeddings", post(embeddings));
            axum::serve(listener, app).await.expect("server");
        });
        let client = PerformanceClientCore::new(
            format!("http://{}", addr),
            Some("bench-key".to_string()),
            1,
            None,
            None,
            None,
        )
        .expect("client builds");
        let base = RequestProcessingPreference::new()
            .with_max_concurrent_requests(1)
            .with_batch_size(1);

        let cases = [
            ("no parent", base.clone()),
            (
                "parent + tracestate forwarded",
                base.clone()
                    .with_traceparent(PARENT.to_string())
                    .with_tracestate("vendor=opaque".to_string()),
            ),
            // The server answers well inside the delay, so this measures arming the hedge race.
            (
                "hedging armed (1 s delay)",
                base.clone()
                    .with_hedge_delay(1.0)
                    .with_hedge_budget_pct(1.0),
            ),
        ];
        for (name, preference) in &cases {
            println!("{name}:");
            measure(&client, preference).await;
        }
    });
}
