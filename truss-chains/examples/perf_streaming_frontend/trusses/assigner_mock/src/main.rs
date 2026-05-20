// AssignerMock: joins transcribed sentences with diarizer segments and
// emits the merged speaker-attributed transcript. TC CS HTTP leaf in Rust,
// mirroring the Python/Node siblings via the same JSON contract.

use axum::{
    extract::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::net::SocketAddr;
use std::time::{Duration, Instant};
use tracing::info;

#[derive(Deserialize)]
struct Sentence {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    order: Option<u64>,
}

#[derive(Deserialize)]
struct DiarizeSegment {
    #[serde(default)]
    speaker: Option<String>,
    #[serde(default)]
    text_hint: Option<String>,
}

#[derive(Deserialize)]
struct Diarize {
    #[serde(default)]
    segments: Vec<DiarizeSegment>,
}

#[derive(Deserialize)]
struct PredictRequest {
    #[serde(default)]
    sentences: Vec<Value>,
    #[serde(default)]
    diarize: Option<Value>,
}

#[derive(Serialize)]
struct AssignedItem {
    speaker: String,
    text: String,
}

#[derive(Serialize)]
struct PredictResponse {
    chainlet: &'static str,
    items: Vec<AssignedItem>,
    sentence_count: usize,
    segment_count: usize,
}

async fn health() -> Json<Value> {
    Json(serde_json::json!({"status": "ok"}))
}

async fn predict(Json(req): Json<PredictRequest>) -> Json<PredictResponse> {
    let start = Instant::now();
    info!(
        sentences = req.sentences.len(),
        has_diarize = req.diarize.is_some(),
        "[STEP] assign_speaker_to_sentence: request received"
    );
    tokio::time::sleep(Duration::from_millis(30)).await;

    let sentences: Vec<Sentence> = req
        .sentences
        .into_iter()
        .filter_map(|v| serde_json::from_value(v).ok())
        .collect();

    let mut sorted: Vec<&Sentence> = sentences.iter().collect();
    sorted.sort_by_key(|s| s.order.unwrap_or(u64::MAX));
    info!(sorted_count = sorted.len(), "[STEP] sentences sorted by order");

    let diarize: Diarize = req
        .diarize
        .and_then(|v| serde_json::from_value(v).ok())
        .unwrap_or(Diarize { segments: vec![] });
    info!(segments = diarize.segments.len(), "[STEP] diarize segments parsed");

    let mut items = Vec::with_capacity(sorted.len());
    for (i, s) in sorted.iter().enumerate() {
        let speaker = diarize
            .segments
            .get(i)
            .and_then(|seg| seg.speaker.clone())
            .unwrap_or_else(|| "UNK".to_string());
        let text = s.text.clone().unwrap_or_default();
        items.push(AssignedItem { speaker, text });
    }
    info!(
        items = items.len(),
        elapsed_ms = start.elapsed().as_millis() as u64,
        "[STEP] assign_speaker_to_sentence: done"
    );

    Json(PredictResponse {
        chainlet: "AssignerMock",
        sentence_count: sorted.len(),
        segment_count: diarize.segments.len(),
        items,
    })
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let app = Router::new()
        .route("/health", get(health))
        .route("/predict", post(predict));
    let addr: SocketAddr = "0.0.0.0:8000".parse().unwrap();
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    info!(addr = %addr, "[INIT] AssignerMock listening");
    axum::serve(listener, app).await.unwrap();
}
