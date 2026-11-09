//! Fake model server and OTLP collector for the client-span export tests. The exporter
//! reads its environment once per process, so each configuration is its own test file.
#![allow(dead_code)] // Each test file uses a different subset.

use axum::{body::Bytes, extract::State, http::HeaderMap, routing::post, Json, Router};
use baseten_performance_client_core::{
    HttpMethod, PerformanceClientCore, RequestProcessingPreference, TraceContext,
};
use serde_json::{json, Value};
use std::{
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::sync::{mpsc, Notify};

pub struct Harness {
    pub url: String,
    /// Notified when `/blocked` receives a request, which never completes.
    pub blocked: Arc<Notify>,
    seen: Arc<Mutex<Vec<HeaderMap>>>,
    spans: mpsc::UnboundedReceiver<Value>,
    server: tokio::task::JoinHandle<()>,
}

impl Harness {
    /// Starts both servers and points the exporter at the collector.
    pub async fn start() -> Self {
        let blocked = Arc::new(Notify::new());
        let seen = Arc::new(Mutex::new(Vec::<HeaderMap>::new()));
        let (tx, spans) = mpsc::unbounded_channel();
        let app = Router::new()
            .route(
                "/blocked",
                post({
                    let blocked = blocked.clone();
                    move || {
                        let blocked = blocked.clone();
                        async move {
                            blocked.notify_one();
                            std::future::pending::<()>().await;
                        }
                    }
                }),
            )
            .route(
                "/model",
                post({
                    let seen = seen.clone();
                    move |headers: HeaderMap| {
                        seen.lock().unwrap().push(headers);
                        async { Json(json!({})) }
                    }
                }),
            )
            .route(
                "/v1/traces",
                post(
                    |State(tx): State<mpsc::UnboundedSender<Value>>,
                     headers: HeaderMap,
                     body: Bytes| async move {
                        let body: Value =
                            serde_json::from_reader(flate2::read::GzDecoder::new(&body[..]))
                                .unwrap();
                        assert_eq!(headers["authorization"], "Bearer collector-only");
                        assert!(!headers.contains_key("x-application-secret"));
                        for span in body["resourceSpans"][0]["scopeSpans"][0]["spans"]
                            .as_array()
                            .unwrap()
                        {
                            tx.send(span.clone()).unwrap();
                        }
                        Json(json!({}))
                    },
                ),
            )
            .with_state(tx);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        std::env::set_var("BASETEN_PERFORMANCE_CLIENT_OTLP_ENDPOINT", &url);
        std::env::set_var(
            "BASETEN_PERFORMANCE_CLIENT_OTLP_HEADERS",
            "authorization=Bearer collector-only",
        );
        Self {
            url,
            blocked,
            seen,
            spans,
            server,
        }
    }

    pub fn client(&self) -> PerformanceClientCore {
        PerformanceClientCore::new(
            self.url.clone(),
            Some("model-key".into()),
            1,
            None,
            None,
            None,
        )
        .unwrap()
    }

    /// Posts once to `/model` and returns the headers the model received.
    pub async fn post(
        &self,
        client: &PerformanceClientCore,
        trace_context: Option<TraceContext>,
    ) -> HeaderMap {
        let preference = RequestProcessingPreference {
            trace_context,
            ..Default::default()
        };
        client
            .process_batch_post_requests(
                "/model".into(),
                vec![json!({})],
                &preference,
                HttpMethod::POST,
            )
            .await
            .unwrap();
        self.seen.lock().unwrap().last().unwrap().clone()
    }

    pub async fn next_span(&mut self) -> Value {
        tokio::time::timeout(Duration::from_secs(5), self.spans.recv())
            .await
            .unwrap()
            .unwrap()
    }

    pub fn has_pending_span(&mut self) -> bool {
        !self.spans.is_empty()
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        self.server.abort();
    }
}
