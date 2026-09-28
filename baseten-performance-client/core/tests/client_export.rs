//! Separate process: the exporter reads environment configuration once.
use axum::{body::Bytes, extract::State, http::HeaderMap, routing::post, Json, Router};
use baseten_performance_client_core::{
    HttpMethod, PerformanceClientCore, RequestProcessingPreference, TraceContext,
};
use serde_json::{json, Value};
use std::{
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::sync::mpsc;

#[tokio::test]
async fn exports_call_spans_with_matching_wire_context() {
    let seen = Arc::new(Mutex::new(Vec::<HeaderMap>::new()));
    let (tx, mut rx) = mpsc::unbounded_channel();
    let app = Router::new()
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
                        serde_json::from_reader(flate2::read::GzDecoder::new(&body[..])).unwrap();
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
    std::env::set_var(
        "OTEL_EXPORTER_OTLP_HEADERS",
        "x-application-secret=must-not-leak",
    );
    let client =
        PerformanceClientCore::new(url, Some("model-key".into()), 1, None, None, None).unwrap();
    std::env::set_var(
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
        "authorization=wrong,x-application-secret=must-not-leak",
    );
    std::env::set_var(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "http://127.0.0.1:1/wrong",
    );
    std::env::set_var("OTEL_EXPORTER_OTLP_TRACES_COMPRESSION", "zstd");
    std::env::set_var("OTEL_TRACES_SAMPLER", "always_off");
    let parent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    for (index, context) in [
        None,
        Some(TraceContext {
            traceparent: parent.into(),
            tracestate: Some("vendor=value".into()),
        }),
    ]
    .into_iter()
    .enumerate()
    {
        let preference = RequestProcessingPreference {
            trace_context: context,
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
        let span = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .unwrap()
            .unwrap();
        let wire = seen.lock().unwrap()[index].clone();
        assert_eq!(
            wire["traceparent"],
            format!(
                "00-{}-{}-01",
                span["traceId"].as_str().unwrap(),
                span["spanId"].as_str().unwrap()
            )
        );
        assert_eq!(wire["authorization"], "Bearer model-key");
        assert_eq!(span["name"], "perfclient.batch_post");
        let nanos = |key: &str| span[key].as_str().unwrap().parse::<u128>().unwrap();
        assert!(nanos("endTimeUnixNano") >= nanos("startTimeUnixNano"));
        if index == 1 {
            assert_eq!(span["parentSpanId"], "00f067aa0ba902b7");
            assert_eq!(span["traceId"], "4bf92f3577b34da6a3ce929d0e0e4736");
            assert_eq!(wire["tracestate"], "vendor=value");
            assert_eq!(span["traceState"], "vendor=value");
        }
    }
    let preference = RequestProcessingPreference::new().with_trace_context(TraceContext {
        traceparent: parent.replace("-01", "-00"),
        tracestate: None,
    });
    client
        .process_batch_post_requests(
            "/model".into(),
            vec![json!({})],
            &preference,
            HttpMethod::POST,
        )
        .await
        .unwrap();
    assert_eq!(
        seen.lock().unwrap()[2]["traceparent"],
        parent.replace("-01", "-00")
    );
    // A missing route fails inference, but still closes and exports the call span.
    assert!(client
        .process_batch_post_requests(
            "/missing".into(),
            vec![json!({})],
            &RequestProcessingPreference::new(),
            HttpMethod::POST
        )
        .await
        .is_err());
    let span = tokio::time::timeout(Duration::from_secs(5), rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(span["status"]["code"], 2);
    assert!(rx.try_recv().is_err());
    server.abort();
}
