use axum::{http::HeaderMap, routing::post, Json, Router};
use baseten_performance_client_core::{
    HttpMethod, PerformanceClientCore, RequestProcessingPreference, TraceContext,
};
use serde_json::{json, Value};
use std::process::Command;

#[tokio::test]
async fn malformed_exporter_config_does_not_block_inference() {
    // Each configuration needs a fresh process because the tracer is initialized once.
    if std::env::var_os("PERFCLIENT_TRACING_TEST_CHILD").is_none() {
        for (endpoint, headers, expected_error) in [
            (
                "not-a-url",
                "",
                "invalid BASETEN_PERFORMANCE_CLIENT_OTLP_ENDPOINT",
            ),
            ("ftp://localhost", "", "must be an HTTP(S) URL"),
            (
                "http://localhost:1",
                "authorization=Basic test-secret,broken",
                "invalid BASETEN_PERFORMANCE_CLIENT_OTLP_HEADERS",
            ),
            (
                "http://localhost:1",
                "bad name=test-secret",
                "invalid BASETEN_PERFORMANCE_CLIENT_OTLP_HEADERS name",
            ),
            (
                "http://localhost:1",
                "authorization=Basic test-secret\ninvalid",
                "invalid BASETEN_PERFORMANCE_CLIENT_OTLP_HEADERS value",
            ),
        ] {
            let output = Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "malformed_exporter_config_does_not_block_inference",
                    "--nocapture",
                ])
                .env("PERFCLIENT_TRACING_TEST_CHILD", "1")
                .env("BASETEN_PERFORMANCE_CLIENT_OTLP_ENDPOINT", endpoint)
                .env("BASETEN_PERFORMANCE_CLIENT_OTLP_HEADERS", headers)
                .env("PERFORMANCE_CLIENT_LOG_LEVEL", "warn")
                .output()
                .unwrap();
            let logs = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(output.status.success(), "{expected_error}: {logs}");
            assert_eq!(
                logs.matches("Performance client tracing disabled").count(),
                1,
                "{logs}"
            );
            assert!(logs.contains(expected_error), "{logs}");
            assert!(
                !logs.contains("test-secret"),
                "credentials must not be logged"
            );
        }
        return;
    }

    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .try_init();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}", listener.local_addr().unwrap());
    let app = Router::new().route(
        "/predict",
        post(
            |headers: HeaderMap, Json(payload): Json<Value>| async move {
                Json(json!({
                    "payload": payload,
                    "traceparent": headers.get("traceparent").map(|v| v.to_str().unwrap()),
                    "tracestate": headers.get("tracestate").map(|v| v.to_str().unwrap()),
                }))
            },
        ),
    );
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client =
        PerformanceClientCore::new(base_url, Some("test-key".into()), 1, None, None, None).unwrap();
    let preference = RequestProcessingPreference::new().with_max_retries(0);
    const PARENT: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    let linked = preference.clone().with_trace_context(TraceContext {
        traceparent: PARENT.into(),
        tracestate: Some("vendor=value".into()),
    });
    for (preference, parent, state) in [
        (&preference, None, None),
        (&linked, Some(PARENT), Some("vendor=value")),
        (&preference, None, None),
    ] {
        let (responses, _) = client
            .process_batch_post_requests(
                "/predict".into(),
                vec![json!({"input": "test"})],
                preference,
                HttpMethod::POST,
            )
            .await
            .expect("tracing setup errors must not block inference");
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].0["payload"]["input"].as_str(), Some("test"));
        assert_eq!(responses[0].0["traceparent"].as_str(), parent);
        assert_eq!(responses[0].0["tracestate"].as_str(), state);
    }
    server.abort();
}
