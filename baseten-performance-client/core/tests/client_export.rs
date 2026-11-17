//! Separate process: the exporter reads environment configuration once.
mod common;

use baseten_performance_client_core::{HttpMethod, RequestProcessingPreference, TraceContext};
use serde_json::json;
use std::time::Duration;

#[tokio::test]
async fn exports_call_spans_with_matching_wire_context() {
    let mut harness = common::Harness::start().await;
    std::env::set_var(
        "OTEL_EXPORTER_OTLP_HEADERS",
        "x-application-secret=must-not-leak",
    );
    let client = harness.client();
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
        Some(TraceContext {
            traceparent: parent.replace("-01", "-03"),
            tracestate: None,
        }),
        Some(TraceContext {
            traceparent: parent.replace("-01", "-ff"),
            tracestate: None,
        }),
    ]
    .into_iter()
    .enumerate()
    {
        let wire = harness.post(&client, context).await;
        let span = harness.next_span().await;
        assert_eq!(
            wire["traceparent"],
            format!(
                "00-{}-{}-{:02x}",
                span["traceId"].as_str().unwrap(),
                span["spanId"].as_str().unwrap(),
                span["flags"].as_u64().unwrap() & 3
            )
        );
        assert_eq!(wire["authorization"], "Bearer model-key");
        assert_eq!(span["name"], "perfclient.batch_post");
        assert_eq!(span["status"]["code"], 1);
        assert_eq!(span["kind"], 3);
        assert_eq!(
            span["flags"].as_u64().unwrap() & 3,
            if index >= 2 { 3 } else { 1 }
        );
        if index != 1 {
            assert!(!wire.contains_key("tracestate"));
        }
        let nanos = |key: &str| span[key].as_str().unwrap().parse::<u128>().unwrap();
        assert!(nanos("endTimeUnixNano") >= nanos("startTimeUnixNano"));
        if index > 0 {
            assert_eq!(span["parentSpanId"], "00f067aa0ba902b7");
            assert_eq!(span["traceId"], "4bf92f3577b34da6a3ce929d0e0e4736");
        }
        if index == 1 {
            assert_eq!(wire["tracestate"], "vendor=value");
            assert_eq!(span["traceState"], "vendor=value");
        }
    }
    let wire = harness
        .post(
            &client,
            Some(TraceContext {
                traceparent: parent.replace("-01", "-02"),
                tracestate: None,
            }),
        )
        .await;
    assert_eq!(wire["traceparent"], parent.replace("-01", "-02"));
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
    let span = harness.next_span().await;
    assert_eq!(span["status"]["code"], 2);
    assert!(!harness.has_pending_span());
    {
        let preference = RequestProcessingPreference::new();
        let request = client.process_batch_post_requests(
            "/blocked".into(),
            vec![json!({})],
            &preference,
            HttpMethod::POST,
        );
        tokio::pin!(request);
        tokio::select! {
            result = &mut request => panic!("request unexpectedly completed: {result:?}"),
            notified = tokio::time::timeout(Duration::from_secs(5), harness.blocked.notified()) => notified.unwrap(),
        }
        // Drop the pending request to exercise SDK span closure on cancellation.
    }
    let span = harness.next_span().await;
    assert_eq!(span["status"]["code"], 2);
}
