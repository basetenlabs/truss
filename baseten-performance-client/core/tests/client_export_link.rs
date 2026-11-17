//! Separate process: the exporter reads environment configuration once.
mod common;

use baseten_performance_client_core::TraceContext;

#[tokio::test]
async fn link_mode_roots_each_call_and_links_the_caller() {
    let mut harness = common::Harness::start().await;
    std::env::set_var("BASETEN_PERFORMANCE_CLIENT_OTLP_CALLER_CONTEXT", "link");
    let client = harness.client();
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
    ]
    .into_iter()
    .enumerate()
    {
        let wire = harness.post(&client, context).await;
        let span = harness.next_span().await;
        assert_eq!(
            wire["traceparent"],
            format!(
                "00-{}-{}-01",
                span["traceId"].as_str().unwrap(),
                span["spanId"].as_str().unwrap()
            )
        );
        assert!(!wire.contains_key("tracestate"));
        assert_eq!(span["parentSpanId"].as_str().unwrap_or_default(), "");
        assert_eq!(span["traceState"].as_str().unwrap_or_default(), "");
        let links = span["links"].as_array().cloned().unwrap_or_default();
        if index == 0 {
            assert!(links.is_empty());
            continue;
        }
        let [link] = &links[..] else {
            panic!("got links {links:?}, want one caller link");
        };
        assert_ne!(span["traceId"], "4bf92f3577b34da6a3ce929d0e0e4736");
        assert_eq!(link["traceId"], "4bf92f3577b34da6a3ce929d0e0e4736");
        assert_eq!(link["spanId"], "00f067aa0ba902b7");
        assert_eq!(
            link["flags"].as_u64().unwrap() & 3,
            if index == 2 { 3 } else { 1 }
        );
        assert_eq!(
            link["traceState"].as_str().unwrap_or_default(),
            if index == 1 { "vendor=value" } else { "" }
        );
    }
    // An unsampled caller is still forwarded unchanged, without recording.
    let unsampled = parent.replace("-01", "-00");
    let wire = harness
        .post(
            &client,
            Some(TraceContext {
                traceparent: unsampled.clone(),
                tracestate: None,
            }),
        )
        .await;
    assert_eq!(wire["traceparent"], unsampled);
}
