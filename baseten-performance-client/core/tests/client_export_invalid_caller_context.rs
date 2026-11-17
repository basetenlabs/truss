//! Separate process: the exporter reads environment configuration once.
mod common;

use baseten_performance_client_core::TraceContext;

#[tokio::test]
async fn unknown_caller_context_disables_export_but_not_inference() {
    let harness = common::Harness::start().await;
    std::env::set_var("BASETEN_PERFORMANCE_CLIENT_OTLP_CALLER_CONTEXT", "child");
    let client = harness.client();
    let parent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    let wire = harness
        .post(
            &client,
            Some(TraceContext {
                traceparent: parent.into(),
                tracestate: Some("vendor=value".into()),
            }),
        )
        .await;
    // Recording would replace the span id, so an unchanged context means export is off.
    assert_eq!(wire["traceparent"], parent);
    assert_eq!(wire["tracestate"], "vendor=value");
}
