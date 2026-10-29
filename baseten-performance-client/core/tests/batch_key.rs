//! Requests of a call carry `x-baseten-batch-key` only when the call asks for it.
mod common;

use baseten_performance_client_core::RequestProcessingPreference;

const BATCH_KEY: &str = "x-baseten-batch-key";

fn batch_keys(headers: &http::HeaderMap) -> Vec<&str> {
    headers
        .get_all(BATCH_KEY)
        .iter()
        .map(|value| value.to_str().unwrap())
        .collect()
}

/// Asserts every request of one call carries the same single key, the batch-level part of its
/// request ID, and returns it.
fn shared_batch_key(requests: &[http::HeaderMap]) -> String {
    let key = batch_keys(&requests[0])
        .first()
        .expect("the call's first request has no batch key")
        .to_string();
    for headers in requests {
        assert_eq!(batch_keys(headers), [key.as_str()]);
        let request_id = headers["x-baseten-customer-request-id"].to_str().unwrap();
        assert!(
            request_id.starts_with(&format!("{key}-")),
            "request id {request_id} does not extend batch key {key}"
        );
    }
    key
}

// One test: the harness sets process-wide exporter variables, so harnesses must not run in
// parallel.
#[tokio::test]
async fn send_batch_key_marks_every_request_of_the_call() {
    let harness = common::Harness::start().await;
    let client = harness.client();

    for headers in harness
        .post_batch(&client, 2, &RequestProcessingPreference::default())
        .await
    {
        assert!(!headers.contains_key(BATCH_KEY));
    }

    let keyed = RequestProcessingPreference::default().with_send_batch_key(true);
    let posted = harness.post_batch(&client, 3, &keyed).await;
    assert_eq!(posted.len(), 3);
    let posted_key = shared_batch_key(&posted);

    // Embeddings take the other request path.
    let embedded = harness
        .embed(&client, 3, &keyed.clone().with_batch_size(1))
        .await;
    assert_eq!(embedded.len(), 3);
    assert_ne!(shared_batch_key(&embedded), posted_key);
}
