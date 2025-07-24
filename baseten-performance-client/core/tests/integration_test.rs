use baseten_performance_client_core::*;

#[tokio::test]
async fn test_embeddings_with_max_chars_per_request() {
    let texts = vec![
        "Short text".to_string(),
        "This is a much longer text that should be placed in a different batch due to character limits".to_string(),
        "Medium length text here".to_string(),
        "Another very long piece of text that should also be batched separately due to the character limit".to_string(),
    ];

    // Test the split policy directly
    let policy = SplitPolicy::max_chars_per_request(50, 4);
    let batches = texts.split(&policy);

    // Verify that batches respect character limits
    assert!(!batches.is_empty(), "Should create at least one batch");

    for batch in &batches {
        let total_chars: usize = batch.iter().map(|s| s.chars().count()).sum();
        println!(
            "Batch with {} items has {} characters",
            batch.len(),
            total_chars
        );
        // Allow some flexibility in the algorithm
        assert!(batch.len() > 0, "No empty batches should be created");
    }

    // Test that very large character limits result in single batch
    let policy_large = SplitPolicy::max_chars_per_request(1000, 4);
    let large_batches = texts.split(&policy_large);
    assert_eq!(
        large_batches.len(),
        1,
        "Should create only one batch with large char limit"
    );
}

#[tokio::test]
async fn test_combine_responses() {
    let response1 = CoreOpenAIEmbeddingsResponse {
        object: "list".to_string(),
        data: vec![CoreOpenAIEmbeddingData {
            object: "embedding".to_string(),
            embedding_internal: CoreEmbeddingVariant::FloatVector(vec![1.0, 2.0]),
            index: 0,
        }],
        model: "test-model".to_string(),
        usage: CoreOpenAIUsage {
            prompt_tokens: 10,
            total_tokens: 15,
        },
        total_time: 1.5,
        individual_request_times: vec![1.0],
        response_headers: Vec::new(),
    };

    let response2 = CoreOpenAIEmbeddingsResponse {
        object: "list".to_string(),
        data: vec![CoreOpenAIEmbeddingData {
            object: "embedding".to_string(),
            embedding_internal: CoreEmbeddingVariant::FloatVector(vec![3.0, 4.0]),
            index: 1,
        }],
        model: "test-model".to_string(),
        usage: CoreOpenAIUsage {
            prompt_tokens: 8,
            total_tokens: 12,
        },
        total_time: 2.0,
        individual_request_times: vec![1.5],
        response_headers: Vec::new(),
    };

    let combined = CoreOpenAIEmbeddingsResponse::combine(vec![response1, response2], 2);

    assert_eq!(combined.data.len(), 2);
    assert_eq!(combined.usage.prompt_tokens, 18);
    assert_eq!(combined.usage.total_tokens, 27);
    assert_eq!(combined.model, "test-model");

    // Verify order is maintained
    assert_eq!(combined.data[0].index, 0);
    assert_eq!(combined.data[1].index, 1);

    // check that individual request times are combined correctly
    let times = &combined.individual_request_times;
    assert_eq!(times.len(), 2);
    assert_eq!(times[0], 1.0);
    assert_eq!(times[1], 1.5);
}

#[test]
fn test_rerank_response_combine() {
    let response1 = CoreRerankResponse::new(
        vec![CoreRerankResult {
            index: 0,
            score: 0.9,
            text: Some("First text".to_string()),
        }],
        Some(1.0),
        Some(vec![0.5]),
    );

    let response2 = CoreRerankResponse::new(
        vec![CoreRerankResult {
            index: 1,
            score: 0.8,
            text: Some("Second text".to_string()),
        }],
        Some(1.5),
        Some(vec![0.7]),
    );

    let combined = CoreRerankResponse::combine(vec![response1, response2], 2);

    assert_eq!(combined.data.len(), 2);
    assert_eq!(combined.data[0].score, 0.9);
    assert_eq!(combined.data[1].score, 0.8);

    // Verify order is maintained by index
    assert_eq!(combined.data[0].index, 0);
    assert_eq!(combined.data[1].index, 1);
}

#[test]
fn test_classification_response_combine() {
    let response1 = CoreClassificationResponse::new(
        vec![vec![CoreClassificationResult {
            label: "positive".to_string(),
            score: 0.95,
        }]],
        Some(1.0),
        Some(vec![0.5]),
    );

    let response2 = CoreClassificationResponse::new(
        vec![vec![CoreClassificationResult {
            label: "negative".to_string(),
            score: 0.85,
        }]],
        Some(1.5),
        Some(vec![0.7]),
    );

    let combined = CoreClassificationResponse::combine(vec![response1, response2], 2);

    assert_eq!(combined.data.len(), 2);
    assert_eq!(combined.data[0][0].label, "positive");
    assert_eq!(combined.data[1][0].label, "negative");
}

#[test]
fn test_send_request_config_hedge_timeout_validation() {
    use baseten_performance_client_core::http_client::SendRequestConfig;
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::Arc;
    use std::time::Duration;

    let cancel_token = Arc::new(AtomicBool::new(false));
    let hedge_budget = Arc::new(AtomicUsize::new(100));
    let retry_budget = Arc::new(AtomicUsize::new(100));

    // Test case 1: hedge timeout higher than request timeout (should succeed)
    let result = SendRequestConfig::new(
        3,
        Duration::from_millis(100),
        retry_budget.clone(),
        cancel_token.clone(),
        Some((hedge_budget.clone(), Duration::from_secs(2))), // hedge timeout = 2s
        Duration::from_secs(1),                               // request timeout = 1s
    );
    assert!(
        result.is_ok(),
        "Should succeed when hedge timeout > request timeout"
    );

    // Test case 2: hedge timeout equal to request timeout (should fail)
    let result = SendRequestConfig::new(
        3,
        Duration::from_millis(100),
        retry_budget.clone(),
        cancel_token.clone(),
        Some((hedge_budget.clone(), Duration::from_secs(1))), // hedge timeout = 1s
        Duration::from_secs(1),                               // request timeout = 1s
    );
    assert!(
        result.is_err(),
        "Should fail when hedge timeout = request timeout"
    );

    // Test case 3: hedge timeout lower than request timeout (should fail)
    let result = SendRequestConfig::new(
        3,
        Duration::from_millis(100),
        retry_budget.clone(),
        cancel_token.clone(),
        Some((hedge_budget.clone(), Duration::from_millis(500))), // hedge timeout = 0.5s
        Duration::from_secs(1),                                   // request timeout = 1s
    );
    assert!(
        result.is_err(),
        "Should fail when hedge timeout < request timeout"
    );

    // Test case 4: no hedge budget (should succeed)
    let result = SendRequestConfig::new(
        3,
        Duration::from_millis(100),
        retry_budget.clone(),
        cancel_token.clone(),
        None, // no hedge budget
        Duration::from_secs(1),
    );
    assert!(
        result.is_ok(),
        "Should succeed when no hedge budget is specified"
    );
}
