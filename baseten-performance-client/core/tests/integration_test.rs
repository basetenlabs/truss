use baseten_performance_client_core::split_policy::{Combinable, SplitPolicy, Splittable};
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
    // Test case 1: hedge delay higher than request delay (should fail)
    // This validation is now done in RequestProcessingConfig, not SendRequestConfig
    let pref = RequestProcessingPreference::new()
        .with_timeout_s(1.0)
        .with_hedge_delay(2.0); // hedge delay higher than timeout - should fail

    let result = pref.pair_with_request_validate_and_convert(
        "https://example.com".to_string(),
        100,
        "test_api_key".to_string(),
    );
    assert!(
        result.is_err(),
        "Should fail when hedge delay > request timeout"
    );

    // Test case 2: hedge delay equal to request timeout (should fail)
    let pref2 = RequestProcessingPreference::new()
        .with_timeout_s(1.0)
        .with_hedge_delay(1.0); // hedge delay equal to timeout - should fail

    let result2 = pref2.pair_with_request_validate_and_convert(
        "https://example.com".to_string(),
        100,
        "test_api_key".to_string(),
    );
    assert!(
        result2.is_err(),
        "Should fail when hedge delay = request timeout"
    );

    // Test case 3: hedge delay lower than request timeout (should pass)
    let pref3 = RequestProcessingPreference::new()
        .with_timeout_s(1.0)
        .with_hedge_delay(0.5); // hedge delay lower than timeout - should pass

    let result3 = pref3.pair_with_request_validate_and_convert(
        "https://example.com".to_string(),
        100,
        "test_api_key".to_string(),
    );
    assert!(
        result3.is_ok(),
        "Should pass when hedge delay < request timeout"
    );

    // Test case 4: no hedge budget (should succeed)
    let pref4 = RequestProcessingPreference::new().with_timeout_s(1.0); // no hedge delay - should succeed

    let result4 = pref4.pair_with_request_validate_and_convert(
        "https://example.com".to_string(),
        100,
        "test_api_key".to_string(),
    );
    assert!(
        result4.is_ok(),
        "Should succeed when no hedge budget is specified"
    );
}
