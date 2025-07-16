const { PerformanceClient } = require('./index.js');

// Example usage of the Baseten Performance Client for Node.js

async function main() {
    // Initialize clients for different endpoints
    const embedBaseUrl = process.env.EMBED_URL || "https://model-yqv4yjjq.api.baseten.co/environments/production/sync";
    const rerankBaseUrl = process.env.RERANK_URL || "https://model-abc123.api.baseten.co/environments/production/sync";
    const apiKey = process.env.BASETEN_API_KEY || process.env.OPENAI_API_KEY;

    if (!apiKey) {
        console.error('Please set BASETEN_API_KEY or OPENAI_API_KEY environment variable');
        process.exit(1);
    }

    // Create separate clients for different endpoints
    const embedClient = new PerformanceClient(embedBaseUrl, apiKey);
    const rerankClient = new PerformanceClient(rerankBaseUrl, apiKey);

    console.log('🚀 Baseten Performance Client Example');
    console.log('=====================================\n');

    // Example 1: Embeddings
    console.log('1. Testing Embeddings...');
    try {
        const texts = ["Hello world", "This is a test", "Node.js is awesome"];
        const embedResponse = embedClient.embed(
            texts,
            "text-embedding-3-small", // or your model name
            null, // encoding_format
            null, // dimensions
            null, // user
            8,    // max_concurrent_requests
            2,    // batch_size
            30    // timeout_s
        );

        console.log(`✓ Embedded ${embedResponse.data.length} texts`);
        console.log(`✓ Model: ${embedResponse.model}`);
        console.log(`✓ Total tokens: ${embedResponse.usage.total_tokens}`);
        console.log(`✓ Total time: ${embedResponse.total_time?.toFixed(3)}s`);

        embedResponse.data.forEach((item, i) => {
            console.log(`  Text ${i}: ${item.embedding.length} dimensions`);
        });
    } catch (error) {
        console.error('❌ Embeddings failed:', error.message);
    }

    console.log('\n2. Testing Reranking...');
    try {
        const query = "What is machine learning?";
        const docs = [
            "Machine learning is a subset of artificial intelligence",
            "JavaScript is a programming language",
            "Deep learning uses neural networks",
            "Python is popular for data science"
        ];

        const rerankResponse = rerankClient.rerank(
            query,
            docs,
            false, // raw_scores
            true,  // return_text
            false, // truncate
            "Right", // truncation_direction
            4,     // max_concurrent_requests
            2,     // batch_size
            30     // timeout_s
        );

        console.log(`✓ Reranked ${rerankResponse.data.length} documents`);
        console.log(`✓ Total time: ${rerankResponse.total_time?.toFixed(3)}s`);

        rerankResponse.data.forEach((result, i) => {
            console.log(`  ${i + 1}. Score: ${result.score.toFixed(3)} - ${result.text?.substring(0, 50)}...`);
        });
    } catch (error) {
        console.error('❌ Reranking failed:', error.message);
    }

    console.log('\n3. Testing Classification...');
    try {
        const textsToClassify = [
            "I love this product!",
            "This is terrible.",
            "It's okay, nothing special."
        ];

        const classifyResponse = rerankClient.classify(
            textsToClassify,
            false, // raw_scores
            false, // truncate
            "Right", // truncation_direction
            4,     // max_concurrent_requests
            2,     // batch_size
            30     // timeout_s
        );

        console.log(`✓ Classified ${classifyResponse.data.length} texts`);
        console.log(`✓ Total time: ${classifyResponse.total_time?.toFixed(3)}s`);

        classifyResponse.data.forEach((group, i) => {
            console.log(`  Text ${i + 1}:`);
            group.forEach(result => {
                console.log(`    ${result.label}: ${result.score.toFixed(3)}`);
            });
        });
    } catch (error) {
        console.error('❌ Classification failed:', error.message);
    }

    console.log('\n4. Testing Batch POST...');
    try {
        const payloads = [
            { "model": "text-embedding-3-small", "input": ["Hello"] },
            { "model": "text-embedding-3-small", "input": ["World"] }
        ];

        const batchResponse = embedClient.batch_post(
            "/v1/embeddings", // URL path
            payloads,
            4,  // max_concurrent_requests
            30  // timeout_s
        );

        console.log(`✓ Processed ${batchResponse.data.length} batch requests`);
        console.log(`✓ Total time: ${batchResponse.total_time.toFixed(3)}s`);

        batchResponse.data.forEach((response, i) => {
            console.log(`  Request ${i + 1}: ${JSON.stringify(response).substring(0, 100)}...`);
        });
    } catch (error) {
        console.error('❌ Batch POST failed:', error.message);
    }

    console.log('\n🎉 Example completed!');
}

// Run the example
if (require.main === module) {
    main().catch(console.error);
}
