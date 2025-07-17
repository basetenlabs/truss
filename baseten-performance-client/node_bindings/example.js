const { PerformanceClient } = require('./index.js');

// Example usage of the Baseten Performance Client for Node.js

async function main() {
    // Initialize clients for different endpoints
    const embedBaseUrl = process.env.EMBED_URL || "https://model-yqv4yjjq.api.baseten.co/environments/production/sync";
    const rerankBaseUrl = process.env.RERANK_URL || "https://model-e3mx5vzq.api.baseten.co/environments/production/sync";
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

        const batchResponse = embedClient.batchPost(
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

// Should print something like this:
// 🚀 Baseten Performance Client Example
// =====================================

// 1. Testing Embeddings...
// ✓ Embedded 3 texts
// ✓ Model: text-embedding-3-small
// ✓ Total tokens: 18
// ✓ Total time: 0.102s
//   Text 0: 384 dimensions
//   Text 1: 384 dimensions
//   Text 2: 384 dimensions

// 2. Testing Reranking...
// ✓ Reranked 4 documents
// ✓ Total time: 0.208s
//   1. Score: 0.999 - Machine learning is a subset of artificial intelli...
//   2. Score: 0.000 - JavaScript is a programming language...
//   3. Score: 0.002 - Deep learning uses neural networks...
//   4. Score: 0.000 - Python is popular for data science...

// 3. Testing Classification...
// ✓ Classified 3 texts
// ✓ Total time: 0.061s
//   Text 1:
//     LABEL_0: 0.037
//   Text 2:
//     LABEL_0: 0.043
//   Text 3:
//     LABEL_0: 0.086

// 4. Testing Batch POST...
// ✓ Processed 2 batch requests
// ✓ Total time: 0.022s
//   Request 1: {"data":[{"embedding":[-0.07749477,-0.012250737,0.055566512,-0.037535366,0.04937588,0.0109734535,0.0...
//   Request 2: {"data":[{"embedding":[0.019163806,-0.013788592,0.012026816,-0.010408859,0.057886917,-0.00013181016,...

// 🎉 Example completed!
