const { PerformanceClient } = require('../performance-client.node');

// Tests requiring valid API keys and endpoints
// These tests are not run automatically in CI but can be run manually

const BASETEN_API_KEY = process.env.BASETEN_API_KEY;
const EMBED_URL = process.env.EMBED_URL || "https://model-yqv4yjjq.api.baseten.co/environments/production/sync";
const RERANK_URL = process.env.RERANK_URL || "https://model-abc123.api.baseten.co/environments/production/sync";

// Simple test framework
class TestRunner {
    constructor() {
        this.tests = [];
        this.passed = 0;
        this.failed = 0;
    }

    test(name, fn) {
        this.tests.push({ name, fn });
    }

    async run() {
        console.log(`Running ${this.tests.length} integration tests...\n`);

        for (const { name, fn } of this.tests) {
            try {
                console.log(`• ${name}...`);
                await fn();
                console.log(`  ✓ PASSED\n`);
                this.passed++;
            } catch (error) {
                console.log(`  ✗ FAILED: ${error.message}\n`);
                this.failed++;
            }
        }

        console.log(`Results: ${this.passed} passed, ${this.failed} failed`);
        return this.failed === 0;
    }
}

// Test utilities
function assert(condition, message) {
    if (!condition) {
        throw new Error(message || 'Assertion failed');
    }
}

const runner = new TestRunner();

// Test 1: Real embedding test
runner.test('Real embedding test', async () => {
    if (!BASETEN_API_KEY) {
        console.log('  ⚠️  SKIPPED: BASETEN_API_KEY not set');
        return;
    }

    const client = new PerformanceClient(EMBED_URL, BASETEN_API_KEY);
    const texts = ["Hello world", "Test embedding"];

    const response = await client.embed(
        texts,
        "text-embedding-3-small",
        null, // encoding_format
        null, // dimensions
        null, // user
        2,    // max_concurrent_requests
        1,    // batch_size
        30    // timeout_s
    );

    assert(response.data && response.data.length === 2, 'Should return 2 embeddings');
    assert(response.data[0].embedding && response.data[0].embedding.length > 0, 'First embedding should have data');
    assert(response.total_time && response.total_time > 0, 'Should have positive total_time');
    assert(response.usage && response.usage.total_tokens > 0, 'Should have token usage');

    console.log(`    Embedded ${texts.length} texts in ${response.total_time.toFixed(3)}s`);
    console.log(`    Used ${response.usage.total_tokens} tokens`);
});

// Test 2: Real reranking test
runner.test('Real reranking test', async () => {
    if (!BASETEN_API_KEY) {
        console.log('  ⚠️  SKIPPED: BASETEN_API_KEY not set');
        return;
    }

    const client = new PerformanceClient(RERANK_URL, BASETEN_API_KEY);
    const query = "What is machine learning?";
    const documents = [
        "Machine learning is a subset of artificial intelligence",
        "JavaScript is a programming language",
        "Deep learning uses neural networks"
    ];

    const response = await client.rerank(
        query,
        documents,
        false, // raw_scores
        true,  // return_text
        false, // truncate
        "Right", // truncation_direction
        2,     // max_concurrent_requests
        1,     // batch_size
        30     // timeout_s
    );

    assert(response.data && response.data.length === 3, 'Should return 3 reranked results');
    assert(response.data[0].score !== undefined, 'First result should have score');
    assert(response.data[0].text !== undefined, 'First result should have text');
    assert(response.total_time && response.total_time > 0, 'Should have positive total_time');

    console.log(`    Reranked ${documents.length} documents in ${response.total_time.toFixed(3)}s`);
    console.log(`    Top result score: ${response.data[0].score.toFixed(3)}`);
});

// Test 3: Real classification test
runner.test('Real classification test', async () => {
    if (!BASETEN_API_KEY) {
        console.log('  ⚠️  SKIPPED: BASETEN_API_KEY not set');
        return;
    }

    const client = new PerformanceClient(RERANK_URL, BASETEN_API_KEY);
    const texts = ["This is great!", "I hate this", "It's okay"];

    const response = await client.classify(
        texts,
        false, // raw_scores
        false, // truncate
        "Right", // truncation_direction
        2,     // max_concurrent_requests
        1,     // batch_size
        30     // timeout_s
    );

    assert(response.data && response.data.length === 3, 'Should return 3 classification results');
    assert(Array.isArray(response.data[0]), 'First result should be an array of classifications');
    assert(response.data[0][0].label !== undefined, 'Should have label');
    assert(response.data[0][0].score !== undefined, 'Should have score');
    assert(response.total_time && response.total_time > 0, 'Should have positive total_time');

    console.log(`    Classified ${texts.length} texts in ${response.total_time.toFixed(3)}s`);
    console.log(`    First text top label: ${response.data[0][0].label} (${response.data[0][0].score.toFixed(3)})`);
});

// Test 4: Batch post test
runner.test('Real batch post test', async () => {
    if (!BASETEN_API_KEY) {
        console.log('  ⚠️  SKIPPED: BASETEN_API_KEY not set');
        return;
    }

    const client = new PerformanceClient(EMBED_URL, BASETEN_API_KEY);
    const payloads = [
        { "model": "text-embedding-3-small", "input": ["Hello"] },
        { "model": "text-embedding-3-small", "input": ["World"] }
    ];

    const response = await client.batchPost(
        "/v1/embeddings",
        payloads,
        2,  // max_concurrent_requests
        30  // timeout_s
    );

    assert(response.data && response.data.length === 2, 'Should return 2 batch results');
    assert(response.total_time && response.total_time > 0, 'Should have positive total_time');
    assert(response.individual_request_times && response.individual_request_times.length === 2, 'Should have individual request times');

    console.log(`    Processed ${payloads.length} batch requests in ${response.total_time.toFixed(3)}s`);
});

// Run tests
async function main() {
    console.log('Baseten Performance Client - Integration Tests (Requires API Key)');
    console.log('====================================================================\n');

    if (!BASETEN_API_KEY) {
        console.log('⚠️  BASETEN_API_KEY environment variable not set.');
        console.log('   These tests require a valid Baseten API key to run.');
        console.log('   Set BASETEN_API_KEY and optionally EMBED_URL and RERANK_URL to run integration tests.\n');
    }

    const success = await runner.run();

    if (success) {
        console.log('\n🎉 All integration tests passed!');
        process.exit(0);
    } else {
        console.log('\n❌ Some integration tests failed!');
        process.exit(1);
    }
}

// Only run if this file is executed directly
if (require.main === module) {
    main().catch(console.error);
}

module.exports = { TestRunner, assert };
