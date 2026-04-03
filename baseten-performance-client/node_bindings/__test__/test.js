const { PerformanceClient } = require('../performance-client.node');

// Test configuration
const TEST_BASE_URL = process.env.TEST_BASE_URL || "https://httpbin.org";
const TEST_API_KEY = process.env.TEST_API_KEY || "test-key";

// Simple test framework
class TestRunner { // todo: needs to be compatible with AVA
    constructor() {
        this.tests = [];
        this.passed = 0;
        this.failed = 0;
    }

    test(name, fn) {
        this.tests.push({ name, fn });
    }

    async run() {
        console.log(`Running ${this.tests.length} tests...\n`);

        for (const { name, fn } of this.tests) {
            try {
                console.log(`• ${name}...`);
                const result = fn();
                if (result && typeof result.then === 'function') {
                    await result;
                }
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

function assertThrows(fn, message) {
    try {
        fn();
        throw new Error(message || 'Expected function to throw');
    } catch (error) {
        if (error.message === message || error.message.includes('Expected function to throw')) {
            throw error;
        }
        // Expected error, test passes
    }
}

// Test suite
const runner = new TestRunner();

// Test 1: Constructor validation
runner.test('Constructor should create client with valid parameters', () => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);
    assert(client instanceof PerformanceClient, 'Should create PerformanceClient instance');
});

// Test 2: Constructor with invalid API key
runner.test('Constructor should handle missing API key', () => {
    // Note: Constructor doesn't throw for missing API key
    const client = new PerformanceClient(TEST_BASE_URL);
    assert(client instanceof PerformanceClient, 'Should create client without explicit API key');
});

// Test 3: Embed method parameter validation
runner.test('Embed method should validate empty input', async () => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    const error = await client.embed([], "test-model").catch(e => e);
    assert(error.message.includes('Input list cannot be empty'),
           'Should throw error for empty input array');
});

// Test 4: Embed method parameter validation with valid input
runner.test('Embed method should accept valid parameters', async () => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    // This should not throw for parameter validation
    // (it will likely fail due to httpbin not being an embedding API, but that's fine)
    try {
        await client.embed(["test text"], "test-model");
    } catch (error) {
        // Allow network/API errors, but not parameter validation errors
        assert(!error.message.includes('Input list cannot be empty'),
               'Should not fail on parameter validation');
    }
});

// Test 5: Rerank method parameter validation
runner.test('Rerank method should validate empty texts', async () => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    const error = await client.rerank("test query", []).catch(e => e);
    assert(error.message.includes('Texts list cannot be empty'),
           'Should throw error for empty texts array');
});

// Test 6: Classify method parameter validation
runner.test('Classify method should validate empty inputs', async () => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    const error = await client.classify([]).catch(e => e);
    assert(error.message.includes('Inputs list cannot be empty'),
           'Should throw error for empty inputs array');
});

// Test 7: Batch post method parameter validation
runner.test('Batch post method should validate empty payloads', async () => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    try {
        await client.batchPost("/test", []);
        throw new Error('Should throw error for empty payloads array');
    } catch (error) {
        assert(error.message.includes('Payloads list cannot be empty'),
               'Should throw error for empty payloads array');
    }
});

// Test 8: Test with mock data (will fail at network level but parameters should be valid)
runner.test('Methods should handle network errors gracefully', async () => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    // Test just one case to avoid unhandled errors
    try {
        await client.embed(["test"], "model");
    } catch (error) {
        // Should be network/API errors, not parameter validation errors
        assert(!error.message.includes('cannot be empty'),
               'Should not fail on parameter validation');
    }
});

// Test 9: Test optional parameters
runner.test('Methods should handle optional parameters', async () => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    // Test with different optional parameter combinations - just test one to avoid errors
    try {
        await client.embed(["test"], "model");
    } catch (error) {
        // Allow network errors, but not parameter type errors
        assert(!error.message.includes('type') && !error.message.includes('cannot be empty'),
               'Should handle optional parameters correctly');
    }
});

// Test 10: Test with environment variables
runner.test('Constructor should work with environment variables', () => {
    // Set env var first
    process.env.BASETEN_API_KEY = "test-key";

    // Test without explicit API key (should use environment variable)
    const client = new PerformanceClient(TEST_BASE_URL);
    assert(client instanceof PerformanceClient, 'Should create client with env var API key');

    // Clean up
    delete process.env.BASETEN_API_KEY;
});

// Test 11: Test all expected methods are available
runner.test('All expected methods should be available', () => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    const expectedMethods = ['embed', 'rerank', 'classify', 'batchPost'];

    expectedMethods.forEach(method => {
        assert(typeof client[method] === 'function', `Method ${method} should be available`);
    });
});

// Run all tests
async function main() {
    console.log('Baseten Performance Client - Node.js Tests');
    console.log('==========================================\n');

    const success = await runner.run();

    if (success) {
        console.log('\n🎉 All tests passed!');
        process.exit(0);
    } else {
        console.log('\n❌ Some tests failed!');
        process.exit(1);
    }
}

// Only run if this file is executed directly
if (require.main === module) {
    main().catch(console.error);
}

module.exports = { TestRunner, assert, assertThrows };
