const { PerformanceClient } = require('../index.js');

// Test configuration
const TEST_BASE_URL = process.env.TEST_BASE_URL || "https://httpbin.org";
const TEST_API_KEY = process.env.TEST_API_KEY || "test-key";

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
        console.log(`Running ${this.tests.length} tests...\n`);

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
    assertThrows(() => {
        new PerformanceClient(TEST_BASE_URL);
    }, 'Should throw error for missing API key');
});

// Test 3: Embed method parameter validation
runner.test('Embed method should validate empty input', () => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    assertThrows(() => {
        client.embed([], "test-model");
    }, 'Should throw error for empty input array');
});

// Test 4: Embed method parameter validation with valid input
runner.test('Embed method should accept valid parameters', () => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    // This should not throw for parameter validation
    // (it will likely fail due to httpbin not being an embedding API, but that's fine)
    try {
        client.embed(["test text"], "test-model");
    } catch (error) {
        // Allow network/API errors, but not parameter validation errors
        assert(!error.message.includes('Input list cannot be empty'),
               'Should not fail on parameter validation');
    }
});

// Test 5: Rerank method parameter validation
runner.test('Rerank method should validate empty texts', () => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    assertThrows(() => {
        client.rerank("test query", []);
    }, 'Should throw error for empty texts array');
});

// Test 6: Classify method parameter validation
runner.test('Classify method should validate empty inputs', () => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    assertThrows(() => {
        client.classify([]);
    }, 'Should throw error for empty inputs array');
});

// Test 7: Batch post method parameter validation
runner.test('Batch post method should validate empty payloads', () => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    assertThrows(() => {
        client.batch_post("/test", []);
    }, 'Should throw error for empty payloads array');
});

// Test 8: Test with mock data (will fail at network level but parameters should be valid)
runner.test('Methods should handle network errors gracefully', () => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    // These should fail with network errors, not parameter errors
    const testCases = [
        () => client.embed(["test"], "model"),
        () => client.rerank("query", ["text1", "text2"]),
        () => client.classify(["text1", "text2"]),
        () => client.batch_post("/test", [{"key": "value"}])
    ];

    testCases.forEach((testCase, index) => {
        try {
            testCase();
        } catch (error) {
            // Should be network/API errors, not parameter validation errors
            assert(!error.message.includes('cannot be empty'),
                   `Test case ${index + 1} should not fail on parameter validation`);
        }
    });
});

// Test 9: Test optional parameters
runner.test('Methods should handle optional parameters', () => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    // Test with different optional parameter combinations
    try {
        client.embed(["test"], "model", null, 384, null, 16, 2, 30);
        client.rerank("query", ["text"], true, false, true, "Left", 8, 1, 30);
        client.classify(["text"], true, false, "Right", 8, 1, 30);
        client.batch_post("/test", [{"test": "data"}], 16, 30);
    } catch (error) {
        // Allow network errors, but not parameter type errors
        assert(!error.message.includes('type'), 'Should handle optional parameters correctly');
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
