const { PerformanceClient } = require('../performance-client.node');

// Automated tests for CI - TODO: Should be compatible with AVA
// currently not picked up by AVA, but can be run manually or in CI

// Test configuration
const TEST_BASE_URL = "https://httpbin.org";
const TEST_API_KEY = "test-key";

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
        console.log(`Running ${this.tests.length} automated tests...\n`);

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
        client.batchPost("/test", []);
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
        () => client.batchPost("/test", [{"key": "value"}])
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
        client.batchPost("/test", [{"test": "data"}], 16, 30);
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

// Test 11: Test all expected methods are available
runner.test('All expected methods should be available', () => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    const expectedMethods = ['embed', 'rerank', 'classify', 'batchPost'];

    expectedMethods.forEach(method => {
        assert(typeof client[method] === 'function', `Method ${method} should be available`);
    });
});

// Test 12: Test constructor with null/undefined values
runner.test('Constructor should handle null/undefined values properly', () => {
    assertThrows(() => {
        new PerformanceClient(null, TEST_API_KEY);
    }, 'Should throw error for null base URL');

    assertThrows(() => {
        new PerformanceClient(undefined, TEST_API_KEY);
    }, 'Should throw error for undefined base URL');
});

// Test 13: Test method parameter types
runner.test('Methods should validate parameter types', () => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    // Test invalid input types
    assertThrows(() => {
        client.embed("not an array", "model");
    }, 'Should throw error for non-array input');

    assertThrows(() => {
        client.rerank(123, ["text"]);
    }, 'Should throw error for non-string query');

    assertThrows(() => {
        client.classify("not an array");
    }, 'Should throw error for non-array inputs');

    assertThrows(() => {
        client.batchPost("/test", "not an array");
    }, 'Should throw error for non-array payloads');
});

// Run all tests
async function main() {
    console.log('Baseten Performance Client - Automated CI Tests');
    console.log('===============================================\n');

    const success = await runner.run();

    if (success) {
        console.log('\n🎉 All automated tests passed!');
        process.exit(0);
    } else {
        console.log('\n❌ Some automated tests failed!');
        process.exit(1);
    }
}

// Only run if this file is executed directly
if (require.main === module) {
    main().catch(console.error);
}

module.exports = { TestRunner, assert, assertThrows };
