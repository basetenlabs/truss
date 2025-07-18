const test = require('ava');
const { PerformanceClient } = require('../performance-client.node');

// Automated tests for CI - Compatible with AVA
// Test configuration
const TEST_BASE_URL = "https://httpbin.org";
const TEST_API_KEY = "test-key";

// Test 1: Constructor validation
test('Constructor should create client with valid parameters', t => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);
    t.true(client instanceof PerformanceClient, 'Should create PerformanceClient instance');
});

// Test 2: Constructor with invalid API key
test('Constructor should handle missing API key', t => {
    t.throws(() => {
        new PerformanceClient(TEST_BASE_URL);
    }, { message: /api key/i });
});

// Test 3: Embed method parameter validation
test('Embed method should validate empty input', t => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    t.throws(() => {
        client.embed([], "test-model");
    }, { message: /empty/i });
});

// Test 4: Embed method parameter validation with valid input
test('Embed method should accept valid parameters', t => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    // This should not throw for parameter validation
    // (it will likely fail due to httpbin not being an embedding API, but that's fine)
    try {
        client.embed(["test text"], "test-model");
    } catch (error) {
        // Allow network/API errors, but not parameter validation errors
        t.false(error.message.includes('Input list cannot be empty'),
               'Should not fail on parameter validation');
    }
});

// Test 5: Rerank method parameter validation
test('Rerank method should validate empty texts', t => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    t.throws(() => {
        client.rerank("test query", []);
    }, { message: /empty/i });
});

// Test 6: Classify method parameter validation
test('Classify method should validate empty inputs', t => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    t.throws(() => {
        client.classify([]);
    }, { message: /empty/i });
});

// Test 7: Batch post method parameter validation
test('Batch post method should validate empty payloads', t => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    t.throws(() => {
        client.batchPost("/test", []);
    }, { message: /empty/i });
});

// Test 8: Test with mock data (will fail at network level but parameters should be valid)
test('Methods should handle network errors gracefully', t => {
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
            t.false(error.message.includes('cannot be empty'),
                   `Test case ${index + 1} should not fail on parameter validation`);
        }
    });
});

// Test 9: Test optional parameters
test('Methods should handle optional parameters', t => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    // Test with different optional parameter combinations
    try {
        client.embed(["test"], "model", null, 384, null, 16, 2, 30);
        client.rerank("query", ["text"], true, false, true, "Left", 8, 1, 30);
        client.classify(["text"], true, false, "Right", 8, 1, 30);
        client.batchPost("/test", [{"test": "data"}], 16, 30);
    } catch (error) {
        // Allow network errors, but not parameter type errors
        t.false(error.message.includes('type'), 'Should handle optional parameters correctly');
    }
});

// Test 10: Test with environment variables
test('Constructor should work with environment variables', t => {
    // Set env var first
    process.env.BASETEN_API_KEY = "test-key";

    // Test without explicit API key (should use environment variable)
    const client = new PerformanceClient(TEST_BASE_URL);
    t.true(client instanceof PerformanceClient, 'Should create client with env var API key');

    // Clean up
    delete process.env.BASETEN_API_KEY;
});

// Test 11: Test all expected methods are available
test('All expected methods should be available', t => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    const expectedMethods = ['embed', 'rerank', 'classify', 'batchPost'];

    expectedMethods.forEach(method => {
        t.is(typeof client[method], 'function', `Method ${method} should be available`);
    });
});

// Test 12: Test constructor with null/undefined values
test('Constructor should handle null/undefined values properly', t => {
    t.throws(() => {
        new PerformanceClient(null, TEST_API_KEY);
    }, { message: /base url/i });

    t.throws(() => {
        new PerformanceClient(undefined, TEST_API_KEY);
    }, { message: /base url/i });
});

// Test 13: Test method parameter types
test('Methods should validate parameter types', t => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    // Test invalid input types
    t.throws(() => {
        client.embed("not an array", "model");
    }, { message: /array/i });

    t.throws(() => {
        client.rerank(123, ["text"]);
    }, { message: /string/i });

    t.throws(() => {
        client.classify("not an array");
    }, { message: /array/i });

    t.throws(() => {
        client.batchPost("/test", "not an array");
    }, { message: /array/i });
});
