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
    // Note: The constructor doesn't throw for missing API key, it uses env var or default
    const client = new PerformanceClient(TEST_BASE_URL);
    t.true(client instanceof PerformanceClient, 'Should create client without explicit API key');
});

// Test 3: Embed method parameter validation
test('Embed method should validate empty input', async t => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    await t.throwsAsync(async () => {
        await client.embed([], "test-model");
    }, { message: /Input list cannot be empty/i });
});

// Test 4: Embed method parameter validation with valid input
test('Embed method should accept valid parameters', async t => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    // This should not throw for parameter validation
    // (it will likely fail due to httpbin not being an embedding API, but that's fine)
    try {
        await client.embed(["test text"], "test-model");
        t.pass('Should not throw for valid parameters');
    } catch (error) {
        // Allow network/API errors, but not parameter validation errors
        t.false(error.message.includes('Input list cannot be empty'),
               'Should not fail on parameter validation');
        t.pass('Network errors are acceptable');
    }
});

// Test 5: Rerank method parameter validation
test('Rerank method should validate empty texts', async t => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    await t.throwsAsync(async () => {
        await client.rerank("test query", []);
    }, { message: /Texts list cannot be empty/i });
});

// Test 6: Classify method parameter validation
test('Classify method should validate empty inputs', async t => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    await t.throwsAsync(async () => {
        await client.classify([]);
    }, { message: /Inputs list cannot be empty/i });
});

// Test 7: Batch post method parameter validation
test('Batch post method should validate empty payloads', async t => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    await t.throwsAsync(async () => {
        await client.batchPost("/test", []);
    }, { message: /Payloads list cannot be empty/i });
});

// Test 8: Test with mock data (will fail at network level but parameters should be valid)
test('Methods should handle network errors gracefully', async t => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    // These should fail with network errors, not parameter errors
    const testCases = [
        () => client.embed(["test"], "model"),
        () => client.rerank("query", ["text1", "text2"]),
        () => client.classify(["text1", "text2"]),
        () => client.batchPost("/test", [{"key": "value"}])
    ];

    for (let i = 0; i < testCases.length; i++) {
        try {
            await testCases[i]();
            t.pass(`Test case ${i + 1} succeeded unexpectedly`);
        } catch (error) {
            // Should be network/API errors, not parameter validation errors
            t.false(error.message.includes('cannot be empty'),
                   `Test case ${i + 1} should not fail on parameter validation`);
            t.pass(`Test case ${i + 1} failed with expected network error`);
        }
    }
});

// Test 9: Test optional parameters
test('Methods should handle optional parameters', async t => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    // Test with different optional parameter combinations
    try {
        await client.embed(["test"], "model", null, 384, null, null, null, null);
        await client.rerank("query", ["text"], true, false, true, "Left", null, null, null);
        await client.classify(["text"], true, false, "Right", null, null, null);
        await client.batchPost("/test", [{"test": "data"}], null, null);
        t.pass('All optional parameter combinations handled correctly');
    } catch (error) {
        // Allow network errors, but not parameter type errors
        if (!error.message.includes('type') && !error.message.includes('cannot be empty')) {
            t.pass('Network errors are acceptable with optional parameters');
        } else {
            t.fail('Should handle optional parameters correctly: ' + error.message);
        }
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
    }, { message: /Failed to convert JavaScript value/i });

    t.throws(() => {
        new PerformanceClient(undefined, TEST_API_KEY);
    }, { message: /Failed to convert JavaScript value/i });
});

// Test 13: Test method parameter types
test('Methods should validate parameter types', async t => {
    const client = new PerformanceClient(TEST_BASE_URL, TEST_API_KEY);

    // Test invalid input types
    await t.throwsAsync(async () => {
        await client.embed("not an array", "model");
    }, { message: /Given napi value is not an array/i });

    await t.throwsAsync(async () => {
        await client.rerank(123, ["text"]);
    }, { message: /Failed to convert JavaScript value/i });

    await t.throwsAsync(async () => {
        await client.classify("not an array");
    }, { message: /Given napi value is not an array/i });

    await t.throwsAsync(async () => {
        await client.batchPost("/test", "not an array");
    }, { message: /Given napi value is not an array/i });
});
