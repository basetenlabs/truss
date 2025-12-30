// Concise parameterized test for HTTP method functionality
const { PerformanceClient } = require('./index.js');

// Test configuration
const testCases = [
  { method: 'GET', path: '/get' },
  { method: 'POST', path: '/post' },
  { method: 'PUT', path: '/put' },
  { method: 'PATCH', path: '/patch' },
  { method: 'DELETE', path: '/delete' },
  { method: null, path: '/post' }, // Default to POST
];

async function runTests() {
  console.log('🧪 Starting HTTP method tests...\n');

  const client = new PerformanceClient('https://httpbin.org', 'test-key');
  const payloads = [{ test: 'data' }];

  let passed = 0;
  let failed = 0;

  // Test valid methods
  for (const { method, path } of testCases) {
    try {
      const args = method ? [path, payloads, undefined, undefined, method] : [path, payloads];
      const response = await client.batchPost(...args);

      if (response.data.length === 1 && response.total_time > 0) {
        console.log(`✅ ${method || 'default'} -> ${path}`);
        passed++;
      } else {
        console.log(`❌ ${method || 'default'} -> Invalid response`);
        failed++;
      }
    } catch (e) {
      console.log(`❌ ${method || 'default'} -> ${e.message}`);
      failed++;
    }
  }

  // Test invalid method
  try {
    await client.batchPost('/post', payloads, undefined, undefined, 'INVALID');
    console.log('❌ INVALID method -> Should have failed');
    failed++;
  } catch (e) {
    if (e.message.includes('Invalid HTTP method')) {
      console.log('✅ INVALID method -> Correctly rejected');
      passed++;
    } else {
      console.log(`❌ INVALID method -> Wrong error: ${e.message}`);
      failed++;
    }
  }

  console.log(`\n📊 Results: ${passed} passed, ${failed} failed`);

  if (failed === 0) {
    console.log('🎉 All tests passed!');
    process.exit(0);
  } else {
    console.log('💥 Some tests failed!');
    process.exit(1);
  }
}

runTests().catch(console.error);
