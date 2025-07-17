import test from 'ava'

import { PerformanceClient } from '../index.js'

// Test client initialization
test('PerformanceClient initialization', (t) => {
  const client = new PerformanceClient('https://api.example.com', 'test-api-key')
  t.truthy(client, 'Client should be initialized')
})
