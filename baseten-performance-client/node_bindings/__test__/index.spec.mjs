import test from 'ava'

import { PerformanceClient } from '../index.js'

// Test client initialization
test('PerformanceClient initialization', (t) => {
  const client = new PerformanceClient('https://api.example.com', 'test-api-key')
  t.truthy(client, 'Client should be initialized')
})

// Test client has embed method
test('client has embed method', (t) => {
  const client = new PerformanceClient('https://api.example.com', 'test-api-key')
  t.is(typeof client.embed, 'function', 'embed should be a function')
})

// Test client has rerank method
test('client has rerank method', (t) => {
  const client = new PerformanceClient('https://api.example.com', 'test-api-key')
  t.is(typeof client.rerank, 'function', 'rerank should be a function')
})

// Test client has classify method
test('client has classify method', (t) => {
  const client = new PerformanceClient('https://api.example.com', 'test-api-key')
  t.is(typeof client.classify, 'function', 'classify should be a function')
})

// Test client has batchPost method
test('client has batchPost method', (t) => {
  const client = new PerformanceClient('https://api.example.com', 'test-api-key')
  t.is(typeof client.batchPost, 'function', 'batchPost should be a function')
})
