import { createServer } from "node:http"
import test from 'ava'

import { PerformanceClient, HttpClientWrapper, RequestProcessingPreference } from '../index.js'

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

// Test HttpClientWrapper initialization
test('HttpClientWrapper initialization', (t) => {
  const wrapper = new HttpClientWrapper()
  t.truthy(wrapper, 'HttpClientWrapper should be initialized with default http_version')

  const wrapper1 = new HttpClientWrapper(1)
  t.truthy(wrapper1, 'HttpClientWrapper should be initialized with http_version=1')

  const wrapper2 = new HttpClientWrapper(2)
  t.truthy(wrapper2, 'HttpClientWrapper should be initialized with http_version=2')
})

// Test client initialization with HttpClientWrapper
test('PerformanceClient initialization with HttpClientWrapper', (t) => {
  const wrapper = new HttpClientWrapper(1)
  const client = new PerformanceClient('https://api.example.com', 'test-api-key', 1, wrapper)
  t.truthy(client, 'Client should be initialized with HttpClientWrapper')
})

// Test getClientWrapper method
test('client has getClientWrapper method', (t) => {
  const client = new PerformanceClient('https://api.example.com', 'test-api-key')
  t.is(typeof client.getClientWrapper, 'function', 'getClientWrapper should be a function')

  const wrapper = client.getClientWrapper()
  t.truthy(wrapper, 'getClientWrapper should return an HttpClientWrapper')
})

// Test sharing HttpClientWrapper between clients
test('HttpClientWrapper can be shared between clients', (t) => {
  const wrapper = new HttpClientWrapper(1)

  const client1 = new PerformanceClient('https://api1.example.com', 'test-api-key-1', 1, wrapper)
  const client2 = new PerformanceClient('https://api2.example.com', 'test-api-key-2', 1, wrapper)

  t.truthy(client1, 'First client should be initialized')
  t.truthy(client2, 'Second client should be initialized')
})

test('RequestProcessingPreference supports non-retryable status codes', (t) => {
  const preference = new RequestProcessingPreference(
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    [529, 429, 529],
  )

  t.deepEqual(preference.nonRetryableStatusCodes, [429, 529])
})

// Test getting wrapper from one client and using it in another
test('HttpClientWrapper from one client can be used in another', (t) => {
  const client1 = new PerformanceClient('https://api1.example.com', 'test-api-key-1')
  const wrapper = client1.getClientWrapper()

  const client2 = new PerformanceClient('https://api2.example.com', 'test-api-key-2', 1, wrapper)
  t.truthy(client2, 'Second client should be initialized with wrapper from first client')
})

test('explicit trace context reaches the server and can be disabled', async (t) => {
  const parent = '00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01'
  const seen = []
  const server = createServer((req, res) => {
    seen.push(req.headers)
    req.resume()
    res.setHeader('Content-Type', 'application/json')
    res.end(JSON.stringify({ok: true}))
  })
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
  t.teardown(() => server.close())
  const client = new PerformanceClient(`http://127.0.0.1:${server.address().port}`, 'test-key')
  const preference = new RequestProcessingPreference()
  preference.traceContext = {traceparent: parent, tracestate: 'vendor=value'}
  await client.batchPost('/echo', [{}], preference)
  preference.traceContext = undefined
  await client.batchPost('/echo', [{}], preference)
  t.is(seen[0].traceparent, parent)
  t.is(seen[0].tracestate, 'vendor=value')
  t.is(seen[1].traceparent, undefined)
  t.is(seen[1].tracestate, undefined)
})
