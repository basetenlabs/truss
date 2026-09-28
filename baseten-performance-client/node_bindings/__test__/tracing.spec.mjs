import test from 'ava'
import http from 'node:http'
import { createRequire } from 'node:module'
import { ROOT_CONTEXT, context, createTraceState, trace } from '@opentelemetry/api'

import { PerformanceClient, RequestProcessingPreference } from '../index.js'

const binding = createRequire(import.meta.url)('../binding.js')

const PARENT = '00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01'

// Stand-in for an SDK's context manager: calls made inside `with` see its context as active.
class StackContextManager {
  constructor() {
    this.current = ROOT_CONTEXT
  }
  active() {
    return this.current
  }
  with(ctx, fn, thisArg, ...args) {
    const previous = this.current
    this.current = ctx
    try {
      return fn.call(thisArg, ...args)
    } finally {
      this.current = previous
    }
  }
  bind(_ctx, target) {
    return target
  }
  enable() {
    return this
  }
  disable() {
    this.current = ROOT_CONTEXT
    return this
  }
}
context.setGlobalContextManager(new StackContextManager())

// Records the trace headers of each request and answers like an embeddings endpoint.
async function startServer() {
  const seen = []
  const server = http.createServer((req, res) => {
    let body = ''
    req.on('data', (chunk) => (body += chunk))
    req.on('end', () => {
      seen.push({ traceparent: req.headers.traceparent, tracestate: req.headers.tracestate })
      const request = JSON.parse(body)
      const payload = JSON.stringify({
        object: 'list',
        data: request.input.map((_, index) => ({ object: 'embedding', embedding: [0.1], index })),
        model: request.model,
        usage: { prompt_tokens: 1, total_tokens: 1 },
      })
      res.writeHead(200, { 'Content-Type': 'application/json' })
      res.end(payload)
    })
  })
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
  const client = new PerformanceClient(`http://127.0.0.1:${server.address().port}`, 'test-key')
  return { client, seen, close: () => new Promise((resolve) => server.close(resolve)) }
}

function inSpan(fn) {
  const span = trace.wrapSpanContext({
    traceId: '0af7651916cd43dd8448eb211c80319c',
    spanId: 'b7ad6b7169203331',
    traceFlags: 0,
    traceState: createTraceState('vendor=opaque'),
  })
  return context.with(trace.setSpan(context.active(), span), fn)
}

test.serial('export is off in these tests', (t) => {
  t.falsy(process.env.BASETEN_PERFORMANCE_CLIENT_OTLP_ENDPOINT)
})

test.serial('the active span is the parent, with its flags and tracestate', async (t) => {
  const { client, seen, close } = await startServer()
  try {
    await inSpan(() => client.embed(['hello'], 'test-model'))
    t.deepEqual(seen, [
      {
        traceparent: '00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-00',
        tracestate: 'vendor=opaque',
      },
    ])
  } finally {
    await close()
  }
})

test.serial('an explicit traceparent beats the active span', async (t) => {
  const { client, seen, close } = await startServer()
  try {
    const preference = new RequestProcessingPreference(
      ...Array(15).fill(undefined),
      PARENT,
    )
    await inSpan(() => client.embed(['hello'], 'test-model', null, null, null, preference))
    t.deepEqual(seen, [{ traceparent: PARENT, tracestate: undefined }])
  } finally {
    await close()
  }
})

test.serial('a traceparent in extraHeaders beats the active span', async (t) => {
  const { client, seen, close } = await startServer()
  try {
    const preference = new RequestProcessingPreference(
      ...Array(13).fill(undefined),
      { Traceparent: PARENT },
    )
    t.true(preference.hasExplicitTraceContext)
    await inSpan(() => client.embed(['hello'], 'test-model', null, null, null, preference))
    t.deepEqual(seen, [{ traceparent: PARENT, tracestate: undefined }])
  } finally {
    await close()
  }
})

test.serial('without an active span no traceparent is sent', async (t) => {
  const { client, seen, close } = await startServer()
  try {
    await client.embed(['hello'], 'test-model')
    t.deepEqual(seen, [{ traceparent: undefined, tracestate: undefined }])
  } finally {
    await close()
  }
})

test('the exported client is still the native class', (t) => {
  const client = new PerformanceClient('https://api.example.com', 'test-key')
  t.true(client instanceof binding.PerformanceClient)
})

test('withTraceContext copies the preference and sets the context', (t) => {
  const original = new RequestProcessingPreference(4)
  const traced = original.withTraceContext(PARENT, 'vendor=opaque')
  t.is(traced.maxConcurrentRequests, 4)
  t.is(traced.traceparent, PARENT)
  t.is(traced.tracestate, 'vendor=opaque')
  t.is(original.traceparent, null)
})
