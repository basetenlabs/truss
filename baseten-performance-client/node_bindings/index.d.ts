// The entry point re-exports the generated binding unchanged; index.js only adds call-time
// capture of the active OpenTelemetry span, which does not change any signature.
export * from './binding'
