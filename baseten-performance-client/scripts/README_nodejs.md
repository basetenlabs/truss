# Node.js Performance Benchmark Script

## Overview

`compare_latency_openai.js` is a Node.js equivalent of the Python `compare_latency_openai.py` script. It benchmarks the performance of the Baseten PerformanceClient against direct OpenAI API calls using fetch.

## Prerequisites

1. **Node.js** (version 10 or higher)
2. **Built Node.js bindings** - Run `npm run build` in the `node_bindings/` directory
3. **Environment variables**:
   - `BASETEN_API_KEY`: Your Baseten API key

## Usage

```bash
# Set your API key
export BASETEN_API_KEY="your_api_key_here"

# Run the benchmark
node scripts/compare_latency_openai.js
```

## Features

The script benchmarks:
- **PerformanceClient**: Using the Rust-based performance client via Node.js bindings
- **AsyncOpenAI equivalent**: Using Node.js fetch for direct API calls

### Benchmark Parameters

- **Test lengths**: 64, 128, 256, 384, 512, 2048, 8192, 32768, 131072 embeddings
- **Micro batch size**: 16 (for splitting large requests)
- **Max concurrent requests**: 512
- **Model**: text-embedding-3-small

### Monitoring

The script monitors:
- **Duration**: Total time for each benchmark
- **Memory usage**: Peak RSS memory usage during execution
- **CPU usage**: Simplified CPU usage approximation (less accurate than Python version)

## Output

Results are saved to `benchmark_results_nodejs.csv` with columns:
- `client`: PerformanceClient or AsyncOpenAI
- `length`: Number of input texts
- `duration`: Time in seconds
- `maxCpu`: Peak CPU usage percentage
- `avgCpu`: Average CPU usage percentage
- `readings`: Number of monitoring samples
- `maxRam`: Peak memory usage in MB

## Differences from Python Version

1. **CPU Monitoring**: Node.js version uses `process.cpuUsage()` which is less precise than Python's `psutil`
2. **HTTP2 Support**: Currently only tests HTTP/1.1 PerformanceClient (HTTP/2 support can be added)
3. **OpenAI Client**: Uses fetch instead of the official OpenAI Node.js client for simpler dependencies

## API Compatibility

The Node.js PerformanceClient API:
```javascript
await client.embed(
    texts,                    // Array<string>
    model,                    // string
    encodingFormat,           // string | null
    dimensions,               // number | null
    user,                     // string | null
    maxConcurrentRequests,    // number
    batchSize,               // number
    timeoutS                 // number
);
```

## Troubleshooting

1. **Module not found**: Ensure Node.js bindings are built (`npm run build` in `node_bindings/`)
2. **API errors**: Verify `BASETEN_API_KEY` is set and valid
3. **Memory issues**: Large benchmark lengths may require sufficient system memory
