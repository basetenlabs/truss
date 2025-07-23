#!/usr/bin/env node

const fs = require('fs');
const os = require('os');
const { PerformanceClient } = require('../node_bindings/performance-client.node');

// Configuration
const apiKey = process.env.BASETEN_API_KEY;
if (!apiKey) {
    throw new Error("BASETEN_API_KEY environment variable not set.");
}

const apiBaseEmbed = "https://model-yqv4yjjq.api.baseten.co/environments/production/sync";

// Benchmark settings
const benchmarkLengths = [64, 128, 256, 384, 512, 2048, 8192, 32768, 131072];
const microBatchSize = 16;

// Create clients
const clientB = new PerformanceClient(apiBaseEmbed, apiKey);
const clientOai = null; // We'll use fetch for OpenAI API calls

// Simplified resource monitoring for Node.js
class ResourceMonitor {
    constructor(intervalMs = 100) {
        this.memReadings = [];
        this.intervalMs = intervalMs;
        this.running = false;
        this.interval = null;
        this.initialCpu = null;
        this.maxMem = 0;
    }

    start() {
        this.running = true;
        this.memReadings = [];
        this.maxMem = 0;
        this.initialCpu = process.cpuUsage();

        this.interval = setInterval(() => {
            if (!this.running) return;

            const memUsage = process.memoryUsage();
            const memMB = memUsage.rss / (1024 * 1024);
            this.memReadings.push(memMB);
            this.maxMem = Math.max(this.maxMem, memMB);
        }, this.intervalMs);
    }

    stop() {
        this.running = false;
        if (this.interval) {
            clearInterval(this.interval);
        }
    }

    getStats() {
        // Simple CPU calculation (not as accurate as Python's psutil)
        const finalCpu = process.cpuUsage(this.initialCpu);
        const cpuPercent = (finalCpu.user + finalCpu.system) / 1000; // Very rough approximation

        return {
            maxCpu: cpuPercent / 10, // Scale down for comparison
            avgCpu: cpuPercent / 10,
            readings: this.memReadings.length,
            maxRam: this.maxMem
        };
    }
}

async function runBasetenBenchmark(length, clientName = "PerformanceClient") {
    console.log(`Running Baseten benchmark for length: ${length}`);

    // Prepare input data
    const fullInputTexts = Array(length).fill("Hello world");

    // Warm-up run
    await clientB.embed(
        Array(1024).fill("Hello world"),
        "text-embedding-3-small",
        null, // encodingFormat
        null, // dimensions
        null, // user
        512,  // maxConcurrentRequests
        1     // batchSize
    );

    // Setup monitoring
    const monitor = new ResourceMonitor();
    monitor.start();

    // Timed run
    const timeStart = performance.now();
    const response = await clientB.embed(
        fullInputTexts,
        "text-embedding-3-small",
        null, // encodingFormat
        null, // dimensions
        null, // user
        512,  // maxConcurrentRequests
        microBatchSize // batchSize
    );
    const timeEnd = performance.now();
    const duration = (timeEnd - timeStart) / 1000; // Convert to seconds

    // Stop monitoring
    monitor.stop();
    const stats = monitor.getStats();

    // Basic validations
    if (!response || !response.data || response.data.length !== length) {
        throw new Error(`Invalid response: expected ${length} embeddings, got ${response?.data?.length || 0}`);
    }

    return {
        client: clientName,
        length,
        duration,
        maxCpu: stats.maxCpu,
        avgCpu: stats.avgCpu,
        readings: stats.readings,
        maxRam: stats.maxRam
    };
}

async function runAsyncOpenAIBenchmark(length) {
    console.log(`Running AsyncOpenAI benchmark for length: ${length}`);

    // Prepare input data
    const inputTexts = Array(microBatchSize).fill("Hello world");
    const numTasks = Math.floor(length / microBatchSize);

    // Rate limiting
    let concurrentRequests = 0;
    const maxConcurrent = 512;

    async function createEmbedding() {
        while (concurrentRequests >= maxConcurrent) {
            await new Promise(resolve => setTimeout(resolve, 1));
        }

        concurrentRequests++;
        try {
            const response = await fetch(`${apiBaseEmbed}/v1/embeddings`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${apiKey}`
                },
                body: JSON.stringify({
                    input: inputTexts,
                    model: "text-embedding-3-small"
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } finally {
            concurrentRequests--;
        }
    }

    // Warm-up
    const warmupTasks = Math.min(numTasks, 16);
    await Promise.all(Array(warmupTasks).fill().map(() => createEmbedding()));

    // Setup monitoring
    const monitor = new ResourceMonitor();
    monitor.start();

    // Timed run
    const timeStart = performance.now();
    const apiResponses = await Promise.all(
        Array(numTasks).fill().map(() => createEmbedding())
    );

    const allEmbeddings = [];
    for (const res of apiResponses) {
        for (const emb of res.data) {
            allEmbeddings.push(emb.embedding);
        }
    }

    const timeEnd = performance.now();
    const duration = (timeEnd - timeStart) / 1000; // Convert to seconds

    // Stop monitoring
    monitor.stop();
    const stats = monitor.getStats();

    // Validation
    if (allEmbeddings.length !== length) {
        throw new Error(`Expected ${length} embeddings, got ${allEmbeddings.length}`);
    }

    return {
        client: "AsyncOpenAI",
        length,
        duration,
        maxCpu: stats.maxCpu,
        avgCpu: stats.avgCpu,
        readings: stats.readings,
        maxRam: stats.maxRam
    };
}

async function runAllBenchmarks() {
    console.log("Starting benchmark comparison for PerformanceClient and AsyncOpenAI");

    const results = [];

    for (const length of benchmarkLengths) {
        console.log(`\nRunning benchmark for length: ${length}, concurrent requests ${Math.floor(length / microBatchSize)}`);

        // Run Baseten Performance Client benchmark
        const resBaseten = await runBasetenBenchmark(length, "PerformanceClient");
        console.log(`PerformanceClient: duration=${resBaseten.duration.toFixed(4)} s, max_cpu=${resBaseten.maxCpu.toFixed(2)}%, max_ram=${resBaseten.maxRam.toFixed(2)} MB`);
        results.push(resBaseten);

        // Run AsyncOpenAI benchmark
        const resAsync = await runAsyncOpenAIBenchmark(length);
        console.log(`AsyncOpenAI      : duration=${resAsync.duration.toFixed(4)} s, max_cpu=${resAsync.maxCpu.toFixed(2)}%, max_ram=${resAsync.maxRam.toFixed(2)} MB`);
        results.push(resAsync);
    }

    return results;
}

function writeResultsCsv(results, filename = "benchmark_results_nodejs.csv") {
    const fieldnames = [
        "client",
        "length",
        "duration",
        "maxCpu",
        "avgCpu",
        "readings",
        "maxRam"
    ];

    let csvContent = fieldnames.join(",") + "\n";

    for (const result of results) {
        const row = fieldnames.map(field => {
            const value = result[field === "maxCpu" ? "maxCpu" :
                               field === "avgCpu" ? "avgCpu" :
                               field === "maxRam" ? "maxRam" : field];
            return typeof value === 'number' ? value.toString() : `"${value}"`;
        });
        csvContent += row.join(",") + "\n";
    }

    fs.writeFileSync(filename, csvContent);
    console.log(`\nBenchmark results saved to ${filename}`);
}

async function main() {
    try {
        const results = await runAllBenchmarks();
        writeResultsCsv(results);
    } catch (error) {
        console.error("Benchmark failed:", error);
        process.exit(1);
    }
}

if (require.main === module) {
    main();
}

module.exports = {
    runBasetenBenchmark,
    runAsyncOpenAIBenchmark,
    runAllBenchmarks,
    writeResultsCsv
};
