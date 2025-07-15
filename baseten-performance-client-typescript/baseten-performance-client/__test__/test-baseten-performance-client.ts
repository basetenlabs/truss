#!/usr/bin/env node
import { performance } from "perf_hooks";
import {
  PerformanceClient,
  type RerankResponse,
  type RerankResult,
} from "../index";

// Configuration
const BASE_URL = process.argv[2];
const API_KEY = process.env.BASETEN_API_KEY;

if (!API_KEY) {
  console.error("Error: BASETEN_API_KEY environment variable is required");
  process.exit(1);
}

// Test data - matching the format from qwen-rerank-api.ts
const RERANKER_INSTRUCTION_TEMPLATE =
  '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n<Instruct>: Retrieve relevant passages that answer or address the query\n<Query>: {query}\n<Document>: {doc}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n';

const TEST_DOCUMENTS = [
  "The capital of China is Beijing.",
  "Paris is the capital of France.",
  "Tokyo is the capital of Japan.",
  "London is the capital of the United Kingdom.",
  "Berlin is the capital of Germany.",
  "Rome is the capital of Italy.",
  "Madrid is the capital of Spain.",
  "Moscow is the capital of Russia.",
  "Ottawa is the capital of Canada.",
  "Washington D.C. is the capital of the United States.",
  "The Great Wall of China is one of the most famous landmarks in China.",
  "Chinese cuisine is known for its diverse flavors and regional specialties.",
  "Mandarin Chinese is the most widely spoken language in China.",
  "The Forbidden City is located in Beijing, China.",
  "Shanghai is the largest city in China by population.",
];

const TEST_QUERY = "What is the capital of China?";

// Helper functions
function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(2)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

async function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Test implementation using PerformanceClient
async function testPerformanceClient(
  documents: string[],
  query: string
): Promise<{ duration: number; response: RerankResponse }> {
  const client = new PerformanceClient(BASE_URL, API_KEY);

  // Format all texts with the instruction template
  const formattedTexts = documents.map((doc) =>
    RERANKER_INSTRUCTION_TEMPLATE.replace("{query}", query).replace(
      "{doc}",
      doc
    )
  );

  const start = performance.now();
  const classifyResponse = await client.classify(
    formattedTexts,
    false, // raw_scores
    true, // truncate
    "Right" // truncation_direction
  );
  const duration = performance.now() - start;

  // Convert ClassificationResponse to RerankResponse format
  const results = classifyResponse.data.map((scoresArr, idx) => ({
    index: idx,
    score: scoresArr.find((s) => s.label === "yes")?.score ?? 0,
    text: undefined,
  }));

  // Sort by score descending
  results.sort((a, b) => b.score - a.score);

  const rerankResponse: RerankResponse = {
    object: "list",
    data: results,
    totalTime: classifyResponse.totalTime,
    individualRequestTimes: classifyResponse.individualRequestTimes,
  };

  return { duration, response: rerankResponse };
}

// Main test runner
async function runTests() {
  console.log("🧪 Baseten Performance Client Integration Test\n");

  // Test 1: Basic functionality test
  console.log(
    "══════════════════════════════════════════════════════════════════════"
  );
  console.log("Test 1: Basic Rerank Functionality");
  console.log(
    "══════════════════════════════════════════════════════════════════════\n"
  );

  try {
    const response = await testPerformanceClient(
      TEST_DOCUMENTS.slice(0, 5),
      TEST_QUERY
    );

    console.log("✅ Basic rerank test passed");
    console.log(
      `   Results: ${response.response.data.length} documents ranked`
    );
    console.log("   Top 3 results:");
    response.response.data.slice(0, 3).forEach((result: RerankResult, i) => {
      console.log(
        `     ${i + 1}. Document ${
          result.index
        }: score = ${result.score.toFixed(4)}`
      );
    });
    console.log();
  } catch (error) {
    console.error(
      "❌ Basic rerank test failed:",
      error instanceof Error ? error.message : String(error)
    );
    process.exit(1);
  }

  // Test 2: Performance measurement
  console.log(
    "══════════════════════════════════════════════════════════════════════"
  );
  console.log("Test 2: Performance Measurement");
  console.log(
    "══════════════════════════════════════════════════════════════════════\n"
  );

  const numRuns = 3;
  let clientTimes: number[] = [];

  for (let run = 1; run <= numRuns; run++) {
    console.log(`📊 Run ${run}/${numRuns}:`);
    const clientResult = await testPerformanceClient(
      TEST_DOCUMENTS,
      TEST_QUERY
    );
    clientTimes.push(clientResult.duration);
    console.log(`    Duration: ${formatDuration(clientResult.duration)}`);
    console.log(
      `    Top result: Document ${
        clientResult.response.data[0].index
      } (score: ${clientResult.response.data[0].score.toFixed(4)})`
    );

    if (clientResult.response.individualRequestTimes) {
      const times = clientResult.response.individualRequestTimes;
      const avgBatchTime = times.reduce((a, b) => a + b, 0) / times.length;
      console.log(
        `    Average batch time: ${formatDuration(avgBatchTime * 1000)}`
      );
    }

    console.log();
    await sleep(1000); // Pause between runs
  }

  // Calculate averages
  const avgClientTime = clientTimes.reduce((a, b) => a + b, 0) / numRuns;

  console.log("📈 Performance Summary:");
  console.log(`   Average time per run: ${formatDuration(avgClientTime)}`);
  console.log(`   Documents processed: ${TEST_DOCUMENTS.length}`);
  console.log(
    `   Average throughput: ${(
      TEST_DOCUMENTS.length /
      (avgClientTime / 1000)
    ).toFixed(1)} docs/sec`
  );
  console.log(
    "\n   Note: PerformanceClient uses batching (4 docs/batch) with concurrent processing\n"
  );

  // Test 3: Error handling
  console.log(
    "══════════════════════════════════════════════════════════════════════"
  );
  console.log("Test 3: Error Handling and Validation");
  console.log(
    "══════════════════════════════════════════════════════════════════════\n"
  );

  const testCases = [
    {
      name: "Invalid API key",
      test: async () => {
        const client = new PerformanceClient(BASE_URL, "invalid-key");
        const formattedText = RERANKER_INSTRUCTION_TEMPLATE.replace(
          "{query}",
          TEST_QUERY
        ).replace("{doc}", "test");
        await client.classify(
          [formattedText],
          false, // raw_scores
          true, // truncate
          "Right" // truncation_direction
        );
      },
      shouldFail: true,
    },
    {
      name: "Empty documents array",
      test: async () => {
        const client = new PerformanceClient(BASE_URL, API_KEY);
        await client.classify(
          [],
          false, // raw_scores
          true, // truncate
          "Right" // truncation_direction
        );
      },
      shouldFail: true,
    },
    {
      name: "Empty document in array",
      test: async () => {
        const client = new PerformanceClient(BASE_URL, API_KEY);
        await client.classify(
          [""],
          false, // raw_scores
          true, // truncate
          "Right" // truncation_direction
        );
      },
      shouldFail: true,
    },
  ];

  for (const testCase of testCases) {
    try {
      await testCase.test();
      if (testCase.shouldFail) {
        console.log(`❌ ${testCase.name}: Expected error but succeeded`);
      } else {
        console.log(`✅ ${testCase.name}: Passed`);
      }
    } catch (error) {
      if (testCase.shouldFail) {
        console.log(
          `✅ ${testCase.name}: Correctly failed with: ${
            error instanceof Error ? error.message : String(error)
          }`
        );
      } else {
        console.log(
          `❌ ${testCase.name}: Unexpected error: ${
            error instanceof Error ? error.message : String(error)
          }`
        );
      }
    }
  }

  console.log();

  // Test 4: Performance at different scales
  console.log(
    "══════════════════════════════════════════════════════════════════════"
  );
  console.log("Test 4: Performance at Different Scales");
  console.log(
    "══════════════════════════════════════════════════════════════════════\n"
  );

  const documentCounts = [25, 50, 100, 200];
  const client = new PerformanceClient(BASE_URL, API_KEY);

  for (const count of documentCounts) {
    console.log(`📊 Testing with ${count} documents:`);

    // Generate test documents
    const testDocs = Array(count)
      .fill(null)
      .map((_, i) => {
        const baseDoc = TEST_DOCUMENTS[i % TEST_DOCUMENTS.length];
        return `Document ${i}: ${baseDoc}`;
      });

    // Test PerformanceClient
    const clientStart = performance.now();

    // Format all documents
    const formattedDocs = testDocs.map((doc) =>
      RERANKER_INSTRUCTION_TEMPLATE.replace("{query}", TEST_QUERY).replace(
        "{doc}",
        doc
      )
    );

    const classifyResponse = await client.classify(
      formattedDocs,
      false, // raw_scores
      true, // truncate
      "Right" // truncation_direction
    );

    const clientDuration = performance.now() - clientStart;
    const clientThroughput = (count / (clientDuration / 1000)).toFixed(1);

    console.log(`    Duration: ${formatDuration(clientDuration)}`);
    console.log(`    Throughput: ${clientThroughput} docs/sec`);

    if (classifyResponse.individualRequestTimes) {
      const times = classifyResponse.individualRequestTimes;
      console.log(
        `    Batches: ${times.length} (${Math.ceil(count / 4)} expected)`
      );
    }

    console.log();
    await sleep(1500); // Longer pause between different document counts
  }

  // Test 5: Large documents
  console.log(
    "══════════════════════════════════════════════════════════════════════"
  );
  console.log("Test 5: Large Documents (100 docs, ~500 words each)");
  console.log(
    "══════════════════════════════════════════════════════════════════════\n"
  );

  const longDocText =
    "This is a long document designed to test the performance of the reranking client with more realistic payloads. Each document will consist of this paragraph repeated multiple times to simulate a text of several hundred words. The goal is to see how the client handles larger request bodies and potentially longer processing times on the server side. We expect the controlled concurrency of the PerformanceClient to be even more beneficial in this scenario, as it prevents overwhelming the server with many large requests simultaneously, which could lead to timeouts or connection errors. This kind of workload is typical in Retrieval-Augmented Generation (RAG) systems where large chunks of text are retrieved from a vector database and then reranked for relevance before being passed to a large language model. The ability to efficiently process these large documents is critical for the overall performance and user experience of the system. Let's see how it performs under this more strenuous test. This is a long document designed to test the performance of the reranking client with more realistic payloads. Each document will consist of this paragraph repeated multiple times to simulate a text of several hundred words. The goal is to see how the client handles larger request bodies and potentially longer processing times on the server side. We expect the controlled concurrency of the PerformanceClient to be even more beneficial in this scenario, as it prevents overwhelming the server with many large requests simultaneously, which could lead to timeouts or connection errors. This kind of workload is typical in Retrieval-Augmented Generation (RAG) systems where large chunks of text are retrieved from a vector database and then reranked for relevance before being passed to a large language model. The ability to efficiently process these large documents is critical for the overall performance and user experience of the system. Let's see how it performs under this more strenuous test. This is a long document designed to test the performance of the reranking client with more realistic payloads. Each document will consist of this paragraph repeated multiple times to simulate a text of several hundred words. The goal is to see how the client handles larger request bodies and potentially longer processing times on the server side. We expect the controlled concurrency of the PerformanceClient to be even more beneficial in this scenario, as it prevents overwhelming the server with many large requests simultaneously, which could lead to timeouts or connection errors. This kind of workload is typical in Retrieval-Augmented Generation (RAG) systems where large chunks of text are retrieved from a vector database and then reranked for relevance before being passed to a large language model. The ability to efficiently process these large documents is critical for the overall performance and user experience of the system. Let's see how it performs under this more strenuous test.";

  const longDocumentSet = Array(100)
    .fill(null)
    .map((_, i) => `Document ${i}: ${longDocText}`);

  console.log("📊 Testing with 100 long documents:");

  // Test PerformanceClient
  const clientStart = performance.now();

  // Format all documents
  const formattedDocs = longDocumentSet.map((doc) =>
    RERANKER_INSTRUCTION_TEMPLATE.replace("{query}", TEST_QUERY).replace(
      "{doc}",
      doc
    )
  );

  const classifyResponse = await client.classify(
    formattedDocs,
    false, // raw_scores
    true, // truncate
    "Right" // truncation_direction
  );

  const clientDuration = performance.now() - clientStart;
  const clientThroughput = (100 / (clientDuration / 1000)).toFixed(1);

  console.log(`    Duration: ${formatDuration(clientDuration)}`);
  console.log(`    Throughput: ${clientThroughput} docs/sec`);

  if (classifyResponse.individualRequestTimes) {
    const times = classifyResponse.individualRequestTimes;
    console.log(
      `    Batches: ${times.length} (${Math.ceil(100 / 4)} expected)`
    );
    const avgBatchTime = times.reduce((a, b) => a + b, 0) / times.length;
    console.log(
      `    Average batch time: ${formatDuration(avgBatchTime * 1000)}`
    );
  }

  console.log();

  console.log(
    "\n══════════════════════════════════════════════════════════════════════"
  );
  console.log("✅ All tests completed successfully!");
  console.log(
    "══════════════════════════════════════════════════════════════════════"
  );
}

// Run the tests
runTests().catch((error) => {
  console.error("\n❌ Fatal error:", error);
  process.exit(1);
});
