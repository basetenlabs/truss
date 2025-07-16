export interface OpenAIEmbeddingData {
  object: string;
  embedding: number[];
  index: number;
}

export interface OpenAIUsage {
  prompt_tokens: number;
  total_tokens: number;
}

export interface OpenAIEmbeddingsResponse {
  object: string;
  data: OpenAIEmbeddingData[];
  model: string;
  usage: OpenAIUsage;
  total_time?: number;
  individual_request_times?: number[];
}

export interface RerankResult {
  index: number;
  score: number;
  text?: string;
}

export interface RerankResponse {
  object: string;
  data: RerankResult[];
  total_time?: number;
  individual_request_times?: number[];
}

export interface ClassificationResult {
  label: string;
  score: number;
}

export interface ClassificationResponse {
  object: string;
  data: ClassificationResult[][];
  total_time?: number;
  individual_request_times?: number[];
}

export interface BatchPostResponse {
  data: any[];
  response_headers: Record<string, string>[];
  individual_request_times: number[];
  total_time: number;
}

export class PerformanceClient {
  constructor(baseUrl: string, apiKey?: string);

  embed(
    input: string[],
    model: string,
    encodingFormat?: string | null,
    dimensions?: number | null,
    user?: string | null,
    maxConcurrentRequests?: number,
    batchSize?: number,
    timeoutS?: number
  ): OpenAIEmbeddingsResponse;

  rerank(
    query: string,
    texts: string[],
    rawScores?: boolean,
    returnText?: boolean,
    truncate?: boolean,
    truncationDirection?: string,
    maxConcurrentRequests?: number,
    batchSize?: number,
    timeoutS?: number
  ): RerankResponse;

  classify(
    inputs: string[],
    rawScores?: boolean,
    truncate?: boolean,
    truncationDirection?: string,
    maxConcurrentRequests?: number,
    batchSize?: number,
    timeoutS?: number
  ): ClassificationResponse;

  batch_post(
    urlPath: string,
    payloads: any[],
    maxConcurrentRequests?: number,
    timeoutS?: number
  ): BatchPostResponse;
}
