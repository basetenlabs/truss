# This file is automatically by AI
# ruff: noqa: E501, F401, F811

import builtins
import typing

class OpenAIEmbeddingData:
    """
    Represents a single embedding object.

    Attributes:
        object: The object type (usually a string identifier).
        index: Position of the embedding in the response.
    """

    object: builtins.str
    index: builtins.int

    @property
    def embedding(self) -> typing.Union[builtins.str, builtins.list[builtins.float]]:
        """
        The embedding vector. It may be either a list of floats or a base64 encoded string.

        Returns:
            Either a list of floats (when the embedding is decoded) or a string.
        """
        ...

class OpenAIUsage:
    """
    Represents token usage information for an API request.

    Attributes:
        prompt_tokens: The number of tokens used for the prompt.
        total_tokens: The total number of tokens used.
    """

    prompt_tokens: builtins.int
    total_tokens: builtins.int

class OpenAIEmbeddingsResponse:
    """
    Represents the response returned by an embeddings API request.

    Attributes:
        object: The object type (e.g. "list").
        data: A list of OpenAIEmbeddingData objects.
        model: The model identifier used for the embedding.
        usage: Usage details such as token counts.
        total_time: Optional total time taken for the operation in seconds.
        individual_request_times: Optional list of individual batch request times in seconds.
        response_headers: A list of dictionaries, where each dictionary contains
                          the response headers for the corresponding batch request.

    Methods:
        numpy() -> _NDArrayF32:
            Converts the embeddings data into a 2D NumPy array.

    Example:
        >>> response = client.embed(["Hello"], model="model-id")
        >>> array = response.numpy()
        >>> print(f"Total time: {response.total_time}")
    """

    object: builtins.str
    data: builtins.list[OpenAIEmbeddingData]
    model: builtins.str
    usage: OpenAIUsage
    total_time: typing.Optional[builtins.float]
    individual_request_times: typing.Optional[builtins.list[builtins.float]]
    response_headers: builtins.list[builtins.dict[builtins.str, builtins.str]]

    def numpy(self) -> typing.Any:
        """
        Converts the embeddings data into a 2D NumPy array.

        Each row in the array corresponds to an embedding. The array's type will
        be float32 and its shape will be (number_of_embeddings, embedding_dimension).

        Returns:
            numpy.ndarray: A 2D NumPy array of f32.

        Raises:
            ValueError: If an embedding is not a float vector, if there are inconsistent dimensions,
                        or if the data is empty.
            ImportError: If NumPy is not installed in the Python environment.

        Example:
            >>> array = response.numpy()
            >>> print(array.shape)
        """
        ...

class RerankResult:
    """
    Represents a single result from a rerank operation.

    Attributes:
        index: The index of the result.
        score: The rerank score.
        text: The optional original text.
    """

    index: builtins.int
    score: builtins.float
    text: typing.Optional[builtins.str]

class RerankResponse:
    """
    Represents the response for a rerank API request.

    Attributes:
        object: The object type (usually "list").
        data: A list of RerankResult objects.
        total_time: Optional total time taken for the operation in seconds.
        individual_request_times: Optional list of individual batch request times in seconds.
        response_headers: A list of dictionaries, where each dictionary contains
                          the response headers for the corresponding batch request.

    Example:
        >>> response = client.rerank("query", ["doc1", "doc2"])
        >>> for result in response.data:
        ...     print(result.index, result.score)
        >>> print(f"Total time: {response.total_time}")
    """

    object: builtins.str
    data: builtins.list[RerankResult]
    total_time: typing.Optional[builtins.float]
    individual_request_times: typing.Optional[builtins.list[builtins.float]]
    response_headers: builtins.list[builtins.dict[builtins.str, builtins.str]]

    def __init__(
        self,
        data: builtins.list[RerankResult],
        total_time: typing.Optional[builtins.float] = None,
        individual_request_times: typing.Optional[builtins.list[builtins.float]] = None,
        response_headers: typing.Optional[
            builtins.list[builtins.dict[builtins.str, builtins.str]]
        ] = None,
    ) -> None:
        """
        Initializes a RerankResponse.

        Args:
            data: A list of RerankResult objects.
            total_time: Optional total time for the operation.
            individual_request_times: Optional list of individual batch request times.
            response_headers: Optional list of response headers per batch request.
        """
        ...

class ClassificationResult:
    """
    Represents a single classification result.

    Attributes:
        label: The predicted classification label.
        score: The confidence score.
    """

    label: builtins.str
    score: builtins.float

class ClassificationResponse:
    """
    Represents the response from a classification API request.

    Attributes:
        object: The object type (usually "list").
        data: A nested list of ClassificationResult objects.
        total_time: Optional total time taken for the operation in seconds.
        individual_request_times: Optional list of individual batch request times in seconds.
        response_headers: A list of dictionaries, where each dictionary contains
                          the response headers for the corresponding batch request.

    Example:
        >>> response = client.classify(["text1", "text2"])
        >>> for result_group in response.data:
        ...     for result in result_group:
        ...         print(result.label, result.score)
        >>> print(f"Total time: {response.total_time}")
    """

    object: builtins.str
    data: builtins.list[builtins.list[ClassificationResult]]
    total_time: typing.Optional[builtins.float]
    individual_request_times: typing.Optional[builtins.list[builtins.float]]
    response_headers: builtins.list[builtins.dict[builtins.str, builtins.str]]

    def __init__(
        self,
        data: builtins.list[builtins.list[ClassificationResult]],
        total_time: typing.Optional[builtins.float] = None,
        individual_request_times: typing.Optional[builtins.list[builtins.float]] = None,
        response_headers: typing.Optional[
            builtins.list[builtins.dict[builtins.str, builtins.str]]
        ] = None,
    ) -> None:
        """
        Initializes a ClassificationResponse.

        Args:
            data: A list where each element is a list of ClassificationResult objects.
            total_time: Optional total time for the operation.
            individual_request_times: Optional list of individual batch request times.
            response_headers: Optional list of response headers per batch request.
        """
        ...

class RequestProcessingPreference:
    """
    Configuration for request processing with user-defined preferences.

    This class allows you to specify custom parameters for request processing,
    including concurrency limits, timeouts, hedging behavior, and budget percentages.
    All parameters are optional - defaults will be applied during processing.

    Attributes:
        max_concurrent_requests: Maximum number of parallel requests (default: 128).
        batch_size: Number of items per batch (default: 128).
        max_chars_per_request: Optional character-based batching limit.
        timeout_s: Per-request timeout in seconds (default: 3600.0).
        hedge_delay: Optional request hedging delay in seconds.
        total_timeout_s: Optional total timeout for the entire operation in seconds.
        hedge_budget_pct: Hedge budget percentage (default: 0.10).
        retry_budget_pct: Retry budget percentage (default: 0.05).
        max_retries: Maximum number of HTTP retries (default: 4).
        initial_backoff_ms: Initial backoff duration in milliseconds (default: 125).
        cancel_token: Optional CancellationToken for cancelling operations.

    Example:
        >>> # Use all defaults
        >>> preference = RequestProcessingPreference()
        >>>
        >>> # Custom concurrency and timeout via constructor
        >>> preference = RequestProcessingPreference(
        ...     max_concurrent_requests=64,
        ...     timeout_s=30.0
        ... )
        >>>
        >>> # Or use property setters for more flexibility
        >>> preference = RequestProcessingPreference()
        >>> preference.max_concurrent_requests = 64
        >>> preference.timeout_s = 30.0
        >>> preference.hedge_delay = 0.5
        >>> preference.hedge_budget_pct = 0.15
        >>> preference.max_retries = 3
        >>> preference.initial_backoff_ms = 250
        >>>
        >>> # With cancellation token
        >>> token = CancellationToken()
        >>> preference = RequestProcessingPreference(
        ...     max_concurrent_requests=64,
        ...     cancel_token=token
        ... )
        >>> # Later cancel the operation
        >>> token.cancel()
    """

    def __init__(
        self,
        max_concurrent_requests: typing.Optional[int] = None,
        batch_size: typing.Optional[int] = None,
        max_chars_per_request: typing.Optional[int] = None,
        timeout_s: typing.Optional[float] = None,
        hedge_delay: typing.Optional[float] = None,
        total_timeout_s: typing.Optional[float] = None,
        hedge_budget_pct: typing.Optional[float] = None,
        retry_budget_pct: typing.Optional[float] = None,
        max_retries: typing.Optional[int] = None,
        initial_backoff_ms: typing.Optional[int] = None,
        cancel_token: typing.Optional[CancellationToken] = None,
    ) -> None:
        """
                Initialize a RequestProcessingPreference with optional parameters.

                Args:
                    max_concurrent_requests: Maximum parallel requests (default: 128).
                    batch_size: Number of items per batch (default: 128).
                    max_chars_per_request: Optional character-based batching limit.
                    timeout_s: Per-request timeout in seconds (default: 3600.0).
                    hedge_delay: Optional request hedging delay in seconds.
                    total_timeout_s: Optional total timeout for the entire operation in seconds.
                    hedge_budget_pct: Hedge budget percentage (default: 0.10).
                    retry_budget_pct: Retry budget percentage (default: 0.05).
                    max_retries: Maximum number of HTTP retries (default: 4).
                    initial_backoff_ms: Initial backoff duration in milliseconds (default: 125).
        cancel_token: Optional CancellationToken for cancelling operations.
        """

    # Property definitions with type hints
    max_concurrent_requests: builtins.int
    batch_size: builtins.int
    max_chars_per_request: typing.Optional[builtins.int]
    timeout_s: builtins.float
    hedge_delay: typing.Optional[builtins.float]
    total_timeout_s: typing.Optional[builtins.float]
    hedge_budget_pct: builtins.float
    retry_budget_pct: builtins.float
    max_retries: builtins.int
    initial_backoff_ms: builtins.int
    cancel_token: typing.Optional[CancellationToken]

    @classmethod
    def default(cls) -> "RequestProcessingPreference":
        """
        Create a new preference with default values.

        Returns:
            RequestProcessingPreference with all default values applied.
        """
        ...

    ...

class CancellationToken:
    """
    CancellationToken for cancelling async operations.

    This token can be used to cancel long-running operations.

    Attributes:
        (No public attributes, only methods)

    Example:
        >>> token = CancellationToken()
        >>> # Pass token to RequestProcessingPreference
        >>> preference = RequestProcessingPreference(cancel_token=token)
        >>> # Later cancel the operation
        >>> token.cancel()
    """

    def __init__(self) -> None:
        """
        Create a new cancellation token.
        """
        ...

    def cancel(self) -> None:
        """
        Cancel all operations using this token.

        Once cancelled, the token cannot be un-cancelled.
        Any operation checking this token will see the cancelled state.
        """
        ...

    def is_cancelled(self) -> builtins.bool:
        """
        Check if cancellation has been requested.

        Returns:
            True if the token has been cancelled, False otherwise.
        """
        ...

class BatchPostResponse:
    """
    Represents the response from a batch_post or async_batch_post operation.

    Attributes:
        data: A list of Python objects, where each object is the deserialized
              JSON response from the server for the corresponding request payload.
        total_time: Total time taken for the entire batch operation in seconds.
        individual_request_times: List of individual request times in seconds for each payload.
        response_headers: A list of dictionaries, where each dictionary contains
                          the response headers for the corresponding request.
    """

    data: builtins.list[typing.Any]
    total_time: builtins.float
    individual_request_times: builtins.list[builtins.float]
    response_headers: builtins.list[builtins.dict[builtins.str, builtins.str]]

class HttpClientWrapper:
    """
    A wrapper around the HTTP client that can be shared between multiple PerformanceClient instances.

    This allows connection pooling to be reused across clients, improving performance
    when making requests to the same region from multiple PerformanceClient instances.

    Example:
        >>> wrapper = HttpClientWrapper(http_version=1)
        >>> client1 = PerformanceClient(base_url="https://api1.example.com", client_wrapper=wrapper)
        >>> client2 = PerformanceClient(base_url="https://api2.example.com", client_wrapper=wrapper)
    """

    def __init__(self, http_version: builtins.int = 1) -> None:
        """
        Create a new HTTP client wrapper.

        Args:
            http_version: HTTP version to use. 1 for HTTP/1.1, 2 for HTTP/2. Defaults to 1.
        """
        ...

class PerformanceClient:
    """
    Baseten.co API client for embedding, reranking, and classification, and custom workloads.

    This client allows you to send text to an embedding model, rerank documents,
    or classify texts through the API.

    Attributes:
        base_url: The base URL for the API.
        api_key: The API key for authentication.

    Example:
        >>> client = PerformanceClient(base_url="https://example.api.baseten.co/environments/production/sync", api_key="your_api_key")
        >>> embeddings = client.embed(["Hello world"], model="BAAI/bge-large-en")
        >>> array = embeddings.numpy()
    """
    def __init__(
        self,
        base_url: builtins.str,
        api_key: typing.Optional[builtins.str] = None,
        http_version: builtins.int = 1,
        client_wrapper: typing.Optional[HttpClientWrapper] = None,
    ) -> None:
        """
        Initialize the sync client with the API base URL and optional API key.

        Args:
            base_url: The base URL of the API.
            api_key: The API key. If not provided, environment variables will be checked.
            http_version: Defaults to 1 for HTTP/1.1. If set to 2, uses HTTP/2.
                Under high concurrency, HTTP/1.1 delivers better performance and is the better default choice.
            client_wrapper: Optional HttpClientWrapper instance to reuse connection pooling
                across multiple PerformanceClient instances.

        Example:
            >>> client = PerformanceClient(base_url="https://example.api.baseten.co/sync", api_key="your_key", http_version=1)
            >>> # Or with shared connection pool:
            >>> wrapper = HttpClientWrapper(http_version=1)
            >>> client = PerformanceClient(base_url="https://example.api.baseten.co/sync", client_wrapper=wrapper)
        """
        ...

    def get_client_wrapper(self) -> HttpClientWrapper:
        """
        Get the HTTP client wrapper used by this client.

        This can be passed to other PerformanceClient instances to share connection pooling.

        Returns:
            The HttpClientWrapper instance used by this client.

        Example:
            >>> wrapper = client.get_client_wrapper()
            >>> client2 = PerformanceClient(base_url="https://other.api.com", client_wrapper=wrapper)
        """
        ...

    @property
    def api_key(self) -> builtins.str:
        """
        Returns the configured API key.

        Example:
            >>> print(client.api_key)
        """
        ...

    def embed(
        self,
        input: builtins.list[builtins.str],
        model: builtins.str,
        encoding_format: typing.Optional[builtins.str] = None,
        dimensions: typing.Optional[builtins.int] = None,
        user: typing.Optional[builtins.str] = None,
        preference: typing.Optional[RequestProcessingPreference] = None,
    ) -> OpenAIEmbeddingsResponse:
        """
        Sends a list of strings to the embedding endpoint to generate embeddings.

        Args:
            input: A list of texts to embed.
            model: The model identifier.
            encoding_format: Optional encoding format.
            dimensions: Optional dimension size of the embeddings.
            user: Optional user identifier.
            preference: Optional RequestProcessingPreference for configuration. If not provided, defaults will be used.

        Returns:
            An OpenAIEmbeddingsResponse object.

        Raises:
            ValueError: If the input list is empty or parameters are invalid.
            requests.exceptions.HTTPError: If the request fails.
            requests.exceptions.Timeout: If a timeout occurs.

        Example:
            >>> response = client.embed(["hello", "world"], model="model-id")
            >>> print(response.data[0].embedding)

        Example with preference:
            >>> preference = RequestProcessingPreference(max_concurrent_requests=64, timeout_s=30.0)
            >>> response = client.embed(["hello", "world"], model="model-id", preference=preference)

        Example with error handling:
            >>> try:
            ...     response = client.embed(["hello", "world"], model="invalid-model")
            ... except requests.exceptions.HTTPError as e:
            ...     print(f"Request failed: {e} with status code {e.response.args[0]}")
        """
        ...

    def rerank(
        self,
        query: builtins.str,
        texts: builtins.list[builtins.str],
        raw_scores: builtins.bool = False,
        model: typing.Optional[builtins.str] = None,
        return_text: builtins.bool = False,
        truncate: builtins.bool = False,
        truncation_direction: builtins.str = "Right",
        preference: typing.Optional[RequestProcessingPreference] = None,
    ) -> RerankResponse:
        """
        Reranks a set of texts based on the provided query.

        Args:
            query: The query string.
            texts: A list of texts to rerank.
            raw_scores: Whether raw scores should be returned.
            model: Optional model identifier.
            return_text: Whether to include the original text.
            truncate: Whether to truncate texts.
            truncation_direction: Direction for truncation ('Right' by default).
            preference: Optional RequestProcessingPreference for configuration. If not provided, defaults will be used.

        Returns:
            A RerankResponse object.

        Raises:
            ValueError: If the texts list is empty or parameters are invalid.
            requests.exceptions.HTTPError: If the request fails.
            requests.exceptions.Timeout: If a timeout occurs.

        Example:
            >>> response = client.rerank("find", ["doc1", "doc2"])
            >>> for result in response.data:
            ...     print(result.index, result.score)

        Example with preference:
            >>> preference = RequestProcessingPreference(max_concurrent_requests=64, hedge_delay=0.5)
            >>> response = client.rerank("find", ["doc1", "doc2"], preference=preference)
        """
        ...

    def classify(
        self,
        inputs: builtins.list[builtins.str],
        model: typing.Optional[builtins.str] = None,
        raw_scores: builtins.bool = False,
        truncate: builtins.bool = False,
        truncation_direction: builtins.str = "Right",
        preference: typing.Optional[RequestProcessingPreference] = None,
    ) -> ClassificationResponse:
        """
        Classifies each input text.

        Args:
            inputs: A list of texts to classify.
            model: Optional model identifier.
            raw_scores: Whether raw scores should be returned.
            truncate: Whether to truncate long texts.
            truncation_direction: Truncation direction ('Right' by default).
            preference: Optional RequestProcessingPreference for configuration. If not provided, defaults will be used.

        Returns:
            A ClassificationResponse object.

        Raises:
            ValueError: If the inputs list is empty or parameters are invalid.
            requests.exceptions.HTTPError: If the request fails.
            requests.exceptions.Timeout: If a timeout occurs.

        Example:
            >>> response = client.classify(["text1", "text2"])
            >>> for group in response.data:
            ...     for result in group:
            ...                 print(result.label, result.score)

        Example with preference:
            >>> preference = RequestProcessingPreference(batch_size=64, timeout_s=30.0)
            >>> response = client.classify(["text1", "text2"], preference=preference)
        """
        ...

    def batch_post(
        self,
        url_path: builtins.str,
        payloads: builtins.list[typing.Any],
        custom_headers: typing.Optional[
            builtins.dict[builtins.str, builtins.str]
        ] = None,
        preference: typing.Optional[RequestProcessingPreference] = None,
    ) -> BatchPostResponse:
        """
        Sends a list of generic JSON payloads to a specified URL path concurrently.

        Each payload is sent as an individual POST request. The responses are
        returned as a BatchPostResponse object.

        Args:
            url_path: The specific API path to post to (e.g., "/v1/custom_endpoint").
            payloads: A list of Python objects that are JSON-serializable.
                      Each object will be the body of a POST request.
            custom_headers: Optional dictionary of custom HTTP headers to include in each request.
            preference: Optional RequestProcessingPreference for configuration. If not provided, defaults will be used.

        Returns:
            A BatchPostResponse object containing the list of responses,
            total time, and individual request times.

        Raises:
            ValueError: If the payloads list is empty or parameters are invalid.
            requests.exceptions.HTTPError: If any of the underlying HTTP requests fail.
            requests.exceptions.Timeout: If a timeout occurs.

        Example:
            >>> client = PerformanceClient(base_url="https://example.api.baseten.co/sync", api_key="your_key")
            >>> custom_payloads = [
            ...     {"data": "request1_data", "id": 1},
            ...     {"data": "request2_data", "id": 2}
            ... ]
            >>> response_obj = client.batch_post("/v1/process_item", custom_payloads)
            >>> for resp_data in response_obj.data:
            ...     print(resp_data)
            >>> print(f"Total time: {response_obj.total_time}")
            >>> # With custom headers:
            >>> response_obj = client.batch_post("/v1/process_item", custom_payloads, custom_headers={"X-Custom-Header": "value"})
            >>> # With preference:
            >>> preference = RequestProcessingPreference(max_concurrent_requests=64, timeout_s=30.0)
            >>> response_obj = client.batch_post("/v1/process_item", custom_payloads, preference=preference)
        """
        ...

    async def async_embed(
        self,
        input: builtins.list[builtins.str],
        model: builtins.str,
        encoding_format: typing.Optional[builtins.str] = None,
        dimensions: typing.Optional[builtins.int] = None,
        user: typing.Optional[builtins.str] = None,
        preference: typing.Optional[RequestProcessingPreference] = None,
    ) -> OpenAIEmbeddingsResponse:
        """
        Asynchronously sends a list of texts to the embedding endpoint to generate embeddings.

        Args:
            input: A list of texts to embed.
            model: The model identifier.
            encoding_format: Optional encoding format.
            dimensions: Optional dimension size of the embeddings.
            user: Optional user identifier.
            preference: Optional RequestProcessingPreference for configuration. If not provided, defaults will be used.

        Returns:
            An awaitable OpenAIEmbeddingsResponse object.

        Raises:
            ValueError: If the input list is empty or parameters are invalid.
            requests.exceptions.HTTPError: If the request fails.
            requests.exceptions.Timeout: If a timeout occurs.

        Example:
            >>> response = await client.async_embed(["hello", "world"], model="model-id")
            >>> print(response.data[0].embedding)

        Example with preference:
            >>> preference = RequestProcessingPreference(max_concurrent_requests=64, timeout_s=30.0)
            >>> response = await client.async_embed(["hello", "world"], model="model-id", preference=preference)
        """
        ...

    async def async_rerank(
        self,
        query: builtins.str,
        texts: builtins.list[builtins.str],
        raw_scores: builtins.bool = False,
        model: typing.Optional[builtins.str] = None,
        return_text: builtins.bool = False,
        truncate: builtins.bool = False,
        truncation_direction: builtins.str = "Right",
        preference: typing.Optional[RequestProcessingPreference] = None,
    ) -> RerankResponse:
        """
        Asynchronously reranks a set of texts based on the provided query.

        Args:
            query: The query string.
            texts: A list of texts to rerank.
            raw_scores: Whether raw scores should be returned.
            model: Optional model identifier.
            return_text: Whether to include the original text.
            truncate: Whether to truncate texts.
            truncation_direction: Direction for truncation ('Right' by default).
            preference: Optional RequestProcessingPreference for configuration. If not provided, defaults will be used.

        Returns:
            An awaitable RerankResponse object.

        Raises:
            ValueError: If the texts list is empty or parameters are invalid.
            requests.exceptions.HTTPError: If the request fails.
            requests.exceptions.Timeout: If a timeout occurs.

        Example:
            >>> response = await client.async_rerank("find", ["doc1", "doc2"])
            >>> for result in response.data:
            ...     print(result.index, result.score)
        """
        ...

    async def async_classify(
        self,
        inputs: builtins.list[builtins.str],
        model: typing.Optional[builtins.str] = None,
        raw_scores: builtins.bool = False,
        truncate: builtins.bool = False,
        truncation_direction: builtins.str = "Right",
        preference: typing.Optional[RequestProcessingPreference] = None,
    ) -> ClassificationResponse:
        """
        Asynchronously classifies each input text.

        Args:
            inputs: A list of texts to classify.
            model: Optional model identifier.
            raw_scores: Whether raw scores should be returned.
            truncate: Whether to truncate long texts.
            truncation_direction: Truncation direction ('Right' by default).
            preference: Optional RequestProcessingPreference for configuration. If not provided, defaults will be used.

        Returns:
            An awaitable ClassificationResponse object.

        Raises:
            ValueError: If the inputs list is empty or parameters are invalid.
            requests.exceptions.HTTPError: If the request fails.
            requests.exceptions.Timeout: If a timeout occurs.

        Example:
            >>> response = await client.async_classify(["text1", "text2"])
            >>> for group in response.data:
            ...     for result in group:
            ...         print(result.label, result.score)
        """
        ...

    async def async_batch_post(
        self,
        url_path: builtins.str,
        payloads: builtins.list[typing.Any],
        custom_headers: typing.Optional[
            builtins.dict[builtins.str, builtins.str]
        ] = None,
        preference: typing.Optional[RequestProcessingPreference] = None,
    ) -> BatchPostResponse:
        """
        Asynchronously sends a list of generic JSON payloads to a specified URL path concurrently.

        Args:
            url_path: The specific API path to post to (e.g., "/v1/custom_endpoint").
            payloads: A list of Python objects that are JSON-serializable.
            custom_headers: Optional dictionary of custom HTTP headers to include in each request.
            preference: Optional RequestProcessingPreference for configuration. If not provided, defaults will be used.

        Returns:
            An awaitable BatchPostResponse object.

        Raises:
            ValueError: If the payloads list is empty or parameters are invalid.
            requests.exceptions.HTTPError: If any underlying HTTP requests fail.
            requests.exceptions.Timeout: If a timeout occurs.

        Example:
            >>> response_obj = await client.async_batch_post("/v1/process_item", [{"data": "r1"}, {"data": "r2"}])
            >>> for resp_data in response_obj.data:
            ...     print(resp_data)
            >>> print(f"Total time: {response_obj.total_time}")
        """
        ...

__version__: builtins.str
"""The version of the  bei_client library."""
