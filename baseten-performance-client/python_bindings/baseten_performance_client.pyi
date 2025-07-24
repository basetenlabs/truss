# This file is automatically by AI
# ruff: noqa: E501, F401, F811

import builtins
import typing

try:
    import numpy
    import numpy.typing as npt

    _NDArrayF32 = npt.NDArray[numpy.float32]
    _NDArrayAny = npt.NDArray[
        typing.Any
    ]  # For a more general ndarray if dtype is not float32
except ImportError:
    _NDArrayF32 = (
        typing.Any
    )  # Fallback if numpy is not available for type checking environment
    _NDArrayAny = typing.Any

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

    def numpy(self) -> _NDArrayF32:
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

    def __init__(
        self,
        data: builtins.list[RerankResult],
        total_time: typing.Optional[builtins.float] = None,
        individual_request_times: typing.Optional[builtins.list[builtins.float]] = None,
    ) -> None:
        """
        Initializes a RerankResponse.

        Args:
            data: A list of RerankResult objects.
            total_time: Optional total time for the operation.
            individual_request_times: Optional list of individual batch request times.
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

    def __init__(
        self,
        data: builtins.list[builtins.list[ClassificationResult]],
        total_time: typing.Optional[builtins.float] = None,
        individual_request_times: typing.Optional[builtins.list[builtins.float]] = None,
    ) -> None:
        """
        Initializes a ClassificationResponse.

        Args:
            data: A list where each element is a list of ClassificationResult objects.
            total_time: Optional total time for the operation.
            individual_request_times: Optional list of individual batch request times.
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
        http_version: typing.Optional[builtins.int] = 1,  # Defaults to HTTP/1.1
    ) -> None:
        """
        Initialize the sync client with the API base URL and optional API key.

        Args:
            base_url: The base URL of the API.
            api_key: The API key. If not provided, environment variables will be checked.
            http_version: Defaults to 1 for HTTP/1.1. If set to 2, uses HTTP/2.
                Under high concurrency, HTTP/1.1 delivers better performance and is the better default choice.

        Example:
            >>> client = PerformanceClient(base_url="https://example.api.baseten.co/sync", api_key="your_key", http_version=1)
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
        max_concurrent_requests: builtins.int = 32,  # DEFAULT_CONCURRENCY
        batch_size: builtins.int = 16,  # DEFAULT_BATCH_SIZE
        timeout_s: builtins.float = 3600.0,  # DEFAULT_REQUEST_TIMEOUT_S
        max_chars_per_request: typing.Optional[builtins.int] = None,
        hedge_delay: typing.Optional[builtins.float] = None,
    ) -> OpenAIEmbeddingsResponse:
        """
        Sends a list of strings to the embedding endpoint to generate embeddings.

        Args:
            input: A list of texts to embed.
            model: The model identifier.
            encoding_format: Optional encoding format.
            dimensions: Optional dimension size of the embeddings.
            user: Optional user identifier.
            max_concurrent_requests: Maximum parallel requests.
            batch_size: Number of texts per batch.
            timeout_s: Total timeout in seconds.
            max_chars_per_request: Optional character-based batching limit.
            hedge_delay: Optional request hedging delay in seconds.

        Returns:
            An OpenAIEmbeddingsResponse object.

        Raises:
            ValueError: If the input list is empty or parameters are invalid.
            requests.exceptions.HTTPError: If the request fails.

        Example:
            >>> response = client.embed(["hello", "world"], model="model-id")
            >>> print(response.data[0].embedding)

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
        return_text: builtins.bool = False,
        truncate: builtins.bool = False,
        truncation_direction: builtins.str = "Right",
        max_concurrent_requests: builtins.int = 32,  # DEFAULT_CONCURRENCY
        batch_size: builtins.int = 16,  # DEFAULT_BATCH_SIZE
        timeout_s: builtins.float = 3600.0,  # DEFAULT_REQUEST_TIMEOUT_S
        max_chars_per_request: typing.Optional[builtins.int] = None,
        hedge_delay: typing.Optional[builtins.float] = None,
    ) -> RerankResponse:
        """
        Reranks a set of texts based on the provided query.

        Args:
            query: The query string.
            texts: A list of texts to rerank.
            raw_scores: Whether raw scores should be returned.
            return_text: Whether to include the original text.
            truncate: Whether to truncate texts.
            truncation_direction: Direction for truncation ('Right' by default).
            max_concurrent_requests: Maximum parallel requests.
            batch_size: Batch size for each request.
            timeout_s: Overall timeout in seconds.
            max_chars_per_request: Optional character-based batching limit.
            hedge_delay: Optional request hedging delay in seconds.

        Returns:
            A RerankResponse object.

        Raises:
            ValueError: If the texts list is empty or parameters are invalid.
            requests.exceptions.HTTPError: If the request fails.

        Example:
            >>> response = client.rerank("find", ["doc1", "doc2"])
            >>> for result in response.data:
            ...     print(result.index, result.score)
        """
        ...

    def classify(
        self,
        inputs: builtins.list[builtins.str],
        raw_scores: builtins.bool = False,
        truncate: builtins.bool = False,
        truncation_direction: builtins.str = "Right",
        max_concurrent_requests: builtins.int = 32,  # DEFAULT_CONCURRENCY
        batch_size: builtins.int = 16,  # DEFAULT_BATCH_SIZE
        timeout_s: builtins.float = 3600.0,  # DEFAULT_REQUEST_TIMEOUT_S
        max_chars_per_request: typing.Optional[builtins.int] = None,
        hedge_delay: typing.Optional[builtins.float] = None,
    ) -> ClassificationResponse:
        """
        Classifies each input text.

        Args:
            inputs: A list of texts to classify.
            raw_scores: Whether raw scores should be returned.
            truncate: Whether to truncate long texts.
            truncation_direction: Truncation direction ('Right' by default).
            max_concurrent_requests: Maximum parallel requests.
            batch_size: Batch size for each request.
            timeout_s: Overall timeout in seconds.
            max_chars_per_request: Optional character-based batching limit.
            hedge_delay: Optional request hedging delay in seconds.

        Returns:
            A ClassificationResponse object.

        Raises:
            ValueError: If the inputs list is empty or parameters are invalid.
            requests.exceptions.HTTPError: If the request fails.

        Example:
            >>> response = client.classify(["text1", "text2"])
            >>> for group in response.data:
            ...     for result in group:
            ...         print(result.label, result.score)
        """
        ...

    def batch_post(
        self,
        url_path: builtins.str,
        payloads: builtins.list[typing.Any],
        max_concurrent_requests: builtins.int = 32,  # DEFAULT_CONCURRENCY
        timeout_s: builtins.float = 3600.0,  # DEFAULT_REQUEST_TIMEOUT_S
        hedge_delay: typing.Optional[builtins.float] = None,
    ) -> BatchPostResponse:
        """
        Sends a list of generic JSON payloads to a specified URL path concurrently.

        Each payload is sent as an individual POST request. The responses are
        returned as a BatchPostResponse object.

        Args:
            url_path: The specific API path to post to (e.g., "/v1/custom_endpoint").
            payloads: A list of Python objects that are JSON-serializable.
                      Each object will be the body of a POST request.
            max_concurrent_requests: Maximum number of parallel requests.
            timeout_s: Total timeout in seconds for the entire batch operation,
                       also used as the timeout for each individual request.
            hedge_delay: Optional request hedging delay in seconds.

        Returns:
            A BatchPostResponse object containing the list of responses,
            total time, and individual request times.

        Raises:
            ValueError: If the payloads list is empty or parameters are invalid.
            requests.exceptions.HTTPError: If any of the underlying HTTP requests fail.

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
        """
        ...

    async def async_embed(
        self,
        input: builtins.list[builtins.str],
        model: builtins.str,
        encoding_format: typing.Optional[builtins.str] = None,
        dimensions: typing.Optional[builtins.int] = None,
        user: typing.Optional[builtins.str] = None,
        max_concurrent_requests: builtins.int = 32,
        batch_size: builtins.int = 16,
        timeout_s: builtins.float = 3600.0,
        max_chars_per_request: typing.Optional[builtins.int] = None,
        hedge_delay: typing.Optional[builtins.float] = None,
    ) -> OpenAIEmbeddingsResponse:
        """
        Asynchronously sends a list of texts to the embedding endpoint to generate embeddings.

        Args:
            input: A list of texts to embed.
            model: The model identifier.
            encoding_format: Optional encoding format.
            dimensions: Optional dimension size of the embeddings.
            user: Optional user identifier.
            max_concurrent_requests: Maximum parallel requests.
            batch_size: Number of texts per batch.
            timeout_s: Total timeout in seconds.
            max_chars_per_request: Optional character-based batching limit.
            hedge_delay: Optional request hedging delay in seconds.

        Returns:
            An awaitable OpenAIEmbeddingsResponse object.

        Raises:
            ValueError: If the input list is empty or parameters are invalid.
            requests.exceptions.HTTPError: If the request fails.

        Example:
            >>> response = await client.async_embed(["hello", "world"], model="model-id")
            >>> print(response.data[0].embedding)
        """
        ...

    async def async_rerank(
        self,
        query: builtins.str,
        texts: builtins.list[builtins.str],
        raw_scores: builtins.bool = False,
        return_text: builtins.bool = False,
        truncate: builtins.bool = False,
        truncation_direction: builtins.str = "Right",
        max_concurrent_requests: builtins.int = 32,
        batch_size: builtins.int = 16,
        timeout_s: builtins.float = 3600.0,
        max_chars_per_request: typing.Optional[builtins.int] = None,
        hedge_delay: typing.Optional[builtins.float] = None,
    ) -> RerankResponse:
        """
        Asynchronously reranks a set of texts based on the provided query.

        Args:
            query: The query string.
            texts: A list of texts to rerank.
            raw_scores: Whether raw scores should be returned.
            return_text: Whether to include the original text.
            truncate: Whether to truncate texts.
            truncation_direction: Direction for truncation ('Right' by default).
            max_concurrent_requests: Maximum parallel requests.
            batch_size: Batch size for each request.
            timeout_s: Overall timeout in seconds.
            max_chars_per_request: Optional character-based batching limit.
            hedge_delay: Optional request hedging delay in seconds.

        Returns:
            An awaitable RerankResponse object.

        Raises:
            ValueError: If the texts list is empty or parameters are invalid.
            requests.exceptions.HTTPError: If the request fails.

        Example:
            >>> response = await client.async_rerank("find", ["doc1", "doc2"])
            >>> for result in response.data:
            ...     print(result.index, result.score)
        """
        ...

    async def async_classify(
        self,
        inputs: builtins.list[builtins.str],
        raw_scores: builtins.bool = False,
        truncate: builtins.bool = False,
        truncation_direction: builtins.str = "Right",
        max_concurrent_requests: builtins.int = 32,
        batch_size: builtins.int = 16,
        timeout_s: builtins.float = 3600.0,
        max_chars_per_request: typing.Optional[builtins.int] = None,
        hedge_delay: typing.Optional[builtins.float] = None,
    ) -> ClassificationResponse:
        """
        Asynchronously classifies each input text.

        Args:
            inputs: A list of texts to classify.
            raw_scores: Whether raw scores should be returned.
            truncate: Whether to truncate long texts.
            truncation_direction: Truncation direction ('Right' by default).
            max_concurrent_requests: Maximum parallel requests.
            batch_size: Batch size for each request.
            timeout_s: Overall timeout in seconds.
            max_chars_per_request: Optional character-based batching limit.
            hedge_delay: Optional request hedging delay in seconds.

        Returns:
            An awaitable ClassificationResponse object.

        Raises:
            ValueError: If the inputs list is empty or parameters are invalid.
            requests.exceptions.HTTPError: If the request fails.

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
        max_concurrent_requests: builtins.int = 32,
        timeout_s: builtins.float = 3600.0,
        hedge_delay: typing.Optional[builtins.float] = None,
    ) -> BatchPostResponse:
        """
        Asynchronously sends a list of generic JSON payloads to a specified URL path concurrently.

        Args:
            url_path: The specific API path to post to (e.g., "/v1/custom_endpoint").
            payloads: A list of Python objects that are JSON-serializable.
            max_concurrent_requests: Maximum number of parallel requests.
            timeout_s: Total timeout in seconds for the batch operation.
            hedge_delay: Optional request hedging delay in seconds.

        Returns:
            An awaitable BatchPostResponse object.

        Raises:
            ValueError: If the payloads list is empty or parameters are invalid.
            requests.exceptions.HTTPError: If any underlying HTTP requests fail.

        Example:
            >>> response_obj = await client.async_batch_post("/v1/process_item", [{"data": "r1"}, {"data": "r2"}])
            >>> for resp_data in response_obj.data:
            ...     print(resp_data)
            >>> print(f"Total time: {response_obj.total_time}")
        """
        ...

__version__: builtins.str
"""The version of the  bei_client library."""
