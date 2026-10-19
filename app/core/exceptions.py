class OllamaError(Exception):
    """Base of every failure talking to Ollama. `status_code` is the HTTP status
    the routes answer with, so the mapping lives here and not in each route."""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class OllamaClientError(OllamaError):
    """A failure that retrying would not change: Ollama rejected the request
    (4xx), could not be reached, or answered something that is not a usable
    reply (malformed JSON, empty content). Maps to HTTP 502."""

    def __init__(self, message: str = "Ollama server error"):
        super().__init__(message=message, status_code=502)


class OllamaUpstreamError(OllamaClientError):
    """Ollama answered 5xx. Transient by assumption — the model is loading, the
    host is out of memory for a moment — so the client retries it before it
    reaches the caller as a 502 like any other client error."""

    def __init__(self, message: str = "Ollama upstream error"):
        super().__init__(message=message)


class OllamaTimeoutError(OllamaError):
    """Local LLM inference exceeded ollama_timeout. Maps to HTTP 504."""

    def __init__(self, message: str = "Ollama request timed out"):
        super().__init__(message=message, status_code=504)
