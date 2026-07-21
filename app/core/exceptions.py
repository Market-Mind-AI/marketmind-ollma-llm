class OllamaError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class OllamaClientError(OllamaError):
    """Non-recoverable failure talking to the local Ollama server (connection
    refused, upstream 5xx after retries, malformed response). Maps to HTTP 502."""

    def __init__(self, message: str = "Ollama server error"):
        super().__init__(message=message, status_code=502)


class OllamaTimeoutError(OllamaError):
    """Local LLM inference exceeded ollama_timeout. Maps to HTTP 504."""

    def __init__(self, message: str = "Ollama request timed out"):
        super().__init__(message=message, status_code=504)
