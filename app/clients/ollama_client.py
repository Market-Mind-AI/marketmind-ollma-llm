import re
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.exceptions import OllamaClientError, OllamaTimeoutError
from app.core.logging import logger

# qwen3 is a reasoning model: even with "think": false the streamed content can
# still contain <think>...</think> blocks. Strip them (DOTALL so they span
# newlines) before returning the narrative to callers.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

# Retry only transient upstream failures (timeouts / 5xx). A non-retryable
# client error (bad request, malformed response) fails fast.
RETRY_DECORATOR = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((OllamaTimeoutError, OllamaClientError)),
    reraise=True,
)


def strip_think(content: str) -> str:
    """Remove any <think>...</think> reasoning blocks and trim surrounding whitespace."""
    return _THINK_BLOCK.sub("", content).strip()


class OllamaClient:
    def __init__(self) -> None:
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_model
        self._timeout = settings.ollama_timeout
        self._num_ctx = settings.ollama_num_ctx

    @RETRY_DECORATOR
    async def chat(self, system_prompt: str, user_message: str) -> str:
        """Call Ollama's chat API and return the model text with reasoning blocks stripped."""
        url = f"{self._base_url}/api/chat"
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.3,
                "num_ctx": self._num_ctx,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=body)

                if response.status_code >= 500:
                    raise OllamaClientError(f"Ollama server error {response.status_code}: {url}")

                response.raise_for_status()

                try:
                    data = response.json()
                except Exception as e:
                    raise OllamaClientError(f"Failed to parse Ollama response: {e}") from e

        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Timeout calling Ollama: {url}") from e
        except (OllamaClientError, OllamaTimeoutError):
            raise
        except httpx.HTTPStatusError as e:
            raise OllamaClientError(f"HTTP error {e.response.status_code} from Ollama: {url}") from e
        except httpx.HTTPError as e:
            raise OllamaClientError(f"HTTP error calling Ollama: {e}") from e

        content = data.get("message", {}).get("content")
        if not content:
            raise OllamaClientError("Ollama response contained no message content")

        narrative = strip_think(content)
        if not narrative:
            raise OllamaClientError("Ollama response was empty after stripping reasoning blocks")

        logger.info(f"Ollama chat completed | model={self._model} | chars={len(narrative)}")
        return narrative

    async def health(self) -> bool:
        """Best-effort availability check against Ollama's tags endpoint."""
        url = f"{self._base_url}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(url)
                return response.status_code == 200
        except httpx.HTTPError:
            return False
