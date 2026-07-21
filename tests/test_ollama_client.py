import httpx
import pytest
import respx

from app.clients.ollama_client import OllamaClient
from app.core.exceptions import OllamaClientError, OllamaTimeoutError
from tests.conftest import ollama_response


@pytest.mark.asyncio
@respx.mock
async def test_chat_success_strips_think(ollama_chat_url: str) -> None:
    content = "<think>raciocinando sobre os mercados</think>Recomendação: BRA (72.4) antes de PRT (61.0)."
    respx.post(ollama_chat_url).mock(return_value=httpx.Response(200, json=ollama_response(content)))

    result = await OllamaClient().chat("system", "user")

    assert result == "Recomendação: BRA (72.4) antes de PRT (61.0)."
    assert "<think>" not in result


@pytest.mark.asyncio
@respx.mock
async def test_chat_sends_expected_body(ollama_chat_url: str) -> None:
    route = respx.post(ollama_chat_url).mock(
        return_value=httpx.Response(200, json=ollama_response("Narrativa."))
    )

    await OllamaClient().chat("SYS", "USR")

    import json

    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "qwen3:14b"
    assert body["stream"] is False
    assert body["think"] is False
    assert body["options"]["temperature"] == 0.3
    assert body["options"]["num_ctx"] == 8192
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    assert body["messages"][1] == {"role": "user", "content": "USR"}


@pytest.mark.asyncio
@respx.mock
async def test_chat_server_error_raises_client_error(ollama_chat_url: str) -> None:
    respx.post(ollama_chat_url).mock(return_value=httpx.Response(500, text="boom"))

    with pytest.raises(OllamaClientError):
        await OllamaClient().chat("system", "user")


@pytest.mark.asyncio
@respx.mock
async def test_chat_timeout_raises_timeout_error(ollama_chat_url: str) -> None:
    respx.post(ollama_chat_url).mock(side_effect=httpx.TimeoutException("timeout"))

    with pytest.raises(OllamaTimeoutError):
        await OllamaClient().chat("system", "user")


@pytest.mark.asyncio
@respx.mock
async def test_chat_empty_content_raises(ollama_chat_url: str) -> None:
    respx.post(ollama_chat_url).mock(return_value=httpx.Response(200, json=ollama_response("")))

    with pytest.raises(OllamaClientError):
        await OllamaClient().chat("system", "user")
