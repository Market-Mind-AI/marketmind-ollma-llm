import json

import httpx
import pytest
import respx

from app.clients.ollama_client import OllamaClient
from app.core.exceptions import OllamaClientError, OllamaTimeoutError, OllamaUpstreamError
from tests.conftest import ollama_response


def _messages() -> list[dict[str, object]]:
    return [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
    ]


@pytest.mark.asyncio
@respx.mock
async def test_chat_success_strips_think(ollama_chat_url: str) -> None:
    content = "<think>reasoning</think>Recommendation: BRA before PRT."
    respx.post(ollama_chat_url).mock(return_value=httpx.Response(200, json=ollama_response(content)))

    result = await OllamaClient().chat("qwen3:14b", _messages())

    assert result == "Recommendation: BRA before PRT."
    assert "<think>" not in result


@pytest.mark.asyncio
@respx.mock
async def test_chat_sends_expected_body(ollama_chat_url: str) -> None:
    route = respx.post(ollama_chat_url).mock(return_value=httpx.Response(200, json=ollama_response("Narrative.")))

    await OllamaClient().chat("qwen3:14b", _messages(), temperature=0.5, num_ctx=2048)

    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "qwen3:14b"
    assert body["stream"] is False
    assert body["think"] is False
    assert body["options"]["temperature"] == 0.5
    assert body["options"]["num_ctx"] == 2048
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    assert body["messages"][1] == {"role": "user", "content": "USR"}
    assert "format" not in body


@pytest.mark.asyncio
@respx.mock
async def test_chat_num_ctx_defaults_to_settings(ollama_chat_url: str) -> None:
    route = respx.post(ollama_chat_url).mock(return_value=httpx.Response(200, json=ollama_response("ok")))

    await OllamaClient().chat("qwen3:14b", _messages())

    body = json.loads(route.calls.last.request.content)
    assert body["options"]["num_ctx"] == OllamaClient()._num_ctx


@pytest.mark.asyncio
@respx.mock
async def test_chat_attaches_format_only_when_given(ollama_chat_url: str) -> None:
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    route = respx.post(ollama_chat_url).mock(return_value=httpx.Response(200, json=ollama_response("ok")))

    await OllamaClient().chat("qwen3:14b", _messages(), format=schema)

    body = json.loads(route.calls.last.request.content)
    assert body["format"] == schema


@pytest.mark.asyncio
@respx.mock
async def test_chat_forwards_images_in_message(ollama_chat_url: str) -> None:
    messages = [{"role": "user", "content": "describe", "images": ["YjY0"]}]
    route = respx.post(ollama_chat_url).mock(return_value=httpx.Response(200, json=ollama_response("ok")))

    await OllamaClient().chat("qwen3-vl:8b-instruct", messages)

    body = json.loads(route.calls.last.request.content)
    assert body["messages"][0]["images"] == ["YjY0"]


@pytest.mark.asyncio
@respx.mock
async def test_chat_server_error_is_retried_three_times(ollama_chat_url: str) -> None:
    route = respx.post(ollama_chat_url).mock(return_value=httpx.Response(500, text="boom"))

    with pytest.raises(OllamaUpstreamError):
        await OllamaClient().chat("qwen3:14b", _messages())

    assert route.call_count == 3


@pytest.mark.asyncio
@respx.mock
async def test_chat_server_error_recovers_on_a_later_attempt(ollama_chat_url: str) -> None:
    route = respx.post(ollama_chat_url).mock(
        side_effect=[httpx.Response(503, text="loading"), httpx.Response(200, json=ollama_response("ok"))]
    )

    assert await OllamaClient().chat("qwen3:14b", _messages()) == "ok"
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_chat_timeout_is_retried_three_times(ollama_chat_url: str) -> None:
    route = respx.post(ollama_chat_url).mock(side_effect=httpx.TimeoutException("timeout"))

    with pytest.raises(OllamaTimeoutError):
        await OllamaClient().chat("qwen3:14b", _messages())

    assert route.call_count == 3


@pytest.mark.asyncio
@respx.mock
async def test_chat_client_error_fails_fast(ollama_chat_url: str) -> None:
    # A 4xx is the request being wrong — an unknown model, say — and asking
    # again changes nothing.
    route = respx.post(ollama_chat_url).mock(return_value=httpx.Response(404, json={"error": "model not found"}))

    with pytest.raises(OllamaClientError) as exc_info:
        await OllamaClient().chat("nope:1b", _messages())

    assert not isinstance(exc_info.value, OllamaUpstreamError)
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_chat_malformed_json_fails_fast(ollama_chat_url: str) -> None:
    route = respx.post(ollama_chat_url).mock(return_value=httpx.Response(200, text="not json"))

    with pytest.raises(OllamaClientError):
        await OllamaClient().chat("qwen3:14b", _messages())

    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_chat_empty_content_fails_fast(ollama_chat_url: str) -> None:
    route = respx.post(ollama_chat_url).mock(return_value=httpx.Response(200, json=ollama_response("")))

    with pytest.raises(OllamaClientError):
        await OllamaClient().chat("qwen3:14b", _messages())

    assert route.call_count == 1
