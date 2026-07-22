import json
from typing import Any

import httpx
import pytest
import respx

from tests.conftest import ollama_response


@pytest.mark.asyncio
@respx.mock
async def test_chat_happy_path_strips_think(
    async_client: httpx.AsyncClient, ollama_chat_url: str, chat_payload: dict[str, Any]
) -> None:
    content = "<think>reasoning about the greeting</think>Hello there!"
    route = respx.post(ollama_chat_url).mock(return_value=httpx.Response(200, json=ollama_response(content)))

    response = await async_client.post("/chat", json=chat_payload)

    assert response.status_code == 200
    data = response.json()
    assert data["content"] == "Hello there!"
    assert "<think>" not in data["content"]
    assert data["model"] == "qwen3:14b"

    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "qwen3:14b"
    assert body["stream"] is False
    assert body["think"] is False
    assert body["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
    assert body["messages"][1] == {"role": "user", "content": "Say hello."}
    assert "format" not in body


@pytest.mark.asyncio
@respx.mock
async def test_chat_forwards_format_schema(
    async_client: httpx.AsyncClient, ollama_chat_url: str, chat_payload: dict[str, Any]
) -> None:
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    payload = {**chat_payload, "format": schema}
    route = respx.post(ollama_chat_url).mock(
        return_value=httpx.Response(200, json=ollama_response('{"answer": "hi"}'))
    )

    response = await async_client.post("/chat", json=payload)

    assert response.status_code == 200
    body = json.loads(route.calls.last.request.content)
    assert body["format"] == schema


@pytest.mark.asyncio
@respx.mock
async def test_chat_no_system_omits_system_message(
    async_client: httpx.AsyncClient, ollama_chat_url: str
) -> None:
    route = respx.post(ollama_chat_url).mock(return_value=httpx.Response(200, json=ollama_response("ok")))

    response = await async_client.post("/chat", json={"model": "qwen3:14b", "prompt": "hi"})

    assert response.status_code == 200
    body = json.loads(route.calls.last.request.content)
    assert len(body["messages"]) == 1
    assert body["messages"][0] == {"role": "user", "content": "hi"}


@pytest.mark.asyncio
@respx.mock
async def test_chat_num_ctx_override(
    async_client: httpx.AsyncClient, ollama_chat_url: str, chat_payload: dict[str, Any]
) -> None:
    route = respx.post(ollama_chat_url).mock(return_value=httpx.Response(200, json=ollama_response("ok")))

    response = await async_client.post("/chat", json={**chat_payload, "num_ctx": 4096, "temperature": 0.7})

    assert response.status_code == 200
    body = json.loads(route.calls.last.request.content)
    assert body["options"]["num_ctx"] == 4096
    assert body["options"]["temperature"] == 0.7


@pytest.mark.asyncio
@respx.mock
async def test_vision_happy_path_sends_image(
    async_client: httpx.AsyncClient, ollama_chat_url: str, vision_payload: dict[str, Any]
) -> None:
    route = respx.post(ollama_chat_url).mock(
        return_value=httpx.Response(200, json=ollama_response("A cat."))
    )

    response = await async_client.post("/vision", json=vision_payload)

    assert response.status_code == 200
    data = response.json()
    assert data["content"] == "A cat."
    assert data["model"] == "qwen3-vl:8b-instruct"

    body = json.loads(route.calls.last.request.content)
    user_msg = body["messages"][-1]
    assert user_msg["role"] == "user"
    assert user_msg["content"] == "What is in this picture?"
    assert user_msg["images"] == ["aGVsbG8td29ybGQ="]


@pytest.mark.asyncio
@respx.mock
async def test_vision_strips_data_uri_prefix(
    async_client: httpx.AsyncClient, ollama_chat_url: str, vision_payload: dict[str, Any]
) -> None:
    payload = {**vision_payload, "image_base64": "data:image/png;base64,aGVsbG8td29ybGQ="}
    route = respx.post(ollama_chat_url).mock(return_value=httpx.Response(200, json=ollama_response("ok")))

    response = await async_client.post("/vision", json=payload)

    assert response.status_code == 200
    body = json.loads(route.calls.last.request.content)
    assert body["messages"][-1]["images"] == ["aGVsbG8td29ybGQ="]


@pytest.mark.asyncio
@respx.mock
async def test_chat_ollama_5xx_returns_502(
    async_client: httpx.AsyncClient, ollama_chat_url: str, chat_payload: dict[str, Any]
) -> None:
    respx.post(ollama_chat_url).mock(return_value=httpx.Response(503, text="unavailable"))

    response = await async_client.post("/chat", json=chat_payload)

    assert response.status_code == 502


@pytest.mark.asyncio
@respx.mock
async def test_chat_timeout_returns_504(
    async_client: httpx.AsyncClient, ollama_chat_url: str, chat_payload: dict[str, Any]
) -> None:
    respx.post(ollama_chat_url).mock(side_effect=httpx.TimeoutException("timeout"))

    response = await async_client.post("/chat", json=chat_payload)

    assert response.status_code == 504


@pytest.mark.asyncio
@respx.mock
async def test_chat_empty_content_returns_502(
    async_client: httpx.AsyncClient, ollama_chat_url: str, chat_payload: dict[str, Any]
) -> None:
    respx.post(ollama_chat_url).mock(return_value=httpx.Response(200, json=ollama_response("")))

    response = await async_client.post("/chat", json=chat_payload)

    assert response.status_code == 502


@pytest.mark.asyncio
async def test_health_ok(async_client: httpx.AsyncClient) -> None:
    with respx.mock:
        response = await async_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "marketmind-ollma-llm"
