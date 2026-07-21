from typing import Any

import httpx
import pytest
import respx

from tests.conftest import ollama_response


@pytest.mark.asyncio
@respx.mock
async def test_explain_happy_path_strips_think(
    async_client: httpx.AsyncClient, ollama_chat_url: str, explain_payload: dict[str, Any]
) -> None:
    content = (
        "<think>vou comparar BRA e PRT usando penetração</think>"
        "Onde vender: 1) BRA (MOS 72.4), 2) PRT (MOS 61.0)."
    )
    respx.post(ollama_chat_url).mock(return_value=httpx.Response(200, json=ollama_response(content)))

    response = await async_client.post("/explain", json=explain_payload)

    assert response.status_code == 200
    data = response.json()
    assert data["narrative"] == "Onde vender: 1) BRA (MOS 72.4), 2) PRT (MOS 61.0)."
    assert "<think>" not in data["narrative"]
    assert data["model"] == "qwen3:14b"
    assert data["product_name"] == "Whey Protein"


@pytest.mark.asyncio
@respx.mock
async def test_explain_ollama_5xx_returns_502(
    async_client: httpx.AsyncClient, ollama_chat_url: str, explain_payload: dict[str, Any]
) -> None:
    respx.post(ollama_chat_url).mock(return_value=httpx.Response(503, text="unavailable"))

    response = await async_client.post("/explain", json=explain_payload)

    assert response.status_code == 502


@pytest.mark.asyncio
@respx.mock
async def test_explain_timeout_returns_504(
    async_client: httpx.AsyncClient, ollama_chat_url: str, explain_payload: dict[str, Any]
) -> None:
    respx.post(ollama_chat_url).mock(side_effect=httpx.TimeoutException("timeout"))

    response = await async_client.post("/explain", json=explain_payload)

    assert response.status_code == 504


@pytest.mark.asyncio
async def test_explain_requires_at_least_one_country(async_client: httpx.AsyncClient) -> None:
    response = await async_client.post(
        "/explain", json={"product_name": "X", "language": "pt", "countries": []}
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_health_ok(async_client: httpx.AsyncClient) -> None:
    with respx.mock:
        response = await async_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "marketmind-ollma-llm"
