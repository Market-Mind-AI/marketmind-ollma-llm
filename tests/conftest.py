from collections.abc import AsyncGenerator
from typing import Any

import httpx
import pytest

from app.core.config import settings
from app.main import app


@pytest.fixture
async def async_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def ollama_chat_url() -> str:
    return f"{settings.ollama_base_url.rstrip('/')}/api/chat"


@pytest.fixture
def chat_payload() -> dict[str, Any]:
    return {
        "model": "qwen3:14b",
        "system": "You are a helpful assistant.",
        "prompt": "Say hello.",
        "temperature": 0.3,
    }


@pytest.fixture
def vision_payload() -> dict[str, Any]:
    return {
        "model": "qwen3-vl:8b-instruct",
        "system": "Describe the image.",
        "prompt": "What is in this picture?",
        "image_base64": "aGVsbG8td29ybGQ=",
        "image_media_type": "image/png",
    }


def ollama_response(content: str) -> dict[str, Any]:
    return {
        "model": settings.ollama_model,
        "created_at": "2026-07-21T10:00:00Z",
        "message": {"role": "assistant", "content": content},
        "done": True,
    }
