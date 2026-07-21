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
def explain_payload() -> dict[str, Any]:
    return {
        "product_name": "Whey Protein",
        "language": "pt",
        "countries": [
            {
                "country": "BRA",
                "mos": 72.4,
                "mos_band": "Alta",
                "dimensions": {"demand": 80.0, "macro": 55.0, "digital": 68.0},
                "signals": [
                    {
                        "code": "demand_volume",
                        "label": "Volume de demanda",
                        "score": 80.0,
                        "band": "Alta",
                        "meaning": "Volume absoluto de buscas mensais",
                        "detail": (
                            "1526130 total monthly searches; population 212,812,405; "
                            "penetration 717.1 searches per 100k inhabitants"
                        ),
                    }
                ],
            },
            {
                "country": "PRT",
                "mos": 61.0,
                "mos_band": "Média",
                "dimensions": {"demand": 45.0, "macro": 78.0, "digital": 82.0},
                "signals": [
                    {
                        "code": "demand_volume",
                        "label": "Volume de demanda",
                        "score": 45.0,
                        "band": "Média",
                        "meaning": "Volume absoluto de buscas mensais",
                        "detail": (
                            "210500 total monthly searches; population 10,305,564; "
                            "penetration 2042.6 searches per 100k inhabitants"
                        ),
                    }
                ],
            },
        ],
    }


def ollama_response(content: str) -> dict[str, Any]:
    return {
        "model": settings.ollama_model,
        "created_at": "2026-07-21T10:00:00Z",
        "message": {"role": "assistant", "content": content},
        "done": True,
    }
