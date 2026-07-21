from fastapi import APIRouter

from app.clients.ollama_client import OllamaClient
from app.core.config import settings
from app.schemas.responses import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns service status and, best-effort, local Ollama availability.",
)
async def health_check() -> HealthResponse:
    # Pinging Ollama is best-effort and never makes the service unhealthy: the
    # host LLM may be down while this proxy is perfectly fine.
    try:
        ollama_ok = await OllamaClient().health()
        ollama_status = "available" if ollama_ok else "unavailable"
    except Exception:
        ollama_status = "unavailable"

    return HealthResponse(
        status="ok",
        service=settings.app_name,
        ollama=ollama_status,
    )
