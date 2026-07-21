from fastapi import APIRouter, HTTPException

from app.core.exceptions import OllamaClientError, OllamaTimeoutError
from app.core.logging import logger
from app.schemas.requests import ExplainRequest
from app.schemas.responses import ExplainResponse
from app.services.explanation_service import ExplanationService

router = APIRouter()


@router.post(
    "/explain",
    response_model=ExplainResponse,
    summary="Explain a MOS ranking",
    description=(
        "Turns a MarketMind MOS evidence packet into a business-facing narrative "
        "via the local Ollama model. Explains the deterministic ranking; never "
        "invents or recomputes numbers."
    ),
)
async def explain(request: ExplainRequest) -> ExplainResponse:
    try:
        return await ExplanationService().explain(request)
    except OllamaTimeoutError as e:
        logger.error(f"Ollama timeout: {e.message}")
        raise HTTPException(status_code=504, detail=e.message) from e
    except OllamaClientError as e:
        logger.error(f"Ollama client error: {e.message}")
        raise HTTPException(status_code=502, detail=e.message) from e
