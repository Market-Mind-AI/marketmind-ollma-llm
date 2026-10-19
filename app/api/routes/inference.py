from typing import Any

from fastapi import APIRouter, HTTPException

from app.clients.ollama_client import OllamaClient
from app.core.config import settings
from app.core.exceptions import OllamaError
from app.core.logging import logger
from app.schemas.requests import ChatRequest, VisionRequest
from app.schemas.responses import InferenceResponse

router = APIRouter()

_DATA_URI_PREFIX = "base64,"


def _build_messages(system: str | None, user: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(user)
    return messages


def _strip_data_uri(image_base64: str) -> str:
    """Ollama wants raw base64; drop any leading `data:<media-type>;base64,` prefix."""
    idx = image_base64.find(_DATA_URI_PREFIX)
    if image_base64.startswith("data:") and idx != -1:
        return image_base64[idx + len(_DATA_URI_PREFIX) :]
    return image_base64


async def _run(messages: list[dict[str, Any]], request: ChatRequest | VisionRequest) -> InferenceResponse:
    # The caller picks the model; OLLAMA_MODEL is only what answers when it
    # does not. Every MarketMind caller sends one.
    model = request.model or settings.ollama_model
    try:
        content = await OllamaClient().chat(
            model=model,
            messages=messages,
            format=request.format,
            temperature=request.temperature,
            num_ctx=request.num_ctx,
        )
    except OllamaError as e:
        # 504 for a timeout, 502 for everything else: the exception carries it.
        logger.error(f"Ollama {type(e).__name__}: {e.message}")
        raise HTTPException(status_code=e.status_code, detail=e.message) from e

    return InferenceResponse(content=content, model=model)


@router.post(
    "/chat",
    response_model=InferenceResponse,
    summary="Text generation",
    description="Generic text inference against a local Ollama model. The caller owns the prompt.",
)
async def chat(request: ChatRequest) -> InferenceResponse:
    messages = _build_messages(request.system, {"role": "user", "content": request.prompt})
    return await _run(messages, request)


@router.post(
    "/vision",
    response_model=InferenceResponse,
    summary="Image + prompt inference",
    description="Generic vision inference against a local Ollama vision model (e.g. qwen3-vl:8b-instruct).",
)
async def vision(request: VisionRequest) -> InferenceResponse:
    user_message: dict[str, Any] = {
        "role": "user",
        "content": request.prompt,
        "images": [_strip_data_uri(request.image_base64)],
    }
    messages = _build_messages(request.system, user_message)
    return await _run(messages, request)
