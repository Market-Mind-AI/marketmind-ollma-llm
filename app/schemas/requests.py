from typing import Any

from pydantic import BaseModel


class ChatRequest(BaseModel):
    # The caller picks the model. Omitted means settings.ollama_model.
    model: str | None = None
    system: str | None = None
    prompt: str
    # A JSON Schema for Ollama structured outputs; sent as top-level "format".
    format: dict[str, Any] | None = None
    temperature: float = 0.3
    # Falls back to settings.ollama_num_ctx when not provided.
    num_ctx: int | None = None


class VisionRequest(BaseModel):
    model: str | None = None
    system: str | None = None
    prompt: str
    # Raw base64 image payload; any "data:...;base64," prefix is stripped downstream.
    image_base64: str
    # Accepted for compatibility and not used: Ollama takes raw base64 and
    # sniffs the type itself. The scoring service's OllamaProvider sends it,
    # and a field it sends must not become a validation error here.
    image_media_type: str = "image/jpeg"
    format: dict[str, Any] | None = None
    temperature: float = 0.3
    num_ctx: int | None = None
