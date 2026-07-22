from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    ollama: str | None = None


class InferenceResponse(BaseModel):
    content: str
    model: str
