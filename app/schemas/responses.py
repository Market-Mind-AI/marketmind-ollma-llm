from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    ollama: str | None = None


class ExplainResponse(BaseModel):
    narrative: str
    model: str
    product_name: str
