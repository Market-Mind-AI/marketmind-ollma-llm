from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.inference import router as inference_router
from app.core.config import settings
from app.core.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info(f"Starting {settings.app_name} v{settings.app_version} in {settings.app_env} mode")
    logger.info(f"Ollama target: {settings.ollama_base_url}")
    yield
    logger.info(f"Shutting down {settings.app_name}")


app = FastAPI(
    title="MarketMind Ollama LLM API",
    description=(
        "Stateless Ollama gateway for MarketMind. A dumb transport that forwards text (/chat) "
        "and vision (/vision) inference requests to a local Ollama model. Callers own the prompt "
        "and pick the model; this service knows nothing about MOS."
    ),
    version=settings.app_version,
    lifespan=lifespan,
)

app.include_router(health_router, tags=["Health"])
app.include_router(inference_router, tags=["Inference"])
