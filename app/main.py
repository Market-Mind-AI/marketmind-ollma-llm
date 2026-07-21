from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.explain import router as explain_router
from app.api.routes.health import router as health_router
from app.core.config import settings
from app.core.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info(f"Starting {settings.app_name} v{settings.app_version} in {settings.app_env} mode")
    logger.info(f"Ollama target: {settings.ollama_base_url} | model={settings.ollama_model}")
    yield
    logger.info(f"Shutting down {settings.app_name}")


app = FastAPI(
    title="MarketMind Ollama LLM API",
    description=(
        "Stateless final-inference layer for MarketMind. Wraps a local Ollama model to turn a "
        "deterministic MOS (Market Opportunity Score) evidence packet into a business-facing "
        "narrative that explains WHERE TO SELL a product. It explains the numbers; it never "
        "invents or recomputes them."
    ),
    version=settings.app_version,
    lifespan=lifespan,
)

app.include_router(health_router, tags=["Health"])
app.include_router(explain_router, tags=["Explain"])
