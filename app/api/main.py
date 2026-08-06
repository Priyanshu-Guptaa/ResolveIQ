"""FastAPI application entrypoint.

Run with:

    uvicorn app.api.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.dependencies import get_knowledge_engine
from app.api.routers import (
    dashboard,
    health,
    investigations,
    knowledge,
    product_intelligence,
    settings as settings_router,
    sql_studio,
)
from app.config import get_settings
from app.logging_config import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("Starting %s", settings.app_name)

    if settings.auto_seed_knowledge:
        knowledge_engine = get_knowledge_engine()
        knowledge_engine.seed_from_directory(settings.sample_knowledge_dir)

    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title="ResolveIQ",
    description="Investigation Intelligence Platform for L2/L3 Product Support Engineers.",
    version="0.1.0-sprint1",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(investigations.router)
app.include_router(dashboard.router)
app.include_router(settings_router.router)
app.include_router(knowledge.router)
app.include_router(sql_studio.router)
app.include_router(product_intelligence.router)
