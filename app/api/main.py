"""FastAPI application entrypoint.

Run with:

    uvicorn app.api.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.dependencies import get_knowledge_engine, run_knowledge_foundation_migration
from app.api.routers import (
    chat,
    dashboard,
    health,
    investigations,
    knowledge,
    product_intelligence,
    settings as settings_router,
    sql_studio,
)
from app.api.routers.admin import knowledge_management as admin_knowledge_management
from app.api.routers.admin import knowledge_objects as admin_knowledge_objects
from app.api.routers.admin import log_knowledge as admin_log_knowledge
from app.api.routers.admin import task_import as admin_task_import
from app.api.routers.admin import knowledge_relationships as admin_knowledge_relationships
from app.api.routers.admin import classification as admin_classification
from app.api.routers.admin import resolution_verification as admin_resolution_verification
from app.config import get_settings
from app.logging_config import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("Starting %s", settings.app_name)

    if settings.auto_seed_knowledge:
        # Phase 3.1: JSON/constant -> governed SQLite tables (idempotent,
        # safe on every startup) *before* Chroma seeding, since Chroma
        # is now seeded from those tables, not from JSON directly.
        run_knowledge_foundation_migration()
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
app.include_router(chat.router)
app.include_router(investigations.router)
app.include_router(dashboard.router)
app.include_router(settings_router.router)
app.include_router(knowledge.router)
app.include_router(sql_studio.router)
app.include_router(product_intelligence.router)
app.include_router(admin_knowledge_management.router)
app.include_router(admin_knowledge_relationships.router)
app.include_router(admin_knowledge_objects.router)
app.include_router(admin_task_import.router)
app.include_router(admin_log_knowledge.router)
app.include_router(admin_classification.router)
app.include_router(admin_resolution_verification.router)
