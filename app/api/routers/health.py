"""Liveness/readiness endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import Settings
from app.api.dependencies import get_settings_dep

router = APIRouter(tags=["health"])


@router.get("/health")
def health(settings: Settings = Depends(get_settings_dep)) -> dict:
    return {"status": "ok", "app": settings.app_name}
