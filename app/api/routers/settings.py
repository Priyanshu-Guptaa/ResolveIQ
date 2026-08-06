"""Read-only view of the running configuration (RFC rev 3, Settings module).

Deliberately excludes anything secret-shaped -- there's nothing secret in
:class:`~app.config.Settings` today, but this endpoint is the boundary
where that distinction gets enforced as the config surface grows.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.dependencies import get_settings_dep
from app.config import Settings

router = APIRouter(prefix="/settings", tags=["settings"])


class SettingsView(BaseModel):
    app_name: str
    log_level: str
    embedding_model_name: str
    similarity_top_k: int
    min_similarity_for_root_cause: float
    auto_seed_knowledge: bool


@router.get("", response_model=SettingsView)
def get_settings_view(settings: Settings = Depends(get_settings_dep)) -> SettingsView:
    return SettingsView(
        app_name=settings.app_name,
        log_level=settings.log_level,
        embedding_model_name=settings.embedding_model_name,
        similarity_top_k=settings.similarity_top_k,
        min_similarity_for_root_cause=settings.min_similarity_for_root_cause,
        auto_seed_knowledge=settings.auto_seed_knowledge,
    )
