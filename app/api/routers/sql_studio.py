"""SQL Studio endpoints.

Sprint 2, Phase 1 scope: the read-only Query Library only, so the
Dashboard's "Open SQL Studio" quick action lands somewhere real. Phase 8
adds saved/recent/favourite query CRUD on top of this same library; query
execution stays out of scope for this sprint (RFC rev 3, §15).
"""

from __future__ import annotations

from fastapi import APIRouter

from app.domain.sql_studio import QUERY_LIBRARY, QueryTemplate

router = APIRouter(prefix="/sql", tags=["sql-studio"])


@router.get("/library", response_model=list[QueryTemplate])
def get_query_library() -> list[QueryTemplate]:
    return QUERY_LIBRARY
