"""SQL Studio endpoints.

Sprint 2, Phase 1 scope: the read-only Query Library only, so the
Dashboard's "Open SQL Studio" quick action lands somewhere real. Phase 8
adds saved/recent/favourite query CRUD on top of this same library; query
execution stays out of scope for this sprint (RFC rev 3, §15).

Sprint 3, Phase 3.1: the library now comes from the governed
``sql_templates`` table via :class:`SqlLibraryEngine`, not the
``QUERY_LIBRARY`` Python constant -- same response shape, same route.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_sql_library_engine
from app.domain.sql_studio import QueryTemplate
from app.engines.sql_library.engine import SqlLibraryEngine

router = APIRouter(prefix="/sql", tags=["sql-studio"])


@router.get("/library", response_model=list[QueryTemplate])
def get_query_library(engine: SqlLibraryEngine = Depends(get_sql_library_engine)) -> list[QueryTemplate]:
    return engine.list_templates()
