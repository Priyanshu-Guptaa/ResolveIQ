"""Dashboard endpoint: the operational hub (RFC rev 3, §08).

One aggregated call so the Streamlit Dashboard page issues a single
request instead of six -- every field is backed by a real engine query,
listed in the module docstring of each engine method it calls.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_investigation_engine, get_knowledge_engine
from app.api.routers.investigations import _to_summary
from app.api.schemas import DashboardResponse
from app.domain.enums import InvestigationStatus
from app.domain.sql_studio import QUERY_LIBRARY
from app.engines.investigation.engine import InvestigationEngine
from app.engines.knowledge.engine import KnowledgeEngine

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("", response_model=DashboardResponse)
def get_dashboard(
    investigation_engine: InvestigationEngine = Depends(get_investigation_engine),
    knowledge_engine: KnowledgeEngine = Depends(get_knowledge_engine),
) -> DashboardResponse:
    # Lightweight summaries -- Dashboard rows never need evidence content.
    all_investigations = investigation_engine.list_investigation_summaries()
    active = [
        inv
        for inv in all_investigations
        if inv.status in (InvestigationStatus.OPEN, InvestigationStatus.IN_PROGRESS)
    ]
    active.sort(key=lambda inv: inv.updated_at, reverse=True)

    return DashboardResponse(
        stats=investigation_engine.get_dashboard_stats(),
        active_investigations=[_to_summary(inv) for inv in active[:8]],
        recently_viewed=[_to_summary(inv) for inv in investigation_engine.list_recently_viewed(5)],
        recent_activity=investigation_engine.list_recent_activity(10),
        recent_knowledge=knowledge_engine.list_recent_documentation(4),
        recent_known_bugs=knowledge_engine.list_recent_known_bugs(4),
        query_library_preview=QUERY_LIBRARY[:4],
    )
