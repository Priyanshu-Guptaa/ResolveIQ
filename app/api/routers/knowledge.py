"""Knowledge search endpoint.

Sprint 2, Phase 1 scope: a single working search across the three
existing Knowledge Engine collections, reusing the exact search methods
the Recommendation Engine already depends on -- no new retrieval logic.
Phase 6 (Knowledge Center) builds the 10-category browsing UI on top of
this same endpoint; it does not replace it.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_knowledge_engine
from app.domain.enums import KnowledgeCollection
from app.domain.recommendation import KnowledgeMatch
from app.engines.knowledge.engine import KnowledgeEngine

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

_SEARCHERS = {
    KnowledgeCollection.HISTORICAL_INVESTIGATIONS: "search_historical_investigations",
    KnowledgeCollection.DOCUMENTATION: "search_documentation",
    KnowledgeCollection.KNOWN_BUGS: "search_known_bugs",
}


@router.get("/search", response_model=list[KnowledgeMatch])
def search_knowledge(
    q: str = Query(min_length=1),
    collection: KnowledgeCollection | None = Query(default=None),
    top_k: int = Query(default=5, ge=1, le=25),
    engine: KnowledgeEngine = Depends(get_knowledge_engine),
) -> list[KnowledgeMatch]:
    """Search one collection if ``collection`` is given, otherwise all
    three, merged and re-sorted by score."""
    collections = [collection] if collection else list(_SEARCHERS.keys())
    results: list[KnowledgeMatch] = []
    for c in collections:
        searcher = getattr(engine, _SEARCHERS[c])
        results.extend(searcher(q, top_k))
    results.sort(key=lambda match: match.score, reverse=True)
    return results[:top_k]
