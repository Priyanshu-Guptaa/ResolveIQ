"""Product Intelligence endpoints -- Component Registry (incremental
start, Phase 2B). Plain lookup only, no matching/reasoning.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_product_intelligence_engine
from app.domain.product_intelligence import ComponentProfile
from app.engines.product_intelligence.engine import ProductIntelligenceEngine

router = APIRouter(prefix="/product-intelligence", tags=["product-intelligence"])


@router.get("/components", response_model=list[ComponentProfile])
def list_components(
    q: str | None = Query(default=None, description="Optional substring filter on component name"),
    engine: ProductIntelligenceEngine = Depends(get_product_intelligence_engine),
) -> list[ComponentProfile]:
    if q:
        return engine.search_components(q)
    return engine.list_components()
