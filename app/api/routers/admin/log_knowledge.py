"""Log Intelligence wiki import endpoint.

One upload (PDF/HTML/text export of an internal wiki page) -> text
extraction (reuses the Evidence Ingestion Pipeline's existing
``FileTypeRegistry`` -- no second PDF/HTML parser here) -> deterministic
extraction (``app.engines.log_knowledge.extractor``) -> idempotent
upsert (``LogWikiImporter``) -> summary.

Generic Knowledge Object CRUD (get/list/edit/publish/archive/delete/
history) for the resulting ``LogSourceApplication``/
``LogCollectionScenario`` records comes free from the existing
``/admin/objects/{type}...`` endpoints (Phase 3.4) -- nothing
type-specific is duplicated here.

See ``app/api/routers/admin/__init__.py`` for why there's no role check
yet.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.api.dependencies import get_file_type_registry, get_log_wiki_importer, get_recommendation_engine, get_lookup_repository
from app.domain.log_intelligence_kb import LogWikiImportSummary
from app.engines.ingestion.file_type_registry import FileTypeRegistry
from app.engines.log_knowledge.extractor import extract
from app.engines.log_knowledge.importer import LogWikiImporter
from app.engines.recommendation.engine import RecommendationEngine
from app.infrastructure.db.lookup_repository import LookupRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/log-knowledge", tags=["admin-log-knowledge"])


@router.get("/technologies", response_model=list[str])
def list_technologies(
    recommendation_engine: RecommendationEngine = Depends(get_recommendation_engine),
) -> list[str]:
    """Distinct technology names actually present in the Log
    Intelligence Knowledge Base -- backs the Recommended Log
    Collection section's manual "browse by technology" filter (real
    imported data, never a hardcoded technology list)."""
    return recommendation_engine.available_log_technologies()


@router.post("/import", response_model=LogWikiImportSummary, status_code=201)
async def import_log_wiki_page(
    file: UploadFile,
    product: str = "Command Center",
    actor: str = "admin",
    file_type_registry: FileTypeRegistry = Depends(get_file_type_registry),
    importer: LogWikiImporter = Depends(get_log_wiki_importer),
    lookup_repo: LookupRepository = Depends(get_lookup_repository),
) -> LogWikiImportSummary:
    """Accepts one exported wiki page (.pdf/.html/.htm/.txt). Every
    application and message-flow scenario the page describes becomes a
    governed ``LogSourceApplication``/``LogCollectionScenario`` record
    -- re-uploading the same page, or a later page describing the same
    applications, enriches those same records rather than duplicating
    them (see ``LogWikiImporter``'s docstring)."""
    filename = file.filename or "upload"
    raw_bytes = await file.read()

    parsed = file_type_registry.parse(filename, raw_bytes)
    text = parsed[0].text if parsed else ""
    if not text.strip():
        raise HTTPException(status_code=422, detail=f"Could not extract any text from {filename}")

    # Approved product decision #6 (Context Dimensions phase,
    # 2026-08-12): a customer-named wiki section heading must never
    # become a component/log-source name -- see extract()'s
    # known_customer_names parameter / extractor._resolve_component.
    known_customer_names = frozenset(
        name.strip().lower()
        for customer in lookup_repo.list_customers()
        for name in (customer.name, *customer.aliases)
        if name.strip()
    )
    sources, scenarios = extract(text, product=product, source_wiki_page=filename, known_customer_names=known_customer_names)
    if not sources and not scenarios:
        raise HTTPException(
            status_code=422,
            detail=f"No log repository entries recognized in {filename} -- nothing to import.",
        )

    return importer.import_page(sources, scenarios, source_wiki_page=filename, actor=actor)
