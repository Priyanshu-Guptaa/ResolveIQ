"""Knowledge Relationship Manager endpoints (Sprint 3, Phase 3.3). See
``app/api/routers/admin/__init__.py`` for why there's no role check yet.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import get_knowledge_relationship_engine
from app.api.schemas import (
    CreateLookupEntityRequest,
    CreatePlaybookRequest,
    CreateRelationshipRequest,
    CreateVersionRequest,
)
from app.domain.knowledge_relationships import (
    ExplorerView,
    ImpactAnalysis,
    KnowledgeHealthReport,
    KnowledgeObjectRef,
    KnowledgeObjectType,
    RelationshipValidationIssue,
    ResolvedRelationship,
)
from app.domain.lookup_entities import Product, Technology, Version
from app.domain.playbook import Playbook
from app.engines.knowledge_relationships.engine import (
    DuplicateRelationshipError,
    KnowledgeObjectNotFoundError,
    KnowledgeRelationshipEngine,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/relationships", tags=["admin-knowledge-relationships"])


def _not_found(exc: KnowledgeObjectNotFoundError) -> HTTPException:
    return HTTPException(status_code=404, detail=str(exc))


def _conflict(exc: DuplicateRelationshipError) -> HTTPException:
    return HTTPException(status_code=409, detail=str(exc))


@router.get("/search-objects", response_model=list[KnowledgeObjectRef])
def search_objects(
    q: str = Query(default=""),
    object_type: KnowledgeObjectType | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    engine: KnowledgeRelationshipEngine = Depends(get_knowledge_relationship_engine),
) -> list[KnowledgeObjectRef]:
    """Powers the Relationship Editor's searchable picker -- every
    result is a real object with a real id, never free text."""
    return engine.search_objects(q, object_type=object_type, limit=limit)


@router.get("", response_model=list[ResolvedRelationship])
def list_relationships(
    object_type: KnowledgeObjectType = Query(...),
    object_id: str = Query(...),
    engine: KnowledgeRelationshipEngine = Depends(get_knowledge_relationship_engine),
) -> list[ResolvedRelationship]:
    return engine.list_relationships(object_type, object_id)


@router.post("", response_model=ResolvedRelationship, status_code=201)
def add_relationship(
    request: CreateRelationshipRequest,
    engine: KnowledgeRelationshipEngine = Depends(get_knowledge_relationship_engine),
) -> ResolvedRelationship:
    try:
        return engine.add_relationship(
            request.from_type,
            request.from_id,
            request.to_type,
            request.to_id,
            request.relationship_type,
            created_by=request.created_by,
        )
    except KnowledgeObjectNotFoundError as exc:
        raise _not_found(exc) from exc
    except DuplicateRelationshipError as exc:
        raise _conflict(exc) from exc


@router.delete("/{relationship_id}", status_code=204, response_model=None)
def remove_relationship(
    relationship_id: str,
    engine: KnowledgeRelationshipEngine = Depends(get_knowledge_relationship_engine),
) -> None:
    """FastAPI's route decorator otherwise infers ``response_model``
    from the ``-> None`` return annotation, which conflicts with a 204
    (no body allowed) -- found at app-startup time, not by pytest,
    since no test actually imports the assembled FastAPI app; this is
    now covered by app.api.main import in the test suite too."""
    engine.remove_relationship(relationship_id)


@router.get("/explorer/{object_type}/{object_id}", response_model=ExplorerView)
def get_explorer_view(
    object_type: KnowledgeObjectType,
    object_id: str,
    engine: KnowledgeRelationshipEngine = Depends(get_knowledge_relationship_engine),
) -> ExplorerView:
    """"A user selecting CommandProcessorHost should immediately see
    every connected object" -- grouped sections, folding in Phase
    3.1's legacy Component-anchored relationships too."""
    try:
        return engine.get_explorer_view(object_type, object_id)
    except KnowledgeObjectNotFoundError as exc:
        raise _not_found(exc) from exc


@router.get("/impact/{object_type}/{object_id}", response_model=ImpactAnalysis)
def get_impact_analysis(
    object_type: KnowledgeObjectType,
    object_id: str,
    engine: KnowledgeRelationshipEngine = Depends(get_knowledge_relationship_engine),
) -> ImpactAnalysis:
    """"Before deleting or modifying any knowledge object, show every
    downstream dependency.\""""
    try:
        return engine.get_impact_analysis(object_type, object_id)
    except KnowledgeObjectNotFoundError as exc:
        raise _not_found(exc) from exc


@router.get("/validate", response_model=list[RelationshipValidationIssue])
def validate_relationships(
    engine: KnowledgeRelationshipEngine = Depends(get_knowledge_relationship_engine),
) -> list[RelationshipValidationIssue]:
    return engine.validate_relationships()


@router.get("/health", response_model=KnowledgeHealthReport)
def get_health_report(
    engine: KnowledgeRelationshipEngine = Depends(get_knowledge_relationship_engine),
) -> KnowledgeHealthReport:
    return engine.get_health_report()


# --- Minimal creation for the graph's newest node types ---------------------
# See KnowledgeRelationshipEngine's docstring on create_playbook/
# create_product/etc.: full management is a later Administration module.


@router.get("/playbooks", response_model=list[Playbook])
def list_playbooks(engine: KnowledgeRelationshipEngine = Depends(get_knowledge_relationship_engine)) -> list[Playbook]:
    return engine.list_playbooks()


@router.post("/playbooks", response_model=Playbook, status_code=201)
def create_playbook(
    request: CreatePlaybookRequest, engine: KnowledgeRelationshipEngine = Depends(get_knowledge_relationship_engine)
) -> Playbook:
    return engine.create_playbook(
        request.title,
        product=request.product,
        description=request.description,
        steps=request.steps,
        created_by=request.created_by,
    )


@router.get("/products", response_model=list[Product])
def list_products(engine: KnowledgeRelationshipEngine = Depends(get_knowledge_relationship_engine)) -> list[Product]:
    return engine.list_products()


@router.post("/products", response_model=Product, status_code=201)
def create_product(
    request: CreateLookupEntityRequest, engine: KnowledgeRelationshipEngine = Depends(get_knowledge_relationship_engine)
) -> Product:
    return engine.create_product(request.name, created_by=request.created_by)


@router.get("/technologies", response_model=list[Technology])
def list_technologies(
    engine: KnowledgeRelationshipEngine = Depends(get_knowledge_relationship_engine),
) -> list[Technology]:
    return engine.list_technologies()


@router.post("/technologies", response_model=Technology, status_code=201)
def create_technology(
    request: CreateLookupEntityRequest, engine: KnowledgeRelationshipEngine = Depends(get_knowledge_relationship_engine)
) -> Technology:
    return engine.create_technology(request.name, created_by=request.created_by)


@router.get("/versions", response_model=list[Version])
def list_versions(engine: KnowledgeRelationshipEngine = Depends(get_knowledge_relationship_engine)) -> list[Version]:
    return engine.list_versions()


@router.post("/versions", response_model=Version, status_code=201)
def create_version(
    request: CreateVersionRequest, engine: KnowledgeRelationshipEngine = Depends(get_knowledge_relationship_engine)
) -> Version:
    return engine.create_version(request.name, product_id=request.product_id, created_by=request.created_by)
