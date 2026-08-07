"""Knowledge Object Framework endpoints (Sprint 3, Phase 3.4).

Deliberately narrow: one generic surface parametrized by
``object_type``, not nine per-type routers. Relationship/Explorer/
Impact/Health/Validation concerns already have their own generic
endpoints from Phase 3.3 (``/admin/relationships/...``) -- this router
does not duplicate them; the UI calls both routers for one object.

See ``app/api/routers/admin/__init__.py`` for why there's no role check
yet.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import get_knowledge_object_service
from app.api.schemas import KnowledgeObjectWriteRequest
from app.domain.entity_version import EntityVersion
from app.domain.knowledge_relationships import KnowledgeObjectRef, KnowledgeObjectType
from app.engines.knowledge_object_framework.service import (
    KnowledgeObjectNotFoundError,
    KnowledgeObjectService,
    ObjectHasDependentsError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/objects", tags=["admin-knowledge-objects"])


def _not_found(exc: KnowledgeObjectNotFoundError) -> HTTPException:
    return HTTPException(status_code=404, detail=str(exc))


def _conflict(exc: ObjectHasDependentsError) -> HTTPException:
    return HTTPException(status_code=409, detail=str(exc))


@router.post("/reindex", response_model=dict)
def reindex_existing(
    object_type: KnowledgeObjectType | None = Query(default=None),
    service: KnowledgeObjectService = Depends(get_knowledge_object_service),
) -> dict:
    """Backfills the search index for rows that already exist but were
    never indexed -- needed after a bulk import (rows created via direct
    repository access, or before this indexing hook existed) or after
    restoring a database backup. Idempotent; safe to call anytime.
    Omit ``object_type`` to reindex all three indexed types."""
    return service.reindex_existing(object_type)


@router.get("/{object_type}", response_model=list[KnowledgeObjectRef])
def list_objects(
    object_type: KnowledgeObjectType,
    service: KnowledgeObjectService = Depends(get_knowledge_object_service),
) -> list[KnowledgeObjectRef]:
    return service.list_refs(object_type)


@router.get("/{object_type}/{object_id}", response_model=None)
def get_object(
    object_type: KnowledgeObjectType,
    object_id: str,
    service: KnowledgeObjectService = Depends(get_knowledge_object_service),
) -> dict:
    """No fixed ``response_model`` -- the nine object types have
    genuinely different shapes and this endpoint serves all of them;
    ``model_dump(mode="json")`` is what actually makes each one
    display-ready, not a shared schema."""
    obj = service.get(object_type, object_id)
    if obj is None:
        raise _not_found(KnowledgeObjectNotFoundError(object_type, object_id))
    return obj.model_dump(mode="json")


@router.post("/{object_type}", response_model=None, status_code=201)
def create_object(
    object_type: KnowledgeObjectType,
    request: KnowledgeObjectWriteRequest,
    service: KnowledgeObjectService = Depends(get_knowledge_object_service),
) -> dict:
    try:
        obj = service.create(object_type, created_by=request.actor, **request.fields)
    except Exception as exc:  # noqa: BLE001 -- surfaces Pydantic's own validation message
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return obj.model_dump(mode="json")


@router.patch("/{object_type}/{object_id}", response_model=None)
def edit_object(
    object_type: KnowledgeObjectType,
    object_id: str,
    request: KnowledgeObjectWriteRequest,
    service: KnowledgeObjectService = Depends(get_knowledge_object_service),
) -> dict:
    try:
        obj = service.edit_metadata(object_type, object_id, updated_by=request.actor, **request.fields)
    except KnowledgeObjectNotFoundError as exc:
        raise _not_found(exc) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return obj.model_dump(mode="json")


def _lifecycle_route(name: str, action):
    @router.post(f"/{{object_type}}/{{object_id}}/{name}", response_model=None, name=f"{name}_object")
    def _handler(
        object_type: KnowledgeObjectType,
        object_id: str,
        request: KnowledgeObjectWriteRequest = KnowledgeObjectWriteRequest(),
        service: KnowledgeObjectService = Depends(get_knowledge_object_service),
    ) -> dict:
        try:
            obj = action(service, object_type, object_id, updated_by=request.actor)
        except KnowledgeObjectNotFoundError as exc:
            raise _not_found(exc) from exc
        return obj.model_dump(mode="json")

    return _handler


publish_object = _lifecycle_route("publish", KnowledgeObjectService.publish)
archive_object = _lifecycle_route("archive", KnowledgeObjectService.archive)
restore_object = _lifecycle_route("restore", KnowledgeObjectService.restore)
deprecate_object = _lifecycle_route("deprecate", KnowledgeObjectService.deprecate)


@router.delete("/{object_type}/{object_id}", status_code=204, response_model=None)
def delete_object(
    object_type: KnowledgeObjectType,
    object_id: str,
    service: KnowledgeObjectService = Depends(get_knowledge_object_service),
) -> None:
    """Gated by Impact Analysis -- refuses (409) if anything still
    depends on this object; archive is always the safe alternative."""
    try:
        service.delete(object_type, object_id)
    except KnowledgeObjectNotFoundError as exc:
        raise _not_found(exc) from exc
    except ObjectHasDependentsError as exc:
        raise _conflict(exc) from exc


@router.get("/{object_type}/{object_id}/history", response_model=list[EntityVersion])
def get_history(
    object_type: KnowledgeObjectType,
    object_id: str,
    service: KnowledgeObjectService = Depends(get_knowledge_object_service),
) -> list[EntityVersion]:
    return service.get_history(object_type, object_id)


@router.get("/{object_type}/{object_id}/validate", response_model=list[str])
def validate_object(
    object_type: KnowledgeObjectType,
    object_id: str,
    service: KnowledgeObjectService = Depends(get_knowledge_object_service),
) -> list[str]:
    try:
        return service.validate(object_type, object_id)
    except KnowledgeObjectNotFoundError as exc:
        raise _not_found(exc) from exc
