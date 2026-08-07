"""Knowledge Object Service (Sprint 3, Phase 3.4) -- the reusable
lifecycle backend every future Administration module (Known Bug
Manager, Playbook Manager, SQL Manager, ...) should become a thin layer
on top of, instead of an independent implementation.

Deliberately does NOT own a second persistence mechanism: every read/
write dispatches through the same adapter registry
(``app.engines.knowledge_object_framework.adapters``) that
``KnowledgeRelationshipEngine`` already uses, and relationship/impact
concerns are delegated straight to ``KnowledgeRelationshipEngine``
itself rather than reimplemented. What this service actually adds on
top of what already existed:

- A uniform Create/Edit/Publish/Archive/Restore/Deprecate/Delete surface
  across all nine object types (previously: Documentation had exactly
  this in Phase 3.2's ``KnowledgeManagementEngine``, hand-written for
  one type; every other type had none at all).
- History -- a version snapshot recorded on every write, generic across
  every domain model via ``model_dump(mode="json")``.
- Generic validation -- deliberately minimal, universal rules only (see
  ``validate``'s docstring); per-type business rules are explicitly a
  later Administration module's job, not this framework's.

No object-specific business rules live here, per the phase's explicit
instruction -- ``create``/``edit_metadata`` accept arbitrary keyword
fields and hand them straight to the target type's Pydantic model,
which is what actually enforces "a Known Bug needs a description" --
this service never hardcodes a required-field list per type.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from pydantic import BaseModel

from app.domain.enums import ObjectLifecycleStatus
from app.domain.entity_version import EntityVersion
from app.domain.knowledge_relationships import (
    ExplorerView,
    ImpactAnalysis,
    KnowledgeObjectRef,
    KnowledgeObjectType,
    ResolvedRelationship,
)

if TYPE_CHECKING:
    from app.engines.knowledge.engine import KnowledgeEngine
    from app.engines.knowledge_object_framework.adapters import KnowledgeObjectAdapter
    from app.engines.knowledge_relationships.engine import KnowledgeRelationshipEngine
    from app.infrastructure.db.version_repository import VersionRepository

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class KnowledgeObjectNotFoundError(Exception):
    def __init__(self, object_type: KnowledgeObjectType, object_id: str) -> None:
        self.object_type = object_type
        self.object_id = object_id
        super().__init__(f"{object_type.value} not found: {object_id}")


class ObjectHasDependentsError(Exception):
    """Raised by ``delete`` -- "before deleting or modifying any
    knowledge object, show every downstream dependency" (Phase 3.3's
    Impact Analysis) becomes an actual gate here: an object with any
    dependent relationship cannot be hard-deleted, only archived."""

    def __init__(self, impact: ImpactAnalysis) -> None:
        self.impact = impact
        super().__init__(
            f"Cannot delete -- {impact.total_dependents} object(s) still depend on this. "
            "Remove those relationships first, or archive instead of deleting."
        )


class KnowledgeObjectService:
    def __init__(
        self,
        adapters: dict[KnowledgeObjectType, "KnowledgeObjectAdapter"],
        relationship_engine: "KnowledgeRelationshipEngine",
        version_repo: "VersionRepository",
        knowledge_engine: "KnowledgeEngine",
    ) -> None:
        self._adapters = adapters
        self._relationships = relationship_engine
        self._versions = version_repo
        self._knowledge = knowledge_engine

    def _adapter(self, object_type: KnowledgeObjectType):
        adapter = self._adapters.get(object_type)
        if adapter is None:
            raise KeyError(f"No adapter registered for {object_type.value}")
        return adapter

    # --- Read ------------------------------------------------------------

    def get(self, object_type: KnowledgeObjectType, object_id: str) -> BaseModel | None:
        return self._adapter(object_type).get(object_id)

    def list_all(self, object_type: KnowledgeObjectType) -> list[BaseModel]:
        return self._adapter(object_type).list_all()

    def list_refs(self, object_type: KnowledgeObjectType) -> list[KnowledgeObjectRef]:
        adapter = self._adapter(object_type)
        return [adapter.to_ref(obj) for obj in adapter.list_all()]

    # --- Create / Edit ---------------------------------------------------

    def create(self, object_type: KnowledgeObjectType, *, created_by: str | None = None, **fields) -> BaseModel:
        """Instantiates ``adapter.model_cls`` directly from ``fields`` --
        Pydantic's own validation is what enforces each type's actual
        required fields (e.g. a Known Bug needs ``description``); this
        method never hardcodes that per type."""
        adapter = self._adapter(object_type)
        instance = adapter.model_cls(id=_new_id(), created_by=created_by, updated_by=created_by, **fields)
        adapter.save(instance)
        self._record_version(object_type, instance.id, instance, created_by, "created")
        logger.info("Created %s %s", object_type.value, instance.id)
        return instance

    def edit_metadata(
        self, object_type: KnowledgeObjectType, object_id: str, *, updated_by: str | None = None, **fields
    ) -> BaseModel:
        adapter = self._adapter(object_type)
        current = adapter.get(object_id)
        if current is None:
            raise KnowledgeObjectNotFoundError(object_type, object_id)
        updated = current.model_copy(update={**fields, "updated_by": updated_by, "updated_at": _utcnow()})
        adapter.save(updated)
        self._record_version(object_type, object_id, updated, updated_by, "edited")
        self._reindex_if_published_document(object_type, updated)
        return updated

    # --- Lifecycle ---------------------------------------------------------

    def publish(self, object_type: KnowledgeObjectType, object_id: str, *, updated_by: str | None = None) -> BaseModel:
        return self._transition(object_type, object_id, ObjectLifecycleStatus.PUBLISHED, updated_by, "published")

    def archive(self, object_type: KnowledgeObjectType, object_id: str, *, updated_by: str | None = None) -> BaseModel:
        return self._transition(object_type, object_id, ObjectLifecycleStatus.ARCHIVED, updated_by, "archived")

    def restore(self, object_type: KnowledgeObjectType, object_id: str, *, updated_by: str | None = None) -> BaseModel:
        return self._transition(object_type, object_id, ObjectLifecycleStatus.DRAFT, updated_by, "restored to draft")

    def deprecate(self, object_type: KnowledgeObjectType, object_id: str, *, updated_by: str | None = None) -> BaseModel:
        return self._transition(object_type, object_id, ObjectLifecycleStatus.DEPRECATED, updated_by, "deprecated")

    def _transition(
        self,
        object_type: KnowledgeObjectType,
        object_id: str,
        new_status: ObjectLifecycleStatus,
        updated_by: str | None,
        summary: str,
    ) -> BaseModel:
        adapter = self._adapter(object_type)
        current = adapter.get(object_id)
        if current is None:
            raise KnowledgeObjectNotFoundError(object_type, object_id)
        was_published = current.status == ObjectLifecycleStatus.PUBLISHED
        updated = current.model_copy(update={"status": new_status, "updated_by": updated_by, "updated_at": _utcnow()})
        adapter.save(updated)
        self._record_version(object_type, object_id, updated, updated_by, summary)

        if object_type == KnowledgeObjectType.DOCUMENT:
            if new_status == ObjectLifecycleStatus.PUBLISHED:
                self._knowledge.index_documentation(updated)
            elif was_published:
                self._knowledge.unindex_documentation(object_id)
        return updated

    def delete(self, object_type: KnowledgeObjectType, object_id: str, *, deleted_by: str | None = None) -> None:
        """Hard delete -- gated by Impact Analysis ("Delete (where
        appropriate)"): refuses if anything still depends on this
        object. Archive is always available as the non-destructive
        alternative regardless of dependents."""
        adapter = self._adapter(object_type)
        current = adapter.get(object_id)
        if current is None:
            raise KnowledgeObjectNotFoundError(object_type, object_id)

        impact = self._relationships.get_impact_analysis(object_type, object_id)
        if impact.total_dependents > 0:
            raise ObjectHasDependentsError(impact)

        self._record_version(object_type, object_id, current, deleted_by, "deleted")
        if object_type == KnowledgeObjectType.DOCUMENT and current.status == ObjectLifecycleStatus.PUBLISHED:
            self._knowledge.unindex_documentation(object_id)
        adapter.delete(object_id)
        logger.info("Deleted %s %s", object_type.value, object_id)

    def _reindex_if_published_document(self, object_type: KnowledgeObjectType, updated: BaseModel) -> None:
        if object_type == KnowledgeObjectType.DOCUMENT and updated.status == ObjectLifecycleStatus.PUBLISHED:
            self._knowledge.index_documentation(updated)

    # --- History -----------------------------------------------------------

    def get_history(self, object_type: KnowledgeObjectType, object_id: str) -> list[EntityVersion]:
        return self._versions.list_versions(object_type, object_id)

    def _record_version(
        self, object_type: KnowledgeObjectType, object_id: str, instance: BaseModel, changed_by: str | None, summary: str
    ) -> None:
        self._versions.record_version(
            object_type, object_id, instance.model_dump(mode="json"), changed_by=changed_by, change_summary=summary
        )

    # --- Relationships / Impact (delegated, not reimplemented) -------------

    def list_relationships(self, object_type: KnowledgeObjectType, object_id: str) -> list[ResolvedRelationship]:
        return self._relationships.list_relationships(object_type, object_id)

    def get_explorer_view(self, object_type: KnowledgeObjectType, object_id: str) -> ExplorerView:
        return self._relationships.get_explorer_view(object_type, object_id)

    def get_impact_analysis(self, object_type: KnowledgeObjectType, object_id: str) -> ImpactAnalysis:
        return self._relationships.get_impact_analysis(object_type, object_id)

    # --- Validation --------------------------------------------------------

    def validate(self, object_type: KnowledgeObjectType, object_id: str) -> list[str]:
        """Deliberately minimal and universal -- per the phase's explicit
        "do not build object-specific business rules yet": only checks
        that apply identically to every one of the nine types. A future
        thin Administration module can add type-specific rules (e.g. "a
        SQL template's sql_text must not be empty") on top of this, not
        instead of it.
        """
        adapter = self._adapter(object_type)
        current = adapter.get(object_id)
        if current is None:
            raise KnowledgeObjectNotFoundError(object_type, object_id)

        ref = adapter.to_ref(current)
        warnings: list[str] = []
        if not ref.title.strip():
            warnings.append("No title/name set.")

        for other in self.list_refs(object_type):
            if other.id != object_id and other.title.strip().lower() == ref.title.strip().lower():
                warnings.append(f'Another {object_type.value} is already titled "{other.title}" ({other.id}) -- possible duplicate.')

        if current.status == ObjectLifecycleStatus.ARCHIVED:
            relationships = self._relationships.list_relationships(object_type, object_id)
            if relationships:
                warnings.append(
                    f"This object is Archived but still has {len(relationships)} active relationship(s) -- "
                    "consider removing them or restoring the object."
                )
        return warnings
