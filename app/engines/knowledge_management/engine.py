"""Knowledge Management Engine (Sprint 3, Phase 3.2) -- the first
Administration module: document upload, preview, metadata, validation,
lifecycle, and the Knowledge Library's search/filter/sort/pagination.

Deliberately reuses rather than duplicates:
- Parsing: the existing Evidence Ingestion Pipeline (``IngestionEngine``)
  -- the exact same pipeline that parses uploaded investigation evidence,
  not a second parser.
- Indexing: the existing ``KnowledgeEngine`` -- publish/archive call its
  ``index_documentation``/``unindex_documentation``, not a second
  embedding path.
- Storage: the governed ``documentation`` table from Phase 3.1/3.2 via
  ``KnowledgeRepository`` -- no JSON, no second persistence mechanism.

This engine owns exactly what those three don't: the document lifecycle
(Draft/Published/Archived), metadata editing, validation, and upload
provenance (saving the original file's bytes for audit).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from app.domain.enums import DocumentStatus
from app.domain.evidence import DocumentationRecord
from app.domain.knowledge_management import DocumentPage, KnowledgeDashboardStats, UploadedDocumentResult
from app.engines.ingestion.engine import IngestionEngine

if TYPE_CHECKING:
    from app.engines.knowledge.engine import KnowledgeEngine
    from app.infrastructure.db.component_repository import ComponentProfileRepository
    from app.infrastructure.db.knowledge_repository import KnowledgeRepository

logger = logging.getLogger(__name__)


class DocumentNotFoundError(Exception):
    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        super().__init__(f"Document not found: {document_id}")


def _new_id() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class KnowledgeManagementEngine:
    def __init__(
        self,
        repository: "KnowledgeRepository",
        component_repository: "ComponentProfileRepository",
        ingestion_engine: IngestionEngine,
        knowledge_engine: "KnowledgeEngine",
        upload_dir: Path,
    ) -> None:
        self._repository = repository
        self._component_repository = component_repository
        self._ingestion = ingestion_engine
        self._knowledge = knowledge_engine
        self._upload_dir = upload_dir

    # --- Dashboard -------------------------------------------------------

    def get_dashboard_stats(self) -> KnowledgeDashboardStats:
        counts = self._repository.count_documentation_by_status()
        return KnowledgeDashboardStats(
            total_count=sum(counts.values()),
            published_count=counts.get(DocumentStatus.PUBLISHED.value, 0),
            draft_count=counts.get(DocumentStatus.DRAFT.value, 0),
            under_review_count=counts.get(DocumentStatus.UNDER_REVIEW.value, 0),
            archived_count=counts.get(DocumentStatus.ARCHIVED.value, 0),
            recently_added=self._repository.list_recently_added_documentation(5),
        )

    # --- Library (search / filter / sort / pagination) ---------------------

    def list_documents(
        self,
        *,
        status: str | None = None,
        product: str | None = None,
        technology: str | None = None,
        component: str | None = None,
        search: str | None = None,
        sort_by: str = "updated_at",
        sort_desc: bool = True,
        page: int = 1,
        page_size: int = 20,
    ) -> DocumentPage:
        return self._repository.list_documentation_summaries(
            status=status,
            product=product,
            technology=technology,
            component=component,
            search=search,
            sort_by=sort_by,
            sort_desc=sort_desc,
            page=page,
            page_size=page_size,
        )

    def get_document(self, document_id: str) -> DocumentationRecord | None:
        return self._repository.get_documentation(document_id)

    # --- Upload (Upload + Extract steps) ------------------------------------

    def upload_document(
        self, filename: str, content: bytes, *, uploaded_by: str | None = None
    ) -> list[UploadedDocumentResult]:
        """Runs the Evidence Ingestion Pipeline, then creates one Draft
        ``DocumentationRecord`` per parsed file (N for a .zip, 1
        otherwise), each with the original bytes saved to disk for
        provenance. Nothing is indexed/searchable yet -- that only
        happens on :meth:`publish_document`."""
        parsed_files = self._ingestion.parse(filename, content)
        original_path = self._save_original_file(filename, content)

        results: list[UploadedDocumentResult] = []
        for parsed in parsed_files:
            record = DocumentationRecord(
                id=_new_id(),
                title=Path(parsed.filename).stem or parsed.filename,
                content=parsed.text,
                source="upload",
                status=DocumentStatus.DRAFT,
                original_filename=parsed.filename,
                file_type=parsed.kind.value,
                file_path=original_path,
                created_by=uploaded_by,
                updated_by=uploaded_by,
            )
            self._repository.save_documentation(record)
            results.append(UploadedDocumentResult(document=record, warnings=parsed.warnings))
            logger.info("Uploaded document %s (%s) from %s", record.id, record.title, filename)
        return results

    def _save_original_file(self, filename: str, content: bytes) -> str:
        """Saves the raw uploaded bytes under a token-prefixed name (not
        a document id -- one .zip upload's bytes are shared by every
        document it fans out into) so the original is always
        retrievable for audit. ``Path(filename).name`` strips any
        directory component a browser might send, preventing writes
        outside ``upload_dir``."""
        self._upload_dir.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex[:12]
        safe_name = f"{token}_{Path(filename).name}"
        (self._upload_dir / safe_name).write_bytes(content)
        return safe_name

    # --- Metadata step ---------------------------------------------------

    def update_metadata(
        self,
        document_id: str,
        *,
        title: str | None = None,
        tags: list[str] | None = None,
        product: str | None = None,
        version: str | None = None,
        technology: str | None = None,
        related_components: list[str] | None = None,
        updated_by: str | None = None,
    ) -> DocumentationRecord:
        record = self._repository.get_documentation(document_id)
        if record is None:
            raise DocumentNotFoundError(document_id)

        updates: dict = {"updated_at": _utcnow(), "updated_by": updated_by}
        if title is not None:
            updates["title"] = title
        if tags is not None:
            updates["tags"] = tags
        if product is not None:
            updates["product"] = product
        if version is not None:
            updates["version"] = version
        if technology is not None:
            updates["technology"] = technology
        if related_components is not None:
            updates["related_components"] = related_components

        updated = record.model_copy(update=updates)
        self._repository.save_documentation(updated)

        # Keep the search index's title/tags/metadata in sync for a
        # document that's already live.
        if updated.status == DocumentStatus.PUBLISHED:
            self._knowledge.index_documentation(updated)
        return updated

    # --- Validate step -----------------------------------------------------

    def validate_document(self, document_id: str) -> list[str]:
        """Deterministic, lightweight checks -- not the full Data
        Quality module (a later Administration phase); just enough to
        stop an administrator from publishing something obviously
        incomplete without blocking them from doing it anyway."""
        record = self._repository.get_documentation(document_id)
        if record is None:
            raise DocumentNotFoundError(document_id)

        warnings: list[str] = []
        if not record.content.strip():
            warnings.append("No extracted text -- this document has nothing for search to match against.")
        if not record.tags:
            warnings.append("No tags set -- tags help this document surface in filtered search.")
        if not (record.product or record.technology or record.related_components):
            warnings.append("No product, technology, or component set -- this document will be hard to find via filters.")

        duplicate = self._repository.list_documentation_summaries(search=record.title, page_size=25)
        for other in duplicate.items:
            if other.id != document_id and other.title.strip().lower() == record.title.strip().lower():
                warnings.append(f'Another document is already titled "{other.title}" ({other.id}) -- possible duplicate.')
        return warnings

    # --- Lifecycle ---------------------------------------------------------

    def publish_document(self, document_id: str, *, updated_by: str | None = None) -> DocumentationRecord:
        record = self._repository.get_documentation(document_id)
        if record is None:
            raise DocumentNotFoundError(document_id)

        updated = record.model_copy(
            update={"status": DocumentStatus.PUBLISHED, "updated_at": _utcnow(), "updated_by": updated_by}
        )
        self._repository.save_documentation(updated)
        self._knowledge.index_documentation(updated)
        logger.info("Published document %s (%s)", document_id, updated.title)
        return updated

    def archive_document(self, document_id: str, *, updated_by: str | None = None) -> DocumentationRecord:
        record = self._repository.get_documentation(document_id)
        if record is None:
            raise DocumentNotFoundError(document_id)

        was_published = record.status == DocumentStatus.PUBLISHED
        updated = record.model_copy(
            update={"status": DocumentStatus.ARCHIVED, "updated_at": _utcnow(), "updated_by": updated_by}
        )
        self._repository.save_documentation(updated)
        if was_published:
            self._knowledge.unindex_documentation(document_id)
        logger.info("Archived document %s (%s)", document_id, updated.title)
        return updated

    def restore_to_draft(self, document_id: str, *, updated_by: str | None = None) -> DocumentationRecord:
        """Archived (or Published) -> Draft. Only present so an
        administrator can undo an accidental archive/publish without a
        hard delete anywhere in this module -- consistent with the rest
        of ResolveIQ never destroying data on a status change."""
        record = self._repository.get_documentation(document_id)
        if record is None:
            raise DocumentNotFoundError(document_id)

        was_published = record.status == DocumentStatus.PUBLISHED
        updated = record.model_copy(
            update={"status": DocumentStatus.DRAFT, "updated_at": _utcnow(), "updated_by": updated_by}
        )
        self._repository.save_documentation(updated)
        if was_published:
            self._knowledge.unindex_documentation(document_id)
        logger.info("Restored document %s (%s) to Draft", document_id, updated.title)
        return updated
