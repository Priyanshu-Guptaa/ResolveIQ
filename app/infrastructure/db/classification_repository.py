"""Repository for ``MetadataClassificationSuggestion`` (Context
Dimensions phase, 2026-08-12). See app/domain/classification.py's
module docstring for why this is a separate table from the
KnowledgeRelationship graph.
"""

from __future__ import annotations

import logging
from typing import Protocol

from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from app.domain.classification import ClassificationDimension, ConfidenceTier, MetadataClassificationSuggestion, SuggestionStatus
from app.infrastructure.db.models import MetadataClassificationSuggestionModel

logger = logging.getLogger(__name__)


class ClassificationRepository(Protocol):
    def save(self, suggestion: MetadataClassificationSuggestion) -> None:
        ...

    def get(self, suggestion_id: str) -> MetadataClassificationSuggestion | None:
        ...

    def list_by_status(self, status: SuggestionStatus) -> list[MetadataClassificationSuggestion]:
        ...

    def list_for_object(self, object_type: str, object_id: str) -> list[MetadataClassificationSuggestion]:
        ...

    def exists_for_object(
        self, object_type: str, object_id: str, dimension: ClassificationDimension, suggested_value_id: str | None
    ) -> bool:
        """True when this exact (object, dimension, value) suggestion
        already has a non-rejected row -- re-running classification
        over the same document must not create duplicate pending items
        every time."""
        ...


class SqlAlchemyClassificationRepository:
    def __init__(self, session_factory: sessionmaker[OrmSession]) -> None:
        self._session_factory = session_factory

    def save(self, suggestion: MetadataClassificationSuggestion) -> None:
        with self._session_factory() as session:
            model = session.get(MetadataClassificationSuggestionModel, suggestion.id)
            if model is None:
                model = MetadataClassificationSuggestionModel(id=suggestion.id)
                session.add(model)
            model.object_type = suggestion.object_type
            model.object_id = suggestion.object_id
            model.dimension = suggestion.dimension.value
            model.suggested_value_id = suggestion.suggested_value_id
            model.suggested_value_text = suggestion.suggested_value_text
            model.confidence_tier = suggestion.confidence_tier.value
            model.evidence_snippet = suggestion.evidence_snippet
            model.evidence_rule = suggestion.evidence_rule
            model.status = suggestion.status.value
            model.reviewed_by = suggestion.reviewed_by
            model.reviewed_at = suggestion.reviewed_at
            model.created_at = suggestion.created_at
            session.commit()

    def get(self, suggestion_id: str) -> MetadataClassificationSuggestion | None:
        with self._session_factory() as session:
            model = session.get(MetadataClassificationSuggestionModel, suggestion_id)
            return _to_domain(model) if model is not None else None

    def list_by_status(self, status: SuggestionStatus) -> list[MetadataClassificationSuggestion]:
        with self._session_factory() as session:
            query = session.query(MetadataClassificationSuggestionModel).filter(
                MetadataClassificationSuggestionModel.status == status.value
            )
            return [_to_domain(m) for m in query.order_by(MetadataClassificationSuggestionModel.created_at).all()]

    def list_for_object(self, object_type: str, object_id: str) -> list[MetadataClassificationSuggestion]:
        with self._session_factory() as session:
            query = session.query(MetadataClassificationSuggestionModel).filter(
                MetadataClassificationSuggestionModel.object_type == object_type,
                MetadataClassificationSuggestionModel.object_id == object_id,
            )
            return [_to_domain(m) for m in query.all()]

    def exists_for_object(
        self, object_type: str, object_id: str, dimension: ClassificationDimension, suggested_value_id: str | None
    ) -> bool:
        with self._session_factory() as session:
            query = session.query(MetadataClassificationSuggestionModel).filter(
                MetadataClassificationSuggestionModel.object_type == object_type,
                MetadataClassificationSuggestionModel.object_id == object_id,
                MetadataClassificationSuggestionModel.dimension == dimension.value,
                MetadataClassificationSuggestionModel.suggested_value_id == suggested_value_id,
                MetadataClassificationSuggestionModel.status != SuggestionStatus.REJECTED.value,
            )
            return session.query(query.exists()).scalar()


def _to_domain(model: MetadataClassificationSuggestionModel) -> MetadataClassificationSuggestion:
    return MetadataClassificationSuggestion(
        id=model.id,
        object_type=model.object_type,
        object_id=model.object_id,
        dimension=ClassificationDimension(model.dimension),
        suggested_value_id=model.suggested_value_id,
        suggested_value_text=model.suggested_value_text,
        confidence_tier=ConfidenceTier(model.confidence_tier),
        evidence_snippet=model.evidence_snippet,
        evidence_rule=model.evidence_rule,
        status=SuggestionStatus(model.status),
        reviewed_by=model.reviewed_by,
        reviewed_at=model.reviewed_at,
        created_at=model.created_at,
    )
