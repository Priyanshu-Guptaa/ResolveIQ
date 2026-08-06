"""InvestigationSession -- the shared-context container.

Per the platform spec: "An investigation should maintain a shared context.
All modules should automatically use this context. The user should never
have to repeatedly enter the same information."

This model is that shared context. Every engine that touches an
investigation reads its evidence and merged entities from here rather than
asking the caller to re-supply them.
"""

from __future__ import annotations

import uuid
from collections import Counter
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.domain.entities import ExtractedEntity
from app.domain.enums import InvestigationStatus
from app.domain.evidence import Evidence


def _new_id() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class InvestigationSession(BaseModel):
    """A single investigation an engineer is working, and everything known
    about it so far.

    ``evidence`` accumulates as the engineer pastes the task description,
    uploads logs, and adds notes. ``merged_entities`` is derived (not
    independently editable) -- it is the de-duplicated union of every
    entity extracted from every piece of evidence, which is what the
    Recommendation Engine actually reasons over.
    """

    id: str = Field(default_factory=_new_id)
    title: str
    status: InvestigationStatus = InvestigationStatus.OPEN
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    evidence: list[Evidence] = Field(default_factory=list)

    @property
    def merged_entities(self) -> list[ExtractedEntity]:
        """De-duplicated entities across all evidence, most-frequent first.

        Frequency matters for recommendations: an entity (e.g. a
        correlation ID) that shows up in both the task description and an
        uploaded log is a much stronger investigation anchor than one that
        appears once.
        """
        counts: Counter[ExtractedEntity] = Counter()
        for item in self.evidence:
            counts.update(item.extracted_entities)
        return [entity for entity, _ in counts.most_common()]

    @property
    def context_text(self) -> str:
        """Flattened text representation of the whole investigation so far,
        used as the query for semantic search against historical knowledge.
        """
        parts: list[str] = [self.title]
        for item in self.evidence:
            if item.raw_content:
                parts.append(item.raw_content)
        return "\n".join(parts)

    def add_evidence(self, evidence: Evidence) -> None:
        self.evidence.append(evidence)
        self.updated_at = _utcnow()
