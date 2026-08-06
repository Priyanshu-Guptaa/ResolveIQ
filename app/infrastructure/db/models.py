"""SQLAlchemy ORM models for structured persistence.

Entities/log-events extracted per piece of evidence are stored as JSON
columns on ``EvidenceModel`` rather than normalized into their own tables.
That's a deliberate Sprint 1 simplification -- there's no cross-
investigation entity search yet (the Recommendation Engine only reasons
over one investigation's merged context at a time). If a later sprint needs
"find every investigation that touched host X", promoting entities to a
proper indexed table is a contained migration, not a redesign.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class InvestigationModel(Base):
    __tablename__ = "investigations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(500))
    status: Mapped[str] = mapped_column(String(20), default="open")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    evidence: Mapped[list["EvidenceModel"]] = relationship(
        back_populates="investigation",
        cascade="all, delete-orphan",
        order_by="EvidenceModel.created_at",
    )


class EvidenceModel(Base):
    __tablename__ = "evidence"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    investigation_id: Mapped[str] = mapped_column(ForeignKey("investigations.id"))
    evidence_type: Mapped[str] = mapped_column(String(50))
    source: Mapped[str] = mapped_column(String(100), default="manual")
    title: Mapped[str] = mapped_column(String(500), default="")
    raw_content: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    # JSON blobs holding serialized ExtractedEntity / LogEvent / metadata
    # lists (see module docstring for why these aren't normalized tables).
    extracted_entities: Mapped[list] = mapped_column(JSON, default=list)
    log_events: Mapped[list] = mapped_column(JSON, default=list)
    evidence_metadata: Mapped[dict] = mapped_column(JSON, default=dict)

    investigation: Mapped["InvestigationModel"] = relationship(back_populates="evidence")
