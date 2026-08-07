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

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Index, String, Table, Text
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
    last_viewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, default=None)

    # Manually-entered case metadata (Phase 2A, persistent Summary Card).
    # Nullable/optional -- see InvestigationSession's docstring for why.
    customer: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    product: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    version: Mapped[str | None] = mapped_column(String(100), nullable=True, default=None)
    technology: Mapped[str | None] = mapped_column(String(100), nullable=True, default=None)
    assigned_engineer: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)

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


# =============================================================================
# Governed knowledge tables (Sprint 3, Phase 3.1 -- Knowledge Foundation &
# Data Model Migration).
#
# These five sources were static JSON seed files / a Python constant
# through Sprint 2 -- fine while nothing needed to change at runtime, not
# fine once an Administration Portal needs to edit, version, and approve
# them. This migration promotes each into a real table; no CRUD/admin UI
# is built against them yet (that's Phase 3.3+) -- this phase only makes
# the data governable and wires every existing read path to use it
# transparently.
#
# Association tables (not mapped classes -- plain link tables, queried
# with explicit joins in the repository layer rather than ORM
# relationship() lazy collections, matching this codebase's existing
# N+1 discipline; see repository.py's list_recent_activity() docstring).
# =============================================================================

known_bug_components = Table(
    "known_bug_components",
    Base.metadata,
    Column("known_bug_id", ForeignKey("known_bugs.id"), primary_key=True),
    Column("component_id", ForeignKey("component_profiles.id"), primary_key=True),
    Index("ix_known_bug_components_component_id", "component_id"),
)

sql_template_components = Table(
    "sql_template_components",
    Base.metadata,
    Column("sql_template_id", ForeignKey("sql_templates.id"), primary_key=True),
    Column("component_id", ForeignKey("component_profiles.id"), primary_key=True),
    Index("ix_sql_template_components_component_id", "component_id"),
)

historical_investigation_components = Table(
    "historical_investigation_components",
    Base.metadata,
    Column("historical_investigation_id", ForeignKey("historical_investigations.id"), primary_key=True),
    Column("component_id", ForeignKey("component_profiles.id"), primary_key=True),
    Index("ix_historical_investigation_components_component_id", "component_id"),
)


# Every governed model below repeats the same five columns identically
# (created_at/updated_at: DateTime(timezone=True); created_by/updated_by:
# String(200), nullable; is_active: Boolean, default True, indexed since
# every list query filters on it). Not a declarative mixin -- SQLAlchemy
# mixins need care with Mapped[] + inheritance ordering, and five plain
# columns repeated five times is simpler to read than the machinery to
# avoid it at this scale.


class ComponentProfileModel(Base):
    __tablename__ = "component_profiles"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    product: Mapped[str] = mapped_column(String(200))

    responsibilities: Mapped[list] = mapped_column(JSON, default=list)
    related_components: Mapped[list] = mapped_column(JSON, default=list)
    dependencies: Mapped[list] = mapped_column(JSON, default=list)
    consumes: Mapped[list] = mapped_column(JSON, default=list)
    produces: Mapped[list] = mapped_column(JSON, default=list)
    related_services: Mapped[list] = mapped_column(JSON, default=list)
    related_queues: Mapped[list] = mapped_column(JSON, default=list)
    database_tables: Mapped[list] = mapped_column(JSON, default=list)
    configuration: Mapped[list] = mapped_column(JSON, default=list)
    events: Mapped[list] = mapped_column(JSON, default=list)
    message_flows: Mapped[list] = mapped_column(JSON, default=list)
    data_flows: Mapped[list] = mapped_column(JSON, default=list)
    version_differences: Mapped[list] = mapped_column(JSON, default=list)
    typical_failures: Mapped[list] = mapped_column(JSON, default=list)
    required_logs: Mapped[list] = mapped_column(JSON, default=list)
    common_sql: Mapped[list] = mapped_column(JSON, default=list)
    known_bugs: Mapped[list] = mapped_column(JSON, default=list)
    """Free-text known-issue strings authored directly on the component
    profile (Phase 2B) -- distinct from the ``known_bugs`` *table*
    below, which is a separate, structured knowledge source. The two
    are intentionally not reconciled in this phase; see
    ``seed_migration.py``'s module docstring."""
    documentation_links: Mapped[list] = mapped_column(JSON, default=list)
    playbooks: Mapped[list] = mapped_column(JSON, default=list)
    historical_investigations: Mapped[list] = mapped_column(JSON, default=list)
    """Free-text references authored on the component profile (Phase
    2B) -- distinct from the ``historical_investigations`` *table*."""

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    created_by: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    updated_by: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)


class KnownBugModel(Base):
    __tablename__ = "known_bugs"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    title: Mapped[str] = mapped_column(String(500))
    description: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(50), default="open")
    affected_components: Mapped[list] = mapped_column(JSON, default=list)
    workaround: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    created_by: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    updated_by: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)


class SqlTemplateModel(Base):
    __tablename__ = "sql_templates"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    title: Mapped[str] = mapped_column(String(500))
    category: Mapped[str] = mapped_column(String(100))
    sql_text: Mapped[str] = mapped_column(Text)
    explanation: Mapped[str] = mapped_column(Text)
    tags: Mapped[list] = mapped_column(JSON, default=list)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    created_by: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    updated_by: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)


class HistoricalInvestigationModel(Base):
    __tablename__ = "historical_investigations"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    title: Mapped[str] = mapped_column(String(500))
    description: Mapped[str] = mapped_column(Text)
    root_cause: Mapped[str] = mapped_column(Text)
    resolution: Mapped[str] = mapped_column(Text)
    next_step: Mapped[str] = mapped_column(Text, default="")
    tags: Mapped[list] = mapped_column(JSON, default=list)
    domain: Mapped[str] = mapped_column(String(100), default="general")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    created_by: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    updated_by: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)


class DocumentationModel(Base):
    """Metadata only, per Phase 3.1 scope -- no ``content`` column. The
    actual document body continues to live wherever it lives today (the
    sample JSON file's ``content`` field, indexed straight into
    ChromaDB); this table exists so Documentation has a governable,
    listable row ready for Phase 3.5's real ingestion pipeline, without
    duplicating or migrating document storage this phase."""

    __tablename__ = "documentation"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    title: Mapped[str] = mapped_column(String(500))
    tags: Mapped[list] = mapped_column(JSON, default=list)
    source: Mapped[str] = mapped_column(String(100), default="sample")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    created_by: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    updated_by: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
