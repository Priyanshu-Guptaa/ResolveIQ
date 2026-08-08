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

    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True, default=None)
    """SHA-256 of raw_content -- the duplicate-upload guard's lookup key.
    See Evidence.content_hash's docstring for why this is nullable and
    never backfilled for pre-existing rows."""

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

documentation_components = Table(
    "documentation_components",
    Base.metadata,
    Column("documentation_id", ForeignKey("documentation.id"), primary_key=True),
    Column("component_id", ForeignKey("component_profiles.id"), primary_key=True),
    Index("ix_documentation_components_component_id", "component_id"),
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
    status: Mapped[str] = mapped_column(String(20), default="published", index=True)
    """Governance lifecycle (Phase 3.4) -- Draft/Published/Archived/...
    See GovernanceFields.status."""


class KnownBugModel(Base):
    """``bug_status`` (Python attribute) deliberately keeps the existing
    DB column name ``"status"`` -- that column already exists with real
    bug-state data ("open", "fixed-in-2.3.2", ...) in any database that
    ran the Phase 3.1 migration, and there is no reason to touch it.
    The new Phase 3.4 governance lifecycle field is a genuinely
    different concept that happens to share the name "status" at the
    domain-model level (``GovernanceFields.status``) -- to avoid two
    columns both named "status" in one table, it's mapped to a
    distinctly-named DB column, ``lifecycle_status``. Every other
    governed table in this file has no such pre-existing collision, so
    their new status column is simply named ``status`` directly."""

    __tablename__ = "known_bugs"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    title: Mapped[str] = mapped_column(String(500))
    description: Mapped[str] = mapped_column(Text)
    bug_status: Mapped[str] = mapped_column("status", String(50), default="open")
    affected_components: Mapped[list] = mapped_column(JSON, default=list)
    workaround: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    created_by: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    updated_by: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    lifecycle_status: Mapped[str] = mapped_column("lifecycle_status", String(20), default="published", index=True)


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
    status: Mapped[str] = mapped_column(String(20), default="published", index=True)


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
    status: Mapped[str] = mapped_column(String(20), default="published", index=True)


class DocumentationModel(Base):
    """The Knowledge Management module's document table (Sprint 3, Phase
    3.2). Phase 3.1 stored metadata only, deferring content; the
    ``content``/``product``/``version``/``technology``/``status``/
    upload-provenance columns below were added this phase via the
    existing additive-column migration (``session.py``'s
    ``_add_missing_columns`` -- no manual ALTER needed for the seven
    pre-existing rows, and their ``content`` is backfilled once by
    ``seed_migration.py``)."""

    __tablename__ = "documentation"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    title: Mapped[str] = mapped_column(String(500))
    content: Mapped[str] = mapped_column(Text, default="")
    tags: Mapped[list] = mapped_column(JSON, default=list)
    source: Mapped[str] = mapped_column(String(100), default="sample")

    product: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    version: Mapped[str | None] = mapped_column(String(100), nullable=True, default=None)
    technology: Mapped[str | None] = mapped_column(String(100), nullable=True, default=None)

    status: Mapped[str] = mapped_column(String(20), default="draft", index=True)

    original_filename: Mapped[str | None] = mapped_column(String(500), nullable=True, default=None)
    file_type: Mapped[str | None] = mapped_column(String(20), nullable=True, default=None)
    file_path: Mapped[str | None] = mapped_column(String(1000), nullable=True, default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    created_by: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    updated_by: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)


# =============================================================================
# Knowledge Relationship Manager (Sprint 3, Phase 3.3).
#
# PlaybookModel/ProductModel/TechnologyModel/VersionModel give three
# previously-free-text-only concepts (and one previously-embedded-only
# concept, Playbook) real identity, so they can be nodes in the
# relationship graph below. KnowledgeRelationshipModel is one generic
# edge table -- the generalization of Phase 3.1's four hardcoded
# "X -> Component" association tables (still present above, untouched)
# to "any of nine types -> any of nine types," since a hand-written
# table per pair would multiply combinatorially. See
# app/domain/knowledge_relationships.py's module docstring for the full
# reasoning.
# =============================================================================


class PlaybookModel(Base):
    __tablename__ = "playbooks"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    title: Mapped[str] = mapped_column(String(500))
    product: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    description: Mapped[str] = mapped_column(Text, default="")
    steps: Mapped[list] = mapped_column(JSON, default=list)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    created_by: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    updated_by: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    status: Mapped[str] = mapped_column(String(20), default="published", index=True)


class ProductModel(Base):
    __tablename__ = "products"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    created_by: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    updated_by: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    status: Mapped[str] = mapped_column(String(20), default="published", index=True)


class TechnologyModel(Base):
    __tablename__ = "technologies"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    created_by: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    updated_by: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    status: Mapped[str] = mapped_column(String(20), default="published", index=True)


class VersionModel(Base):
    __tablename__ = "versions"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    product_id: Mapped[str | None] = mapped_column(ForeignKey("products.id"), nullable=True, default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    created_by: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    updated_by: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    status: Mapped[str] = mapped_column(String(20), default="published", index=True)

    __table_args__ = (Index("ix_versions_product_id", "product_id"),)


class KnowledgeRelationshipModel(Base):
    """One row = one edge. ``from_type``/``to_type`` hold a
    ``KnowledgeObjectType`` value (plain string, not a DB-level FK --
    the referenced id can live in any of nine different tables, which a
    real foreign key can't span). Existence of both ends is enforced in
    the engine at write time (``KnowledgeRelationshipEngine.
    add_relationship`` -- "no orphan relationships"), and re-checked by
    ``validate_relationships`` as an integrity report, the same
    application-level-not-DB-level-constraint approach already used for
    ComponentProfile's self-referential related_components (Phase 3.1)."""

    __tablename__ = "knowledge_relationships"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    from_type: Mapped[str] = mapped_column(String(30))
    from_id: Mapped[str] = mapped_column(String(100))
    to_type: Mapped[str] = mapped_column(String(30))
    to_id: Mapped[str] = mapped_column(String(100))
    relationship_type: Mapped[str] = mapped_column(String(30), default="related_to")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    created_by: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)

    __table_args__ = (
        Index("ix_knowledge_relationships_from", "from_type", "from_id"),
        Index("ix_knowledge_relationships_to", "to_type", "to_id"),
    )


class EntityVersionModel(Base):
    """History (Sprint 3, Phase 3.4 -- Knowledge Object Framework). One
    row per save of any governed object, across all nine types -- the
    same generalization move as ``KnowledgeRelationshipModel`` in Phase
    3.3 (one shared table instead of nine per-type history tables).
    ``snapshot`` is the object's full ``model_dump(mode="json")`` at
    save time -- generic across every Pydantic domain model, so this
    table needed zero per-type schema knowledge to be added."""

    __tablename__ = "entity_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    object_type: Mapped[str] = mapped_column(String(30))
    object_id: Mapped[str] = mapped_column(String(100))
    version_number: Mapped[int] = mapped_column()
    snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    changed_by: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    changed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    change_summary: Mapped[str] = mapped_column(String(500), default="")

    __table_args__ = (Index("ix_entity_versions_object", "object_type", "object_id"),)


class LogSourceApplicationModel(Base):
    """Log Intelligence (Sprint 3 follow-up) -- one log-producing
    component/service extracted from operational wiki content. See
    app/domain/log_intelligence_kb.py's module docstring for why this
    is deliberately separate from ComponentProfile."""

    __tablename__ = "log_source_applications"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    name: Mapped[str] = mapped_column(String(300), index=True)

    # LogRepositoryLocation, flattened -- structured per explicit
    # instruction, not one opaque path string.
    location_platform: Mapped[str] = mapped_column(String(20), default="unknown")
    location_root_path: Mapped[str] = mapped_column(String(1000), default="")
    location_subdirectory: Mapped[str | None] = mapped_column(String(300), nullable=True, default=None)
    location_filename_patterns: Mapped[list] = mapped_column(JSON, default=list)
    location_raw_paths: Mapped[list] = mapped_column(JSON, default=list)

    technology: Mapped[list] = mapped_column(JSON, default=list)
    product: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    purpose: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    log_level_support: Mapped[list] = mapped_column(JSON, default=list)
    typical_issues: Mapped[list] = mapped_column(JSON, default=list)
    common_errors: Mapped[list] = mapped_column(JSON, default=list)
    related_sql: Mapped[list] = mapped_column(JSON, default=list)
    related_documentation: Mapped[list] = mapped_column(JSON, default=list)
    related_known_bugs: Mapped[list] = mapped_column(JSON, default=list)
    related_playbooks: Mapped[list] = mapped_column(JSON, default=list)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    source_wiki_pages: Mapped[list] = mapped_column(JSON, default=list)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    created_by: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    updated_by: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    status: Mapped[str] = mapped_column(String(20), default="published", index=True)


class LogCollectionScenarioModel(Base):
    """Log Intelligence (Sprint 3 follow-up) -- the primary
    recommendation object: an ordered set of logs to collect for one
    product + technology + operation."""

    __tablename__ = "log_collection_scenarios"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    product: Mapped[str] = mapped_column(String(200), index=True)
    technology: Mapped[str] = mapped_column(String(200), index=True)
    version: Mapped[str | None] = mapped_column(String(100), nullable=True, default=None)
    scenario_type: Mapped[str] = mapped_column(String(200))
    region: Mapped[str | None] = mapped_column(String(50), nullable=True, default=None)
    steps: Mapped[list] = mapped_column(JSON, default=list)
    """Serialized list[LogCollectionStep] -- id/component_name/priority/
    explanation per entry."""
    notes: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    source_wiki_page: Mapped[str] = mapped_column(String(500), default="")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    created_by: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    updated_by: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    status: Mapped[str] = mapped_column(String(20), default="published", index=True)
