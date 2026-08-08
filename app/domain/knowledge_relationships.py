"""Knowledge Relationship Manager domain models (Sprint 3, Phase 3.3).

Phase 3.1 gave four specific entity types (Known Bug, SQL Template,
Historical Investigation, Documentation) a foreign-key relationship to
exactly one thing: a Component. That was the right, minimal shape for
what existed then. This phase needs relationships between *any* pair of
nine object types -- Components, Documents, Known Bugs, SQL Templates,
Historical Investigations, Playbooks, Technologies, Products, Versions
-- which a hand-written association table per pair would multiply
combinatorially for no real benefit. ``KnowledgeRelationship`` is the
generalization of the exact same idea: one edge table, typed at both
ends, instead of N tables each hardcoded to one pair.

The four Phase 3.1 tables are **not** touched or migrated into this --
they keep working exactly as before (zero regression risk to already-
approved code). This is the mechanism for everything new this phase
introduces.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class KnowledgeObjectType(str, Enum):
    COMPONENT = "component"
    DOCUMENT = "document"
    KNOWN_BUG = "known_bug"
    SQL_TEMPLATE = "sql_template"
    HISTORICAL_INVESTIGATION = "historical_investigation"
    PLAYBOOK = "playbook"
    TECHNOLOGY = "technology"
    PRODUCT = "product"
    VERSION = "version"
    LOG_SOURCE_APPLICATION = "log_source_application"
    """A log-producing component/service extracted from operational
    wiki content (Log Intelligence). Deliberately separate from
    COMPONENT (the curated Product Intelligence Architecture Explorer)
    -- linked to it via IMPLEMENTS_LOGGING_FOR where a deterministic
    name match exists, never merged."""
    LOG_COLLECTION_SCENARIO = "log_collection_scenario"
    """An ordered set of logs to collect for one technology + operation
    (e.g. "RF Mesh, Command Request (Outbound)") -- the primary object
    the Recommendation Engine's log-collection guidance is built from."""


class RelationshipType(str, Enum):
    """A small, controlled vocabulary rather than free text -- kept
    deliberately short for this phase; expand the enum, not the schema,
    if a new meaning is needed later."""

    RELATED_TO = "related_to"
    DOCUMENTS = "documents"
    FIXES = "fixes"
    REQUIRES = "requires"
    USES = "uses"
    APPLIES_TO = "applies_to"
    IMPLEMENTS_LOGGING_FOR = "implements_logging_for"
    """LogSourceApplication -> Component: this log-producing service
    (from wiki-derived Log Intelligence knowledge) is how the given
    Component's activity gets logged. Additive only, created at import
    time via deterministic name matching against the full live
    Component Registry -- never merges the two entities."""


class KnowledgeObjectRef(BaseModel):
    """A uniform, lightweight reference to any of the nine knowledge
    object types -- what search results, pickers, and the Explorer
    render, without every caller needing to know each entity's full
    shape."""

    type: KnowledgeObjectType
    id: str
    title: str
    subtitle: str = ""
    """Extra disambiguating context for a picker -- a component's
    product, a document's status, a bug's severity, etc."""


class KnowledgeRelationship(BaseModel):
    """One edge in the knowledge graph. Undirected in spirit (either
    end can be "the thing you're looking from"), stored with a single
    direction chosen at creation time -- ``list_for_object`` matches an
    id against *either* end, and duplicate-detection treats (A,B) and
    (B,A) of the same ``relationship_type`` as the same edge."""

    id: str = ""
    from_type: KnowledgeObjectType
    from_id: str
    to_type: KnowledgeObjectType
    to_id: str
    relationship_type: RelationshipType = RelationshipType.RELATED_TO
    created_at: datetime = Field(default_factory=_utcnow)
    created_by: str | None = None


class ResolvedRelationship(BaseModel):
    """A ``KnowledgeRelationship`` with both ends resolved to a
    display-ready ``KnowledgeObjectRef`` -- what the UI actually
    renders; the raw id-pair form is an implementation detail."""

    relationship: KnowledgeRelationship
    from_object: KnowledgeObjectRef
    to_object: KnowledgeObjectRef


class ExplorerGroup(BaseModel):
    """One grouped section of the Relationship Explorer -- e.g. every
    Known Bug connected to the selected Component."""

    object_type: KnowledgeObjectType
    objects: list[KnowledgeObjectRef] = Field(default_factory=list)


class ExplorerView(BaseModel):
    """Everything connected to one selected object, grouped by the
    connected object's type -- "a user selecting CommandProcessorHost
    should immediately see every connected object," grouped exactly as
    specified: Components / Known Bugs / SQL / Playbooks / Documents /
    Historical Investigations (plus Technologies/Products/Versions,
    the three new object types)."""

    center: KnowledgeObjectRef
    groups: list[ExplorerGroup] = Field(default_factory=list)


class RelationshipValidationIssue(BaseModel):
    issue_type: str
    """"broken_link" | "duplicate" | "circular_reference" -- see
    KnowledgeRelationshipEngine.validate_relationships()."""
    severity: str = "warning"
    """"warning" | "error" -- error means the graph is inconsistent
    (a broken link, a real cycle); warning means redundant, not wrong
    (a duplicate)."""
    description: str
    relationship_id: str | None = None


class KnowledgeHealthReport(BaseModel):
    """Real counts and real object lists only -- no illustrative
    numbers, same discipline as every other Dashboard in ResolveIQ."""

    total_relationships: int = 0
    broken_relationships: list[RelationshipValidationIssue] = Field(default_factory=list)
    duplicate_relationships: list[RelationshipValidationIssue] = Field(default_factory=list)
    circular_references: list[RelationshipValidationIssue] = Field(default_factory=list)

    unused_documents: list[KnowledgeObjectRef] = Field(default_factory=list)
    """Published documents with zero relationships -- not unreachable
    (still searchable), just disconnected from the graph."""
    unused_sql_templates: list[KnowledgeObjectRef] = Field(default_factory=list)
    unused_components: list[KnowledgeObjectRef] = Field(default_factory=list)

    components_missing_playbooks: list[KnowledgeObjectRef] = Field(default_factory=list)
    components_missing_documentation: list[KnowledgeObjectRef] = Field(default_factory=list)
    components_missing_historical_investigations: list[KnowledgeObjectRef] = Field(default_factory=list)

    component_count: int = 0
    component_relationship_coverage: float = 0.0
    """Fraction of components with at least one relationship of any
    kind -- 1.0 means every component is connected to something."""


class ImpactAnalysis(BaseModel):
    """"Before deleting or modifying any knowledge object, show every
    downstream dependency" -- everything that references the selected
    object, grouped the same way the Explorer is."""

    object: KnowledgeObjectRef
    dependents: list[ExplorerGroup] = Field(default_factory=list)
    total_dependents: int = 0
