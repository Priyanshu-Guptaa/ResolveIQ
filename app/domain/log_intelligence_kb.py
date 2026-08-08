"""Log Intelligence Knowledge Base -- structured knowledge extracted
from operational wiki content (log repository locations, message-flow
workflows), evolving ResolveIQ's log guidance from generic "upload
logs" into "collect exactly these logs, in this order, and here's why."

Two governed entities, following the same Knowledge Object Framework
(Phase 3.4) every other knowledge type uses -- lifecycle, history, and
generic CRUD come from that framework, not reimplemented here:

- :class:`LogSourceApplication` -- one log-producing component/service.
  Deliberately separate from Product Intelligence's ``ComponentProfile``
  (the curated Architecture Explorer): the wiki's log-producing units
  are far more granular (~80 in the first import) than the curated
  registry. Linked to a real Component via an ``IMPLEMENTS_LOGGING_FOR``
  relationship where a deterministic name match exists at import time
  -- additive only, never merged.
- :class:`LogCollectionScenario` -- the *primary* recommendation object
  (per explicit design decision): an ordered set of
  :class:`LogCollectionStep` entries for one product+technology+
  operation, e.g. "Command Center, RF Mesh, Command Request (Outbound)".
  This is what the Recommendation Engine matches an investigation
  against and walks in order.

Every field here is either literally present in imported wiki content
or a straightforward, clearly-deterministic derivation from it (a
component name parsed out of its log path, a priority number from the
order log paths were listed in) -- nothing is fabricated, and nothing
here involves LLM reasoning. Fields that legitimately don't exist in
today's single imported page (purpose, typical_issues, common_errors,
related_documentation, related_known_bugs, related_playbooks) stay
empty until a future wiki import actually supplies them; the schema is
designed not to need a redesign when that happens (see
``source_wiki_pages`` -- future pages enrich the same records instead
of creating duplicates).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.domain.governance import GovernanceFields


class LogRepositoryLocation(BaseModel):
    """Structured breakdown of where a log lives -- never a single
    opaque path string. Parsed deterministically from the wiki's own
    path syntax (backslash + ``~``/``..`` prefix => Windows;
    ``/var``/``/etc`` prefix => Linux)."""

    platform: str = "unknown"
    """"windows" | "linux" | "unknown" -- inferred from path syntax,
    never guessed beyond what the path itself indicates."""
    root_path: str
    """The path up to (not including) the component-specific segment,
    e.g. "~\\Logs" or "/var/log/landisgyr/LGNetworkManager"."""
    subdirectory: str | None = None
    """The component-specific folder segment, e.g. "CommandProcessor",
    when the wiki's path has one distinct from root_path. None when the
    root_path already fully identifies the location (e.g. NMS's Linux
    path has no further subdirectory)."""
    filename_patterns: list[str] = Field(default_factory=list)
    """One or more literal filenames as written in the wiki (e.g.
    ["logfile.log"], or ["NMS_Listener.log", "NMS_Manager.log", ...]
    for a component with several distinct log files) -- never a
    fabricated glob; only what the wiki actually lists."""
    raw_paths: list[str] = Field(default_factory=list)
    """The original wiki text for each path, verbatim, for traceability
    back to the source."""


class LogSourceApplication(GovernanceFields):
    """One log-producing component/service."""

    id: str
    name: str
    """"CommandProcessor", "NMS", "Kafka" -- as named in the wiki."""
    location: LogRepositoryLocation
    technology: list[str] = Field(default_factory=list)
    """Every technology this component's logging has been documented
    under so far, e.g. ["RF Mesh", "RF Mesh IP"] -- a component can
    appear in more than one scenario."""
    product: str | None = None
    purpose: str | None = None
    """Empty from a logs-repository-only import; a future per-
    application wiki page can supply this without any schema change."""
    log_level_support: list[str] = Field(default_factory=list)
    typical_issues: list[str] = Field(default_factory=list)
    common_errors: list[str] = Field(default_factory=list)
    related_sql: list[str] = Field(default_factory=list)
    """Real table/reference names the wiki gives (e.g. "IntervalData"),
    not fabricated queries."""
    related_documentation: list[str] = Field(default_factory=list)
    related_known_bugs: list[str] = Field(default_factory=list)
    related_playbooks: list[str] = Field(default_factory=list)
    notes: str | None = None
    """Freeform caveats the wiki states next to this log, verbatim
    (e.g. "applicable for Security functionalities like Encryption/
    Decryption/Signing")."""
    source_wiki_pages: list[str] = Field(default_factory=list)
    """Every wiki page/import that has contributed to this record --
    provenance, and how re-importing/enriching (rather than
    duplicating) is detected at import time."""


class LogCollectionStep(BaseModel):
    """One ordered entry within a scenario -- the actual recommendation
    unit: which log, at what position, and why."""

    log_source_id: str
    component_name: str
    """Denormalized for display even if the log_source_id lookup ever
    fails -- never leaves a recommendation with a blank name."""
    priority: int
    """Explicit collection order, 1 = first -- preserved from the order
    log paths were listed in the wiki (not implicit list position, so
    it survives serialization/reordering safely)."""
    explanation: str
    """Deterministically generated at import time: describes this
    step's real, extracted position in the message flow (component,
    scenario, technology, step N of M), plus the wiki's own note text
    verbatim when one exists. Never LLM-generated, never fabricated
    beyond restating the structure that was actually extracted."""


class LogCollectionScenario(GovernanceFields):
    """The primary recommendation object: for this product + technology
    + operation, collect these logs, in this order."""

    id: str
    product: str
    technology: str
    version: str | None = None
    """Empty for this import (the wiki doesn't version-scope this
    content) -- present so a future product/version-specific import
    doesn't need a schema change."""
    scenario_type: str
    """"Command Request (Outbound)", "Reads (Push)", "Events", "Firmware
    Download", etc. -- as named in the wiki."""
    region: str | None = None
    """"NAM" | "APAC" -- only set where the wiki explicitly scopes a
    section to a region."""
    steps: list[LogCollectionStep] = Field(default_factory=list)
    notes: str | None = None
    source_wiki_page: str


class LogWikiImportSummary(BaseModel):
    """What the admin sees after importing one wiki page --
    ``app.engines.log_knowledge.importer.LogWikiImporter``'s result.
    Mirrors ``TaskImportSummary``'s shape (see ``app/domain/task_import.py``)
    for the same reason: a consistent "what happened" report across every
    bulk-import feature in the app."""

    source_wiki_page: str
    sources_created: int = 0
    sources_enriched: int = 0
    scenarios_created: int = 0
    scenarios_updated: int = 0
    component_relationships_created: int = 0
    errors: list[str] = Field(default_factory=list)
