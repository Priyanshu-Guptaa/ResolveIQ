"""External Knowledge (TFS + Wiki live connectors).

Deliberately the opposite shape of every other knowledge source in this
codebase: Historical Investigations/Known Bugs/Documentation/Log
Intelligence are all *imported* into ResolveIQ's own governed tables and
ChromaDB collections, because they're either already-exported files or a
data source ResolveIQ owns being the system of record for. TFS and the
Landis+Gyr Wiki are neither -- they're live, external systems of record
ResolveIQ has no business copying wholesale (explicit design constraint:
TFS alone has 80,000+ Bug work items in the Command Center project;
bulk-importing and re-embedding that much content was considered and
rejected).

Instead, these models exist only for the lifetime of one Analyze
request (or a short-TTL cache entry, see
``app/engines/external_knowledge/cache.py``) -- never written to SQLite,
never embedded into ChromaDB. TFS/Wiki remain the systems of record;
ResolveIQ only ever holds a transient, scored view of "what's relevant
to *this* investigation right now."
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class ExternalSource(str, Enum):
    TFS = "tfs"
    WIKI = "wiki"


class TfsCase(BaseModel):
    """One TFS work item (Bug or Issue), normalized from whatever
    fields that work item type actually carries -- Bug and Issue have
    materially different field sets (confirmed against the real TFS
    instance; Issue has no ``LandisGyr.CRMID``/``Landis.BugRootCause``/
    severity fields Bug has), so callers should not assume every field
    here is populated for every case."""

    tfs_id: int
    work_item_type: str
    title: str
    state: str
    reason: str | None = None
    area_path: str
    team_project: str
    changed_date: datetime
    resolved_date: datetime | None = None
    description_text: str = ""
    """HTML-stripped, truncated -- see extraction.py. Empty string, not
    None, when the source field was empty; never fabricated."""
    resolution_text: str | None = None
    """Extracted from Microsoft.VSTS.Common.ResolvedReason (preferred,
    already short) or the most recent relevant System.History entry.
    None when TFS genuinely has no resolution content -- never
    fabricated, never left as an empty string standing in for "we
    don't know"."""
    root_cause: str | None = None
    """Landis.BugRootCause -- Bug only, real field, never present on
    Issue."""
    severity: str | None = None
    target_release: str | None = None
    crm_id: str | None = None
    """LandisGyr.CRMID -- Bug only. Cross-references the exact ticket-
    number convention Task Import already uses for Historical
    Investigations (see app/engines/task_import/importer.py's
    ``tags: ["ticket:<number>"]``), so a TFS Bug and an already-imported
    ServiceNow case can be recognized as the same real-world incident."""
    url: str
    """Deep link back to the real TFS work item -- required for every
    match shown (source traceability)."""


class WikiPage(BaseModel):
    """One Confluence page, normalized from the REST API's content
    response. Field shape follows Confluence's standard, documented
    REST API contract (id/title/space/body.storage.value/_links) --
    this project's own instance has not yet been live-verified against
    real authenticated content (see ``wiki_rest_client.py``'s module
    docstring), so treat this shape as the best-effort documented
    contract, not a proven-against-production fact, until that
    verification happens."""

    page_id: str
    title: str
    space_key: str
    excerpt: str = ""
    """HTML-stripped, truncated body content -- see extraction.py."""
    url: str


class ExternalMatch(BaseModel):
    """One TFS case or Wiki page, scored against the current
    investigation. Exactly one of ``tfs_case``/``wiki_page`` is set,
    matching ``source``."""

    source: ExternalSource
    tfs_case: TfsCase | None = None
    wiki_page: WikiPage | None = None
    score: float = Field(ge=0.0, le=1.0)
    confidence: str
    """"High" | "Medium" | "Low" -- derived from ``score`` by fixed
    bands, see ranking.py. Never an LLM's own confidence claim."""
    match_reasons: list[str] = Field(default_factory=list)
    """Deterministically built from which scoring signals fired (e.g.
    "Same component: CommandProcessor", "Same technology: RF Mesh") --
    the same discipline ``_match_reason()`` already uses for Log
    Intelligence scenario matching. Never LLM-phrased."""
    recommended_action: str | None = None
    """TFS matches only: a template that quotes ``resolution_text``
    directly -- never a paraphrase or invention. None when there's no
    real resolution text to quote."""


class ExternalKnowledgeResult(BaseModel):
    """What one connector (TFS or Wiki) contributed to one Analyze
    request -- always present on ``InvestigationStrategy`` even when
    the source was unreachable, so the UI can render a clear status
    line instead of silently omitting the section."""

    source: ExternalSource
    matches: list[ExternalMatch] = Field(default_factory=list)
    available: bool = True
    """False when the connector could not be reached/authenticated/
    timed out for this request -- the rest of the Investigation
    Strategy must still render normally when this is False (explicit
    requirement: TFS/Wiki are never a hard dependency for Analyze)."""
    error: str | None = None
    """Short, human-readable reason when ``available`` is False --
    never a raw stack trace or a string that could contain credential
    material."""
    query_summary: str = ""
    """What was actually searched for, e.g. "CommandProcessor, RF
    Mesh, Discovered" -- shown so an engineer can judge for themselves
    whether the search terms were reasonable, not just trust the
    ranked output blindly."""
    from_cache: bool = False
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
