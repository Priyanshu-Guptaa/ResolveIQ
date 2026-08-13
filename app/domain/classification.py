"""Metadata Classification -- confidence-tiered, review-gated extraction
of Product/Technology/Component/Customer/Region tags for already-
imported knowledge (Context Dimensions phase, 2026-08-12 design
assessment, Part C.1 / approved product decision #2).

A :class:`MetadataClassificationSuggestion` is deliberately *not* the
same thing as an applied tag. Applying a tag means either setting a
scalar field that already exists on the target record (``product``/
``technology``/``version`` on ``DocumentationRecord``) or creating a
real ``KnowledgeRelationship`` (Customer/Region, via the existing
``APPLIES_TO`` relationship type) -- both genuinely asserted facts. A
suggestion is provisional: the classification engine's own guess at
what a real tag *might* be, with the literal evidence attached, kept
separate so a Medium-confidence guess can sit in a human review queue
without ever being mistaken for an accepted fact, and a Low-confidence
guess can be recorded for traceability without ever being applied at
all. This mirrors the Draft-vs-Published discipline already used
elsewhere in this codebase (e.g. Documentation's own lifecycle) applied
to metadata instead of content.

Every suggestion this engine produces is extractive against an
already-governed value (a real ``Technology``/``Component``/``Customer``/
``Region``/``Product``/``Version`` row) -- never a proposal to create a
*new* dimension value from free text. This is the structural guarantee
behind "a customer name is never interpreted as a technology/component/
product": those candidate lists are curated from real product-
architecture and TFS/Wiki evidence, never from arbitrary document
titles, so a customer name occurring frequently in document titles has
no path to ever becoming a Technology/Component/Product suggestion --
it can only ever match (and be suggested against) the Customer table.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


def _new_id() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ClassificationDimension(str, Enum):
    PRODUCT = "product"
    TECHNOLOGY = "technology"
    VERSION = "version"
    COMPONENT = "component"
    CUSTOMER = "customer"
    REGION = "region"


class ConfidenceTier(str, Enum):
    HIGH = "high"
    """Auto-applied immediately -- the evidence rule that fired is
    strong enough (the governed value's name/alias appears verbatim in
    the record's own title) that no human review adds real value."""
    MEDIUM = "medium"
    """Held for human review -- a real signal exists (the value appears
    in body content, or matches only a partial/weaker rule) but isn't
    strong enough to auto-apply without a second pair of eyes."""
    LOW = "low"
    """Recorded for traceability only, never applied -- a single weak,
    ambiguous, or coincidental signal. Explicit product decision: "low-
    confidence metadata should remain unclassified rather than being
    guessed" -- this tier exists so that decision is auditable (we saw
    the weak signal and chose not to act on it), not so it can be
    silently promoted later without new evidence."""


class SuggestionStatus(str, Enum):
    PENDING = "pending"
    AUTO_ACCEPTED = "auto_accepted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    RECORDED = "recorded"
    """Low-confidence suggestions land here, not PENDING -- they were
    never queued for review in the first place (see ConfidenceTier.LOW)."""


class MetadataClassificationSuggestion(BaseModel):
    """One candidate tag for one knowledge object, with the literal
    evidence that produced it. See module docstring for the provisional-
    vs-applied distinction."""

    id: str = Field(default_factory=_new_id)
    object_type: str
    """A ``KnowledgeObjectType`` value, e.g. "document" -- plain string
    (not the enum type itself) for the same reason
    ``KnowledgeRelationship.from_type`` is a plain string: this table
    needs to reference rows across several different tables, which a
    real DB-level foreign key can't span."""
    object_id: str
    dimension: ClassificationDimension
    suggested_value_id: str | None = None
    """FK into the relevant governed table (customers/regions/
    technologies/component_profiles/products/versions), when the
    matched value is already a real row -- always the case for this
    engine, since it never proposes new dimension values (see module
    docstring)."""
    suggested_value_text: str
    """The governed value's canonical name, kept redundantly even
    though ``suggested_value_id`` resolves it -- so a suggestion is
    still legible without a second lookup, and remains legible even if
    the referenced row is later renamed or removed."""
    confidence_tier: ConfidenceTier
    evidence_snippet: str
    """The literal source text that triggered this suggestion (e.g. the
    document's own title, or the exact sentence a body match came
    from) -- traceability, never a paraphrase."""
    evidence_rule: str
    """Which extraction rule fired, e.g. "title_exact_match" /
    "body_repeated_mention" -- lets a human (or a later re-run) see
    exactly why this was suggested, not just that it was."""
    status: SuggestionStatus = SuggestionStatus.PENDING
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class ClassificationRunSummary(BaseModel):
    """What an admin sees after running classification over a batch of
    documents -- same "what happened" reporting discipline as
    ``TaskImportSummary``/``LogWikiImportSummary``."""

    documents_scanned: int = 0
    suggestions_created: int = 0
    auto_accepted: int = 0
    pending_review: int = 0
    recorded_low_confidence: int = 0
    fields_set: int = 0
    """Scalar product/technology/version fields actually written --
    distinct from relationships created, and always 0 for a field that
    was already non-null (never overwritten, see engine docstring)."""
    relationships_created: int = 0
    errors: list[str] = Field(default_factory=list)
