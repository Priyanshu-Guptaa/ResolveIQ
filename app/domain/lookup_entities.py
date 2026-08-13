"""Product / Technology / Version / Customer / Region -- lightweight,
governed lookup entities.

Product/Technology/Version originated in Sprint 3, Phase 3.3. Before that
phase, "product"/"technology"/"version" only existed as free-text string
columns scattered across several tables (``component_profiles.product``,
``documentation.product``/``technology``/``version``, ...) -- fine for
display, useless for "what else is Command Center 9.2 related to?" Those
existing string columns are left exactly as they are (no migration, no
risk to Phase 3.1/3.2's already-approved code); these tables give
Product/Technology/Version real identity so they can be first-class nodes
in the knowledge relationship graph, seeded from the distinct values
already in use (see ``seed_migration.py``) rather than starting empty.

Customer/Region were added in the Context Dimensions phase (2026-08-12
design assessment, Part C) -- the same governed-lookup shape, extended
with the provenance fields a *controlled, verified* list requires
(explicit product decision: never infer a customer from a text pattern
alone; every real row must trace to real evidence). Deliberately two
small typed tables, not one polymorphic "dimension value" table -- this
codebase already established that convention with Product/Technology/
Version being three near-identical tables rather than one shared shape,
and there's no real benefit to breaking it here. A future dimension
(Protocol, Environment) gets the identical treatment when real data
actually needs it, not before.
"""

from __future__ import annotations

from pydantic import Field

from app.domain.governance import GovernanceFields


class Product(GovernanceFields):
    id: str
    name: str


class Technology(GovernanceFields):
    id: str
    name: str
    parent_technology_id: str | None = None
    """Set when this technology is a more specific variant of another
    already-governed technology, e.g. "RF Mesh IP"'s parent is "RF
    Mesh" -- an ``is-a-more-specific-variant-of`` edge, never a merge.
    This is the direct, data-level fix for "RF Mesh and Mesh IP are not
    automatically the same thing": a matcher can now ask "is this an
    exact match, a parent/child match, or unrelated?" instead of
    colliding on prefix/substring overlap (see
    ``app.engines.shared.text_matching.keyword_match_score``'s own
    documented weakness, and ``RecommendationEngine._recommend_logs``,
    which this field's matching logic now guards). None for a
    technology with no known broader family, or one that IS the
    broadest form of its own family (e.g. "RF Mesh" itself)."""


class Version(GovernanceFields):
    id: str
    name: str
    product_id: str | None = None
    """Which product this version belongs to, e.g. "9.2" under Command
    Center -- optional since not every version is known to be scoped to
    one product yet."""


class Customer(GovernanceFields):
    """A real customer/organization -- deliberately never a Component,
    Technology, or Product (that conflation, discovered live in a real
    team demo, is exactly what this type exists to prevent). Only ever
    created from real, cited evidence -- never auto-inferred from a
    document title pattern alone; see ``source_type``/``source_reference``.
    """

    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    """Known alternate spellings that refer to the same real customer,
    e.g. ["TEPCO", "Tepco Japan", "Tokyo Electric Power"] -- so a
    near-duplicate spelling resolves to one governed record instead of
    silently becoming two. Matching (classification, ranking) checks
    name and every alias."""
    verified: bool = True
    """Always True for anything actually seeded through the governed
    creation path -- there is deliberately no "unverified customer"
    state a document can be tagged against; a candidate that hasn't
    been confirmed stays a ``MetadataClassificationSuggestion``
    (app/domain/classification.py), never a Customer row, until a human
    or a strong structural signal confirms it (see ``source_type``)."""
    source_type: str | None = None
    """"tfs_area_path" | "wiki_space" | "document_title_corpus" |
    "admin" -- what kind of real evidence justified creating this row.
    Never None for a properly-seeded row; None only for legacy/manual
    testing paths."""
    source_reference: str | None = None
    """The literal evidence, e.g. the real Area Path segment, wiki space
    key, or a short note of which documents corroborated this name --
    provenance, same discipline as ``LogSourceApplication.raw_paths``."""


class Region(GovernanceFields):
    """A real geography/market region a customer or piece of knowledge
    is scoped to, e.g. "Japan", "NAM", "APAC". Same governance shape as
    :class:`Customer`, smaller -- regions are lower-cardinality and the
    existing ``LogCollectionScenario.region`` field ("NAM"/"APAC") is
    already real, extracted evidence to seed from."""

    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    source_type: str | None = None
    source_reference: str | None = None
