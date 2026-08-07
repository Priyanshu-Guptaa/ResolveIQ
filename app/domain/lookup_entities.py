"""Product / Technology / Version -- lightweight, governed lookup
entities (Sprint 3, Phase 3.3).

Before this phase, "product"/"technology"/"version" only existed as
free-text string columns scattered across several tables (``component_
profiles.product``, ``documentation.product``/``technology``/
``version``, ...) -- fine for display, useless for "what else is
Command Center 9.2 related to?" Those existing string columns are left
exactly as they are (no migration, no risk to Phase 3.1/3.2's already-
approved code); these new tables give Product/Technology/Version real
identity so they can be first-class nodes in the knowledge relationship
graph, seeded from the distinct values already in use (see
``seed_migration.py``) rather than starting empty.
"""

from __future__ import annotations

from app.domain.governance import GovernanceFields


class Product(GovernanceFields):
    id: str
    name: str


class Technology(GovernanceFields):
    id: str
    name: str


class Version(GovernanceFields):
    id: str
    name: str
    product_id: str | None = None
    """Which product this version belongs to, e.g. "9.2" under Command
    Center -- optional since not every version is known to be scoped to
    one product yet."""
