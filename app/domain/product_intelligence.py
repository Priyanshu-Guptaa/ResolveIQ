"""Product Intelligence domain model -- Component Registry (incremental
start, per RFC rev 2/3).

Scope discipline for this increment: a clean, extensible data shape and a
seeded registry of real Command Center components. No matching/reasoning
logic lives here or anywhere else yet -- an engineer looks a component up
by name (Workspace's Product Intelligence panel, a plain dropdown).
Recommendation Engine V2 (a later phase) is what will eventually decide
*which* component is relevant to an investigation; this phase only makes
sure there's real data to look up once it does.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ComponentProfile(BaseModel):
    """Everything known about one Command Center (or other product)
    component. All list fields default to empty rather than requiring
    every profile to fill in every field -- a sparse profile is still a
    real, useful profile."""

    id: str
    name: str
    product: str
    responsibilities: list[str] = Field(default_factory=list)
    related_components: list[str] = Field(default_factory=list)
    """Names of other ComponentProfile entries -- plain string references,
    not enforced foreign keys. Deliberately simple for this increment."""
    typical_failures: list[str] = Field(default_factory=list)
    required_logs: list[str] = Field(default_factory=list)
    common_sql: list[str] = Field(default_factory=list)
    known_bugs: list[str] = Field(default_factory=list)
    documentation_links: list[str] = Field(default_factory=list)
    playbooks: list[str] = Field(default_factory=list)
