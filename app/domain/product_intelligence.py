"""Domain Intelligence -- Component Registry (Phase 2B, expanded).

Started as a flat "Component Profile" (Phase 2B increment 1: static
documentation lookup). This expansion turns it into a knowledge graph: a
component isn't just described, it's connected -- to other components, to
the services/queues/tables it touches, to how its behavior has changed
across versions, and to the investigations that have previously involved
it.

Scope discipline still applies. This is data, not reasoning:
- Every relationship field is a list of plain strings (or, for version
  differences, a small structured record). Nothing here is an enforced
  foreign key -- a value in ``related_components`` is only ever *treated*
  as a link to another node when the UI happens to find a matching
  ``ComponentProfile.name`` in the registry at render time. That lookup
  lives in the UI layer (ui/components/component_profile.py), not here.
- No scoring, matching, or inference lives in this module or the engine
  in front of it. Recommendation Engine V2 (Phase 2C) is the intended
  consumer of this richer shape; this phase only makes sure the data
  exists and is honestly populated (sparse/empty where we don't have
  real information, not filled with invented specifics).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.domain.governance import GovernanceFields


class VersionDifference(BaseModel):
    """One behavior change for a component between versions. Free-text,
    not a diff -- just enough structure for the UI (and later,
    Recommendation Engine V2) to know *which* version a change applies
    to without parsing it out of a sentence."""

    version: str
    change: str


class ComponentProfile(GovernanceFields):
    """Everything known about one Command Center (or other product)
    component. All list fields default to empty rather than requiring
    every profile to fill in every field -- a sparse profile is still a
    real, useful profile, and an empty relationship list is an honest
    "we don't know that yet," not a placeholder to hide.

    Fields are grouped below by what they describe, matching how the
    Architecture Explorer UI renders them as sections rather than one
    long flat list.
    """

    # --- Identity -----------------------------------------------------
    id: str
    name: str
    product: str

    # --- Overview -------------------------------------------------------
    responsibilities: list[str] = Field(default_factory=list)

    # --- Relationships (the graph edges) ---------------------------------
    # Plain string references to other nodes -- other ComponentProfile
    # names, or names of services/queues/tables that aren't components
    # in this registry. Not enforced foreign keys; see module docstring.
    related_components: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    """Other components or services this one requires to function."""
    consumes: list[str] = Field(default_factory=list)
    """Queues, topics, or data this component reads as input."""
    produces: list[str] = Field(default_factory=list)
    """Queues, topics, or data this component writes as output."""
    related_services: list[str] = Field(default_factory=list)
    related_queues: list[str] = Field(default_factory=list)
    database_tables: list[str] = Field(default_factory=list)

    # --- Operational shape ------------------------------------------------
    configuration: list[str] = Field(default_factory=list)
    """Config keys/settings that materially change this component's behavior."""
    events: list[str] = Field(default_factory=list)
    """Domain events this component publishes or subscribes to."""
    message_flows: list[str] = Field(default_factory=list)
    """Step-by-step message hand-offs this component participates in,
    written as "A -> B: what happens" for readability."""
    data_flows: list[str] = Field(default_factory=list)
    """Where this component's data comes from / goes to, written as
    "source -> destination (read|write, when)"."""

    # --- Behavior over time -------------------------------------------
    version_differences: list[VersionDifference] = Field(default_factory=list)
    typical_failures: list[str] = Field(default_factory=list)

    # --- Support knowledge ----------------------------------------------
    required_logs: list[str] = Field(default_factory=list)
    common_sql: list[str] = Field(default_factory=list)
    known_bugs: list[str] = Field(default_factory=list)
    documentation_links: list[str] = Field(default_factory=list)
    playbooks: list[str] = Field(default_factory=list)
    historical_investigations: list[str] = Field(default_factory=list)
    """Human-readable references to past investigations that involved
    this component. Populated manually/sparsely for now -- automatic
    linking against real investigation history is Recommendation Engine
    V2's job (Phase 2C), not this registry's."""
