"""Playbook -- promoted to a first-class governed entity (Sprint 3,
Phase 3.3), so it can participate in the knowledge graph like every
other object (Known Bug, Document, SQL Template, Historical
Investigation).

Deliberately minimal: id/title/product/description/steps. RFC-003's
richer vision (per-step required evidence, decision points, SQL/log
links per checklist item) is Playbook *Management* -- a later
Administration module. This phase only needs playbooks to have real
identity so they can be related to other objects; ``ComponentProfile.
playbooks`` (Phase 2B, free-text steps embedded directly on a
component) is untouched and coexists, same as every other "new FK
field alongside preserved free text" pattern from Phase 3.1.
"""

from __future__ import annotations

from pydantic import Field

from app.domain.governance import GovernanceFields


class Playbook(GovernanceFields):
    id: str
    title: str
    product: str | None = None
    description: str = ""
    steps: list[str] = Field(default_factory=list)
