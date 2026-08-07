"""Shared governance fields for knowledge entities migrating from static
files/code constants to governed database tables (Sprint 3, Phase 3.1).

This is deliberately the minimum foundation, not the full RFC-003
Administration Platform: ``created_at``/``updated_at``/``created_by``/
``updated_by``/``is_active`` only. Full versioning (history, compare,
restore) and the Draft/Under Review/Approved/Published/Archived approval
workflow are later phases (RFC-003 Phase 3.7+) -- this phase only makes
sure every governed record already carries the columns that workflow
will need, so adding it later is additive, not a migration.

``created_by``/``updated_by`` are plain optional strings, not a foreign
key to a users table -- there is no user system yet (authentication is
explicitly out of scope for this phase). Migrated records are stamped
``"system:migration"`` (see ``seed_migration.py``); real attribution
starts once RFC-003 Phase 3.1's User model exists.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class GovernanceFields(BaseModel):
    """Mixin -- every governed entity (ComponentProfile, KnownBugRecord,
    HistoricalInvestigationRecord, DocumentationRecord, QueryTemplate)
    inherits this alongside its own fields."""

    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    created_by: str | None = None
    updated_by: str | None = None
    is_active: bool = True
