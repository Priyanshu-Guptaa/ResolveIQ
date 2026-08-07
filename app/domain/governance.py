"""Shared governance fields for knowledge entities migrating from static
files/code constants to governed database tables (Sprint 3, Phase 3.1;
lifecycle status generalized in Phase 3.4 -- Knowledge Object Framework).

This is deliberately the minimum foundation, not the full RFC-003
Administration Platform: ``created_at``/``updated_at``/``created_by``/
``updated_by``/``is_active``/``status``. Full versioning (history,
compare, restore) exists since Phase 3.4 too (see
``app/domain/entity_version.py``); the Draft/Under Review/Approved/
Published/Archived *approval workflow* itself (who can transition what,
maker-checker) is still a later phase -- this mixin only makes sure
every governed record already carries the state that workflow will
need, so adding it later is additive, not a migration.

``created_by``/``updated_by`` are plain optional strings, not a foreign
key to a users table -- there is no user system yet (authentication is
explicitly out of scope for this phase). Migrated records are stamped
``"system:migration"`` (see ``seed_migration.py``); real attribution
starts once RFC-003's User model exists.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.domain.enums import ObjectLifecycleStatus


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class GovernanceFields(BaseModel):
    """Mixin -- every governed entity (ComponentProfile, KnownBugRecord,
    HistoricalInvestigationRecord, DocumentationRecord, QueryTemplate,
    Playbook, Product, Technology, Version) inherits this alongside its
    own fields.

    ``status`` defaults to ``PUBLISHED``: every one of these object
    types except Documentation was, before Phase 3.4, always visible/
    usable with no draft concept at all -- defaulting new instances to
    Published preserves that exact existing behavior for anything that
    doesn't opt into drafting. ``DocumentationRecord`` overrides this
    default back to ``DRAFT`` on its own field, preserving Phase 3.2's
    upload-starts-as-draft behavior exactly; Pydantic subclasses are
    free to override an inherited field's default, so both defaults
    coexist correctly.
    """

    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    created_by: str | None = None
    updated_by: str | None = None
    is_active: bool = True
    status: ObjectLifecycleStatus = ObjectLifecycleStatus.PUBLISHED
