"""Backward-compat alias -- Sprint 3, Phase 3.4 generalized this into
``object_status_badge.py`` (all nine object types, incl. ``deprecated``).
Kept so existing imports (``document_detail.py``,
``9_Knowledge_Management.py``) keep working unchanged, same pattern as
``app.domain.enums.DocumentStatus = ObjectLifecycleStatus``.
"""

from __future__ import annotations

from components.object_status_badge import object_status_badge_html

document_status_badge_html = object_status_badge_html
