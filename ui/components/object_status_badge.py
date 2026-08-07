"""Status badge for any governed knowledge object (Sprint 3, Phase 3.4
Knowledge Object Framework) -- generalizes Phase 3.2's
``document_status_badge.py`` from ``DocumentStatus`` (Documentation
only) to ``ObjectLifecycleStatus`` (all nine object types), adding the
one new value (``deprecated``). ``document_status_badge.py`` now
delegates here rather than duplicating the style map, so both callers
stay visually identical.
"""

from __future__ import annotations

_STATUS_STYLE: dict[str, tuple[str, str, str]] = {
    # status_value: (display_label, background, text_color)
    "draft": ("Draft", "#EEF1F4", "#545F6E"),
    "under_review": ("Under Review", "#F5E9D6", "#B8791A"),
    "published": ("Published", "#E3F0E7", "#3E8B5C"),
    "archived": ("Archived", "#EEF1F4", "#8892A0"),
    "deprecated": ("Deprecated", "#FBE7E5", "#B84A3E"),
}


def object_status_badge_html(status: str) -> str:
    label, bg, fg = _STATUS_STYLE.get(status, (status.replace("_", " ").title(), "#EEF1F4", "#545F6E"))
    return (
        f'<span style="display:inline-flex;align-items:center;gap:5px;'
        f'font-family:ui-monospace,Consolas,monospace;font-size:0.72rem;'
        f'padding:3px 9px;border-radius:99px;font-weight:600;'
        f'background:{bg};color:{fg};">'
        f'<span style="width:6px;height:6px;border-radius:50%;background:{fg};"></span>'
        f"{label}</span>"
    )
