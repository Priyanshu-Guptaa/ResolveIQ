"""Status badge for a Knowledge Management document (Sprint 3, Phase
3.2) -- same colored-pill pattern as ``status_badge.py``'s investigation
badge, over ``DocumentStatus`` instead of ``InvestigationStatus``.
"""

from __future__ import annotations

_STATUS_STYLE: dict[str, tuple[str, str, str]] = {
    # status_value: (display_label, background, text_color)
    "draft": ("Draft", "#EEF1F4", "#545F6E"),
    "under_review": ("Under Review", "#F5E9D6", "#B8791A"),
    "published": ("Published", "#E3F0E7", "#3E8B5C"),
    "archived": ("Archived", "#EEF1F4", "#8892A0"),
}


def document_status_badge_html(status: str) -> str:
    label, bg, fg = _STATUS_STYLE.get(status, (status.replace("_", " ").title(), "#EEF1F4", "#545F6E"))
    return (
        f'<span style="display:inline-flex;align-items:center;gap:5px;'
        f'font-family:ui-monospace,Consolas,monospace;font-size:0.72rem;'
        f'padding:3px 9px;border-radius:99px;font-weight:600;'
        f'background:{bg};color:{fg};">'
        f'<span style="width:6px;height:6px;border-radius:50%;background:{fg};"></span>'
        f"{label}</span>"
    )
