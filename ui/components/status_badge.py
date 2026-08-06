"""Status badge rendering.

Maps the backend's ``InvestigationStatus`` values to a colored pill,
matching the RFC rev 3 §12 status table. Today's backend enum has 4
values (open, in_progress, resolved, closed); Phase 5 extends it to 6
(waiting_customer, waiting_l3) -- this file only needs two more dict
entries at that point, not a rewrite.
"""

from __future__ import annotations

_STATUS_STYLE: dict[str, tuple[str, str, str]] = {
    # status_value: (display_label, background, text_color)
    "open": ("Open", "#EEF1F4", "#545F6E"),
    "in_progress": ("Investigating", "#E4EDF1", "#1F5772"),
    "waiting_customer": ("Waiting: Customer", "#F5E9D6", "#B8791A"),
    "waiting_l3": ("Waiting: L3", "#F5E9D6", "#B8791A"),
    "resolved": ("Resolved", "#E3F0E7", "#3E8B5C"),
    "closed": ("Closed", "#EEF1F4", "#8892A0"),
}


def status_badge_html(status: str) -> str:
    label, bg, fg = _STATUS_STYLE.get(status, (status.replace("_", " ").title(), "#EEF1F4", "#545F6E"))
    return (
        f'<span style="display:inline-flex;align-items:center;gap:5px;'
        f'font-family:ui-monospace,Consolas,monospace;font-size:0.72rem;'
        f'padding:3px 9px;border-radius:99px;font-weight:600;'
        f'background:{bg};color:{fg};">'
        f'<span style="width:6px;height:6px;border-radius:50%;background:{fg};"></span>'
        f"{label}</span>"
    )
