"""Small, shared text-formatting helpers used across UI components --
kept here (not duplicated per-component) once the same bug (see
``truncate_words``) turned up independently in more than one place.
"""

from __future__ import annotations


def truncate_words(text: str, max_len: int = 60) -> str:
    """Truncates at the last word boundary at or before ``max_len``
    characters and appends an ellipsis -- never a bare ``text[:max_len]``
    slice, which cuts mid-word with no visual indicator that anything
    was cut at all (confirmed the real cause of garbled-looking
    historical-match titles like "...Each Incremental interval e").

    Falls back to a hard character cut only when there's no space in a
    reasonable window (e.g. one long unbroken token) -- still appends
    the ellipsis either way, so truncation is always visually obvious.
    """
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    last_space = cut.rfind(" ")
    if last_space >= max_len * 0.4:
        cut = cut[:last_space]
    return cut.rstrip() + "…"


def ticket_number_from_tags(tags: str | list[str] | None) -> str | None:
    """Pulls the real ticket/task number back out of a record's tags --
    Task Import stores it as a ``"ticket:<number>"`` tag (see
    ``app/engines/task_import/importer.py``'s ``_tags_for``), which
    every current UI surface reads ``tags`` from without ever parsing
    back out. Accepts either the comma-joined string
    ``KnowledgeMatch.metadata["tags"]`` carries or a real list (the
    admin object's own ``tags`` field), so callers don't each need to
    know which shape they have.
    """
    if not tags:
        return None
    parts = tags if isinstance(tags, list) else [p.strip() for p in tags.split(",")]
    for part in parts:
        part = part.strip()
        if part.startswith("ticket:"):
            number = part[len("ticket:") :].strip()
            if number:
                return number
    return None
