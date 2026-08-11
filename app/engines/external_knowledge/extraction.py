"""Deterministic text extraction shared by both connectors -- HTML
stripping and capped truncation. No LLM, no summarization beyond
"take the real text and cut it to a sane length": both TFS and
Confluence store rich-text fields as HTML, and every downstream
consumer (ranking, the UI, ``recommended_action`` templates) needs
plain text.
"""

from __future__ import annotations

import re

_TAG_RE = re.compile(r"<[^>]+>")
_ENTITY_MAP = {
    "&nbsp;": " ",
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&#39;": "'",
}
_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")

_MAX_EXTRACT_CHARS = 600
"""Same cap discipline as the rest of this codebase (e.g.
app/engines/recommendation/engine.py's _MAX_SNIPPET_CHARS) -- long
enough to be genuinely useful context, short enough to stay
scannable."""


def strip_html(raw: str | None) -> str:
    """Real, if minimal, HTML-to-text: drops tags, decodes the small
    set of entities actually observed in TFS/Confluence rich-text
    fields, collapses whitespace. Not a full HTML parser -- doesn't
    need to be, since the output is only ever read as plain text, and
    a slightly-imperfect strip on an edge case (e.g. an embedded
    <script>, which none of these fields legitimately contain) is
    still strictly safer to display than raw markup."""
    if not raw:
        return ""
    text = _TAG_RE.sub(" ", raw)
    for entity, replacement in _ENTITY_MAP.items():
        text = text.replace(entity, replacement)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def truncate(text: str, max_len: int = _MAX_EXTRACT_CHARS) -> str:
    """Word-boundary truncation with a real ellipsis -- same fix as
    ui/formatting.py's truncate_words, applied here to plain body text
    rather than titles (longer cap, no aggressive word-boundary
    snapping needed since body text isn't a single-line label)."""
    text = text.strip()
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    last_space = cut.rfind(" ")
    if last_space >= max_len * 0.6:
        cut = cut[:last_space]
    return cut.rstrip() + "…"


def extract_and_truncate(raw_html: str | None, max_len: int = _MAX_EXTRACT_CHARS) -> str:
    return truncate(strip_html(raw_html), max_len)
