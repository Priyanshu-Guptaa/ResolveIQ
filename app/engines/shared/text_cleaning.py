"""Strips low-signal boilerplate from ticket/task text before it is used
for semantic embedding or vector search -- never for anything the user
sees directly (DB records, detail views, Evidence tab, etc. always keep
the original, unmodified text; only what gets embedded/searched changes).

Found live (2026-08-11): the embedding model used for local
historical-investigation/documentation/known-bug search
(``all-MiniLM-L6-v2``) has an effective window of only 256 tokens. A
real ServiceNow-style task-description disclaimer paragraph
("This Template is expected to be reviewed in full...") alone consumes
86 of those tokens, and roughly a quarter of the historical
investigations corpus (171 of 728) opens with a numbered header block
(Organization Name / Priority / Environment Setup / Browser / ...)
where most fields are blank -- both compete with the actual defect
description for that same small budget. A real new investigation and a
real historical match that happened to share the exact same 86-token
boilerplate prefix scored 86% "similar" despite describing unrelated
defects (CLECO meter-program-change failure vs. an unrelated
full-meter-read request) -- the reported "vague/irrelevant match"
symptom traced all the way back to this.

Purely subtractive and conservative: only removes text carrying
essentially zero discriminating signal (the known disclaimer
paragraph, or a "Label:" line with nothing -- or only whitespace --
after the colon). Never touches a line that has an actual value, even
a low-quality one (e.g. a placeholder date) -- that boundary is
deliberate, to avoid a second, more fragile problem: guessing at every
possible "this value doesn't really count" pattern across real
customer data.
"""

from __future__ import annotations

import re

_DISCLAIMER_RE = re.compile(
    r"This Template is expected to be reviewed in full\.[\s\S]{0,400}?Task Template",
    re.IGNORECASE,
)

_EMPTY_LABEL_LINE_RE = re.compile(r"^\s*(?:\d+\.\s*)?[A-Za-z][\w /()&+.,-]{0,60}:\s*$")


def strip_low_signal_boilerplate(text: str) -> str:
    """Removes the known template disclaimer paragraph and any
    "Label:" line with no value, leaving everything else -- including
    label lines that DO have a value -- untouched. Safe to call on any
    text; a no-op when neither pattern is present."""
    if not text:
        return text
    cleaned = _DISCLAIMER_RE.sub("", text)
    lines = [line for line in cleaned.splitlines() if not _EMPTY_LABEL_LINE_RE.match(line)]
    return "\n".join(lines).strip()
