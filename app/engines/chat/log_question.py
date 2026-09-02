"""Log-analysis question detection (Chat + Log Intelligence integration).

Real usage gap this closes: an uploaded log's real, already-parsed
``LogEvent``/``ExtractedEntity`` data (timestamps, severities, messages,
meter/endpoint/correlation identifiers -- see ``app.domain.entities.
LogEvent``) was already fully computed by the existing, unmodified
``LogIntelligenceEngine`` at upload time, but Chat's own answer never
surfaced any of it beyond aggregate counts (``LogObservationSummary``,
Chat Assistant Phase 33) -- a user asking "what happened in this log?"
or "show me the timeline" got the same generic investigation-confidence
boilerplate as any other question, with the real, already-computed
per-event detail sitting unused.

This module is the same closed-list, deterministic classifier idiom as
``knowledge_question``/``troubleshooting_question``/``scope_question``:
never an LLM, conservative by construction (only ever True on an exact
phrase match), no opinion about whether a log actually exists for this
turn -- that is entirely ``ChatOrchestrator._compose_log_analysis_
answer``'s decision, made only when real ``LOG_FILE`` evidence is
present.
"""

from __future__ import annotations

from app.engines.shared.text_matching import phrase_present

LOG_ANALYSIS_PHRASES: list[str] = [
    "analyze this log",
    "analyze this meter log",
    "analyze the log",
    "analyse this log",
    "what happened in this log",
    "what happened here",
    "what happened in the log",
    "summary of this log",
    "summary of this meter communication",
    "summarize this log",
    "give me a summary of this log",
    "what errors do you see",
    "what errors are present",
    "what errors are present in this log",
    "where did the communication fail",
    "which command failed",
    "what happened before the failure",
    "what happened after the failure",
    "is there a timeout",
    "are there retries",
    "did the meter respond",
    "what identifiers are present",
    "show me the timeline",
    "give me the timeline",
    "what should i check next",
    "is this a known issue",
    "is this related to a known issue",
    "have we seen this before",
    "have we seen it before",
]
"""Note: the last four entries deliberately overlap with
``knowledge_question.HISTORICAL_PHRASES`` -- when a log is attached,
"have we seen this before?" should be answered WITH the log's own
observed content plus cross-source correlation (Chat + Log
Intelligence integration report, Steps 19/39), not treated as a bare
historical-knowledge question with no log context at all. The two
classifiers are independent and both may legitimately match the same
question; ``ChatOrchestrator._compose_answer`` checks log-analysis
first specifically so a log, when present, is never ignored."""
"""Closed phrase table -- see module docstring. Whole-phrase matching
via ``phrase_present`` only, same discipline as every other question
classifier in this codebase -- a question that merely mentions "log"
or "error" in an unrelated way is never misclassified."""


def contains_log_analysis_question(question: str) -> bool:
    """True if ``question`` contains any whole-phrase match from
    ``LOG_ANALYSIS_PHRASES``. Conservative by construction: only ever
    True on an exact, closed-list phrase match, never a heuristic
    guess. Callers must separately verify real LOG_FILE evidence
    actually exists for this turn before treating this as "answer from
    the log" -- this function only classifies the QUESTION."""
    return any(phrase_present(phrase, question) for phrase in LOG_ANALYSIS_PHRASES)
