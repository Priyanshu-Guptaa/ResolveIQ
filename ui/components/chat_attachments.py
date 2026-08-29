"""Pure, Streamlit-free state/formatting helpers for the AI Assistant
page's log-attachment list and async-enhancement display (Chat
Assistant Phase 42).

Deliberately has no ``streamlit``/``api_client`` import at all -- unlike
every other module under ``ui/components/``, this one holds only pure
functions over plain dicts/lists so it can be unit-tested exactly like
``ui/formatting.py`` already is (see ``tests/test_ui_formatting.py``'s
own ``sys.path.insert(0, ".../ui")`` + bare-import pattern, mirrored by
``tests/test_chat_attachments.py``). ``ui/views/7_AI_Assistant.py`` is
the only caller; it owns all the actual ``st.*``/``api_get``/
``api_post`` calls.

Why this exists (Phase 41's audit, items carried into Phase 42):
1. The page used to track only the single most-recently-uploaded log's
   metadata in session state, even though up to 5 are retained
   server-side for a standalone session -- this module's
   ``record_log_attachment`` maintains the full, truthful list instead.
2. The backend's real FIFO cap (``app.engines.chat.log_upload.
   ChatLogUploadService``, ``_MAX_LOGS_PER_SESSION = 5``) applies ONLY
   to standalone chat sessions -- investigation-scoped uploads go
   through ``InvestigationEngine.add_file_evidence``, which has no cap
   at all. ``record_log_attachment`` takes an explicit ``capped`` flag
   for exactly this reason: applying the cap unconditionally would
   fabricate an eviction that never happened server-side for an
   investigation-scoped session, which Absolute Rule 18 (Phase 42)
   forbids outright.
3. The evicted entry name is never guessed: this module never learns
   about an upload except through its own ``record_log_attachment``
   call for that exact upload, so the "oldest" entry in the tracked
   list -- the only thing ever evicted -- is always a real, previously
   recorded response, never a placeholder.
4. Async enhancement responses go through ``format_enhancement_status``
   so the page never has to duplicate the "what counts as success"
   logic inline -- a response is only ever reported as ``"completed"``
   when the backend's own status says so AND a non-empty answer_text is
   actually present, matching ``ChatEnhancementResponse``'s own
   documented contract (``app/api/schemas.py``) that ``answer_text`` is
   populated only once the backend's existing safety validators already
   passed it.
"""

from __future__ import annotations

MAX_TRACKED_LOG_ATTACHMENTS = 5
"""Mirrors ``app.engines.chat.log_upload._MAX_LOGS_PER_SESSION`` exactly.
Must stay in sync with that constant by inspection (both are small,
stable, and documented) -- not imported directly, since this module is
deliberately kept free of any backend import so it stays testable in
total isolation, the same "no cross-layer import" discipline
``ui/formatting.py`` already follows for domain enums it displays."""

_TERMINAL_ENHANCEMENT_STATUSES = {"completed", "failed", "rejected", "timed_out"}


def record_log_attachment(
    attachments: list[dict],
    upload_response: dict,
    *,
    capped: bool,
    max_attachments: int = MAX_TRACKED_LOG_ATTACHMENTS,
) -> tuple[list[dict], dict | None]:
    """Appends ``upload_response`` (the real ``ChatLogUploadResponse``
    dict the API returned) to a NEW list (the input is never mutated),
    applying the backend's own FIFO cap only when ``capped`` is True --
    pass ``capped=True`` for a standalone chat session, ``capped=False``
    for an investigation-scoped one (see module docstring point 2).

    Returns ``(new_list, evicted_entry_or_None)``. ``evicted_entry`` is
    the exact dict this module previously appended for that upload --
    never reconstructed, never a placeholder -- so callers can safely
    read its ``title`` to build a truthful eviction notice.
    """
    updated = [*attachments, dict(upload_response)]
    evicted: dict | None = None
    if capped and len(updated) > max_attachments:
        evicted = updated.pop(0)
    return updated, evicted


def format_eviction_notice(evicted: dict | None) -> str | None:
    """``None`` when nothing was evicted -- callers should overwrite
    whatever notice they are holding with THIS return value on every
    new upload attempt (never leave a stale notice from an earlier
    upload standing), which is what keeps the message from "remaining
    permanently misleading after subsequent uploads" (Phase 42 Step 4).

    Never fabricates a filename: only ever names the evicted entry when
    its recorded ``title`` is a real, non-empty string; falls back to a
    generic-but-truthful sentence otherwise."""
    if evicted is None:
        return None
    title = evicted.get("title")
    if title:
        return f"Maximum {MAX_TRACKED_LOG_ATTACHMENTS} logs are retained per chat. Oldest attachment removed: **{title}**."
    return f"Maximum {MAX_TRACKED_LOG_ATTACHMENTS} logs are retained per chat. The oldest attachment was removed."


def persistence_message(*, investigation_scoped: bool) -> str:
    """Chat Assistant Phase 42 Step 5 -- the distinction is reliably
    determinable from information the page already has BEFORE any
    upload even happens (the user's own Scope radio selection / the
    resolved ``investigation_id``), so this never has to guess."""
    if investigation_scoped:
        return "Logs attached here are saved to this investigation's evidence and remain available after a restart."
    return (
        f"Logs attached in a standalone chat are temporary -- held in memory only, lost if the "
        f"application restarts, and only the most recent {MAX_TRACKED_LOG_ATTACHMENTS} are kept."
    )


def format_enhancement_status(response: dict | None) -> dict:
    """Turns one (possibly ``None``/pending/terminal) poll of
    ``GET /chat/enhancements/{job_id}`` into a small, render-ready,
    always-safe outcome dict:

    - ``{"outcome": "completed", "answer_text": "..."}`` -- ONLY when
      the backend says ``status == "completed"`` AND a real, non-empty
      ``answer_text`` is present. A malformed/empty completion is never
      reported as success (Phase 42 Absolute Rule: never claim an
      enhancement succeeded when it did not).
    - ``{"outcome": "unavailable", "message": "..."}`` -- for
      failed/rejected/timed_out; ``message`` is the backend's own
      ``error`` string when present (``ChatEnhancementJob.error`` is
      already documented as always-safe-to-show text -- never a raw
      traceback -- see ``app/engines/chat/enhancement.py``), or a safe
      generic fallback otherwise.
    - ``{"outcome": "pending", "message": None}`` -- for ``None``,
      pending, running, an unrecognized status, or a "completed"
      response with no usable ``answer_text`` -- always the safe
      default, never silently upgraded to anything else.
    """
    if not isinstance(response, dict):
        return {"outcome": "pending", "message": None}
    status = response.get("status")
    if status == "completed":
        answer_text = response.get("answer_text")
        if isinstance(answer_text, str) and answer_text.strip():
            return {"outcome": "completed", "answer_text": answer_text}
        return {"outcome": "pending", "message": None}
    if status in ("failed", "rejected", "timed_out"):
        message = response.get("error")
        if not isinstance(message, str) or not message.strip():
            message = "AI-enhanced phrasing is unavailable; the answer above stands."
        return {"outcome": "unavailable", "message": message}
    return {"outcome": "pending", "message": None}


def is_enhancement_terminal(response: dict | None) -> bool:
    """True once a poll response has reached a status this page will
    never need to re-check -- used to decide whether to keep the job_id
    around for a future rerun's one-shot check, or drop it. Bounded by
    construction: a page only ever calls this on the result of ONE
    ``api_get`` per rerun (see ``ui/views/7_AI_Assistant.py``), never in
    a loop of its own."""
    return isinstance(response, dict) and response.get("status") in _TERMINAL_ENHANCEMENT_STATUSES
