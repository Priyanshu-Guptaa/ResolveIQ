"""Unsupported generic-troubleshooting-action detection (Chat Assistant
Phase 46).

``app.engines.chat.troubleshooting_question`` solves fabrication in
response to an EXPLICIT troubleshooting QUESTION ("what should I
check?") by removing that question's clause from the LLM's input
entirely, whenever zero evidence-backed checks exist (Phase 32). Phase
45's real qwen2.5:3b replay -- reproducing Phase 44's own captured
real-call output -- found the exact same structural gap this project
already solved once for customer scope (Phase 35B): a DIFFERENTLY
PHRASED troubleshooting request ("What can I do to resolve this?",
"How should I troubleshoot this issue?", "What action should I take?",
and others) does not match ``troubleshooting_question``'s closed phrase
table, so the clause is never removed, the LLM IS called, and its raw
answer -- unprotected by any output-side check -- reached the simulated
user verbatim, containing exactly the kind of generic, non-evidence-
backed suggestion ("ensure it is properly configured") Rule 9's own
prompt text already tells the model never to substitute.

This module is that output-side gate -- the Rule 9 analogue of
``app.engines.chat.scope_expansion.contains_unsupported_scope_
expansion``, built on the identical idiom: closed phrase list, fixed
phrase table, never an LLM, never embeddings/similarity, conservative
default (trades recall for zero false positives on legitimate,
evidence-backed checks).

Why this is safe to gate narrowly: this function is only ever meant to
be consulted by ``ChatOrchestrator._attempt_llm_answer`` when
``available_checks(structured)`` is EMPTY (see that method's own call
site) -- exactly the same condition ``troubleshooting_question``'s
input-side removal already requires. When zero evidence-backed checks
exist for this answer, there is, by construction, no legitimate
troubleshooting recommendation the LLM could correctly be making at
all -- any troubleshooting-shaped sentence appearing in that exact
context was necessarily invented, not derived from AVAILABLE EVIDENCE-
BACKED CHECKS (which the model was told, verbatim, says NONE). This
mirrors ``contains_unsupported_scope_expansion``'s own reasoning
exactly (that gate's phrase list is safe because ``PromptBuilder``'s
own APPLICABILITY block never itself renders expansion language --
here, the analogous fact is that AVAILABLE EVIDENCE-BACKED CHECKS never
renders anything but "NONE" or the real, already-decided checks, so no
cross-referencing against the fixture is needed to know an unlisted
generic suggestion is unsupported).

The closed phrase list below is deliberately not invented from
scratch: it corresponds directly to the six generic categories Rule 9's
own system-prompt text (``app/engines/llm/prompt_builder.py``) already
names as forbidden substitutes -- "checking power, connections, cables,
signal, configuration, or restarting a device" -- plus the "plugged in"
idiom Phase 44/45's own investigation named explicitly, and the exact
"properly configured" phrase Phase 44's real captured output actually
used. Every entry is a verb-plus-object action phrase (never a lone
topic word like "power" or "configuration" alone), the same discipline
``SCOPE_EXPANSION_PHRASES`` already follows, so a legitimate answer
that merely mentions one of these topics without recommending a
generic action on it is never matched.
"""

from __future__ import annotations

from app.engines.shared.text_matching import phrase_present

UNSUPPORTED_TROUBLESHOOTING_PHRASES: list[str] = [
    # Power -- Rule 9's "checking power" category.
    "check the power",
    "check power",
    "verify the power",
    "verify power",
    "ensure power",
    "ensure it is powered",
    "ensure it is properly powered",
    "properly powered",
    # Connections/connectivity -- Rule 9's "connections" category, plus
    # the "plugged in" idiom named explicitly in Phase 44/45's own
    # investigation.
    "check the connections",
    "check connections",
    "check the connection",
    "check connection",
    "verify the connections",
    "verify connections",
    "verify the connection",
    "verify connection",
    "check connectivity",
    "verify connectivity",
    "check network connectivity",
    "ensure it is plugged in",
    "verify it is plugged in",
    "verify whether it is plugged in",
    "check if it is plugged in",
    "make sure it is plugged in",
    # Cables/wiring -- Rule 9's "cables" category.
    "check the cables",
    "check cables",
    "check the cable",
    "check cable",
    "verify the cables",
    "verify cables",
    "verify the cable",
    "verify cable",
    "check the wiring",
    "check wiring",
    "verify the wiring",
    "verify wiring",
    # Signal -- Rule 9's "signal" category.
    "check the signal",
    "check signal",
    "verify the signal",
    "verify signal",
    "signal strength",
    # Configuration -- Rule 9's "configuration" category, plus the exact
    # phrase Phase 44's real captured qwen2.5:3b output actually used.
    "properly configured",
    "ensure it is configured",
    "ensure it is properly configured",
    "check the configuration",
    "check configuration",
    "verify the configuration",
    "verify configuration",
    # Restarting a device -- Rule 9's "restarting a device" category.
    # Deliberately qualified with "the/your device" (never a bare
    # "restart it"/"restart the service") so this never matches a real,
    # evidence-backed instruction naming a specific service (e.g. "restart
    # the collector service"), which this codebase's own fixtures always
    # phrase as a named service, never as "the device".
    "restart the device",
    "reboot the device",
    "restart your device",
    "reboot your device",
    "power cycle the device",
    "power-cycle the device",
    "power cycle your device",
    "power-cycle your device",
]
"""Closed phrase table -- see module docstring. Every entry is a
generic, non-evidence-specific action recommendation; none names a
specific service, log field, or fixture-supplied value, so a real,
evidence-backed check (which this codebase's own StructuredResolution
fixtures always phrase concretely, e.g. "Restart the collector
service.", "Confirm the collector's route table is restored.") never
matches here."""


def contains_unsupported_troubleshooting_action(text: str) -> bool:
    """True if ``text`` (an LLM-generated final answer) contains any
    whole-phrase match from ``UNSUPPORTED_TROUBLESHOOTING_PHRASES``.
    Conservative by construction: only ever True on an exact, closed-
    list phrase match, never a general heuristic guess, and never
    triggered merely by a real, specific evidence-backed check being
    named (however it is phrased).

    IMPORTANT (integration-order requirement, mirroring
    ``contains_unsupported_scope_expansion`` exactly): callers must
    apply this ONLY when ``available_checks(structured)`` is empty for
    this answer, and ONLY to the raw text ``_generate_llm_answer``
    returns, BEFORE any deterministic statement is appended -- see
    ``ChatOrchestrator._attempt_llm_answer``'s call site. This function
    itself has no way to know whether checks exist; it only answers
    "does this text contain a generic troubleshooting phrase," which is
    unsupported precisely because the caller already knows there is
    nothing evidence-backed for it to legitimately be describing.

    Known, explicit limitation (identical in kind to
    ``contains_unsupported_scope_expansion``'s own, and to
    ``troubleshooting_question``'s closed phrase table): a paraphrase
    outside this list (e.g. "make sure it has an active internet
    connection" instead of "check connectivity") will not be caught.
    This is a deterministic phrase-matching gate, not a semantic
    hallucination classifier -- it does not, and cannot, guarantee
    complete coverage of every way a model could phrase an invented
    generic suggestion."""
    return any(phrase_present(phrase, text) for phrase in UNSUPPORTED_TROUBLESHOOTING_PHRASES)
