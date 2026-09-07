"""Evidence-to-claim grounding validator (Final Hardening Pass,
Objective 1).

Rule 4/9/11's existing output gates (``confidence_expansion.py``,
``scope_expansion.py``, ``troubleshooting_expansion.py``) already catch
three SPECIFIC, real, previously-observed fabrication shapes: an
unsupported confidence-tier upgrade, an unsupported customer-scope
claim, and an unsupported generic troubleshooting action. This module
is deliberately NOT a fourth copy of that idiom aimed at a fourth fixed
phrase -- it catches a different, complementary risk: the LLM citing a
SPECIFIC FACT VALUE (an identifier, a timestamp, a computed duration,
an error code, a case number, a configuration value) that does not
actually appear anywhere in the evidence it was given. Those three
existing gates ask "did the model use forbidden WORDS"; this one asks
"did the model cite a VALUE that isn't real" -- genuinely new coverage,
not a duplicate.

Deliberately provider-agnostic (Step 4 of this phase): ``validate()``
takes plain text (the generated answer, and the exact evidence text
surface the provider was given -- in this codebase, ``PromptBuilder.
build()``'s own ``user_prompt`` return value, so the "known evidence"
this module checks against is guaranteed byte-identical to what the
model actually saw, not a second, potentially-drifting reconstruction
of it) and returns a plain, structured result. Nothing here imports or
knows about ``OllamaProvider``/Ollama at all.

Deliberately NOT a universal hallucination detector (see this module's
own docstring sections below, and Step 1G/1I of this phase's own
brief: "a practical validator is sufficient", "do not over-engineer
this"): it is a closed set of deterministic, regex/entity-extraction-
based checks over the specific claim shapes support engineers actually
rely on (identifiers, timestamps, timing deltas, error codes, case
IDs, configuration values, and unsupported definitive-causation
language) -- the same "closed list, trades recall for zero false
positives, never an LLM" idiom every other safety gate in this
codebase already uses (see ``scope_expansion.py``'s own docstring for
why that trade-off is deliberate here too). A claim shape outside this
list is not caught; that is a known, explicit limitation, not a silent
gap.

FAILURE POLICY (Step 1J): every issue this module finds is currently
either REPAIRED (configuration-value claims: the specific unsupported
sentence is replaced with a safe, honest fallback clause -- Step 1F's
own worked example) or causes the WHOLE answer to be REJECTED (every
other claim type: identifiers, timestamps, timing deltas, error codes,
case IDs, unsupported definitive-causation language). Surgically
editing a fabricated identifier or timestamp out of otherwise-fluent
prose cannot be done safely/deterministically without risking a
grammatically intact but still misleading sentence; full rejection
(the caller falls back to the existing, unmodified deterministic
answer -- exactly like every other ``_attempt_llm_answer`` gate already
does) is the safe default per Step 1J's own instruction: "The user must
NEVER receive an unsupported critical fact merely because the LLM
generated it fluently."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.domain.entities import EntityType
from app.domain.provenance import ResolutionProvenance
from app.engines.external_knowledge.service import _TICKET_NUMBER_RE
from app.engines.log_intelligence.entity_extractor import RegexEntityExtractor
from app.engines.shared.text_matching import phrase_present

_IDENTIFIER_ENTITY_TYPES = (
    EntityType.METER_NUMBER,
    EntityType.SERIAL_NUMBER,
    EntityType.ENDPOINT_ID,
    EntityType.COMMAND_LOG_ID,
    EntityType.REQUEST_ID,
    EntityType.CORRELATION_ID,
    EntityType.SESSION_ID,
)
"""Step 1A's own list ("meter IDs, meter serial numbers, device IDs,
endpoint IDs, collector IDs, correlation IDs, transaction IDs, request
IDs, command IDs, command-log IDs") mapped onto the real, closed
``EntityType`` set this codebase's ``RegexEntityExtractor`` actually
recognizes -- see [[log-intelligence-feature]]. There is no separate
"device ID"/"collector ID"/"transaction ID" ``EntityType`` (confirmed
by reading ``app/domain/enums.py`` before writing this module, per
Step 1's audit instruction) -- those support-engineering terms map onto
the identifier types that really exist here (endpoint/serial/
correlation/request). Deliberately does NOT include THREAD_ID/
PROCESS_ID/POD_NAME/HOST_NAME/IP_ADDRESS/etc.: Step 1A's own examples
are all "which meter/transaction/command failed" identifiers, not
general infrastructure identifiers, and treating every recognized
entity type as a validated "identifier" would risk false positives on
values this validator's evidence-text surface was never meant to
enumerate exhaustively."""

_LOOSE_IDENTIFIER_PATTERNS: dict[EntityType, re.Pattern[str]] = {
    EntityType.METER_NUMBER: re.compile(r"\bmeter(?:[-_ ]*(?:number|no|#))?\s*[:=]?\s*([\w\-]{4,})\b", re.IGNORECASE),
    EntityType.SERIAL_NUMBER: re.compile(r"\bserial(?:[-_ ]*(?:number|no|#))?\s*[:=]?\s*([\w\-]{4,})\b", re.IGNORECASE),
    EntityType.ENDPOINT_ID: re.compile(r"\bendpoint(?:[-_ ]?id)?\s*[:=]?\s*([\w\-]{4,})\b", re.IGNORECASE),
    EntityType.COMMAND_LOG_ID: re.compile(r"\bcommand[-_ ]?log(?:[-_ ]?id)?\s*[:=]?\s*([\w\-]{4,})\b", re.IGNORECASE),
    EntityType.REQUEST_ID: re.compile(r"\brequest[-_ ]?id\s*[:=]?\s*([\w\-]{4,})\b", re.IGNORECASE),
    EntityType.CORRELATION_ID: re.compile(r"\bcorrelation[-_ ]?id\s*[:=]?\s*([\w\-]{4,})\b", re.IGNORECASE),
    EntityType.SESSION_ID: re.compile(r"\bsession[-_ ]?id\s*[:=]?\s*([\w\-]{4,})\b", re.IGNORECASE),
}
"""``RegexEntityExtractor``'s own patterns (reused unchanged by
``_known_identifier_values``/the 1A check below) all require a literal
``:``/``=`` separator -- exactly right for the LOG TEXT they were
designed to parse, but an LLM's natural-language answer prose (and,
just as importantly, this codebase's own ``StructuredResolution.
problem``/``root_cause`` free text, which real investigations populate
from a human-written title/description, not a log line) routinely
states the same fact WITHOUT one: "Meter 99999999 failed", "the request
carried command log ID CMD-1". These patterns are the identical keyword
association with the separator made OPTIONAL, applied to BOTH the
answer text and the evidence text (never only one side -- an asymmetric
loosening would either miss a real fabrication phrased in prose, or
reject a real, legitimately-evidenced identifier merely because the
evidence stated it in prose too). A minimum 4-character value
(matching ``RegexEntityExtractor``'s own real patterns' effective
floor) avoids matching a short, common word that happens to follow one
of these keywords."""

_TIME_RE = re.compile(r"\b(\d{2}):(\d{2}):(\d{2})\b")
"""Matches the time-of-day portion of a timestamp regardless of
whether it is rendered as a full ISO datetime (``2026-08-29T10:00:10``,
``LogObservationSummary``'s own ``.isoformat()`` rendering) or a bare
``10:00:10`` (this codebase's own log-analysis composer's own
rendering convention, per [[resolveiq-project]]) -- comparing only the
H:M:S tuple sidesteps a date-format/separator mismatch that has nothing
to do with whether the underlying moment is real evidence."""

_DURATION_CLAIM_RE = re.compile(
    r"\b(?:after|within|in)\s+(\d+(?:\.\d+)?)\s*(seconds?|secs?|s\b|minutes?|mins?)\b",
    re.IGNORECASE,
)
"""Step 1C: a claimed elapsed-time duration ("timed out after 9
seconds", "responded within 30 seconds")."""

_HTTP_STATUS_RE = re.compile(r"\bHTTP\s*(\d{3})\b", re.IGNORECASE)
"""Step 1D's own worked example. Deliberately narrow (a real HTTP
status code, not any 3-digit number) -- see this module's docstring on
scope: a closed, specific claim shape, not a general numeric-fact
checker."""

_ANSWER_CONFIG_INSTRUCTION_RE = re.compile(
    r"[^.\n]*\b(?:set|change|adjust|configure)\b[^.\n]*?\bto\s+(\d+(?:\.\d+)?)\s*"
    r"(seconds?|secs?|minutes?|mins?|ms|milliseconds?)\b[^.\n]*[.\n]?",
    re.IGNORECASE,
)
"""Step 1F's own worked example ("Set timeout to 60 seconds.") -- the
INSTRUCTION shape this validator looks for in the answer. The match
span is the WHOLE containing sentence (up to the nearest sentence
boundary), not just the number, so a repair can safely replace the
entire unsupported instruction rather than leaving a grammatically
broken fragment behind."""

_EVIDENCE_CONFIG_VALUE_RE = re.compile(
    r"\b(?:timeout|interval|threshold|delay|retry|retries|limit|value|setting)\b"
    r"[^.\n]{0,25}?[:=]?\s*(?:is\s+|of\s+|=\s*)?(\d+(?:\.\d+)?)\s*"
    r"(seconds?|secs?|minutes?|mins?|ms|milliseconds?)\b",
    re.IGNORECASE,
)
"""What counts as a real, DOCUMENTED value in evidence text -- a wider
net than ``_ANSWER_CONFIG_INSTRUCTION_RE`` on purpose: documentation
states a fact ("Documented timeout = 30 seconds", "the configured
timeout is 30 seconds", "retry interval: 30 seconds"), it does not
phrase itself as an instruction the way an LLM's own overclaim does.
Anchored to a real configuration-shaped keyword (never a bare number)
so an unrelated number/unit pair elsewhere in the evidence (a
timestamp, an event count) is never mistaken for a documented setting."""

_SAFE_ROOT_CAUSE_HEDGE_PHRASES: list[str] = [
    "root cause is not confirmed",
    "root cause is not established",
    "root cause is not yet established",
    "root cause is not yet confirmed",
    "root cause is not yet known",
    "root cause is unknown",
    "root cause is not known",
    "root cause is not determined",
    "root cause is not yet determined",
]
"""The same "strip safe hedges before matching" idiom
``confidence_expansion.py``'s ``SAFE_CONFIDENCE_HEDGE_PHRASES`` already
established (see that module's docstring) -- without it, the honest,
correctly-hedged sentence "...the root cause is not confirmed from the
current evidence" (Step 1H's own GOOD example) would itself trip the
positive-assertion check below, since "root cause is" is a literal
substring of "root cause is not confirmed"."""

_DEFINITIVE_CAUSATION_PHRASES: list[str] = [
    "root cause is",
    "definitely caused by",
    "is definitely caused by",
    "resolved by",
    "is resolved by",
    "fixed by",
    "is fixed by",
    "guaranteed to",
    "guaranteed fix",
]
"""Step 1H's own list, minus ``confirmed root cause``/``configuration
must be`` (handled separately below -- see ``_CONFIRMED_ROOT_CAUSE_RE``
and ``_MANDATE_PHRASES``). Deliberately a DIFFERENT phrase set from
``confidence_expansion.py``'s ``contains_unsupported_confidence_claim``
(which gates the literal words "confirmed"/"verified") -- this list
gates definitive-CAUSATION language that carries the same overclaiming
risk without using either of those two words at all (e.g. "resolved by
restarting the collector" makes exactly as strong an unsupported claim
as "confirmed", just without the gated word), so the two gates do not
overlap and neither makes the other redundant, per this phase's own
explicit "must be separate from and not duplicate" instruction. Safe
ONLY at CONFIRMED tier (mirroring ``contains_unsupported_confidence_
claim``'s own, already-proven-safe "gate on tier, not on wording"
design) -- checked by ``validate()`` against the real ``confidence``
tier passed in, never guessed from the answer text itself."""

_MANDATE_PHRASES: list[str] = ["configuration must be", "must be set to"]
"""Step 1H's "configuration must be" -- prescriptive/directive language
kept separate from ``_DEFINITIVE_CAUSATION_PHRASES`` because it is
risky independent of confidence tier (a diagnostic conclusion earned at
CONFIRMED tier is not the same thing as a configuration INSTRUCTION,
which Step 1F treats as "potentially dangerous" regardless of tier) --
always checked, at every tier."""


@dataclass
class ClaimIssue:
    """One unsupported claim this validator found. ``claim_type`` is
    one of: "identifier", "timestamp", "timing_delta", "error",
    "case_id", "configuration_value", "root_cause_language",
    "mandate_language" -- see ``validate()``'s docstring for what each
    checks. ``severity`` is "high" (the whole answer is rejected) or
    "low" (the specific claim is repaired in place)."""

    claim_type: str
    excerpt: str
    severity: str
    detail: str


@dataclass
class GroundingResult:
    """Structured result Step 1I asks for: ``valid``,
    ``unsupported_claims``, plus ``repaired_text``/``severity`` this
    module adds so a caller never has to re-derive them. ``valid`` is
    True iff no HIGH-severity issue was found (LOW-severity issues are
    already repaired into ``repaired_text``, so they never make an
    answer invalid on their own). ``affected_claims`` (Step 1I) is
    ``unsupported_claims`` itself -- kept as one list rather than two
    parallel ones, since every issue found IS an affected claim."""

    valid: bool
    unsupported_claims: list[ClaimIssue] = field(default_factory=list)
    repaired_text: str | None = None
    severity: str = "none"  # "none" | "low" | "high"

    @property
    def affected_claims(self) -> list[ClaimIssue]:
        return self.unsupported_claims


def _extract_identifiers(text: str) -> set[str]:
    """Union of ``RegexEntityExtractor``'s own strict (colon/equals-
    separated) patterns and the loose, separator-optional patterns
    above -- applied identically regardless of whether ``text`` is the
    evidence surface or the answer text (see ``_LOOSE_IDENTIFIER_
    PATTERNS``'s own docstring for why symmetry matters here). Values
    are uppercased for comparison only, so "cmd-999" and "CMD-999" are
    recognized as the same real identifier regardless of which case the
    evidence happened to use versus how the model echoed it."""
    values = {
        e.value.upper()
        for e in RegexEntityExtractor().extract(text)
        if e.entity_type in _IDENTIFIER_ENTITY_TYPES
    }
    for pattern in _LOOSE_IDENTIFIER_PATTERNS.values():
        values.update(m.group(1).upper() for m in pattern.finditer(text))
    return values


def _known_times(evidence_text: str) -> set[tuple[str, str, str]]:
    return {m.groups() for m in _TIME_RE.finditer(evidence_text)}


def _known_http_statuses(evidence_text: str) -> set[str]:
    return {m.group(1) for m in _HTTP_STATUS_RE.finditer(evidence_text)}


def _known_case_ids(evidence_text: str) -> set[str]:
    return {m.group(0).upper() for m in _TICKET_NUMBER_RE.finditer(evidence_text)}


def _seconds(value: str, unit: str) -> float:
    n = float(value)
    return n * 60 if unit.lower().startswith("min") else n


def _known_duration_seconds(known_times: set[tuple[str, str, str]]) -> set[float]:
    """Every real elapsed-time gap (in seconds) between any two known
    evidence timestamps -- Step 1C's own example verifies one specific
    request/timeout pair, but the tier-based LLM prompt this validator
    guards (``PromptBuilder``'s own ``ROOT CAUSE``/``LOG OBSERVATIONS``
    text) carries no per-event role labels to pin down WHICH two
    moments a duration claim refers to (that richer, per-role
    association already exists, deterministically, in this codebase's
    separate log-analysis composer -- see [[resolveiq-project]] --
    which does not call the LLM at all). Checking every pair
    conservatively: if the claimed duration doesn't match ANY real gap
    between two known moments, it cannot be grounded in this evidence
    at all."""
    seconds_values = []
    for h, m, s in known_times:
        seconds_values.append(int(h) * 3600 + int(m) * 60 + int(s))
    gaps: set[float] = set()
    for i, a in enumerate(seconds_values):
        for b in seconds_values[i + 1 :]:
            gaps.add(float(abs(a - b)))
    return gaps


def _sentence_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Widens ``[start, end)`` out to the nearest sentence boundary
    (a preceding ``. ``/newline/start-of-text, and a following
    ``.``/newline/end-of-text) so a repair replaces one complete
    sentence, never a mid-sentence fragment."""
    left = text.rfind(".", 0, start)
    left = left + 1 if left != -1 else 0
    nl = text.rfind("\n", 0, start)
    left = max(left, nl + 1 if nl != -1 else 0)
    right = end
    for boundary in (".", "\n"):
        idx = text.find(boundary, end)
        if idx != -1:
            right = max(right, idx + (1 if boundary == "." else 0))
    return left, min(right, len(text))


_UNSUPPORTED_CONFIG_FALLBACK = "This value is not established from current evidence."


def validate(
    answer_text: str,
    evidence_text: str,
    *,
    confidence: "ResolutionProvenance | None" = None,
) -> GroundingResult:
    """The validator's single entry point (Step 1I: ``validate(answer,
    evidence)``). ``evidence_text`` must be the exact text surface the
    provider was given (this codebase's callers pass ``PromptBuilder.
    build()``'s own ``user_prompt`` -- see ``ChatOrchestrator.
    _validate_llm_answer_grounding``) -- never a re-summarized or
    re-derived version of it, so "known" here always means "the model
    could actually have copied this from what it was shown," not a
    broader or narrower set. ``confidence`` (the real, already-decided
    ``StructuredResolution.confidence`` tier) gates the definitive-
    causation check (see ``_DEFINITIVE_CAUSATION_PHRASES``'s own
    docstring); ``None`` is treated the same as any non-CONFIRMED tier.

    Deterministic and cheap: only regex/closed-entity-extraction over
    plain strings, no network I/O, no LLM call of its own (Step 8 of
    this phase: "the validator must be lightweight... prefer
    deterministic validation")."""
    issues: list[ClaimIssue] = []
    repaired = answer_text

    # 1A -- identifiers.
    known_ids = _extract_identifiers(evidence_text)
    for claimed in _extract_identifiers(answer_text):
        if claimed not in known_ids:
            issues.append(
                ClaimIssue(
                    "identifier",
                    claimed,
                    "high",
                    f"Identifier {claimed!r} does not appear in the supplied evidence.",
                )
            )

    # 1B -- timestamps (compared as bare H:M:S -- see _TIME_RE docstring).
    known_times = _known_times(evidence_text)
    for h, m, s in _TIME_RE.findall(answer_text):
        if (h, m, s) not in known_times:
            issues.append(
                ClaimIssue(
                    "timestamp",
                    f"{h}:{m}:{s}",
                    "high",
                    f"Timestamp {h}:{m}:{s} does not appear in the supplied evidence.",
                )
            )

    # 1C -- timing deltas: the claimed duration must match some real gap
    # between two known evidence timestamps (see _known_duration_seconds).
    if known_times:
        known_gaps = _known_duration_seconds(known_times)
        for value, unit in _DURATION_CLAIM_RE.findall(answer_text):
            claimed = _seconds(value, unit)
            if not any(abs(claimed - gap) < 0.5 for gap in known_gaps):
                issues.append(
                    ClaimIssue(
                        "timing_delta",
                        f"{value} {unit}",
                        "high",
                        f"A {value} {unit} duration does not match any real interval between evidence timestamps.",
                    )
                )

    # 1D -- HTTP error codes.
    known_statuses = _known_http_statuses(evidence_text)
    for code in _HTTP_STATUS_RE.findall(answer_text):
        if code not in known_statuses:
            issues.append(
                ClaimIssue("error", f"HTTP {code}", "high", f"HTTP {code} does not appear in the supplied evidence.")
            )

    # 1E -- case/ticket IDs.
    known_cases = _known_case_ids(evidence_text)
    for case_id in _TICKET_NUMBER_RE.findall(answer_text):
        if case_id.upper() not in known_cases:
            issues.append(
                ClaimIssue("case_id", case_id, "high", f"Case/ticket {case_id!r} does not appear in the supplied evidence.")
            )

    # 1F -- configuration values: REPAIRED, not rejected (Step 1F's own
    # worked example is a rewrite, not a whole-answer fallback).
    evidence_config_values = {(v, u.lower().rstrip("s")) for v, u in _EVIDENCE_CONFIG_VALUE_RE.findall(evidence_text)}
    for m in list(_ANSWER_CONFIG_INSTRUCTION_RE.finditer(repaired)):
        value, unit = m.group(1), m.group(2)
        if (value, unit.lower().rstrip("s")) in evidence_config_values:
            continue
        start, end = _sentence_span(repaired, m.start(), m.end())
        issues.append(
            ClaimIssue(
                "configuration_value",
                repaired[m.start() : m.end()],
                "low",
                f"A configuration value of {value} {unit} is not documented in the supplied evidence; repaired.",
            )
        )
        repaired = repaired[:start] + _UNSUPPORTED_CONFIG_FALLBACK + repaired[end:]

    # 1H -- definitive-causation language, safe only at CONFIRMED tier.
    # Safe hedge phrases (e.g. "root cause is not confirmed") are
    # stripped from a working copy first -- see _SAFE_ROOT_CAUSE_HEDGE_
    # PHRASES's own docstring -- so a correctly-hedged sentence is never
    # flagged merely for containing "root cause is" as a substring.
    if confidence != ResolutionProvenance.CONFIRMED:
        stripped = answer_text
        for hedge in _SAFE_ROOT_CAUSE_HEDGE_PHRASES:
            stripped = re.sub(re.escape(hedge), " ", stripped, flags=re.IGNORECASE)
        for phrase in _DEFINITIVE_CAUSATION_PHRASES:
            if phrase_present(phrase, stripped):
                issues.append(
                    ClaimIssue(
                        "root_cause_language",
                        phrase,
                        "high",
                        f"Definitive-causation phrase {phrase!r} is not supported at the current confidence tier.",
                    )
                )
    # "configuration must be"/"must be set to" -- always checked
    # (regardless of tier), unless the exact value it names is itself a
    # documented configuration value (then 1F's own check above already
    # repaired or accepted it).
    for phrase in _MANDATE_PHRASES:
        if phrase_present(phrase, answer_text) and not evidence_config_values:
            issues.append(
                ClaimIssue(
                    "mandate_language",
                    phrase,
                    "high",
                    f"Prescriptive phrase {phrase!r} is not backed by any documented configuration value.",
                )
            )

    high = [i for i in issues if i.severity == "high"]
    severity = "high" if high else ("low" if issues else "none")
    return GroundingResult(valid=not high, unsupported_claims=issues, repaired_text=repaired, severity=severity)
