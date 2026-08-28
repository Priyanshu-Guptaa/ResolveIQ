"""Unsupported customer-scope-expansion detection (Chat Assistant
Phase 35B).

Phases 27-31 solved fabrication in response to an EXPLICIT
customer-scope QUESTION ("does this affect other customers?") by
removing that question's clause from the LLM's input entirely (see
``app.engines.chat.scope_question``). Phase 35's real-call validation
of qwen2.5:3b as the final model (40 BENCH1 + 10 MULTIPART calls)
found a DIFFERENT trigger for the same underlying claim: on a normal,
non-scope multi-part question ("What is the root cause, has this
happened before, and what should I check first?"), the model
spontaneously volunteered *"This issue has likely occurred before with
other customers in the APAC region using Network Hub technology"* --
unprompted, with no scope question anywhere in the input, and with the
fixture's own supplied evidence naming exactly one customer (TEPCO)
and nothing about any other. Phase 35B's own 20-call reproduction of
the identical fixture/question found the same pattern once more (1/20,
"This issue applies to customers in the APAC region..."), plus three
milder near-misses that stayed anchored to the real customer name but
used ambiguous plural phrasing ("TEPCO customers"). Combined: 2/30
real, freshly-observed unsupported-scope-expansion claims (Wilson 95%
CI [1.8%, 21.3%] -- rare, but real and reproducible, not a fluke).

``scope_question.py``'s own architecture (remove the triggering input
clause before the LLM ever sees it) does not apply here: there is no
removable clause to strip -- the offending language appears while the
model is legitimately answering a real, necessary question ("has this
happened before"), which cannot itself be omitted from the prompt.
The analogous move, proven necessary by this project's own repeated
finding that prompt-only asks cannot reliably prevent this class of
fabrication (Phases 27-30's four straight failed prompt-only attempts
at the *explicit*-question version of this exact problem), is to
validate the OUTPUT deterministically instead of trusting the prompt
to prevent it.

This module is that deterministic, prompt-independent gate. It is
deliberately NOT a blanket filter on any mention of "customer" --
Absolute Rule 19 of this phase explicitly forbids solving this by
suppressing all customer-related language, and legitimate,
evidence-backed applicability (however many real customers are
actually supplied) must remain fully answerable (Rule 20). The closed
phrase list below contains only GENERIC scope-EXPANSION language --
"other customers", "all customers", "broader deployment", and so on --
never a customer name, and never the word "customer" alone. Because
``PromptBuilder``'s own ``_applicability_block`` never itself renders
generic expansion language (it always names specific supplied values
or the literal word "UNKNOWN"), any occurrence of one of these phrases
in the model's own final answer was necessarily invented by the model,
not copied from the prompt -- no cross-referencing against the actual
supplied customer/region list is needed to know it is unsupported.

Same "closed list, fixed phrase table, never an LLM, conservative
default, trades recall for zero false positives" idiom already
established by ``scope_question.py`` and
``app.engines.chat.troubleshooting_question`` -- reusing the same
underlying whole-phrase primitive rather than a third implementation
of phrase matching. A paraphrase outside this list will not be caught;
that is a known, explicit limitation, not a silent gap, identical in
kind to the limitation both of those modules already carry.
"""

from __future__ import annotations

import re

from app.engines.shared.text_matching import phrase_present

SCOPE_EXPANSION_PHRASES: list[str] = [
    "other customers",
    "other customer",
    "additional customers",
    "additional customer",
    "any other customer",
    "any other customers",
    "multiple customers",
    "several customers",
    "many customers",
    "all customers",
    "customers in the region",
    "customers across the region",
    "customers in that region",
    "other regions",
    "broader deployment",
    "wider deployment",
    "industry-wide",
    "across the industry",
    "widespread issue",
    "widespread problem",
    "customers such as",
    "customers like",
]
"""Closed phrase table -- see module docstring. Every entry names a
GENERIC scope beyond what any real ``ApplicabilitySummary`` ever
literally states; none names a specific customer, so a fixture with
several real, legitimately-supplied customers (e.g. TEPCO and PG&E
both actually present in the evidence) never matches here -- only
language claiming scope the evidence does NOT establish does.

``customers such as``/``customers like`` were added after Step 10's
own 100-call validation surfaced "affecting customers such as TEPCO" --
grammatically implying TEPCO is one EXAMPLE from a larger set, the same
underlying claim as "other customers" in different words, evidenced
directly from this phase's own real-call data, not hypothesized."""


_CUSTOMERS_IN_NAMED_REGION_RE = re.compile(
    r"\bcustomers\s+(?:in|across|throughout)\s+(?:the\s+|that\s+)?[\w-]+(?:\s+[\w-]+){0,2}\s+region\b",
    re.IGNORECASE,
)
"""Chat Assistant Phase 35B Step 3's own reproduction ("This issue
applies to customers in the APAC region...") named a SPECIFIC region
("APAC"), which a closed literal-phrase list cannot enumerate in
advance (a real fixture's region name is arbitrary, unbounded data,
never a fixed vocabulary the way "other customers" is a fixed English
idiom). This one narrowly-scoped regex catches "customers in/across/
throughout [the/that] <region-name> region" for any region name.

Deliberately PLURAL "customers" only (not "customer") -- Step 10's
100-call real validation found the singular form produces a real,
measured false-positive class: "the TEPCO customer in the APAC
region" (a real, singular, correctly-named customer) is a safe,
common, legitimate way to state real applicability, and matching it
would reject 8/20 otherwise-safe MULTIPART responses in that run alone.
The plural form is what every actually-unsafe real call used ("customers
in the APAC region", with no name attached) -- restricting to plural
preserved every true positive from that same run while eliminating
every one of those 8 false positives (re-verified directly against the
saved real-call text, not assumed)."""


def contains_unsupported_scope_expansion(text: str) -> bool:
    """IMPORTANT (integration-order requirement): callers must apply
    this ONLY to the raw text ``_generate_llm_answer`` returns, BEFORE
    ``customer_scope_statement()``/``no_evidence_backed_check_statement()``
    are appended. Those deterministic statements deliberately use
    overlapping vocabulary ("...whether any other customer is also
    affected is not established") to correctly express the ABSENCE of
    broader scope -- running this check after they are appended would
    misfire on safe, already-correct deterministic text. See
    ``ChatOrchestrator._generate_answer``'s call site.

    True if ``text`` (an LLM-generated final answer) contains any
    whole-phrase match from ``SCOPE_EXPANSION_PHRASES``, or the
    "customers in <any named region>" pattern above. Conservative by
    construction: only ever True on an exact, closed-list phrase match
    or that one narrowly-scoped regex, never a general heuristic
    guess, and never triggered merely by a real customer/region name
    appearing on its own (however many are legitimately supplied)."""
    if any(phrase_present(phrase, text) for phrase in SCOPE_EXPANSION_PHRASES):
        return True
    return _CUSTOMERS_IN_NAMED_REGION_RE.search(text) is not None
