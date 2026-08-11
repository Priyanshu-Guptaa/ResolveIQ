"""Entity extraction: recognizing IDs, hosts, exceptions, etc. in free text.

The extractor is a pluggable Protocol so the regex-based implementation
here can later be swapped for (or augmented by) an NLP/ML-based extractor
without touching any caller -- callers only depend on
:class:`EntityExtractor`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, Protocol

from app.domain.entities import ExtractedEntity
from app.domain.enums import EntityType

logger = logging.getLogger(__name__)

_CONTEXT_WINDOW = 40
"""Characters of surrounding text captured on each side of a match, for
the human-readable ``context_snippet``."""


class EntityExtractor(Protocol):
    """Anything that can pull entities out of raw text."""

    def extract(self, text: str, *, source_evidence_id: str | None = None) -> list[ExtractedEntity]:
        ...


@dataclass(frozen=True)
class _Pattern:
    """One recognition rule for a given :class:`EntityType`.

    ``group`` selects which regex group becomes the entity value (0 = whole
    match). Multiple patterns may exist per entity type -- e.g. a
    "meter number" can appear as ``meter=123`` or as a bare 9-12 digit
    token near the word "meter".
    """

    regex: re.Pattern[str]
    group: int = 1


def _compile(pattern: str, flags: int = re.IGNORECASE) -> re.Pattern[str]:
    return re.compile(pattern, flags)


# Registry of recognition rules per entity type. Kept as module-level data
# (not hardcoded into the extraction loop) so adding support for a new
# entity type, or tightening a noisy pattern, never requires touching the
# extractor's control flow -- open/closed principle.
_PATTERN_REGISTRY: dict[EntityType, list[_Pattern]] = {
    EntityType.THREAD_ID: [
        _Pattern(_compile(r"\bthread[-_ ]?(?:id|name)?\s*[:=]\s*([\w.\-]+)")),
        _Pattern(_compile(r"\[([\w.\-]*(?:exec|pool|thread)-[\w.\-]+)\]")),
        _Pattern(_compile(r"\b(Thread-\d+)\b"), group=1),
    ],
    EntityType.PROCESS_ID: [
        _Pattern(_compile(r"\b(?:process[-_ ]?id|pid)\s*[:=]\s*(\d+)")),
    ],
    EntityType.SESSION_ID: [
        _Pattern(_compile(r"\bsession[-_ ]?id\s*[:=]\s*([\w\-]+)")),
        _Pattern(_compile(r"\bJSESSIONID=([\w]+)")),
    ],
    EntityType.ENDPOINT_ID: [
        _Pattern(_compile(r"\bendpoint[-_ ]?id\s*[:=]\s*([\w\-]+)")),
    ],
    EntityType.METER_NUMBER: [
        _Pattern(_compile(r"\bmeter(?:[-_ ]*(?:number|no|#))?\s*[:=]\s*([\w\-]+)")),
    ],
    EntityType.SERIAL_NUMBER: [
        _Pattern(_compile(r"\bserial(?:[-_ ]*(?:number|no|#))?\s*[:=]\s*([\w\-]+)")),
    ],
    EntityType.COMMAND_LOG_ID: [
        _Pattern(_compile(r"\bcommand[-_ ]?log[-_ ]?id\s*[:=]\s*([\w\-]+)")),
    ],
    EntityType.REQUEST_ID: [
        _Pattern(_compile(r"\brequest[-_ ]?id\s*[:=]\s*([\w\-]+)")),
        _Pattern(_compile(r"\bx-request-id\s*[:=]\s*([\w\-]+)")),
    ],
    EntityType.CORRELATION_ID: [
        _Pattern(_compile(r"\bcorrelation[-_ ]?id\s*[:=]\s*([\w\-]+)")),
        _Pattern(_compile(r"\bx-correlation-id\s*[:=]\s*([\w\-]+)")),
    ],
    EntityType.EXCEPTION_TYPE: [
        # Negative lookahead excludes a ServiceNow-template-style empty
        # field label (e.g. "ContainsError = ;") -- found live: a real
        # investigation's "Likely issue" ended up naming an empty field
        # label as the exception, because the field happened to be
        # named "...Error" and this pattern doesn't otherwise care
        # whether it was ever actually thrown. A real exception/error
        # name is never immediately followed by "= ;", "= ,", or "="
        # at end of text -- that's specifically the shape of an unset
        # template field.
        _Pattern(_compile(r"\b([\w.$]+(?:Exception|Error))\b(?!\s*=\s*(?:[;,]|$))", flags=0)),
        # A domain error-code convention seen in real ticket text
        # ("DCWErr_Invalid_Response_Length") that the pattern above
        # never catches, since it doesn't end in the literal suffix
        # "Exception"/"Error" -- general "<Prefix>Err_<detail>" shape,
        # not specific to any one product's naming.
        _Pattern(_compile(r"\b([A-Za-z]+Err_[A-Za-z0-9_]+)\b", flags=0)),
    ],
    EntityType.STACK_TRACE: [
        _Pattern(_compile(r"^\s*(at\s+[\w.$]+\([^)]*\))", flags=re.MULTILINE), group=1),
    ],
    EntityType.SQL_SESSION: [
        _Pattern(_compile(r"\bspid\s*[:=]?\s*(\d+)")),
        _Pattern(_compile(r"\bsql[-_ ]?session[-_ ]?id\s*[:=]\s*(\d+)")),
    ],
    EntityType.POD_NAME: [
        _Pattern(_compile(r"\bpod(?:[-_ ]?name)?\s*[:=]\s*([\w\-]+)")),
        _Pattern(_compile(r"\b([\w-]+-[a-f0-9]{8,10}-[a-z0-9]{5})\b", flags=0)),
    ],
    EntityType.HOST_NAME: [
        _Pattern(_compile(r"\bhost(?:name)?\s*[:=]\s*([\w.\-]+)")),
    ],
    EntityType.IP_ADDRESS: [
        _Pattern(
            _compile(
                r"\b((?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d))\b",
                flags=0,
            ),
            group=1,
        ),
    ],
    EntityType.SERVICE_NAME: [
        _Pattern(_compile(r"\bservice(?:[-_ ]?name)?\s*[:=]\s*([\w\-]+)")),
    ],
    EntityType.KAFKA_TOPIC: [
        _Pattern(_compile(r"\btopic\s*[:=]\s*([\w.\-]+)")),
    ],
    EntityType.RABBITMQ_QUEUE: [
        _Pattern(_compile(r"\bqueue\s*[:=]\s*([\w.\-]+)")),
    ],
    EntityType.CONSUMER_GROUP: [
        _Pattern(_compile(r"\bconsumer[-_ ]?group\s*[:=]\s*([\w.\-]+)")),
    ],
    EntityType.URL: [
        _Pattern(_compile(r"(https?://[^\s'\"<>]+)", flags=0), group=1),
    ],
    EntityType.API_ENDPOINT: [
        _Pattern(
            _compile(r"\b(?:GET|POST|PUT|DELETE|PATCH)\s+(/[\w/\-{}.]*)", flags=0),
        ),
    ],
    EntityType.USER_ID: [
        _Pattern(_compile(r"\buser[-_ ]?id\s*[:=]\s*([\w\-@.]+)")),
    ],
    EntityType.EVENT_ID: [
        _Pattern(_compile(r"\bevent[-_ ]?id\s*[:=]\s*(\d+)")),
    ],
}


class RegexEntityExtractor:
    """Pattern-registry based :class:`EntityExtractor`.

    Deliberately simple and dependency-free (stdlib ``re`` only) so it runs
    with zero setup cost -- appropriate for a first implementation that
    already covers every entity type in the platform spec. A future sprint
    can add an ML-based extractor behind the same Protocol.
    """

    def __init__(self, registry: dict[EntityType, list[_Pattern]] | None = None) -> None:
        self._registry = registry or _PATTERN_REGISTRY

    def extract(self, text: str, *, source_evidence_id: str | None = None) -> list[ExtractedEntity]:
        if not text:
            return []

        found: list[ExtractedEntity] = []
        seen: set[tuple[EntityType, str]] = set()

        for entity_type, patterns in self._registry.items():
            for pattern in patterns:
                for match in pattern.regex.finditer(text):
                    value = match.group(pattern.group).strip()
                    if not value:
                        continue
                    key = (entity_type, value.lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    found.append(
                        ExtractedEntity(
                            entity_type=entity_type,
                            value=value,
                            context_snippet=_context_snippet(text, match.start(), match.end()),
                            source_evidence_id=source_evidence_id,
                        )
                    )

        logger.debug("Extracted %d entities from %d chars of text", len(found), len(text))
        return found

    def extract_many(
        self, texts: Iterable[str], *, source_evidence_id: str | None = None
    ) -> list[ExtractedEntity]:
        results: list[ExtractedEntity] = []
        for text in texts:
            results.extend(self.extract(text, source_evidence_id=source_evidence_id))
        return results


def _context_snippet(text: str, start: int, end: int) -> str:
    lo = max(0, start - _CONTEXT_WINDOW)
    hi = min(len(text), end + _CONTEXT_WINDOW)
    snippet = text[lo:hi].replace("\n", " ").strip()
    prefix = "..." if lo > 0 else ""
    suffix = "..." if hi < len(text) else ""
    return f"{prefix}{snippet}{suffix}"
