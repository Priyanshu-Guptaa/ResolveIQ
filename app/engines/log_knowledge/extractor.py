"""Deterministic extraction of Log Intelligence Knowledge Base records
from an operational wiki export (Confluence "CC Logs Repository" page,
exported to PDF/HTML/text).

No LLM reasoning anywhere in this module -- everything here is regex/
state-machine parsing over the document's own structure. Every field on
every extracted record is either literally present in the source text
or a straightforward, traceable derivation from it:

- component name: parsed from the log path itself (the folder segment),
  never guessed from prose.
- platform: inferred purely from path syntax (backslash + ``~``/``..``
  prefix => Windows; ``/var``/``/etc`` prefix => Linux).
- priority: the order log paths were listed in the source, 1 = first --
  this is the single most reliable structural signal in the document
  (the "Logs:" block's line order survives even where the summary
  arrow-chain line above it was garbled by PDF text extraction losing
  the arrow glyph -- see ``_split_chain``).
- explanation: a template describing the real extracted position
  (component, scenario, technology, step N of M), optionally appending
  the wiki's own note text verbatim when the source has one next to
  that specific log line.

``raw_paths`` on every :class:`LogRepositoryLocation` preserves the
original source line verbatim, so anything this parser gets wrong is
always traceable back to exactly what was actually written.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.domain.log_intelligence_kb import LogCollectionScenario, LogCollectionStep, LogRepositoryLocation, LogSourceApplication

# --- Recognized structural markers ------------------------------------------
# A generic pattern, not specific literal section names: any heading that
# talks about a "workflow" and either "log location" or "logs" introduces
# one technology's table of scenarios. Matches every technology heading
# observed in the real document (RF Mesh, RF Mesh IP variants, Wi-Sun,
# DAS/Kafka, DLMS/APAC, M2M/SBS NPL, LTE/PSTN, Series 5 Water Meter, ...)
# without hardcoding any of those specific names.
_TECHNOLOGY_HEADER_RE = re.compile(r"workflow.*(log location|logs)", re.IGNORECASE)

# Standalone reference sections (no scenario table, just "here are the
# logs for X") -- their own heading pattern.
_REFERENCE_HEADER_RE = re.compile(r"^([A-Za-z][\w /()&+.-]{1,40}?)\s+logs:?$", re.IGNORECASE)

_REGION_RE = re.compile(r"^(NAM|APAC)\s+Region$", re.IGNORECASE)

_LOGS_MARKER_RE = re.compile(r"^Logs:?$", re.IGNORECASE)

_LOCATION_PREFIX_RE = re.compile(r"^Log(?:ger)?\s+location:\s*", re.IGNORECASE)
"""A handful of reference-section entries prefix the path with "Log
location:"/"Logger location:" instead of listing it bare -- stripped
before path matching, not discarded (the path itself is unaffected)."""

# A path line. The document mixes two styles: the per-scenario tables
# use full prefixed paths (~\Logs\X\logfile.log, /var/log/...), while
# the later standalone reference sections often list bare
# "Component\logfile.log" or even a bare "ComponentName.log" with no
# separator at all. One permissive pattern covers both: any line that
# is *entirely* path-and-filename-shaped, ending in a real extension,
# optionally trailed by a " - "/" – " note. Anchored full-line (^...$)
# so it can't accidentally match prose that merely contains a period.
_PATH_RE = re.compile(
    r"^\{?(?P<path>[\w~./][\w\\/.\- ]*\.\w{2,10}\*?|/[\w./-]+/)\s*\}?\s*(?:[-–(]\s*(?P<note>.+?)\)?\s*)?$"
)

_SCENARIO_KEYWORDS = (
    "command request",
    "command response",
    "command",
    "response",
    "reads",
    "events",
    "firmware download",
    "firmware",
    "login",
    "topology events",
    "collector communication",
)


@dataclass
class _RawLogEntry:
    path_text: str
    note: str | None
    technology: str
    region: str | None
    scenario_type: str


@dataclass
class ExtractedWikiPage:
    log_entries: list[_RawLogEntry] = field(default_factory=list)


def _classify_platform(path: str) -> str:
    if path.startswith("~") or path.startswith(".."):
        return "windows"
    if path.startswith("/"):
        return "linux"
    if "\\" in path:
        return "windows"
    return "unknown"


_GENERIC_DIR_NAMES = {"log", "logs"}
"""Directory segments that are a generic OS log root, not a component-
identifying folder -- e.g. "/var/log/RemoteAccess.log" names the
*component* via its filename ("RemoteAccess"), not via "log" (that
would incorrectly group every top-level Linux log file that has no
app-specific subfolder under one fake "log" component). Treated the
same as a truly bare filename: component falls back to None so the
caller uses the enclosing section's own name (_resolve_component)."""


def _split_path(path: str) -> tuple[str, str | None, str | None]:
    """Returns (root_path, component_name, filename) parsed purely from
    the path's own segments -- never inferred from surrounding prose.
    ``component_name`` is None for a bare filename with no directory
    segment at all (e.g. "NMS_Listener.log", listed directly under a
    reference section with no per-file subfolder), or for a filename
    directly under a generic "log"/"logs" root with no app-specific
    subfolder (see ``_GENERIC_DIR_NAMES``) -- the caller falls back to
    the enclosing section's own name in both cases, so several such
    files correctly group under one LogSourceApplication (e.g. "NMS")
    instead of each becoming its own component."""
    normalized = path.replace("\\", "/")
    # A stray space next to a separator (a real formatting slip observed
    # in the source, e.g. "MsgProcCmdRspHost \logfile.log") must not
    # produce a second, whitespace-distinct component name.
    parts = [p.strip() for p in normalized.split("/") if p.strip()]
    if not parts:
        return path, None, None
    if normalized.endswith("/"):
        # Directory-only root (e.g. NMS's Linux path) -- the last
        # segment is the most specific component-identifying folder.
        return "/".join(parts[:-1]) if len(parts) > 1 else parts[0], parts[-1], None
    filename = parts[-1]
    if len(parts) >= 2:
        component = parts[-2]
        root = "/".join(parts[:-2]) or parts[0]
        if component.lower() in _GENERIC_DIR_NAMES:
            root = "/".join(parts[:-1])
            component = None
    else:
        component = None
        root = ""
    return root, component, filename


def _normalize_scenario_label(buffer_text: str) -> str | None:
    lowered = buffer_text.lower()
    for keyword in _SCENARIO_KEYWORDS:
        if keyword in lowered:
            # Re-derive a clean, title-cased label from what was
            # actually found, preferring the longer/more specific match
            # already ordered first in _SCENARIO_KEYWORDS.
            cleaned = re.sub(r"\s+", " ", buffer_text).strip(" :.-")
            return cleaned[:80] if cleaned else keyword.title()
    return None


_BROKEN_PATH_TAIL_RE = re.compile(r"^\\[\w.]+\.\w+\*?$")
_BARE_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]+$")


def _repair_broken_paths(lines: list[str]) -> list[str]:
    """A narrow, observed PDF-export artifact (2 occurrences in the real
    document): a path's middle (component) segment gets pushed onto its
    own line *after* the root+filename fragments instead of staying
    inline, e.g.::

        ~\\Logs\\
        \\logfile.log
        MsgProcPushdatahost

    instead of the intended ``~\\Logs\\MsgProcPushdatahost\\logfile.log``.
    Detected by the exact three-line shape (dangling backslash, a
    ``\\filename.ext`` continuation, then a single bare word with no
    spaces) and reassembled into one clean path line -- never invents a
    component name, only relocates one that is genuinely present a few
    lines away."""
    repaired: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if (
            stripped.endswith("\\")
            and i + 2 < len(lines)
            and _BROKEN_PATH_TAIL_RE.match(lines[i + 1].strip())
            and _BARE_WORD_RE.match(lines[i + 2].strip())
        ):
            root = stripped
            tail = lines[i + 1].strip()  # "\logfile.log"
            component = lines[i + 2].strip()
            repaired.append(f"{root}{component}{tail}")
            i += 3
            continue
        repaired.append(lines[i])
        i += 1
    return repaired


def extract_log_entries(text: str) -> list[_RawLogEntry]:
    """Single-pass state machine over the document's lines. Returns one
    entry per log path line found, tagged with the technology/region/
    scenario context active at that point in the document."""
    lines = _repair_broken_paths([ln.rstrip() for ln in text.split("\n")])

    current_technology: str | None = None
    current_region: str | None = None
    current_scenario = "General"
    label_buffer: list[str] = []
    entries: list[_RawLogEntry] = []

    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        stripped = raw.strip()
        i += 1
        if not stripped:
            continue

        region_match = _REGION_RE.match(stripped)
        if region_match:
            current_region = region_match.group(1).upper()
            label_buffer.clear()
            continue

        if _TECHNOLOGY_HEADER_RE.search(stripped):
            # e.g. "RF Mesh IP workflow (with ANSIAdapter) and log
            # location (Applicable for NAM Customers)" -- take the text
            # before "workflow" as the technology name. Excludes bare
            # ".rstrip('()')" so a real closing paren in the technology
            # name (e.g. "RF Mesh (DAS implementation)") survives.
            tech = re.split(r"\bworkflow\b", stripped, flags=re.IGNORECASE)[0].strip(" ,-")
            # The table's own "Workflow and Logs" column header also
            # matches this pattern (it contains "workflow" + "logs") but
            # has nothing before "workflow" -- it must never overwrite
            # the real technology heading that came just before it. It
            # still marks a genuine table-structure boundary though (it
            # sits right before the first scenario row), so it must
            # clear label_buffer even when tech is empty -- otherwise
            # stray short lines between the technology heading and this
            # column-header row (observed: a repeated "RF Mesh" cell and
            # a "Type" column header) leak into the first scenario's
            # label as a messy prefix.
            if tech:
                current_technology = tech
                current_scenario = "General"
            label_buffer.clear()
            continue

        ref_match = _REFERENCE_HEADER_RE.match(stripped)
        if ref_match:
            current_technology = ref_match.group(1).strip()
            current_scenario = "General"
            label_buffer.clear()
            continue

        if _LOGS_MARKER_RE.match(stripped):
            label = _normalize_scenario_label(" ".join(label_buffer)) if label_buffer else None
            if label:
                current_scenario = label
            label_buffer.clear()
            continue

        path_candidate = _LOCATION_PREFIX_RE.sub("", stripped)
        path_match = _PATH_RE.match(path_candidate)
        if path_match and current_technology is not None:
            matched_path = path_match.group("path")
            # A space inside the matched path itself (not the trailing
            # note) means this line was really two things concatenated
            # (observed once: two service endpoints on one line with no
            # clear separator) -- reject rather than import a
            # nonsensical merged component name.
            if " " not in matched_path.strip():
                entries.append(
                    _RawLogEntry(
                        path_text=matched_path,
                        note=path_match.group("note"),
                        technology=current_technology,
                        region=current_region,
                        scenario_type=current_scenario,
                    )
                )
            label_buffer.clear()
            continue

        # Not a recognized marker or path -- a candidate scenario-label
        # fragment (these arrive split across several short lines in the
        # PDF export, e.g. "Command" / "Request" / "(Outbound)").
        if len(stripped) <= 40 and not stripped.lower().startswith("note"):
            label_buffer.append(stripped)
        elif _normalize_scenario_label(" ".join(label_buffer)) is None:
            label_buffer.clear()
        # else: the fragments collected so far already resolve to a
        # real scenario label (e.g. "Command" + "Request" +
        # "(Outbound)" -> "Command Request (Outbound)"), and this long
        # line is that scenario's arrow-chain flow summary -- confirmed
        # structural content (sometimes prefixed "Command Flow:"/
        # "Response Flow:", sometimes bare component names with no
        # prefix at all, and sometimes wrapped across several such long
        # physical lines by the PDF export) but never further label
        # text. Leave label_buffer untouched rather than letting it
        # clear the label already collected -- this was a real bug:
        # every scenario in the document has this exact shape, and the
        # flow summary previously wiped the label right before "Logs:"
        # consumed it, collapsing the type to "General".

    return entries


def _resolve_component(entry: "_RawLogEntry") -> tuple[str, str, str | None]:
    """(root, component, filename), falling back to the entry's own
    technology/section name when the path has no directory segment at
    all -- see _split_path's docstring for why."""
    root, component, filename = _split_path(entry.path_text)
    return root, component or entry.technology, filename


def build_records(
    entries: list[_RawLogEntry], *, product: str, source_wiki_page: str
) -> tuple[list[LogSourceApplication], list[LogCollectionScenario]]:
    """Turns raw tagged path entries into governed domain records.
    Multiple entries for the same component name are merged into one
    LogSourceApplication (a component can appear in several scenarios);
    entries sharing (technology, scenario_type, region) become one
    LogCollectionScenario with steps in the order they were found."""
    sources_by_name: dict[str, LogSourceApplication] = {}
    scenario_groups: dict[tuple[str, str, str | None], list[_RawLogEntry]] = {}

    for entry in entries:
        root, component, filename = _resolve_component(entry)
        platform = _classify_platform(entry.path_text)
        existing = sources_by_name.get(component)
        if existing is None:
            existing = LogSourceApplication(
                id=f"log-src-{_slug(component)}",
                name=component,
                location=LogRepositoryLocation(platform=platform, root_path=root, filename_patterns=[], raw_paths=[]),
                product=product,
                source_wiki_pages=[source_wiki_page],
            )
            sources_by_name[component] = existing
        if filename and filename not in existing.location.filename_patterns:
            existing.location.filename_patterns.append(filename)
        if entry.path_text not in existing.location.raw_paths:
            existing.location.raw_paths.append(entry.path_text)
        if entry.technology not in existing.technology:
            existing.technology.append(entry.technology)
        if entry.note and (existing.notes is None or entry.note not in existing.notes):
            existing.notes = f"{existing.notes}; {entry.note}" if existing.notes else entry.note

        key = (entry.technology, entry.scenario_type, entry.region)
        scenario_groups.setdefault(key, []).append(entry)

    scenarios: list[LogCollectionScenario] = []
    for (technology, scenario_type, region), group_entries in scenario_groups.items():
        steps: list[LogCollectionStep] = []
        seen_components: set[str] = set()
        total_components = len({_resolve_component(e)[1] for e in group_entries})
        for entry in group_entries:
            _, component, _ = _resolve_component(entry)
            if component in seen_components:
                continue
            seen_components.add(component)
            explanation = (
                f"Produced by {component} during the {scenario_type} flow for {technology}"
                f"{f' ({region})' if region else ''} -- step {len(steps) + 1} of "
                f"{total_components} in the message flow."
            )
            if entry.note:
                explanation += f" {entry.note.strip()}."
            steps.append(
                LogCollectionStep(
                    log_source_id=sources_by_name[component].id,
                    component_name=component,
                    priority=len(steps) + 1,
                    explanation=explanation,
                )
            )
        scenarios.append(
            LogCollectionScenario(
                id=f"log-scenario-{_slug(product)}-{_slug(technology)}-{_slug(scenario_type)}-{_slug(region or 'any')}",
                product=product,
                technology=technology,
                scenario_type=scenario_type,
                region=region,
                steps=steps,
                source_wiki_page=source_wiki_page,
            )
        )

    return list(sources_by_name.values()), scenarios


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def extract(text: str, *, product: str, source_wiki_page: str) -> tuple[list[LogSourceApplication], list[LogCollectionScenario]]:
    entries = extract_log_entries(text)
    return build_records(entries, product=product, source_wiki_page=source_wiki_page)
