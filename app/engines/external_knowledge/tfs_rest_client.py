"""Real TFS connector -- WIQL search + batched work-item detail fetch,
against the on-prem TFS REST API (api-version=5.0, confirmed live
against am.tfs.landisgyr.net this session).

Field mapping is grounded in two real work items pulled live from the
actual server, not the generic Azure DevOps docs: a Bug (#2467475) and
a resolved Issue (#2465364). The two types have materially different
field sets -- Bug carries ``Landis.BugRootCause``, ``LandisGyr.CRMID``,
``Microsoft.VSTS.Common.Severity``/``ResolvedReason``; Issue has none
of those, only ``System.Tags`` and ``Microsoft.VSTS.Common.ClosedDate``.
This client branches per type rather than assuming one shared shape.

Auth: NTLM/Negotiate via Windows SSPI (``requests-negotiate-sspi``),
confirmed live this session -- authenticates transparently as whatever
Windows identity the ResolveIQ process runs under, no password/token
ever touches this code. When ``personal_access_token`` is supplied
(hosted/Linux deployments, where SSPI does not exist) it is used instead
as HTTP Basic auth with an empty username -- the PAT path has NOT been
verified against the real TFS server.
"""

from __future__ import annotations

import logging
import threading
import time

import requests

from app.domain.external_knowledge import TfsCase
from app.engines.external_knowledge.extraction import extract_and_truncate

logger = logging.getLogger(__name__)

_API_VERSION = "5.0"
_RESOLVED_DATE_FIELDS = ("Microsoft.VSTS.Common.ResolvedDate", "Microsoft.VSTS.Common.ClosedDate")


class TfsConnectorError(Exception):
    """Raised on any network/auth/HTTP failure -- callers (see
    ``service.py``) catch this and degrade to ``available=False``
    rather than letting it propagate into a broken Analyze request.
    The message is always safe to show in the UI (no credential
    material, no raw response bodies)."""


def _escape_wiql_literal(term: str) -> str:
    return term.replace("'", "''")


def _or_clause(terms: list[str]) -> str | None:
    clauses = []
    for term in terms:
        escaped = _escape_wiql_literal(term.strip())
        if not escaped:
            continue
        clauses.append(f"[System.Title] CONTAINS WORDS '{escaped}'")
        clauses.append(f"[System.Description] CONTAINS WORDS '{escaped}'")
    return " OR ".join(clauses) if clauses else None


def _build_wiql(*, team_project: str, work_item_types: list[str], anchor_terms: list[str], supporting_terms: list[str]) -> str:
    """Anchor terms (component/technology) scope the query when
    present -- a real candidate must match at least one of them.
    Supporting terms (title keywords, entities) are used as the query
    filter only when there's no anchor to scope by; when an anchor
    exists, supporting terms are deliberately left out of the WIQL
    WHERE clause entirely and used only for local ranking afterward --
    requiring them here too would just as easily exclude a real match
    that doesn't happen to share the investigation's exact wording."""
    types_clause = ", ".join(f"'{_escape_wiql_literal(t)}'" for t in work_item_types)
    anchor_clause = _or_clause(anchor_terms)
    if anchor_clause:
        where_terms = anchor_clause
    else:
        where_terms = _or_clause(supporting_terms) or "1 = 1"
    return (
        "SELECT [System.Id] FROM WorkItems WHERE "
        f"[System.TeamProject] = '{_escape_wiql_literal(team_project)}' "
        f"AND [System.WorkItemType] IN ({types_clause}) "
        f"AND ({where_terms}) "
        "ORDER BY [System.ChangedDate] DESC"
    )


def _resolution_text(fields: dict) -> str | None:
    """Prefers Microsoft.VSTS.Common.ResolvedReason (Bug-only, already
    short and TFS's own summary of the fix) when present. Falls back
    to a capped prefix of System.History -- deliberately NOT claimed
    to be "the most recent entry": this client doesn't parse History's
    per-entry HTML structure, so no ordering assumption is made about
    which part of that field is newest."""
    resolved_reason = fields.get("Microsoft.VSTS.Common.ResolvedReason")
    if resolved_reason and str(resolved_reason).strip():
        text = extract_and_truncate(str(resolved_reason))
        if text:
            return text
    history = fields.get("System.History")
    if history and str(history).strip():
        text = extract_and_truncate(str(history))
        if text:
            return text
    return None


def _case_from_work_item(raw: dict, *, base_url: str) -> TfsCase:
    fields = raw.get("fields", {})
    work_item_id = raw["id"]
    resolved_date = None
    for field_name in _RESOLVED_DATE_FIELDS:
        if fields.get(field_name):
            resolved_date = fields[field_name]
            break
    return TfsCase(
        tfs_id=work_item_id,
        work_item_type=fields.get("System.WorkItemType", ""),
        title=fields.get("System.Title", ""),
        state=fields.get("System.State", ""),
        reason=fields.get("System.Reason"),
        area_path=fields.get("System.AreaPath", ""),
        team_project=fields.get("System.TeamProject", ""),
        changed_date=fields.get("System.ChangedDate"),
        resolved_date=resolved_date,
        description_text=extract_and_truncate(fields.get("System.Description")),
        resolution_text=_resolution_text(fields),
        root_cause=fields.get("Landis.BugRootCause"),
        severity=fields.get("Microsoft.VSTS.Common.Severity"),
        target_release=fields.get("LandisGyr.TargetRelease"),
        crm_id=fields.get("LandisGyr.CRMID"),
        url=f"{base_url}/_workitems/edit/{work_item_id}",
    )


class TfsRestConnector:
    def __init__(
        self,
        *,
        base_url: str | None,
        project: str | None,
        timeout_seconds: float,
        personal_access_token: str | None = None,
    ) -> None:
        self._pat = personal_access_token
        self._base_url = base_url.rstrip("/") if base_url else None
        self._project = project
        self._timeout = timeout_seconds
        self._session: requests.Session | None = None
        self._session_lock = threading.Lock()
        """Live-measured, real finding (not a theoretical optimization):
        a fresh ``requests.post``/``requests.get`` per call with a
        freshly-instantiated ``HttpNegotiateAuth()`` re-runs the full
        NTLM challenge/response handshake on every single request --
        against the real server, the WIQL call + one batched detail
        call (two requests) took 10-42s, highly variable, clearly
        dominated by repeated handshake cost rather than payload size.
        A persistent ``requests.Session`` with auth mounted once lets
        the underlying TCP connection (and, for NTLM, the
        connection-bound authenticated state) be reused across both
        calls within one search and across searches. Lazily created
        (not in __init__) so a misconfigured/missing
        requests-negotiate-sspi install still allows the object to be
        constructed -- the ImportError only surfaces on first real use,
        same graceful-degradation contract as everything else here."""

    def is_configured(self) -> bool:
        return bool(self._base_url and self._project)

    def _get_session(self) -> requests.Session:
        if self._session is not None:
            return self._session
        with self._session_lock:
            if self._session is None and self._pat:
                # Hosted/non-Windows deployments: a TFS/Azure DevOps personal
                # access token over Basic auth (empty username), no SSPI.
                session = requests.Session()
                session.auth = ("", self._pat)
                self._session = session
            if self._session is None:
                try:
                    from requests_negotiate_sspi import HttpNegotiateAuth
                except ImportError as exc:  # pragma: no cover -- Windows-only dependency
                    raise TfsConnectorError(
                        "TFS connector requires 'requests-negotiate-sspi' (Windows-only) -- not installed."
                    ) from exc
                session = requests.Session()
                session.auth = HttpNegotiateAuth()
                self._session = session
        return self._session

    def _project_url(self) -> str:
        assert self._base_url and self._project
        return f"{self._base_url}/{requests.utils.quote(self._project)}"

    def search_work_items(
        self, *, anchor_terms: list[str], supporting_terms: list[str], work_item_types: list[str], max_results: int
    ) -> list[TfsCase]:
        """Makes two sequential real requests (WIQL search, then a
        batched detail fetch) -- ``self._timeout`` is treated as a
        TOTAL budget across both, not a per-request allowance: the
        second call's timeout is whatever's left of the budget after
        the first, floored at a small minimum so it always gets a real
        chance to complete rather than being starved to ~0s. Real bug
        this fixes: giving each call its own full ``self._timeout``
        let worst-case total latency reach ~2x the configured value,
        which silently exceeded the caller's own wrapper timeout (see
        service.py's ``_safe_result``) and surfaced as a spurious
        "TFS lookup failed unexpectedly" even though TFS itself never
        actually timed out on its own terms."""
        if not self.is_configured():
            return []
        budget_start = time.monotonic()
        wiql = _build_wiql(
            team_project=self._project,
            work_item_types=work_item_types,
            anchor_terms=anchor_terms,
            supporting_terms=supporting_terms,
        )
        session = self._get_session()
        try:
            resp = session.post(
                f"{self._project_url()}/_apis/wit/wiql?api-version={_API_VERSION}",
                json={"query": wiql},
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise TfsConnectorError(f"TFS search request failed: {exc.__class__.__name__}") from exc

        # WIQL order is recency (ChangedDate DESC), not relevance -- TFS's
        # own CONTAINS WORDS matching is coarse (confirmed: two loosely
        # related terms returned 993 candidates against the real server),
        # so an over-fetch here gives the local ranker (ranking.py) real
        # candidates to actually choose the best few from, rather than
        # just accepting whichever happened to change most recently.
        # Capped at 3x (not higher): live-measured against the real
        # server, a 5x/50-cap overfetch (40 candidates for the default
        # max_results=8) took ~10s end to end -- too slow for an
        # interactive Analyze click. 3x/20 trades some candidate-pool
        # breadth for latency; still a real improvement over WIQL's own
        # (coarse, recency-only) ordering.
        candidate_ids = [item["id"] for item in resp.json().get("workItems", [])]
        overfetch = min(max(max_results * 3, max_results), 20)
        candidate_ids = candidate_ids[:overfetch]
        if not candidate_ids:
            return []

        remaining_budget = max(self._timeout - (time.monotonic() - budget_start), 2.0)
        try:
            detail_resp = session.get(
                f"{self._project_url()}/_apis/wit/workitems",
                params={"ids": ",".join(str(i) for i in candidate_ids), "api-version": _API_VERSION, "$expand": "all"},
                timeout=remaining_budget,
            )
            detail_resp.raise_for_status()
        except requests.RequestException as exc:
            raise TfsConnectorError(f"TFS detail fetch failed: {exc.__class__.__name__}") from exc

        return [
            _case_from_work_item(item, base_url=self._project_url())
            for item in detail_resp.json().get("value", [])
        ]

    def get_work_item(self, tfs_id: int) -> TfsCase | None:
        if not self.is_configured():
            return None
        try:
            resp = self._get_session().get(
                f"{self._project_url()}/_apis/wit/workitems/{tfs_id}",
                params={"api-version": _API_VERSION, "$expand": "all"},
                timeout=self._timeout,
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise TfsConnectorError(f"TFS work item fetch failed: {exc.__class__.__name__}") from exc
        return _case_from_work_item(resp.json(), base_url=self._project_url())

    def get_work_item_history(self, tfs_id: int) -> str | None:
        """The raw System.History field (activity/comment log),
        HTML-stripped and truncated -- distinct from
        ``TfsCase.resolution_text``, which prefers the shorter
        ResolvedReason field when present. Kept as a separate fetch
        since most callers (ranking, the summary card) only need
        ``resolution_text``, not the full activity log."""
        if not self.is_configured():
            return None
        try:
            resp = self._get_session().get(
                f"{self._project_url()}/_apis/wit/workitems/{tfs_id}",
                params={"api-version": _API_VERSION, "fields": "System.History"},
                timeout=self._timeout,
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise TfsConnectorError(f"TFS history fetch failed: {exc.__class__.__name__}") from exc
        history = resp.json().get("fields", {}).get("System.History")
        return extract_and_truncate(history) if history else None
