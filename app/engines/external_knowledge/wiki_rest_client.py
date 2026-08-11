"""Real Wiki connector -- Confluence REST API (CQL search + content
fetch).

IMPORTANT / not yet live-verified: unlike the TFS client (built against
two real work items pulled live from am.tfs.landisgyr.net this
session), this client is built against Confluence's standard,
publicly-documented REST API contract
(``/rest/api/content/search?cql=...``, ``body.storage.value`` for page
HTML, ``_links.webui`` for the page URL) -- not confirmed against
wiki.landisgyr.net's actual response shape, because that requires an
authenticated request and no Wiki credential was available at
implementation time (confirmed live: this instance's auth is NOT
NTLM/SSPI like TFS -- a dummy Basic-auth probe returned Confluence's
own ``X-Seraph-LoginReason: AUTHENTICATED_FAILED``, meaning Basic auth
with a real username + Personal Access Token is the accepted path, an
OAuth-only probe alone was rejected outright). The field names and CQL
syntax below are the real, stable Confluence contract, but this
specific instance's version/configuration has not exercised them.
Live-verify before trusting this in production -- see the setup
instructions in the delivery report for what's needed from a Wiki
administrator.
"""

from __future__ import annotations

import requests

from app.domain.external_knowledge import WikiPage
from app.engines.external_knowledge.extraction import extract_and_truncate

_API_LIMIT_PATH = "/rest/api/content/search"


class WikiConnectorError(Exception):
    """Raised on any network/auth/HTTP failure -- callers catch this
    and degrade to ``available=False``. Message is always safe to
    show in the UI."""


def _escape_cql_literal(term: str) -> str:
    return term.replace('"', '\\"')


def _text_or_clause(terms: list[str]) -> str | None:
    clauses = [f'text ~ "{_escape_cql_literal(t.strip())}"' for t in terms if t.strip()]
    return " OR ".join(clauses) if clauses else None


def _build_cql(*, anchor_terms: list[str], supporting_terms: list[str], space_key: str | None) -> str:
    """Same anchor/supporting split as the TFS WIQL builder, same
    reason -- an anchor term (component/technology), when present,
    scopes the query on its own; supporting terms (title keywords,
    entities) are the query filter only when there's no anchor."""
    anchor_clause = _text_or_clause(anchor_terms)
    query = anchor_clause or _text_or_clause(supporting_terms) or 'type = "page"'
    if space_key:
        query = f'space = "{_escape_cql_literal(space_key)}" AND ({query})'
    return query


def _page_from_result(raw: dict, *, base_url: str) -> WikiPage:
    body_html = raw.get("body", {}).get("storage", {}).get("value")
    webui_path = raw.get("_links", {}).get("webui", "")
    return WikiPage(
        page_id=str(raw.get("id", "")),
        title=raw.get("title", ""),
        space_key=raw.get("space", {}).get("key", ""),
        excerpt=extract_and_truncate(body_html),
        url=f"{base_url}{webui_path}" if webui_path else base_url,
    )


class WikiRestConnector:
    def __init__(
        self,
        *,
        base_url: str | None,
        username: str | None,
        api_token: str | None,
        space_key: str | None,
        timeout_seconds: float,
    ) -> None:
        self._base_url = base_url.rstrip("/") if base_url else None
        self._username = username
        self._api_token = api_token
        self._space_key = space_key
        self._timeout = timeout_seconds
        self._session: requests.Session | None = None
        """Persistent session, same reasoning as TfsRestConnector's
        (real, live-measured finding there: connection reuse matters).
        Basic auth doesn't pay NTLM's multi-round-trip handshake cost,
        but reusing one session still avoids re-establishing the TCP/
        TLS connection on every call."""

    def is_configured(self) -> bool:
        return bool(self._base_url and self._username and self._api_token)

    def _get_session(self) -> requests.Session:
        if self._session is None:
            assert self._username and self._api_token
            session = requests.Session()
            session.auth = (self._username, self._api_token)
            self._session = session
        return self._session

    def search(self, *, anchor_terms: list[str], supporting_terms: list[str], max_results: int) -> list[WikiPage]:
        if not self.is_configured():
            return []
        cql = _build_cql(anchor_terms=anchor_terms, supporting_terms=supporting_terms, space_key=self._space_key)
        try:
            resp = self._get_session().get(
                f"{self._base_url}{_API_LIMIT_PATH}",
                params={"cql": cql, "limit": max_results, "expand": "body.storage,space"},
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise WikiConnectorError(f"Wiki search request failed: {exc.__class__.__name__}") from exc
        results = resp.json().get("results", [])
        return [_page_from_result(item, base_url=self._base_url) for item in results]

    def get_page(self, page_id: str) -> WikiPage | None:
        if not self.is_configured():
            return None
        try:
            resp = self._get_session().get(
                f"{self._base_url}/rest/api/content/{page_id}",
                params={"expand": "body.storage,space"},
                timeout=self._timeout,
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise WikiConnectorError(f"Wiki page fetch failed: {exc.__class__.__name__}") from exc
        return _page_from_result(resp.json(), base_url=self._base_url)
