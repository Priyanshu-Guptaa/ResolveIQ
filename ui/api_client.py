"""Shared HTTP client for every Streamlit page.

Extracted from Sprint 1's single-page app so the multi-page shell doesn't
duplicate request/error-handling logic across 8 pages. Every page imports
from here rather than calling ``requests`` directly.

Caching (Phase 1.5): ``st.cache_data`` wraps the four endpoints RFC rev 3
Phase 1.5 named as safe candidates -- Dashboard, Settings, Knowledge
search, Query Library. Nothing investigation-specific is cached (an open
Workspace must always show the true current state). See each cached
function's docstring for its TTL and why.

Cache scope: ``st.cache_data`` caches at the Streamlit *process* level,
shared across every browser session hitting this server -- there is no
per-user cache partitioning, consistent with the rest of Sprint 2's
single-tenant assumption (no auth yet).
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("RESOLVEIQ_API_URL", "http://localhost:8000")


def api_get(path: str, params: dict | None = None) -> dict | list | None:
    try:
        response = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        detail = ""
        if getattr(exc, "response", None) is not None:
            detail = f" -- {exc.response.text}"
        st.error(f"API request failed: GET {path} -- {exc}{detail}")
        return None


def api_post(path: str, json_body: dict | None = None, files=None, *, timeout: int = 60) -> dict | list | None:
    """``timeout`` defaults to 60s for ordinary JSON calls -- callers that
    upload files (which synchronously run the Evidence Ingestion
    Pipeline / Knowledge Management's parser server-side, potentially a
    zip fanning into many files) should pass a longer one explicitly.
    60s was found too short for a real 1.5MB zip upload -- see the two
    ``files=`` call sites, which now pass 300s."""
    try:
        response = requests.post(f"{API_BASE_URL}{path}", json=json_body, files=files, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        detail = ""
        if getattr(exc, "response", None) is not None:
            detail = f" -- {exc.response.text}"
        st.error(f"API request failed: POST {path} -- {exc}{detail}")
        return None


def api_patch(path: str, json_body: dict | None = None) -> dict | list | None:
    try:
        response = requests.patch(f"{API_BASE_URL}{path}", json=json_body, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        detail = ""
        if getattr(exc, "response", None) is not None:
            detail = f" -- {exc.response.text}"
        st.error(f"API request failed: PATCH {path} -- {exc}{detail}")
        return None


def api_delete(path: str) -> bool:
    """Returns True on success -- most DELETE endpoints (e.g. removing a
    relationship, Sprint 3 Phase 3.3) return 204 with no body, so there's
    nothing to hand back the way api_get/api_post/api_patch do."""
    try:
        response = requests.delete(f"{API_BASE_URL}{path}", timeout=30)
        response.raise_for_status()
        return True
    except requests.RequestException as exc:
        detail = ""
        if getattr(exc, "response", None) is not None:
            detail = f" -- {exc.response.text}"
        st.error(f"API request failed: DELETE {path} -- {exc}{detail}")
        return False


def api_available() -> bool:
    """Cheap liveness check -- pages use this to show one clear banner
    instead of a wall of individual request errors when the API is down."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return response.ok
    except requests.RequestException:
        return False


def ensure_api_available() -> None:
    """Call once at the top of a page. Stops page execution with one clear
    message if the API is unreachable -- replaces the 4-line
    check/error/stop block that was copy-pasted across all 8 pages."""
    if not api_available():
        st.error(
            "Can't reach the ResolveIQ API. Start it with "
            "`uvicorn app.api.main:app --reload` and reload this page."
        )
        st.stop()


# --- Cached reads (Phase 1.5 caching strategy) ------------------------------


@st.cache_data(ttl=15, show_spinner=False)
def get_dashboard_cached() -> dict | None:
    """15s TTL: Dashboard changes often (every evidence upload, every
    status change), so this is the shortest TTL of the four. Paired with
    an explicit Refresh button on the Dashboard page (calls
    ``get_dashboard_cached.clear()``) for when 15s isn't fast enough --
    e.g. right after uploading evidence in Workspace and clicking back."""
    return api_get("/dashboard")


@st.cache_data(ttl=300, show_spinner=False)
def get_settings_cached() -> dict | None:
    """5min TTL: nothing in Sprint 2 changes Settings at runtime (it's a
    read-only view), so this is effectively "cache until manually
    cleared," bounded generously in case that changes later."""
    return api_get("/settings")


@st.cache_data(ttl=300, show_spinner=False)
def get_query_library_cached() -> list | None:
    """5min TTL: the Query Library is a static code-defined list in
    Sprint 2 (see app/domain/sql_studio.py) -- it cannot change without a
    deploy, which restarts the process and clears this cache anyway."""
    return api_get("/sql/library")


@st.cache_data(ttl=30, show_spinner=False)
def search_knowledge_cached(q: str, collection: str | None = None, top_k: int = 10) -> list | None:
    """30s TTL, keyed on (q, collection, top_k): protects against
    re-running the same search repeatedly within a session. Longer than
    Dashboard's TTL because there's no live knowledge-import UI yet in
    Sprint 2 -- the knowledge base only changes via a manual reseed."""
    return api_get("/knowledge/search", params={"q": q, "collection": collection, "top_k": top_k})


@st.cache_data(ttl=15, show_spinner=False)
def list_investigations_cached() -> list | None:
    """Same 15s TTL as Dashboard, for the same reason -- this backs the
    shared investigation picker (components/investigation_picker.py),
    which every investigation-scoped page renders. Not in the RFC's
    explicit "safe candidates" list, but it's the same shape (lightweight
    summary list, not investigation-specific state) so the same reasoning
    applies; the actual Workspace/Log Intelligence/AI Assistant *content*
    for a selected investigation is never cached."""
    return api_get("/investigations")


@st.cache_data(ttl=300, show_spinner=False)
def get_component_profiles_cached() -> list | None:
    """5min TTL, same reasoning as the Query Library: the Component
    Registry is a static JSON seed file in this phase -- it can't change
    without a deploy, which restarts the process and clears this cache."""
    return api_get("/product-intelligence/components")


@st.cache_data(show_spinner=False)
def get_evidence_preview_cached(investigation_id: str, evidence_id: str, max_chars: int = 2000) -> dict | None:
    """No TTL (cached until the process restarts or is cleared) --
    evidence is immutable once created, so a given (evidence_id,
    max_chars) pair's preview never changes. Investigation loading
    redesign: the Workspace's Overview/Notes tabs call this on every
    Streamlit rerun (st.tabs executes every tab body every rerun,
    regardless of which tab is visually active); without caching, an
    unrelated click anywhere on the page would re-fetch every note's
    preview needlessly."""
    return api_get(f"/investigations/{investigation_id}/evidence/{evidence_id}/preview", params={"max_chars": max_chars})
