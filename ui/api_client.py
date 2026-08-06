"""Shared HTTP client for every Streamlit page.

Extracted from Sprint 1's single-page app so the multi-page shell (RFC
rev 3) doesn't duplicate request/error-handling logic across 8 pages.
Every page imports from here rather than calling ``requests`` directly.
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


def api_post(path: str, json_body: dict | None = None, files=None) -> dict | list | None:
    try:
        response = requests.post(f"{API_BASE_URL}{path}", json=json_body, files=files, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        detail = ""
        if getattr(exc, "response", None) is not None:
            detail = f" -- {exc.response.text}"
        st.error(f"API request failed: POST {path} -- {exc}{detail}")
        return None


def api_available() -> bool:
    """Cheap liveness check -- pages use this to show one clear banner
    instead of a wall of individual request errors when the API is down."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return response.ok
    except requests.RequestException:
        return False
