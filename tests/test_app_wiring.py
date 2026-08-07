"""Smoke test that the assembled FastAPI app actually imports and every
router registers without error.

Added Sprint 3, Phase 3.3 after finding a real bug this way: a DELETE
endpoint returning ``-> None`` with ``status_code=204`` fails FastAPI's
own route-construction assertion ("Status code 204 must not have a
response body") -- an error that only happens at import time, when
``APIRouter.delete(...)`` actually runs. No other test in this suite
imports ``app.api.main`` (every other test exercises an engine or
repository directly against a real or fake dependency), so this whole
class of "the app doesn't actually start" bug had no regression guard
until now. Cheap to run, catches a real class of mistake.
"""

from __future__ import annotations


def test_app_imports_and_registers_every_admin_router():
    from app.api.main import app

    paths = {getattr(route, "path", "") for route in app.routes}

    # One path per admin router, enough to prove each actually mounted
    # (not that every single endpoint exists -- that's the router-level
    # tests' job where they exist, and live verification otherwise).
    assert "/admin/knowledge/dashboard" in paths
    assert "/admin/relationships" in paths
    assert "/admin/relationships/health" in paths
    assert "/admin/relationships/playbooks" in paths

    # And the pre-existing, non-admin routers are still there too.
    assert "/health" in paths
    assert "/investigations" in paths
    assert "/dashboard" in paths
    assert "/product-intelligence/components" in paths
    assert "/sql/library" in paths


def test_every_route_has_a_valid_response_configuration():
    """Importing app.api.main already exercises every
    ``@router.<method>(...)`` decorator call -- if any route's
    status_code/response_model combination were invalid (like the 204
    bug above), the import in the previous test would already have
    raised. This test exists to name that guarantee explicitly, so a
    future reader knows why "just importing the app" is the assertion."""
    import app.api.main  # noqa: F401
