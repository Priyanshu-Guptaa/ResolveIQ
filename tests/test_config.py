"""Tests for Settings validators (Phase 2E — RRF / Identifier Protection
Compatibility Review, Option A safety guard).

Pure unit tests -- no database, no HTTP, no unittest.mock. Uses
``caplog``, the same idiom already established in
tests/test_hybrid_knowledge_store.py's failure-degradation tests.
"""

from __future__ import annotations

import logging

from app.config import Settings


def _settings(*, identifier_protection_enabled: bool, hybrid_fusion_mode: str) -> Settings:
    return Settings(
        identifier_protection_enabled=identifier_protection_enabled,
        hybrid_fusion_mode=hybrid_fusion_mode,
    )


# --- Case A: identifier protection off, RRF -- no warning -------------------


def test_no_warning_when_identifier_protection_disabled_under_rrf(caplog):
    caplog.set_level(logging.WARNING, logger="app.config")
    _settings(identifier_protection_enabled=False, hybrid_fusion_mode="rrf")
    assert "identifier_protection_enabled=True with hybrid_fusion_mode='rrf'" not in caplog.text


# --- Case B: identifier protection on, score fusion -- no warning -----------


def test_no_warning_when_identifier_protection_enabled_under_score_fusion(caplog):
    caplog.set_level(logging.WARNING, logger="app.config")
    _settings(identifier_protection_enabled=True, hybrid_fusion_mode="score")
    assert "identifier_protection_enabled=True with hybrid_fusion_mode='rrf'" not in caplog.text


# --- Case C: identifier protection off, score fusion -- no warning ----------


def test_no_warning_when_identifier_protection_disabled_under_score_fusion(caplog):
    caplog.set_level(logging.WARNING, logger="app.config")
    _settings(identifier_protection_enabled=False, hybrid_fusion_mode="score")
    assert "identifier_protection_enabled=True with hybrid_fusion_mode='rrf'" not in caplog.text


# --- Case D: identifier protection on, RRF -- warning emitted ---------------


def test_warning_emitted_when_identifier_protection_enabled_under_rrf(caplog):
    caplog.set_level(logging.WARNING, logger="app.config")
    _settings(identifier_protection_enabled=True, hybrid_fusion_mode="rrf")
    assert "identifier_protection_enabled=True with hybrid_fusion_mode='rrf'" in caplog.text


# --- Critical no-behavior-change guarantees ----------------------------------


def test_case_d_construction_succeeds_without_exception():
    settings = _settings(identifier_protection_enabled=True, hybrid_fusion_mode="rrf")
    assert settings is not None


def test_case_d_settings_retain_exactly_the_requested_values():
    settings = _settings(identifier_protection_enabled=True, hybrid_fusion_mode="rrf")
    assert settings.identifier_protection_enabled is True
    assert settings.hybrid_fusion_mode == "rrf"


def test_warning_never_modifies_either_setting_for_any_combination():
    for identifier_protection_enabled in (True, False):
        for hybrid_fusion_mode in ("rrf", "score"):
            settings = _settings(
                identifier_protection_enabled=identifier_protection_enabled, hybrid_fusion_mode=hybrid_fusion_mode
            )
            assert settings.identifier_protection_enabled == identifier_protection_enabled
            assert settings.hybrid_fusion_mode == hybrid_fusion_mode


def test_default_settings_construct_without_warning(caplog):
    # rrf_enabled/identifier_protection_enabled/hybrid_fusion_mode all
    # at their real shipped defaults -- must never warn.
    caplog.set_level(logging.WARNING, logger="app.config")
    Settings()
    assert "identifier_protection_enabled=True with hybrid_fusion_mode='rrf'" not in caplog.text
