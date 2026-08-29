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


# --- Chat model default lock (Phase 43) --------------------------------------
# qwen2.5:3b has been the deliberate, documented Settings.ollama_model default
# since Chat Assistant Phase 21 (see the field's own docstring for the real,
# cited 40-call/GEN3 validation history behind the switch from qwen3:4b) --
# no prior test asserted this directly. llm_enabled/llm_async_enabled must
# both remain False by default so a fresh checkout with no Ollama installed
# runs with zero behavior change (ChatOrchestrator's existing deterministic
# _compose_answer() path, unconditionally) -- this is the one guarantee this
# phase exists to lock down and regression-test, not to newly invent.
# Never touches a live Ollama server and never invokes a real LLM.


def test_default_chat_model_is_qwen2_5_3b():
    assert Settings().ollama_model == "qwen2.5:3b"


def test_llm_disabled_by_default():
    assert Settings().llm_enabled is False


def test_llm_async_disabled_by_default():
    assert Settings().llm_async_enabled is False
