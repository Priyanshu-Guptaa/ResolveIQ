"""Tests for Settings defaults (Phase 43 chat model default lock)."""

from __future__ import annotations

from app.config import Settings


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
