"""Tests for OllamaProvider (Chat Assistant Phase 1 -- Qwen 4B/Ollama
integration).

Every HTTP interaction is mocked via httpx.MockTransport -- no real
Ollama server, no port 11434, no internet, no database of any kind.
Ollama is confirmed not installed on this machine (see Step 1 of the
Phase 1 implementation report); these tests are the only validation
performed for this phase.
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.config import Settings
from app.engines.llm.ollama_provider import OllamaProvider
from app.engines.llm.provider import LLMProviderError


def _provider(transport: httpx.MockTransport, **overrides) -> OllamaProvider:
    provider = OllamaProvider(
        base_url=overrides.pop("base_url", "http://localhost:11434"),
        model=overrides.pop("model", "qwen3:4b"),
        timeout_seconds=overrides.pop("timeout_seconds", 5.0),
        num_predict=overrides.pop("num_predict", 0),
    )
    # Swap in a mocked transport on the same reusable httpx.Client the
    # provider already constructed -- keeps __init__'s real client-reuse
    # behavior under test rather than replacing the client entirely.
    provider._client = httpx.Client(transport=transport, timeout=provider._client.timeout)
    return provider


# --- A. Not configured -----------------------------------------------------


def test_not_configured_when_base_url_missing():
    provider = OllamaProvider(base_url="", model="qwen3:4b", timeout_seconds=5.0)
    assert provider.is_configured() is False


def test_not_configured_when_model_missing():
    provider = OllamaProvider(base_url="http://localhost:11434", model="", timeout_seconds=5.0)
    assert provider.is_configured() is False


def test_configured_when_both_present():
    provider = OllamaProvider(base_url="http://localhost:11434", model="qwen3:4b", timeout_seconds=5.0)
    assert provider.is_configured() is True


def test_generate_raises_when_not_configured():
    provider = OllamaProvider(base_url="", model="", timeout_seconds=5.0)
    with pytest.raises(LLMProviderError):
        provider.generate("hello")


# --- B. Correct request construction ----------------------------------------


def test_request_construction_with_system_prompt():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "the answer"}})

    provider = _provider(httpx.MockTransport(handler))
    result = provider.generate("What is the root cause?", system_prompt="You are grounded.")

    assert result == "the answer"
    assert captured["url"] == "http://localhost:11434/api/chat"
    body = captured["body"]
    assert body["model"] == "qwen3:4b"
    assert body["stream"] is False
    assert body["messages"] == [
        {"role": "system", "content": "You are grounded."},
        {"role": "user", "content": "What is the root cause?"},
    ]


def test_request_construction_without_system_prompt():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "ok"}})

    provider = _provider(httpx.MockTransport(handler))
    provider.generate("no system prompt here")

    assert captured["body"]["messages"] == [{"role": "user", "content": "no system prompt here"}]


# --- C. Correct response parsing --------------------------------------------


def test_successful_response_parsing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "Root cause: collector lost route."}})

    provider = _provider(httpx.MockTransport(handler))
    assert provider.generate("question") == "Root cause: collector lost route."


# --- D. Connection failure ---------------------------------------------------


def test_connection_failure_raises_llm_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = _provider(httpx.MockTransport(handler))
    with pytest.raises(LLMProviderError, match="Could not reach Ollama"):
        provider.generate("question")


# --- E. Timeout ----------------------------------------------------------------


def test_timeout_raises_llm_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    provider = _provider(httpx.MockTransport(handler))
    with pytest.raises(LLMProviderError, match="timed out"):
        provider.generate("question")


# --- F. HTTP error ---------------------------------------------------------------


def test_http_error_raises_llm_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal server error")

    provider = _provider(httpx.MockTransport(handler))
    with pytest.raises(LLMProviderError, match="HTTP 500"):
        provider.generate("question")


# --- G. Malformed response ----------------------------------------------------


def test_malformed_json_raises_llm_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json at all")

    provider = _provider(httpx.MockTransport(handler))
    with pytest.raises(LLMProviderError, match="not valid JSON"):
        provider.generate("question")


# --- H. Missing message.content -------------------------------------------------


def test_missing_message_content_raises_llm_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    provider = _provider(httpx.MockTransport(handler))
    with pytest.raises(LLMProviderError, match="message.content"):
        provider.generate("question")


def test_message_without_content_key_raises_llm_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"role": "assistant"}})

    provider = _provider(httpx.MockTransport(handler))
    with pytest.raises(LLMProviderError, match="message.content"):
        provider.generate("question")


# --- I. Empty response --------------------------------------------------------


def test_empty_response_raises_llm_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": ""}})

    provider = _provider(httpx.MockTransport(handler))
    with pytest.raises(LLMProviderError, match="empty response"):
        provider.generate("question")


def test_whitespace_only_response_raises_llm_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "   \n  "}})

    provider = _provider(httpx.MockTransport(handler))
    with pytest.raises(LLMProviderError, match="empty response"):
        provider.generate("question")


# --- J. num_predict cap (Chat Assistant Phase 3E) -----------------------------
#
# Phase 3D real-Ollama investigation proved num_predict bounds Qwen's TOTAL
# generated tokens (reasoning included) -- see ollama_provider.py's own
# "num_predict CAP" docstring section. These tests cover the request-
# construction contract only; the empty-content-after-truncation failure
# mode is already covered by test_empty_response_raises_llm_provider_error
# above -- no new error-handling code exists to test separately.


def test_num_predict_positive_value_is_sent_in_options():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "answer"}})

    provider = _provider(httpx.MockTransport(handler), num_predict=2048)
    provider.generate("question")

    assert captured["body"]["options"]["num_predict"] == 2048


def test_num_predict_zero_sends_no_options_key():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "answer"}})

    provider = _provider(httpx.MockTransport(handler), num_predict=0)
    provider.generate("question")

    assert "options" not in captured["body"]


def test_num_predict_arbitrary_positive_value_transmitted_exactly():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "answer"}})

    provider = _provider(httpx.MockTransport(handler), num_predict=512)
    provider.generate("question")

    assert captured["body"]["options"]["num_predict"] == 512


def test_num_predict_default_constructor_value_sends_no_options_key():
    # OllamaProvider's own default (num_predict not passed at all) must
    # remain the same "no cap" behavior as explicit 0 -- byte-identical
    # to every pre-Phase-3E caller/test that never mentions this parameter.
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "answer"}})

    provider = OllamaProvider(base_url="http://localhost:11434", model="qwen3:4b", timeout_seconds=5.0)
    provider._client = httpx.Client(transport=httpx.MockTransport(handler), timeout=provider._client.timeout)
    provider.generate("question")

    assert "options" not in captured["body"]


def test_num_predict_cap_truncation_still_raises_via_existing_empty_response_check():
    # Simulates exactly what Phase 3D's real num_predict=50 probe
    # observed: done_reason="length" with an empty message.content.
    # No new error-handling code exists for this -- it must be caught by
    # the same, unmodified "Ollama returned an empty response" check
    # every other empty-content case already goes through.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": ""}, "done_reason": "length"})

    provider = _provider(httpx.MockTransport(handler), num_predict=50)
    with pytest.raises(LLMProviderError, match="empty response"):
        provider.generate("question")


def test_settings_negative_ollama_num_predict_raises_validation_error():
    # A Settings-level test, deliberately placed here (not a new file) --
    # Phase 3E's approved scope is exactly the four files this change
    # touches; ollama_num_predict's own fail-fast validator lives in
    # app/config.py, next to _validate_hybrid_fusion_weights's identical
    # "prefer failing fast" idiom.
    import pydantic

    with pytest.raises(pydantic.ValidationError, match="ollama_num_predict"):
        Settings(ollama_num_predict=-1)
