"""LLM provider abstraction (Chat Assistant Phase 1 -- Qwen 4B/Ollama
integration).

``LLMProvider`` is the only thing the rest of the application (the Chat
Orchestrator, eventually anything else that wants generated text) ever
depends on -- never a concrete implementation. Same "one new class
here, no changes anywhere else" pattern already established by every
other swappable external dependency in this codebase
(``EmbeddingProvider``, ``KnowledgeStore``, ``TfsConnector``,
``WikiConnector``): ``OllamaProvider`` (``app.engines.llm.ollama_provider``)
is the first, and so far only, concrete implementation; a future
``OpenAIProvider``/``AnthropicProvider``/another local provider is a
second class implementing this same, unchanged Protocol.

Deliberately minimal: ``generate()`` takes only a prompt and an
optional system prompt. Generation knobs that have exactly one caller
and one fixed need today (temperature, max tokens, timeout, model
name) are NOT part of this contract -- they are concrete-provider
constructor/configuration concerns (see ``OllamaProvider``), not
something the orchestration layer should know exists. Widening this
Protocol later is a compatible change if a real second need for it
ever appears; nothing about it is guessed at now.
"""

from __future__ import annotations

from typing import Protocol


class LLMProviderError(Exception):
    """Raised by any ``LLMProvider`` implementation on a failure to
    generate text -- missing configuration, connection failure,
    timeout, an HTTP-level failure, or a malformed/empty response.
    Callers (``ChatOrchestrator``) catch this and fall back to the
    existing, unchanged, deterministic ``_compose_answer()`` -- the
    same graceful-degradation contract ``TfsConnectorError``/
    ``WikiConnectorError`` already establish for external dependencies
    in this codebase. The message is always safe to log/show (no
    credential material, no raw response bodies beyond a short,
    truncated excerpt)."""


class LLMProvider(Protocol):
    """Anything that can turn a prompt into generated text.
    Provider-agnostic by design -- see module docstring."""

    def is_configured(self) -> bool:
        """True when this provider has everything it needs to attempt
        a real call (e.g. a base URL and model name) -- never performs
        network I/O itself. Mirrors ``TfsConnector.is_configured()``/
        ``WikiConnector.is_configured()`` exactly."""
        ...

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        """Returns the model's generated text. Raises
        ``LLMProviderError`` on any failure -- never returns a
        partial/placeholder string, never lets a raw/unrelated
        exception type escape the implementation."""
        ...
