"""``OllamaProvider`` -- the first concrete ``LLMProvider``
implementation, talking to a locally running Ollama instance over its
native REST API (``/api/chat``).

Owns every Ollama-specific detail (base URL, model tag, request/
response JSON shape, timeout, connection handling) -- nothing above
this class ever sees an Ollama request or response directly, matching
how ``TfsRestConnector``/``WikiRestConnector`` own every TFS/
Confluence-specific detail behind ``TfsConnector``/``WikiConnector``.

Uses ``httpx`` (already a pinned dependency, 0.28.1) directly -- no new
package, same "hand-rolled REST client over an existing HTTP library"
idiom every other external connector in this codebase already follows.
Holds one reusable ``httpx.Client`` rather than issuing a fresh request
per call -- the same connection-reuse reasoning already documented for
``TfsRestConnector``'s persistent, auth-mounted ``requests.Session``
(a fresh connection per call was live-measured there to add real,
avoidable latency).

No streaming in Phase 1 -- ``stream: false`` is sent explicitly on
every request.
"""

from __future__ import annotations

import logging

import httpx

from app.engines.llm.provider import LLMProviderError

logger = logging.getLogger(__name__)


class OllamaProvider:
    """See module docstring. Implements the ``LLMProvider`` Protocol
    (``app.engines.llm.provider``) -- structurally, not by
    inheritance, same as every other Protocol-satisfying class in this
    codebase."""

    def __init__(self, *, base_url: str, model: str, timeout_seconds: float) -> None:
        self._base_url = base_url.rstrip("/") if base_url else ""
        self._model = model
        self._client = httpx.Client(timeout=timeout_seconds)

    def is_configured(self) -> bool:
        return bool(self._base_url) and bool(self._model)

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        if not self.is_configured():
            raise LLMProviderError(
                "Ollama provider is not configured (missing base URL or model) -- "
                "set RESOLVEIQ_OLLAMA_BASE_URL / RESOLVEIQ_OLLAMA_MODEL."
            )

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.post(
                f"{self._base_url}/api/chat",
                json={"model": self._model, "messages": messages, "stream": False},
            )
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise LLMProviderError(
                f"Could not reach Ollama at {self._base_url} -- is it running?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMProviderError(f"Ollama request to {self._base_url} timed out.") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMProviderError(
                f"Ollama returned HTTP {exc.response.status_code} for model '{self._model}'."
            ) from exc
        except httpx.HTTPError as exc:
            # Catch-all for any other httpx-raised transport failure (e.g. a
            # malformed URL, a protocol error) -- never let a raw httpx
            # exception type escape this provider, per the Protocol's own
            # contract.
            raise LLMProviderError(f"Ollama request failed: {exc.__class__.__name__}") from exc

        try:
            body = response.json()
        except ValueError as exc:
            raise LLMProviderError("Ollama returned a response that was not valid JSON.") from exc

        try:
            text = body["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMProviderError(
                "Ollama's response did not contain the expected message.content field."
            ) from exc

        if not isinstance(text, str) or not text.strip():
            raise LLMProviderError("Ollama returned an empty response.")

        return text
