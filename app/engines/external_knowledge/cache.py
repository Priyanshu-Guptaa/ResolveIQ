"""Bounded, in-memory, TTL-only cache for external-knowledge query
results -- explicitly not a data store: nothing here is ever written
to SQLite or ChromaDB, and everything is lost on process restart. See
``app/domain/external_knowledge.py``'s module docstring for why TFS/
Wiki content is never persisted.

One instance is shared by both connectors (keyed by a caller-supplied
cache key that already includes the source), not one cache per
connector -- simpler lifetime management, same bound either way.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Generic, TypeVar

T = TypeVar("T")


class TtlCache(Generic[T]):
    def __init__(self, *, max_entries: int = 50, ttl_seconds: float = 900) -> None:
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._entries: "OrderedDict[str, tuple[float, T]]" = OrderedDict()

    def get(self, key: str) -> T | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            stored_at, value = entry
            if time.monotonic() - stored_at > self._ttl_seconds:
                del self._entries[key]
                return None
            # Move-to-end marks this key as recently used for the
            # bounded-size eviction below (plain LRU discipline).
            self._entries.move_to_end(key)
            return value

    def set(self, key: str, value: T) -> None:
        with self._lock:
            self._entries[key] = (time.monotonic(), value)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
