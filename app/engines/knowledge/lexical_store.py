"""``LexicalKnowledgeStore`` -- a BM25-backed ``KnowledgeStore`` (Chat
Assistant Phase 2 -- Hybrid Retrieval + RRF foundation).

Satisfies the exact same ``KnowledgeStore`` Protocol
(``app.engines.knowledge.knowledge_store``) ``ChromaKnowledgeStore``
already does -- structurally, no modification to that Protocol.
Standalone (never directly wired into ``KnowledgeEngine`` on its own in
this phase); consumed only by ``HybridKnowledgeStore``
(``app.engines.knowledge.hybrid_store``), which composes this alongside
the existing, unchanged ``ChromaKnowledgeStore``.

Uses ``rank_bm25.BM25Okapi`` (one instance per ``KnowledgeCollection``,
never mixed across collections, mirroring
``ChromaKnowledgeStore._collection()``'s own per-collection
separation). Corpus is held in memory (``dict[KnowledgeCollection,
dict[record_id, _LexicalDocument]]``) and each collection's BM25 index
is rebuilt lazily -- only on the next ``query()`` after that
collection was marked dirty by an ``upsert()``/``delete()`` -- the same
"fine at this corpus scale (hundreds of records), revisit if it grows
large" precedent ``ChromaKnowledgeStore.list_recent()``'s own docstring
already establishes for exactly this scale.

Tokenization is deliberately simple: lowercase, alphanumeric-run
extraction (``re.findall(r"[a-z0-9]+", text.lower())``) -- the same
alnum-run-extraction philosophy already established in
``app.engines.shared.text_matching``, not a new idiom. No stemming, no
lemmatization, no query expansion (out of Phase 2 scope by design).

Score contract: BM25's own raw scores are unbounded, non-negative
floats with no natural [0,1] ceiling -- min-max normalized to [0,1]
here, entirely internally, only so this class independently satisfies
``KnowledgeMatch.score``'s Pydantic constraint if ever queried
standalone. This normalized value is purely informational
(``metadata["lexical_score"]`` once surfaced by ``HybridKnowledgeStore``)
and is NEVER treated as a substitute for real vector/cosine similarity
by any caller -- see ``hybrid_store.py``'s own module docstring for the
full score contract this phase's design requires.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from app.domain.enums import KnowledgeCollection
from app.domain.recommendation import KnowledgeMatch

_K1 = 1.5
_B = 0.75
"""Okapi BM25's own literature-standard defaults -- not guessed, kept
as constructor-level overrides (never hardcoded with no escape hatch),
but deliberately not exposed as Settings in this phase (no real tuning
need has arisen yet -- see this phase's approved design)."""

_MAX_SNIPPET_CHARS = 400
"""Matches ChromaKnowledgeStore.query()'s own display-snippet cap
exactly, so a record's snippet reads identically regardless of which
backing store happens to represent it in a fused result."""

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-run extraction -- case-insensitive and
    punctuation-insensitive by construction (only alnum runs are ever
    captured). Empty/whitespace text yields an empty token list, never
    an error."""
    return _TOKEN_RE.findall(text.lower())


class _LexicalDocument:
    """One record's full, untruncated searchable text plus its display
    metadata -- never the same as a 400-char display snippet."""

    __slots__ = ("title", "text", "metadata")

    def __init__(self, title: str, text: str, metadata: dict) -> None:
        self.title = title
        self.text = text
        self.metadata = metadata


class LexicalKnowledgeStore:
    """See module docstring. Implements the ``KnowledgeStore`` Protocol
    structurally, not by inheritance -- same convention every other
    Protocol-satisfying class in this codebase already follows."""

    def __init__(self, *, k1: float = _K1, b: float = _B) -> None:
        self._k1 = k1
        self._b = b
        self._documents: dict[KnowledgeCollection, dict[str, _LexicalDocument]] = {}
        self._index_cache: dict[KnowledgeCollection, BM25Okapi] = {}
        self._dirty: set[KnowledgeCollection] = set()

    # --- KnowledgeStore Protocol -------------------------------------------

    def upsert(
        self,
        collection: KnowledgeCollection,
        record_id: str,
        text: str,
        title: str,
        metadata: dict,
    ) -> None:
        """Never truncates ``text`` -- the full, original searchable
        string (the same one passed to the vector store) is retained
        for indexing. An empty/whitespace ``text`` still registers the
        record's identity (so count()/delete() stay correct for it),
        it just contributes no real terms to the BM25 index."""
        self._documents.setdefault(collection, {})[record_id] = _LexicalDocument(title, text, dict(metadata))
        self._dirty.add(collection)

    def delete(self, collection: KnowledgeCollection, record_id: str) -> None:
        """No-op if the id was never indexed -- same contract as
        ChromaKnowledgeStore.delete()."""
        documents = self._documents.get(collection)
        if documents is not None and record_id in documents:
            del documents[record_id]
            self._dirty.add(collection)

    def count(self, collection: KnowledgeCollection) -> int:
        return len(self._documents.get(collection, {}))

    def query(self, collection: KnowledgeCollection, text: str, top_k: int = 5) -> list[KnowledgeMatch]:
        if not text.strip():
            return []
        documents = self._documents.get(collection)
        if not documents:
            return []

        index = self._index_for(collection)
        record_ids = list(documents.keys())
        query_tokens = _tokenize(text)
        if not query_tokens:
            return []

        raw_scores = index.get_scores(query_tokens)
        query_term_set = set(query_tokens)
        document_tokens = [_tokenize(documents[record_id].text) for record_id in record_ids]
        normalized = _normalize_scores(raw_scores, query_term_set, document_tokens)

        scored: list[tuple[float, str]] = list(zip(normalized, record_ids))
        # Deterministic: ties broken by record_id ascending, never by
        # whatever order dict.keys()/get_scores() happened to iterate in.
        scored.sort(key=lambda pair: (-pair[0], pair[1]))

        matches: list[KnowledgeMatch] = []
        for score, record_id in scored[:top_k]:
            document = documents[record_id]
            matches.append(
                KnowledgeMatch(
                    collection=collection,
                    record_id=record_id,
                    title=document.title,
                    snippet=document.text[:_MAX_SNIPPET_CHARS],
                    score=score,
                    metadata=dict(document.metadata),
                )
            )
        return matches

    def list_recent(self, collection: KnowledgeCollection, limit: int = 5) -> list[KnowledgeMatch]:
        """A pure lexical index has no recency concept of its own --
        honestly returns [] rather than guessing (e.g. from arbitrary
        dict order). Never actually called through the hybrid
        composition -- HybridKnowledgeStore.list_recent() delegates to
        the vector store only (see that module)."""
        return []

    # --- Internal ------------------------------------------------------

    def _index_for(self, collection: KnowledgeCollection) -> BM25Okapi:
        """Rebuilds lazily -- only when this collection was marked
        dirty by an upsert()/delete() since the last rebuild, or has
        never been built at all."""
        if collection not in self._index_cache or collection in self._dirty:
            documents = self._documents.get(collection, {})
            tokenized_corpus = [_tokenize(doc.text) for doc in documents.values()]
            self._index_cache[collection] = BM25Okapi(tokenized_corpus, k1=self._k1, b=self._b)
            self._dirty.discard(collection)
        return self._index_cache[collection]


def _normalize_scores(raw_scores, query_term_set: set[str], document_tokens: list[list[str]]) -> list[float]:
    """Min-max normalizes BM25's own raw scores to [0,1] -- purely so
    this class independently satisfies KnowledgeMatch.score's Pydantic
    constraint if queried standalone.

    IMPORTANT, verified empirically during this phase's own test-writing
    (not assumed): BM25's classic Robertson-Sparck-Jones IDF term,
    ``log((N - n + 0.5) / (n + 0.5))``, is NEGATIVE whenever a term
    appears in more than half the corpus -- confirmed directly against
    rank_bm25.BM25Okapi with a real single-document corpus where the
    query term IS present: raw score -0.549, not a small positive
    number. This means a raw score's sign alone cannot be trusted to
    mean "matched" vs. "didn't match," especially in a small/degenerate
    corpus (a single-document corpus is the extreme case, but any term
    common enough to appear in over half of even a larger corpus hits
    the same negative-IDF behavior).

    All-identical raw scores (min == max -- e.g. a single-document
    corpus, or every remaining document sharing the exact same overlap)
    can't be range-normalized by division, and picking a fallback value
    by the raw score's sign would be wrong per the paragraph above --
    so this falls back to a direct token-overlap check instead: if a
    document shares at least one real token with the query, it
    genuinely matched (real signal) and normalizes to 1.0 (max
    relevance within this degenerate, all-tied batch); if it shares
    none, there is no real signal and it normalizes to 0.0."""
    scores = list(raw_scores)
    if not scores:
        return []
    lowest = min(scores)
    highest = max(scores)
    spread = highest - lowest
    if spread == 0:
        return [1.0 if (query_term_set & set(tokens)) else 0.0 for tokens in document_tokens]
    return [(value - lowest) / spread for value in scores]
