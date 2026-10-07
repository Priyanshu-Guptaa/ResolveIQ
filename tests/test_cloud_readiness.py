"""Hosted-deployment seams: external DB URL, Chroma server mode, blob
storage, TFS PAT auth, slim Chroma documents, and the slim-index script.

Everything runs locally against temp dirs / fakes -- nothing here talks to
PostgreSQL, S3 or a real TFS server.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.domain.enums import KnowledgeCollection
from app.engines.external_knowledge.tfs_rest_client import TfsRestConnector
from app.engines.knowledge.knowledge_store import SNIPPET_CHARS, ChromaKnowledgeStore
from app.infrastructure.db.session import _scalar_default_clause
from app.infrastructure.storage.blob_store import LocalBlobStore, S3BlobStore

# --- settings ---------------------------------------------------------------


def test_database_url_overrides_sqlite_path(tmp_path):
    assert Settings(sqlite_path=tmp_path / "a.db").sqlite_url.startswith("sqlite:///")
    assert Settings(database_url="postgresql+psycopg://u:p@h/db").sqlite_url == "postgresql+psycopg://u:p@h/db"


def test_hosted_defaults_preserve_local_behavior():
    s = Settings()
    assert s.database_url is None
    assert s.chroma_host is None
    assert s.upload_backend == "local"
    assert s.embedding_backend == "sentence-transformers"
    assert s.tfs_personal_access_token is None


def test_s3_backend_requires_bucket():
    with pytest.raises(ValidationError):
        Settings(upload_backend="s3")
    assert Settings(upload_backend="s3", upload_s3_bucket="b").upload_s3_bucket == "b"


# --- database dialect handling ----------------------------------------------


def test_boolean_default_clause_is_dialect_aware():
    from sqlalchemy import Boolean, Column

    col = Column("is_active", Boolean, default=True)
    assert _scalar_default_clause(col, "sqlite") == "DEFAULT 1"
    assert _scalar_default_clause(col, "postgresql") == "DEFAULT TRUE"


def test_orphaned_column_reconciliation_skips_non_sqlite():
    from sqlalchemy import create_engine

    from app.infrastructure.db.session import _reconcile_orphaned_columns

    class _FakeDialect:
        name = "postgresql"

    class _FakeEngine:
        dialect = _FakeDialect()

    # Must return before touching the engine at all.
    _reconcile_orphaned_columns(_FakeEngine())  # type: ignore[arg-type]
    assert create_engine  # (import used so a missing sqlalchemy fails loudly)


# --- blob storage -----------------------------------------------------------


def test_local_blob_store_creates_directory_and_writes(tmp_path):
    store = LocalBlobStore(tmp_path / "nested" / "uploads")
    store.save("abc_file.txt", b"hello")
    assert (tmp_path / "nested" / "uploads" / "abc_file.txt").read_bytes() == b"hello"


def test_s3_blob_store_puts_object_under_prefix():
    calls = []

    class FakeS3:
        def put_object(self, **kwargs):
            calls.append(kwargs)

    S3BlobStore("my-bucket", "knowledge/", client=FakeS3()).save("tok_doc.pdf", b"bytes")
    assert calls == [{"Bucket": "my-bucket", "Key": "knowledge/tok_doc.pdf", "Body": b"bytes"}]


def test_knowledge_management_engine_accepts_blob_store_or_path(tmp_path):
    from app.engines.knowledge_management.engine import KnowledgeManagementEngine

    saved = []

    class Recorder:
        def save(self, name, content):
            saved.append((name, content))

    names = []
    for target in (tmp_path / "u", Recorder()):
        engine = KnowledgeManagementEngine(None, None, None, None, target)  # type: ignore[arg-type]
        name = engine._save_original_file("../../evil/name.txt", b"x")
        assert "/" not in name and ".." not in name and name.endswith("_name.txt")
        names.append(name)
    assert saved == [(names[1], b"x")]
    assert (tmp_path / "u" / names[0]).read_bytes() == b"x"


# --- TFS auth ---------------------------------------------------------------


def test_tfs_pat_uses_basic_auth_without_sspi():
    connector = TfsRestConnector(base_url="https://tfs/x", project="P", timeout_seconds=5, personal_access_token="tok")
    session = connector._get_session()
    assert session.auth == ("", "tok")


# --- Chroma: slim documents, server mode, slim script ------------------------


class _FixedEmbedder:
    dimensions = 3

    def embed(self, texts):
        return [[1.0, float(len(t) % 7), 0.5] for t in texts]


def test_store_keeps_only_snippet_but_embeds_full_text(tmp_path):
    store = ChromaKnowledgeStore(tmp_path / "c", _FixedEmbedder())
    long_text = "word " * 500
    store.upsert(KnowledgeCollection.DOCUMENTATION, "d1", long_text, "T", {"imported_at": "1"})
    raw = store._collection(KnowledgeCollection.DOCUMENTATION).get(ids=["d1"], include=["documents"])
    assert len(raw["documents"][0]) == SNIPPET_CHARS
    match = store.query(KnowledgeCollection.DOCUMENTATION, "word", 1)[0]
    assert match.snippet == long_text[:SNIPPET_CHARS]


def test_store_uses_injected_client(monkeypatch, tmp_path):
    sentinel = object()
    store = ChromaKnowledgeStore(tmp_path / "unused", _FixedEmbedder(), client=sentinel)
    assert store._client is sentinel
    assert not (tmp_path / "unused").exists()


def test_http_client_factory_passes_connection_settings(monkeypatch):
    import chromadb

    from app.infrastructure.vectorstore import chroma_client

    seen = {}

    def fake_http_client(**kwargs):
        seen.update(kwargs)
        return "client"

    monkeypatch.setattr(chromadb, "HttpClient", fake_http_client)
    chroma_client._client_cache.clear()
    client = chroma_client.get_chroma_http_client("chroma.internal", 9000, ssl=True, auth_token="t")
    assert client == "client"
    assert (seen["host"], seen["port"], seen["ssl"]) == ("chroma.internal", 9000, True)
    assert seen["headers"] == {"Authorization": "Bearer t"}
    chroma_client._client_cache.clear()


def test_slim_script_copies_ids_embeddings_metadata_and_truncates(tmp_path):
    from scripts.slim_chroma_index import slim_index

    source = tmp_path / "src"
    store = ChromaKnowledgeStore(source, _FixedEmbedder())
    # Simulate an OLD index that stored the full text.
    col = store._collection(KnowledgeCollection.DOCUMENTATION)
    full = "x" * 2000
    col.upsert(ids=["a"], embeddings=[[1.0, 2.0, 0.5]], documents=[full], metadatas=[{"title": "A", "k": "v"}])

    report = slim_index(source, tmp_path / "dst")
    assert report == {"documentation": (1, 1)}

    dst = ChromaKnowledgeStore(tmp_path / "dst", _FixedEmbedder())
    got = dst._collection(KnowledgeCollection.DOCUMENTATION).get(ids=["a"], include=["documents", "metadatas", "embeddings"])
    assert len(got["documents"][0]) == SNIPPET_CHARS
    assert got["metadatas"][0] == {"title": "A", "k": "v"}
    assert [round(float(x), 4) for x in got["embeddings"][0]] == [1.0, 2.0, 0.5]
    # the source is untouched
    assert len(col.get(ids=["a"], include=["documents"])["documents"][0]) == 2000


def test_slim_script_refuses_in_place_and_non_empty_destination(tmp_path):
    from scripts.slim_chroma_index import slim_index

    source = tmp_path / "src"
    ChromaKnowledgeStore(source, _FixedEmbedder())._collection(KnowledgeCollection.DOCUMENTATION)
    with pytest.raises(SystemExit):
        slim_index(source, source)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "file").write_text("x")
    with pytest.raises(SystemExit):
        slim_index(source, occupied)


def test_slim_script_can_load_into_an_existing_client(tmp_path):
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    from scripts.slim_chroma_index import slim_index

    source = tmp_path / "src"
    store = ChromaKnowledgeStore(source, _FixedEmbedder())
    store._collection(KnowledgeCollection.DOCUMENTATION).upsert(
        ids=["a"], embeddings=[[1.0, 2.0, 0.5]], documents=["y" * 900], metadatas=[{"title": "A"}]
    )
    dest = chromadb.PersistentClient(path=str(tmp_path / "other"), settings=ChromaSettings(anonymized_telemetry=False))
    assert slim_index(source, dest_client=dest) == {"documentation": (1, 1)}
    with pytest.raises(SystemExit):  # now non-empty -> refuses to merge
        slim_index(source, dest_client=dest)


# --- database migration script ------------------------------------------------


def test_migrate_database_copies_every_row_and_refuses_non_empty_target(tmp_path):
    from sqlalchemy import func, select

    from app.infrastructure.db.models import PlaybookModel, UserModel
    from app.infrastructure.db.session import get_engine, get_session_factory
    from scripts.migrate_database import migrate

    src_url = f"sqlite:///{(tmp_path / 'src.db').as_posix()}"
    dst_url = f"sqlite:///{(tmp_path / 'dst.db').as_posix()}"
    get_engine(src_url)
    with get_session_factory(src_url)() as session:
        session.add(UserModel(username="u", password_hash="h", role="admin"))
        session.add(PlaybookModel(id="p1", title="T", description="d", steps=["a", "b"]))
        session.commit()

    report = migrate(src_url, dst_url)
    assert report["users"] == (1, 1) and report["playbooks"] == (1, 1)
    with get_session_factory(dst_url)() as session:
        assert session.get(PlaybookModel, "p1").steps == ["a", "b"]
        assert session.scalar(select(func.count()).select_from(UserModel)) == 1

    with pytest.raises(SystemExit):
        migrate(src_url, dst_url)  # target now has rows
    with pytest.raises(SystemExit):
        migrate(f"sqlite:///{(tmp_path / 'missing.db').as_posix()}", f"sqlite:///{(tmp_path / 'x.db').as_posix()}")
