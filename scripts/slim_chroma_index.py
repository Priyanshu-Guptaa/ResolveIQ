"""Copy a Chroma index into a new directory with slim stored documents.

Older indexes stored every document's full text inside Chroma even though
the app only ever reads back the first 400 characters (the full text lives
in SQLite). This copies each collection -- ids, embeddings and metadata
exactly as they are, distance metric included -- storing only that
snippet, so the vector index (and the container image/volume carrying it)
shrinks without re-embedding anything or changing any search result.

The source index is opened read-only in intent and never modified; the
destination must not already contain data:

    python -m scripts.slim_chroma_index --source data/chroma --dest data/chroma_slim
    python -m scripts.slim_chroma_index --source data/chroma --dest-host chroma.internal --dest-token ...

Then point ``RESOLVEIQ_CHROMA_PERSIST_DIR`` at the new directory (or set
``RESOLVEIQ_CHROMA_HOST`` for the server).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engines.knowledge.knowledge_store import HNSW_METADATA, SNIPPET_CHARS  # noqa: E402

_BATCH = 200


def slim_index(source: Path, dest: Path | None = None, *, dest_client=None) -> dict[str, tuple[int, int]]:
    """Returns ``{collection: (source_count, dest_count)}``.

    The destination is either a new local directory (``dest``) or an
    already-connected Chroma client (``dest_client``, e.g. an HTTP client to
    the hosted Chroma server).
    """
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    if not source.exists():
        raise SystemExit(f"Source index not found: {source}")
    src_client = chromadb.PersistentClient(path=str(source), settings=ChromaSettings(anonymized_telemetry=False))

    if dest_client is not None:
        if any(dest_client.get_collection(c.name).count() for c in dest_client.list_collections()):
            raise SystemExit("Destination Chroma server already holds data -- refusing to merge into it.")
        dst_client = dest_client
    else:
        if dest is None:
            raise SystemExit("Give a destination directory or a destination client.")
        if source.resolve() == dest.resolve():
            raise SystemExit("Destination must differ from the source -- this tool never rewrites an index in place.")
        if dest.exists() and any(dest.iterdir()):
            raise SystemExit(f"Destination {dest} is not empty -- refusing to merge into existing data.")
        dest.mkdir(parents=True, exist_ok=True)
        dst_client = chromadb.PersistentClient(path=str(dest), settings=ChromaSettings(anonymized_telemetry=False))

    report: dict[str, tuple[int, int]] = {}
    for listed in src_client.list_collections():
        src = src_client.get_collection(listed.name)
        dst = dst_client.get_or_create_collection(name=listed.name, metadata={**(src.metadata or {}), **HNSW_METADATA})
        total = src.count()
        for offset in range(0, total, _BATCH):
            page = src.get(limit=_BATCH, offset=offset, include=["embeddings", "documents", "metadatas"])
            if not page["ids"]:
                break
            dst.upsert(
                ids=page["ids"],
                embeddings=[list(map(float, e)) for e in page["embeddings"]],
                documents=[(d or "")[:SNIPPET_CHARS] for d in page["documents"]],
                metadatas=page["metadatas"],
            )
        report[listed.name] = (total, dst.count())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy a Chroma index with slim stored documents.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dest", type=Path, help="new local directory for the slim index")
    parser.add_argument("--dest-host", help="OR: host of a Chroma server to load the slim index into")
    parser.add_argument("--dest-port", type=int, default=8000)
    parser.add_argument("--dest-ssl", action="store_true")
    parser.add_argument("--dest-token", help="bearer token for the Chroma server, if it requires one")
    args = parser.parse_args()
    if bool(args.dest) == bool(args.dest_host):
        parser.error("give exactly one of --dest or --dest-host")

    if args.dest_host:
        from app.infrastructure.vectorstore.chroma_client import get_chroma_http_client

        client = get_chroma_http_client(args.dest_host, args.dest_port, ssl=args.dest_ssl, auth_token=args.dest_token)
        report = slim_index(args.source, dest_client=client)
        args.dest = f"{args.dest_host}:{args.dest_port}"
    else:
        report = slim_index(args.source, args.dest)
    ok = True
    for name, (before, after) in report.items():
        flag = "ok" if before == after else "MISMATCH"
        ok &= before == after
        print(f"{name}: {before} -> {after} [{flag}]")
    if not ok:
        raise SystemExit("Record counts differ -- do not use the destination index.")
    print(f"Done. Destination: {args.dest}")


if __name__ == "__main__":
    main()
