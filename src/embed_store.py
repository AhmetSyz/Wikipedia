"""
Vector store: Chroma (persistent) + Ollama embeddings (nomic-embed-text).

Design choice — Option B from the assignment:
  ONE collection with a `type` metadata field ("person" or "place").

Why?
  * Single source of truth, simpler ops.
  * Chroma's `where` filter cleanly handles both filtered and mixed queries.
  * For "compare a person and a place" or "which place is in Turkey", we
    just run two filtered queries and merge — easier than juggling two
    collections.
  * Embedding model is shared either way, so there's no quality difference.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import chromadb
import ollama
from chromadb.config import Settings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL, OLLAMA_HOST, TOP_K,
)
from src.chunk import Chunk


# --------------------------------------------------------------------------
# Embedding via Ollama (local, no network)
# --------------------------------------------------------------------------

_ollama_client = ollama.Client(host=OLLAMA_HOST)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using the local Ollama embedding model."""
    vectors: list[list[float]] = []
    for t in texts:
        # ollama-python exposes one embedding at a time; we batch in a loop
        resp = _ollama_client.embeddings(model=EMBED_MODEL, prompt=t)
        vectors.append(resp["embedding"])
    return vectors


def embed_one(text: str) -> list[float]:
    """Embed a single string (used for queries)."""
    resp = _ollama_client.embeddings(model=EMBED_MODEL, prompt=text)
    return resp["embedding"]


# --------------------------------------------------------------------------
# Chroma client
# --------------------------------------------------------------------------

def get_client() -> chromadb.api.ClientAPI:
    """Persistent Chroma client rooted at data/chroma."""
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def get_or_create_collection(client: chromadb.api.ClientAPI):
    """Get the collection; cosine distance is what we want for embeddings."""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection() -> None:
    """Wipe and recreate the collection. Used when --rebuild is passed."""
    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------

def index_chunks(chunks: list[Chunk], batch_size: int = 32) -> None:
    """
    Embed and store chunks in the vector DB. Idempotent: if a chunk_id
    already exists, it is upserted.
    """
    client = get_client()
    coll = get_or_create_collection(client)

    total = len(chunks)
    print(f"Embedding & indexing {total} chunks (batch={batch_size})...")

    for start in range(0, total, batch_size):
        batch = chunks[start:start + batch_size]
        texts = [c.text for c in batch]
        ids = [c.chunk_id for c in batch]
        metadatas = [
            {
                "entity": c.entity,
                "type": c.type,
                "title": c.title,
                "url": c.url,
                "chunk_index": c.chunk_index,
            }
            for c in batch
        ]
        embeddings = embed_texts(texts)
        coll.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        done = min(start + batch_size, total)
        print(f"  ...{done}/{total}")

    print(f"Done. Collection '{COLLECTION_NAME}' now has {coll.count()} vectors.")


# --------------------------------------------------------------------------
# Querying
# --------------------------------------------------------------------------

def query(text: str, k: int = TOP_K, type_filter: str | None = None) -> list[dict]:
    """
    Retrieve top-k chunks. If type_filter is set ("person" or "place"),
    only chunks of that type are considered.

    Returns a list of dicts: {text, entity, type, title, url, score}.
    """
    client = get_client()
    coll = get_or_create_collection(client)

    qvec = embed_one(text)
    where = {"type": type_filter} if type_filter else None

    res = coll.query(
        query_embeddings=[qvec],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    out: list[dict] = []
    if not res["documents"] or not res["documents"][0]:
        return out

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    for doc, meta, dist in zip(docs, metas, dists):
        # cosine distance ∈ [0, 2]; convert to similarity ∈ [-1, 1]
        out.append({
            "text": doc,
            "entity": meta.get("entity", ""),
            "type": meta.get("type", ""),
            "title": meta.get("title", ""),
            "url": meta.get("url", ""),
            "score": 1.0 - dist,
        })
    return out
