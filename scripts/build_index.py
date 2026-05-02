"""
One-shot script: ingest Wikipedia → chunk → embed → store in Chroma.

Usage:
    python scripts/build_index.py            # incremental (skips cached)
    python scripts/build_index.py --rebuild  # wipes and rebuilds the index
    python scripts/build_index.py --force    # also re-downloads Wikipedia
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import ingest_all, load_all
from src.chunk import chunk_all
from src.embed_store import index_chunks, reset_collection, get_client, get_or_create_collection
from config import COLLECTION_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Wikipedia RAG index.")
    parser.add_argument("--force", action="store_true",
                        help="Re-download Wikipedia articles even if cached.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Drop and recreate the vector collection.")
    args = parser.parse_args()

    print("=" * 60)
    print(" Wikipedia RAG — Index Builder")
    print("=" * 60)

    t0 = time.perf_counter()

    # Step 1: ingest
    print("\n[1/3] Ingesting Wikipedia articles...")
    ingest_all(force=args.force)

    # Step 2: chunk
    print("\n[2/3] Chunking documents...")
    chunks = chunk_all(load_all())
    print(f"      Produced {len(chunks)} chunks.")

    # Step 3: embed + store
    print("\n[3/3] Embedding and indexing in Chroma...")
    if args.rebuild:
        print("      (rebuilding collection from scratch)")
        reset_collection()

    index_chunks(chunks)

    # Summary
    client = get_client()
    coll = get_or_create_collection(client)
    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 60)
    print(f" Done in {elapsed:.1f}s. Collection '{COLLECTION_NAME}' "
          f"holds {coll.count()} vectors.")
    print("=" * 60)
    print("\nNext: run the chat UI with `streamlit run app.py`")


if __name__ == "__main__":
    main()
