"""
Ingest Wikipedia articles for all required entities.

We use the `wikipedia-api` library (read-only, no API key needed). Each
article is saved as a UTF-8 text file plus a JSON sidecar with metadata.
The ingest is idempotent: existing files are skipped unless `force=True`.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Iterable

import wikipediaapi

# Add project root to path so this module works whether imported or run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR, PEOPLE, PLACES


# A descriptive User-Agent is requested by Wikipedia's API guidelines.
USER_AGENT = "WikipediaRAG-EduProject/1.0 (BLG483E HW3; contact: student@itu.edu.tr)"


def slugify(name: str) -> str:
    """Make a filesystem-safe slug from an entity name."""
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def fetch_one(wiki: wikipediaapi.Wikipedia, title: str, kind: str,
              force: bool = False) -> dict | None:
    """
    Fetch a single Wikipedia page.

    Returns metadata dict on success (whether downloaded or already cached),
    None if the page does not exist.
    """
    slug = f"{kind}_{slugify(title)}"
    txt_path = RAW_DIR / f"{slug}.txt"
    meta_path = RAW_DIR / f"{slug}.json"

    if txt_path.exists() and meta_path.exists() and not force:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    page = wiki.page(title)
    if not page.exists():
        print(f"  [MISS] '{title}' — page not found on Wikipedia")
        return None

    text = page.text
    if not text or len(text) < 200:
        print(f"  [WARN] '{title}' — suspiciously short article ({len(text)} chars)")

    meta = {
        "entity": title,
        "canonical_title": page.title,
        "type": kind,        # "person" or "place"
        "url": page.fullurl,
        "summary": page.summary[:500],
        "char_count": len(text),
    }

    txt_path.write_text(text, encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [OK]   '{title}' → {len(text):,} chars")
    return meta


def ingest_all(force: bool = False) -> list[dict]:
    """Ingest every entity in PEOPLE + PLACES. Returns metadata for each."""
    wiki = wikipediaapi.Wikipedia(user_agent=USER_AGENT, language="en")
    results: list[dict] = []

    print(f"Ingesting {len(PEOPLE)} people...")
    for name in PEOPLE:
        meta = fetch_one(wiki, name, "person", force=force)
        if meta:
            results.append(meta)
        time.sleep(0.1)  # be polite to Wikipedia

    print(f"\nIngesting {len(PLACES)} places...")
    for name in PLACES:
        meta = fetch_one(wiki, name, "place", force=force)
        if meta:
            results.append(meta)
        time.sleep(0.1)

    print(f"\nIngested {len(results)} / {len(PEOPLE) + len(PLACES)} entities total.")
    return results


def load_all() -> Iterable[tuple[dict, str]]:
    """
    Yield (metadata, full_text) for every successfully ingested entity.
    Used downstream by the chunker.
    """
    for meta_path in sorted(RAW_DIR.glob("*.json")):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        txt_path = meta_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        text = txt_path.read_text(encoding="utf-8")
        yield meta, text


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest Wikipedia articles.")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if cached.")
    args = parser.parse_args()
    ingest_all(force=args.force)
