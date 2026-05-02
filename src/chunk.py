"""
Chunking strategy: sentence-aware fixed-window with overlap.

Why this design?
  * Documents can be large (Wikipedia articles run 30k+ chars), so we MUST
    split or we'll blow past the embedding model's context window.
  * Chunking on raw character boundaries can split words and sentences,
    hurting retrieval quality.
  * Pure paragraph chunking is uneven (some paragraphs are 2 lines, others
    are 5,000 chars).
  * Solution: split into sentences first, then greedily pack sentences into
    fixed-size windows with a small overlap so facts at chunk boundaries
    aren't lost.

Implemented from scratch — no LangChain or other "magic" splitters — per
the assignment's preference for language-native code.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    """A single chunk of text, with metadata for retrieval."""
    chunk_id: str
    text: str
    entity: str
    type: str        # "person" or "place"
    title: str       # canonical Wikipedia title
    url: str
    chunk_index: int


# A simple but effective sentence splitter. Avoids splitting on common
# abbreviations (Mr., Dr., etc.) and on decimals.
_ABBREV = {"mr", "mrs", "ms", "dr", "st", "vs", "etc", "e.g", "i.e", "no",
           "fig", "vol", "jr", "sr"}
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")


def split_sentences(text: str) -> list[str]:
    """
    Naive but robust sentence splitter. We split on ., !, ? followed by
    whitespace and a capital letter, then post-process to glue back common
    abbreviations.
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    raw = _SENT_END.split(text)

    # Glue back fragments that ended with an abbreviation
    sentences: list[str] = []
    for s in raw:
        s = s.strip()
        if not s:
            continue
        if sentences:
            prev = sentences[-1]
            last_token = prev.split()[-1].rstrip(".").lower() if prev.split() else ""
            if last_token in _ABBREV:
                sentences[-1] = prev + " " + s
                continue
        sentences.append(s)
    return sentences


def chunk_text(text: str, size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Greedy sentence-packed chunker with character overlap.

    Algorithm:
      1. Split into sentences.
      2. Greedily append sentences to the current chunk until adding the
         next one would exceed `size`.
      3. Emit the chunk, then start the next chunk by re-including the
         tail of the previous one (`overlap` chars) so context isn't lost
         at boundaries.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current = ""

    for sent in sentences:
        # If a single sentence is already huge, hard-split it on whitespace.
        if len(sent) > size:
            sub = _hard_split(sent, size)
            for piece in sub:
                if current:
                    chunks.append(current.strip())
                    current = _tail(current, overlap) + " " + piece
                else:
                    current = piece
            continue

        if len(current) + len(sent) + 1 <= size:
            current = (current + " " + sent).strip() if current else sent
        else:
            chunks.append(current.strip())
            current = (_tail(current, overlap) + " " + sent).strip()

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _tail(s: str, n: int) -> str:
    """Return the last n characters of s, snapped to a word boundary."""
    if not s or n <= 0:
        return ""
    tail = s[-n:]
    # Snap to next space so we don't start mid-word
    space = tail.find(" ")
    return tail[space + 1:] if space != -1 else tail


def _hard_split(s: str, size: int) -> list[str]:
    """Fallback splitter for pathologically long sentences."""
    words = s.split()
    out, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= size:
            cur = (cur + " " + w).strip() if cur else w
        else:
            out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out


def chunk_document(meta: dict, text: str) -> list[Chunk]:
    """Apply chunking to a single Wikipedia document, returning Chunk objects."""
    pieces = chunk_text(text)
    out: list[Chunk] = []
    entity = meta["entity"]
    for i, piece in enumerate(pieces):
        out.append(Chunk(
            chunk_id=f"{meta['type']}_{_slug(entity)}_{i:04d}",
            text=piece,
            entity=entity,
            type=meta["type"],
            title=meta.get("canonical_title", entity),
            url=meta.get("url", ""),
            chunk_index=i,
        ))
    return out


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def chunk_all(documents: Iterable[tuple[dict, str]]) -> list[Chunk]:
    """Chunk a stream of (meta, text) pairs. Returns a flat list."""
    all_chunks: list[Chunk] = []
    for meta, text in documents:
        all_chunks.extend(chunk_document(meta, text))
    return all_chunks
