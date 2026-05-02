"""
Top-level RAG orchestration: query → route → retrieve → generate → answer.

Includes latency measurement (one of the assignment's optional extensions).
"""
from __future__ import annotations

import time
from typing import Iterator

from src.retrieve import retrieve
from src.generate import generate, generate_stream


def answer(user_query: str, k: int = 5) -> dict:
    """
    Synchronous end-to-end RAG. Returns:
        {
          "answer": str,
          "route": str,
          "chunks": [...],
          "matched_people": [...],
          "matched_places": [...],
          "latency_ms": {
            "retrieve": int,
            "generate": int,
            "total": int,
          }
        }
    """
    t0 = time.perf_counter()

    t_retr_start = time.perf_counter()
    retrieval = retrieve(user_query, k=k)
    t_retr = (time.perf_counter() - t_retr_start) * 1000

    t_gen_start = time.perf_counter()
    text = generate(user_query, retrieval["chunks"])
    t_gen = (time.perf_counter() - t_gen_start) * 1000

    total = (time.perf_counter() - t0) * 1000

    return {
        "answer": text,
        "route": retrieval["route"],
        "chunks": retrieval["chunks"],
        "matched_people": retrieval["matched_people"],
        "matched_places": retrieval["matched_places"],
        "latency_ms": {
            "retrieve": int(t_retr),
            "generate": int(t_gen),
            "total": int(total),
        },
    }


def answer_stream(user_query: str, k: int = 5):
    """
    Streaming variant. Yields a dict with retrieval info first, then
    string tokens, then a final dict with timing info.

    Yields:
        {"event": "retrieved", "route": ..., "chunks": [...], "latency_ms": int}
        {"event": "token", "text": "..."}
        ...
        {"event": "done", "latency_ms": {"retrieve": ..., "generate": ..., "total": ...}}
    """
    t0 = time.perf_counter()

    t_retr_start = time.perf_counter()
    retrieval = retrieve(user_query, k=k)
    t_retr = (time.perf_counter() - t_retr_start) * 1000

    yield {
        "event": "retrieved",
        "route": retrieval["route"],
        "chunks": retrieval["chunks"],
        "matched_people": retrieval["matched_people"],
        "matched_places": retrieval["matched_places"],
        "latency_ms": int(t_retr),
    }

    t_gen_start = time.perf_counter()
    for tok in generate_stream(user_query, retrieval["chunks"]):
        yield {"event": "token", "text": tok}
    t_gen = (time.perf_counter() - t_gen_start) * 1000

    total = (time.perf_counter() - t0) * 1000
    yield {
        "event": "done",
        "latency_ms": {
            "retrieve": int(t_retr),
            "generate": int(t_gen),
            "total": int(total),
        },
    }
