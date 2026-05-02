"""
Answer generation using a local LLM via Ollama.

The system prompt is engineered to:
  * Force grounding in the provided context.
  * Encourage "I don't know" when the context is insufficient
    (the assignment explicitly tests this with "president of Mars").
  * Discourage made-up citations or facts.

Streaming is supported for the UI (counts as an optional extension).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import ollama

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import LLM_MODEL, OLLAMA_HOST


_client = ollama.Client(host=OLLAMA_HOST)


SYSTEM_PROMPT = """\
You are a helpful research assistant. You answer questions about famous \
people and famous places using the context passages provided to you.

Rules — follow them strictly:
  1. Base your answer primarily on the supplied CONTEXT passages.
  2. If the CONTEXT contains relevant information, use it to answer — \
even if the match is not perfect. Partial information is better than no answer.
  3. Only say "I don't know based on the available information" when the \
CONTEXT has absolutely no relevant information (e.g. completely unknown \
entities like fictional planets or people not in the database).
  4. Be concise but complete. Prefer 2–6 sentences unless the user asks \
for more depth.
  5. When comparing two subjects, structure the answer clearly (e.g. a \
short paragraph per subject).
  6. Do not fabricate quotes, dates, or statistics not present in the CONTEXT.
"""


def build_user_message(query: str, chunks: list[dict]) -> str:
    """Format the retrieved chunks into a single user-message string."""
    if not chunks:
        return (
            f"QUESTION: {query}\n\n"
            f"CONTEXT: (no relevant passages were retrieved)\n\n"
            f"Answer using the rules above."
        )

    blocks = []
    for i, c in enumerate(chunks, start=1):
        label = f"[{i}] {c['entity']} ({c['type']})"
        blocks.append(f"{label}\n{c['text']}")
    context = "\n\n---\n\n".join(blocks)

    return (
        f"QUESTION: {query}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"Answer the question above using only the CONTEXT. "
        f"If the answer is not present, say you don't know."
    )


def generate(query: str, chunks: list[dict]) -> str:
    """Synchronous generation: returns the full answer string."""
    msg = build_user_message(query, chunks)
    resp = _client.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": msg},
        ],
        options={"temperature": 0.2},  # low temp = more faithful to context
    )
    return resp["message"]["content"].strip()


def generate_stream(query: str, chunks: list[dict]) -> Iterator[str]:
    """Streaming generation: yields token chunks as they arrive."""
    msg = build_user_message(query, chunks)
    stream = _client.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": msg},
        ],
        options={"temperature": 0.2},
        stream=True,
    )
    for part in stream:
        token = part.get("message", {}).get("content", "")
        if token:
            yield token