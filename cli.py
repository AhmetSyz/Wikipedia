"""
Simple CLI for the Wikipedia RAG system. Use this if you can't run Streamlit.

Commands inside the chat loop:
    /sources    toggle showing retrieved chunks
    /clear      clear screen
    /quit       exit
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.rag import answer_stream
from config import LLM_MODEL


def main() -> None:
    print("=" * 60)
    print(" Wikipedia RAG — CLI (model:", LLM_MODEL, ")")
    print(" Type /quit to exit, /sources to toggle context display")
    print("=" * 60)

    show_sources = False

    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not q:
            continue
        if q == "/quit":
            break
        if q == "/clear":
            os.system("clear" if os.name != "nt" else "cls")
            continue
        if q == "/sources":
            show_sources = not show_sources
            print(f"  (showing sources: {show_sources})")
            continue

        meta: dict = {}
        print("bot> ", end="", flush=True)
        for ev in answer_stream(q):
            if ev["event"] == "retrieved":
                meta = {
                    "route": ev["route"],
                    "chunks": ev["chunks"],
                }
            elif ev["event"] == "token":
                print(ev["text"], end="", flush=True)
            elif ev["event"] == "done":
                lat = ev["latency_ms"]
                print(
                    f"\n     [route={meta.get('route')} · "
                    f"retrieve={lat['retrieve']}ms · "
                    f"generate={lat['generate']}ms · "
                    f"total={lat['total']}ms]"
                )

        if show_sources and meta.get("chunks"):
            print("\n  Sources:")
            for i, c in enumerate(meta["chunks"], 1):
                print(f"    [{i}] {c['entity']} ({c['type']}) score={c['score']:.3f}")


if __name__ == "__main__":
    main()
