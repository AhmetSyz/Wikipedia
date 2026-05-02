"""
Streamlit chat UI for the Wikipedia RAG system.

Features:
  * Chat-style message history (session-state powered)
  * Streaming answers as the local LLM produces them
  * Toggle to view retrieved source chunks
  * Per-message latency display
  * Reset / clear button
  * Sidebar shows the system status, model, and entity counts
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.rag import answer_stream
from src.embed_store import get_client, get_or_create_collection
from config import COLLECTION_NAME, LLM_MODEL, EMBED_MODEL, PEOPLE, PLACES


# --------------------------------------------------------------------------
# Helpers (defined BEFORE they are called below)
# --------------------------------------------------------------------------

def render_meta(meta: dict, show_sources: bool) -> None:
    """Render route + latency + (optionally) sources for an assistant message."""
    cols = st.columns(4)
    cols[0].caption(f"🔀 Route: **{meta.get('route', '?')}**")
    lat = meta.get("latency_ms", {})
    cols[1].caption(f"🔍 Retrieve: {lat.get('retrieve', '?')} ms")
    cols[2].caption(f"🧠 Generate: {lat.get('generate', '?')} ms")
    cols[3].caption(f"⏱️ Total: {lat.get('total', '?')} ms")

    if show_sources and meta.get("chunks"):
        with st.expander(f"📄 Retrieved context ({len(meta['chunks'])} chunks)"):
            for i, c in enumerate(meta["chunks"], start=1):
                st.markdown(
                    f"**[{i}] {c['entity']}** ({c['type']}) "
                    f"· score `{c['score']:.3f}` · "
                    f"[Wikipedia ↗]({c['url']})"
                )
                st.write(c["text"])
                if i < len(meta["chunks"]):
                    st.divider()


# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(page_title="Wikipedia RAG", page_icon="📚", layout="wide")

st.title("📚 Local Wikipedia RAG Assistant")
st.caption(
    "Retrieval-augmented Q&A over Wikipedia, running 100% on your laptop. "
    f"LLM: `{LLM_MODEL}` · Embeddings: `{EMBED_MODEL}` · Vector DB: Chroma"
)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    show_sources = st.toggle(
        "Show retrieved context", value=True,
        help="Display the source chunks used to ground each answer.",
    )

    top_k = st.slider(
        "Top-K chunks", 1, 10, 5,
        help="How many chunks to retrieve per query.",
    )

    st.divider()
    st.subheader("📊 Index status")
    try:
        client = get_client()
        coll = get_or_create_collection(client)
        n_vectors = coll.count()
        st.metric("Vectors indexed", f"{n_vectors:,}")
    except Exception as e:
        st.error(f"Chroma not available: {e}")
        n_vectors = 0

    st.metric("People", len(PEOPLE))
    st.metric("Places", len(PLACES))

    if n_vectors == 0:
        st.warning(
            "Index is empty. Run "
            "`python scripts/build_index.py` first."
        )

    st.divider()
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("💡 Try asking")
    examples = [
        "What did Marie Curie discover?",
        "Where is the Eiffel Tower located?",
        "Compare Lionel Messi and Cristiano Ronaldo.",
        "Which famous place is located in Turkey?",
        "Who is the president of Mars?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
            st.session_state.queued_query = ex
            st.rerun()


# --------------------------------------------------------------------------
# Chat state & history replay
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            render_meta(msg["meta"], show_sources)


# --------------------------------------------------------------------------
# Input & streamed response
# --------------------------------------------------------------------------
prompt = st.chat_input("Ask about a person or place...")

# Allow sidebar example buttons to populate the chat
if not prompt and st.session_state.get("queued_query"):
    prompt = st.session_state.pop("queued_query")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        meta_placeholder = st.empty()
        full_text = ""
        meta: dict = {"route": None, "chunks": [], "latency_ms": {}}

        try:
            for event in answer_stream(prompt, k=top_k):
                if event["event"] == "retrieved":
                    meta["route"] = event["route"]
                    meta["chunks"] = event["chunks"]
                    meta["matched_people"] = event["matched_people"]
                    meta["matched_places"] = event["matched_places"]
                elif event["event"] == "token":
                    full_text += event["text"]
                    placeholder.markdown(full_text + "▌")
                elif event["event"] == "done":
                    meta["latency_ms"] = event["latency_ms"]

            placeholder.markdown(full_text)
            with meta_placeholder.container():
                render_meta(meta, show_sources)

        except Exception as e:
            placeholder.error(
                f"Error talking to Ollama: {e}\n\n"
                "Make sure Ollama is running and the models are pulled "
                f"(`ollama pull {LLM_MODEL}` and "
                f"`ollama pull {EMBED_MODEL}`)."
            )
            full_text = "(error)"

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_text,
        "meta": meta,
    })
