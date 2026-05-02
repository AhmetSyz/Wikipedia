# 📚 Local Wikipedia RAG Assistant

A fully-local ChatGPT-style system that answers questions about famous people
and famous places, using only resources running on your laptop:

- **Wikipedia** as the knowledge source
- **Ollama** for the local LLM (`llama3.2:3b`) and embeddings (`nomic-embed-text`)
- **Chroma** as the persistent vector database
- **Streamlit** for the chat interface

No external APIs. No cloud. Built for **BLG483E HW3**.

---

## 🧱 Architecture

```
                     ┌──────────────────────┐
   user query ─────► │   Streamlit chat UI  │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │   Router (rules)     │  → person / place / mixed
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │  Chroma vector DB    │  ← Ollama embeddings
                     │  (single collection, │     (nomic-embed-text)
                     │   metadata: type)    │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │  Local LLM (Ollama)  │  llama3.2:3b
                     │  grounded prompt     │  temperature 0.2
                     └──────────┬───────────┘
                                │
                                ▼
                          answer + sources
```

---

## ⚙️ Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.10+ | Tested with 3.11 |
| [Ollama](https://ollama.com/download) | latest | Runs the local models |
| RAM | ≥ 8 GB | `llama3.2:3b` needs ~2.5 GB; embeddings ~600 MB |
| Disk | ~5 GB | Models + Chroma index + raw text |

Install Ollama:

- **macOS / Windows**: download the installer from <https://ollama.com/download>
- **Linux**: `curl -fsSL https://ollama.com/install.sh | sh`

After installing, make sure the Ollama service is running (it auto-starts on
macOS/Windows). On Linux you may need: `ollama serve &`.

---

## 🚀 Setup (5 commands)

```bash
# 1. Clone and enter the repo
git clone https://github.com/<your-username>/wikipedia-rag.git
cd wikipedia-rag

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Pull the local models (runs once, ~2 GB download)
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# 5. Build the index — this fetches Wikipedia, chunks, embeds, persists.
#    Takes ~3–5 minutes the first time.
python scripts/build_index.py
```

Then launch the chat UI:

```bash
streamlit run app.py
```

The app opens at <http://localhost:8501>.

> Prefer the terminal? `python cli.py` works too.

---

## 🗂️ How the system works

1. **Ingest** — `src/ingest.py` downloads each entity's Wikipedia article via
   `wikipedia-api` (no key required) and saves the raw text plus a small JSON
   metadata file in `data/raw/`.
2. **Chunk** — `src/chunk.py` splits each article into ~500-character,
   sentence-aware chunks with 80 characters of overlap, written from scratch
   (no LangChain or other "magic" splitter).
3. **Embed & store** — `src/embed_store.py` calls Ollama's `nomic-embed-text`
   model on every chunk and writes the vectors to a single Chroma collection
   tagged with `{type: "person" | "place"}` metadata.
4. **Route** — `src/router.py` uses a fast rule-based check (entity-name match
   + question-stem keywords) to decide whether each query is about a person,
   a place, or both. No extra LLM call.
5. **Retrieve** — `src/retrieve.py` queries Chroma with the type filter from
   the router. For **mixed** queries it runs two filtered queries and
   interleaves results.
6. **Generate** — `src/generate.py` prompts `llama3.2:3b` with a strict
   "use only the context, otherwise say 'I don't know'" instruction and
   streams tokens back to the UI.
7. **UI** — `app.py` wires it all together: chat history, streaming
   responses, expandable source panel, latency metrics, reset button.

---

## 🧪 Example queries

Try these in the UI (or click them in the sidebar):

**People**
- *Who was Albert Einstein and what is he known for?*
- *What did Marie Curie discover?*
- *Why is Nikola Tesla famous?*
- *Compare Lionel Messi and Cristiano Ronaldo.*
- *What is Frida Kahlo known for?*

**Places**
- *Where is the Eiffel Tower located?*
- *Why is the Great Wall of China important?*
- *What is Machu Picchu?*
- *What was the Colosseum used for?*
- *Where is Mount Everest?*

**Mixed**
- *Which famous place is located in Turkey?*
- *Which person is associated with electricity?*
- *Compare Albert Einstein and Nikola Tesla.*
- *Compare the Eiffel Tower and the Statue of Liberty.*

**Failure cases** (should respond "I don't know")
- *Who is the president of Mars?*
- *Tell me about a random unknown person John Doe.*

---

## 🔄 Re-building the index

```bash
# Rebuild the vector store from cached articles
python scripts/build_index.py --rebuild

# Re-download the Wikipedia articles AND rebuild
python scripts/build_index.py --force --rebuild
```

---

## 📁 Repository layout

```
wikipedia-rag/
├── README.md
├── product_prd.md             ← Product Requirements Document
├── recommendation.md          ← Production deployment notes
├── requirements.txt
├── config.py                  ← entity lists, model names, knobs
├── app.py                     ← Streamlit UI
├── cli.py                     ← optional CLI fallback
├── src/
│   ├── ingest.py              ← Wikipedia fetcher
│   ├── chunk.py               ← sentence-aware chunker
│   ├── embed_store.py         ← Chroma + Ollama embeddings
│   ├── router.py              ← person / place / mixed classifier
│   ├── retrieve.py            ← retrieval orchestration
│   ├── generate.py            ← Ollama LLM call (with streaming)
│   └── rag.py                 ← end-to-end orchestrator
├── scripts/
│   └── build_index.py         ← one-shot ingest + chunk + embed
└── data/                      ← created at runtime, git-ignored
    ├── raw/                   ← .txt + .json per article
    └── chroma/                ← persistent vector DB
```

---

## 🧰 Tradeoffs & limitations

- **Small LLM**: `llama3.2:3b` is fast on a laptop but weaker than larger
  models — occasional ungrounded phrasings can slip through. The strict
  "I don't know" rule in the system prompt mitigates this.
- **Rule-based router**: simple and transparent, but fails on entities the
  rules don't know about (e.g. "Frida Kahlo" matches by last name; an
  unknown person won't match either pool, so we fall through to a "mixed"
  search). An LLM-based router would be more flexible at the cost of latency.
- **English-only**: Wikipedia is fetched from `en.wikipedia.org`. Multilingual
  support would require swapping the embedding model and the wiki language.
- **No reranking**: vector top-K only. For tougher queries a cross-encoder
  reranker would improve relevance.

See `recommendation.md` for a full production-readiness analysis.

---

## 🎬 Demo video

> Demo link will be added here after recording.

---

## 📜 License

Educational project for ITU BLG483E. Not for commercial use.
