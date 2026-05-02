# Product Requirements Document — Local Wikipedia RAG Assistant

> Version 1.0 · BLG483E HW3

## 1. Problem statement

People often want quick, conversational answers to questions about famous
people and famous places. Public chatbots do this well — but they require
sending queries to third-party servers, which raises privacy, cost, and
availability concerns. We need a system that delivers a similar experience
**entirely on the user's own machine**, with no external API calls, while
remaining accurate and grounded.

## 2. Goals

- **G1.** Run end-to-end on `localhost`. No cloud, no external LLM API.
- **G2.** Cover at minimum the 20 famous people and 20 famous places listed
  in the assignment, with extension room for more.
- **G3.** Produce answers that are grounded in the retrieved Wikipedia
  context. When the context is insufficient, the system must say so rather
  than hallucinate.
- **G4.** Provide a chat-style interface so the experience feels like
  ChatGPT.
- **G5.** Be reproducible: a clean machine should be able to run the project
  by following the `README.md` only.

## 3. Non-goals

- Multi-user, multi-tenant deployment.
- Auth, billing, rate limiting.
- Content beyond the 20+20 entity scope (the architecture supports more,
  but quality outside-of-scope is best-effort).
- Multilingual support (English Wikipedia only in v1).
- Real-time updates as Wikipedia changes (re-ingestion is manual).

## 4. Users & user stories

The primary user is a **curious learner or researcher** running the system
locally. Representative stories:

- *As a student*, I want to ask "What did Marie Curie discover?" and get a
  concise, factual answer with the Wikipedia source.
- *As a teacher*, I want to compare two figures ("Compare Einstein and
  Tesla") and see grounded points for each.
- *As a privacy-conscious user*, I want the assurance that no part of my
  question leaves my machine.
- *As a developer*, I want to read the source code and understand how each
  RAG stage works, without fighting through opaque library magic.

## 5. Functional requirements

| ID | Requirement |
|---|---|
| FR-1 | Ingest at least 20 people + 20 places from Wikipedia, storing raw text + metadata locally. |
| FR-2 | Chunk each document into smaller pieces, with a documented strategy. |
| FR-3 | Embed all chunks with a local embedding model and persist them in a vector store. |
| FR-4 | Given a query, classify it as person / place / both, then retrieve relevant chunks accordingly. |
| FR-5 | Generate a grounded natural-language answer using a local LLM. |
| FR-6 | Return "I don't know based on the available information" when the context is insufficient. |
| FR-7 | Provide a chat-style UI that supports asking, viewing answers, viewing context, and resetting. |
| FR-8 | Stream tokens as they are produced (UX requirement). |
| FR-9 | Display per-query latency (retrieval + generation). |
| FR-10 | Be runnable via the `README` alone, with no extra guidance. |

## 6. Non-functional requirements

- **NFR-1 — Locality**: zero network calls during query time (Wikipedia
  fetches happen only at ingest time).
- **NFR-2 — Reproducibility**: pinned dependency versions; fresh-clone
  setup must succeed in <10 minutes including model downloads.
- **NFR-3 — Latency target**: median end-to-end query under 8 seconds on a
  modern laptop with `llama3.2:3b`.
- **NFR-4 — Code quality**: hand-written chunking, routing, and orchestration
  rather than off-the-shelf "RAG-in-a-box" libraries, per the assignment's
  preference for language-native code.
- **NFR-5 — Transparency**: every answer can show the exact source chunks
  it was generated from.

## 7. Tech stack & rationale

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.10+ | Required by stack; easiest Ollama + Chroma + Streamlit story. |
| LLM | Ollama + `llama3.2:3b` | Small enough for laptops, instruction-tuned, freely available. |
| Embeddings | Ollama + `nomic-embed-text` | Local, fast, 768-dim, no extra dependency. |
| Vector DB | Chroma (persistent) | Native Python, simple metadata filtering, no server to manage. |
| UI | Streamlit | Fastest path to a polished chat UI; works well in the demo video. |
| Wikipedia | `wikipedia-api` library | Read-only, no API key, simple. |

## 8. Storage design — Option B (single collection, metadata)

The assignment offers two options. We chose **Option B**:

> One vector store with metadata (e.g., `type=person` or `type=place`).

**Why:**
- A single collection keeps ops simple — one place to embed, one place to
  query, one cache to manage.
- Chroma's `where` filter cleanly handles the three routing cases:
  filtered-person, filtered-place, and the *mixed* case (two filtered
  queries merged).
- The same embedding model is used for both kinds of entities, so there is
  no representational benefit to splitting collections.
- Adds extensibility: new entity types (events, organizations) are a
  one-line metadata change, not a new collection.

The trade-off is that filtering happens at query time rather than being
implicit in the data layout. With Chroma's HNSW index this cost is
negligible at our scale (a few thousand vectors).

## 9. Chunking strategy

Sentence-aware fixed-window chunks of ~500 characters with 80 characters of
overlap. Implemented from scratch:
1. Split into sentences (regex with abbreviation safeguards).
2. Greedily pack sentences into chunks until the size limit.
3. When emitting a chunk, prepend the tail of the previous chunk
   (snapped to a word boundary) so factual context survives boundaries.

This gives clean retrieval units that respect natural language structure
and avoids the loss-at-boundary problem of pure fixed-character splitting.

## 10. Routing logic

Rule-based, no LLM call:
1. Match known person-names and place-names against the query (using
   diacritic-stripped lowercase comparison and last-name shortcuts).
2. If only people match → `person`. Only places → `place`. Both → `mixed`.
3. If nothing matches, score against question-stem hint sets ("who",
   "biography", "where", "tower", etc.) and route to whichever scores
   higher.
4. Default to `mixed` when truly ambiguous so we still attempt an answer.

The trade-off is that the router is bounded by the rules we wrote.
Real-world unknown entities will hit the `mixed` fallback, which still
produces a sensible search but may waste retrieval budget. Documented
in `recommendation.md` as a place to upgrade for production.

## 11. Generation prompt

The system prompt enforces three things:
1. Answer **only** from the provided context.
2. Say "I don't know based on the available information" when the context
   is insufficient — this is the assignment's failure-case requirement.
3. Don't fabricate quotes, dates, or sources.

Temperature is set to 0.2 to bias the model toward faithful repetition of
context over creative invention.

## 12. Success criteria

- All 40+ entities ingest successfully on a clean run.
- All 14 example queries from the assignment produce correct, grounded
  answers.
- Both failure-case queries return an "I don't know" response.
- Median end-to-end latency is under 8 seconds on a laptop.
- The grader can run the project from the `README` alone.
- Demo video covers system overview, live demo, technical decisions,
  trade-offs, and improvements within 5 minutes.

## 13. Optional extensions implemented

- ✅ Streaming responses
- ✅ Citations / source highlighting
- ✅ Chat history memory (Streamlit session state)
- ✅ Latency measurement (per query, broken into retrieve / generate / total)
- 🔜 (Documented in `recommendation.md`) reranking, multi-model comparison,
  caching.
