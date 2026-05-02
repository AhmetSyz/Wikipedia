# Production Deployment Recommendations

> What would change to take this from a homework project to a real product?

This document discusses how the localhost-only Wikipedia RAG system would
need to evolve for production use, and the trade-offs involved.

## 1. Move the LLM to a dedicated inference server

**Why** — On a single laptop, Ollama + `llama3.2:3b` is fine. In production
the LLM has to serve many concurrent users, with tail-latency guarantees
and proper batching.

**What changes**

- Replace direct Ollama calls with a real serving layer: **vLLM**, **TGI
  (Hugging Face Text Generation Inference)**, or a managed endpoint like
  Anthropic / Bedrock / Vertex AI if "fully local" is no longer a hard
  constraint.
- Use a larger, better-aligned model (e.g. Llama 3.1 70B Instruct or a
  hosted frontier model). The 3B model is a survival choice for laptops,
  not a production choice.
- Put the inference layer behind a thin gateway service so application
  code stays unchanged when the backend swaps.

## 2. Move the vector store off-host

**Why** — Local Chroma is great for a single user. Production needs
multi-replica reads, write throughput, durable storage, and backups.

**Options**

| Option | Strengths | When |
|---|---|---|
| Hosted Chroma / **Chroma Cloud** | Closest migration path | If sticking with Chroma's API |
| **Pinecone** | Battle-tested, managed | If team wants zero ops |
| **Weaviate** | Strong filtering & hybrid search | If you need BM25 + vector |
| **pgvector** on managed Postgres | Operational familiarity | If your stack is already Postgres |
| **Qdrant** | Excellent filtering, open source | If you want to self-host |

The single-collection-with-metadata design transfers cleanly to all of
these. Only the client code in `src/embed_store.py` needs to change.

## 3. Replace rule-based routing with hybrid retrieval

The current rule-based router is transparent and zero-latency, but it can't
handle entities outside the seed list. In production:

- **Hybrid retrieval**: combine BM25 (keyword) + vector retrieval and merge
  with reciprocal-rank-fusion. Catches both lexical and semantic matches.
- **Cross-encoder reranking**: take the top-50 from hybrid, rerank with a
  cross-encoder (e.g. `bge-reranker`) to pick the top-5 actually fed to the
  LLM. Big quality bump for a small latency cost.
- **LLM-based router** *only* when hybrid fails — keeps the fast path fast
  while still handling weird queries.

## 4. Continuous ingestion

Wikipedia changes daily; a frozen index gets stale.

- Move ingestion behind a **scheduled job** (Airflow / Prefect / cron-on-k8s)
  that re-fetches changed articles using Wikipedia's recent-changes feed.
- Use **content hashes** to skip re-embedding unchanged chunks.
- Keep a **versioned index**: write to a new collection, swap atomically
  once it's verified. Lets you roll back a bad ingest in seconds.

## 5. Observability

A production RAG system without observability is impossible to debug.
At minimum:

- **Structured logs** for each query: route, retrieved chunk IDs, scores,
  prompt tokens, generation tokens, latencies, downstream errors.
- **Metrics** (Prometheus or hosted equivalent): p50/p95/p99 latency for
  retrieve and generate; cache hit rate; "I don't know" rate per route.
- **Traces** (OpenTelemetry): one span per stage so you can see where time
  goes on slow queries.
- **Eval harness**: a CI job that runs a fixed test set of questions on
  every commit to the prompt or the retriever, and fails the build on
  regressions. This is the single highest-leverage thing to add.

## 6. Caching

- **Query-result cache** (Redis): hash `(normalized_query, top_k, route)`
  → cached final answer with a TTL of a few hours. Huge win for popular
  questions.
- **Embedding cache**: per-document hash → vector. Saves recomputation
  when re-running ingest.
- **Prompt cache** at the LLM layer if the serving stack supports it
  (vLLM prefix caching, Anthropic prompt caching, etc.) since the system
  prompt is identical for every request.

## 7. Streaming, concurrency, and the API surface

- Promote the orchestrator to a real **FastAPI** service with `/chat`
  (SSE streaming) and `/health` endpoints. Streamlit becomes a thin
  client; mobile / web / Slack frontends can also use the same API.
- Run multiple worker replicas behind a load balancer.
- Use **async I/O** for the embedding and LLM calls so a single replica
  can handle dozens of concurrent users.

## 8. Multi-tenancy & auth

If serving real users:

- Auth: OAuth / OIDC via the user's existing identity provider, or simple
  JWT for early stages.
- Tenant isolation in the vector store: either separate collections per
  tenant or a `tenant_id` metadata field on every chunk with a forced
  filter.
- Rate limiting at the API gateway, with per-tenant quotas.

## 9. Safety, privacy, and prompt injection

- **PII filtering** on user queries before sending them to the LLM,
  especially if users can paste arbitrary text.
- **Prompt-injection defenses**: treat retrieved chunks as untrusted text.
  Use the system prompt to remind the model that instructions inside
  retrieved context must be ignored. For high-stakes use, run an
  additional classifier on retrieved chunks.
- **Output safety**: a lightweight content classifier on responses, with
  a clear "this content was withheld" UX rather than a silent failure.
- **Audit log** of every query and response, encrypted at rest.

## 10. Cost & scaling

For a managed-LLM scenario:

- Track **tokens-in / tokens-out** per query to forecast costs.
- Use a **tier strategy**: cheap small model for routing and short
  factual queries, larger model only when complexity warrants it.
- For self-hosted: GPU-hour budgeting, autoscaling on queue depth, and
  a fallback to a smaller model when load spikes.

## 11. Internationalization

The current build is English-only. To globalize:

- Per-language Wikipedia ingest (separate collections or a `lang`
  metadata field).
- Multilingual embedding model (e.g. `multilingual-e5-large`).
- Language detection on incoming queries to pick the right embedding +
  filter.

## 12. Evaluation strategy

Before any of the above lands in front of users:

- **Golden set**: 100–200 hand-curated `(question, expected-key-facts)`
  pairs per domain. Run on every PR.
- **LLM-as-judge** for nuanced answers, sanity-checked by humans on a
  weekly sample.
- **Retrieval metrics** independent of generation: recall@k against a
  labeled set, so you can detect retriever regressions even when the
  generator masks them.
- **Failure-mode tracking**: what % of queries return "I don't know"
  when they shouldn't (false negatives) and vice-versa.

## 13. Suggested rollout order

If you had to pick the smallest bite that delivers production value, in
order:

1. FastAPI service + structured logging + a basic eval set.
2. Hosted vector store with versioned indexes.
3. Hybrid retrieval + cross-encoder reranker.
4. Continuous ingestion pipeline.
5. Caching tiers.
6. Auth + multi-tenancy.
7. Larger LLM swap.

Each step compounds: by the time you swap the LLM, the rest of the system
is already production-grade and the upgrade is a config change.
