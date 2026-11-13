# Observability in RAG

Observability means capturing every step of a RAG request — query, retrieved chunks, generation — in a way that lets you debug failures, track latency, and monitor quality in production.

Author

Benedict Thekkel

## 1. What to Log

Every RAG request is a chain of operations. Observability requires logging the full chain together so you can replay and debug any failure.

**Minimum log record per request:**

``` json
{
  "request_id": "uuid",
  "timestamp": "2024-11-01T12:00:00Z",
  "query": "What are the refund policies?",
  "query_embedding_ms": 18,
  "retrieved_chunks": [
    {"id": "doc42_chunk3", "score": 0.91, "text": "..."},
    {"id": "doc17_chunk1", "score": 0.88, "text": "..."}
  ],
  "retrieval_ms": 42,
  "reranker_ms": 120,
  "prompt_tokens": 1240,
  "completion_tokens": 180,
  "generation_ms": 890,
  "answer": "Our refund policy allows returns within 30 days...",
  "total_latency_ms": 1070
}
```

**Never log PII without masking.** Strip or hash user identifiers before writing to observability stores.

------------------------------------------------------------------------

## 2. Tracing the RAG Chain

A single user request spans multiple services (embedding API, vector DB, LLM). Distributed tracing ties them together with a shared `trace_id`.

**OpenTelemetry integration pattern:**

``` python
from opentelemetry import trace

tracer = trace.get_tracer("rag-pipeline")

with tracer.start_as_current_span("rag_request") as span:
    span.set_attribute("query", user_query)

    with tracer.start_as_current_span("embed_query"):
        query_vector = embed(user_query)

    with tracer.start_as_current_span("vector_search"):
        chunks = index.search(query_vector, top_k=5)

    with tracer.start_as_current_span("generate"):
        answer = llm.generate(build_prompt(query, chunks))
```

**Tools that provide RAG-aware tracing out of the box:** - **LangSmith** (LangChain) — full chain visualization - **Arize Phoenix** — open-source, LLM-first observability - **Weave** (Weights & Biases) — trace + eval combined - **Helicone / Braintrust** — LLM call logging with replay

------------------------------------------------------------------------

## 3. Latency Breakdown

For production SLAs you need to know *where* time is spent, not just total latency.

| Step            | Typical Latency | Notes                                   |
|-----------------|-----------------|-----------------------------------------|
| Query embedding | 10–30 ms        | Cached after first call                 |
| Vector search   | 5–50 ms         | Depends on index size and ANN params    |
| Re-ranking      | 50–300 ms       | Cross-encoder is expensive              |
| LLM generation  | 500–3000 ms     | Largest contributor; streaming hides it |
| Total           | ~800–3500 ms    | Target \< 2 s for interactive UX        |

**p50 / p95 / p99 breakdown is more informative than mean** — a slow tail at p99 often signals retrieval timeouts or cold LLM instances.

------------------------------------------------------------------------

## 4. Quality Monitoring in Production

Beyond latency, you need signals that answer quality is not degrading.

**Metric streams to track:** - **Inline LLM judge score** — sample 5–10% of requests; score Faithfulness + Answer Relevance; alert on rolling average drop - **User feedback signals** — thumbs up/down, copy-paste rate, follow-up question rate (indirect quality signal) - **Retrieval zero-hit rate** — fraction of queries where top-1 chunk score \< threshold (signals corpus gaps) - **Token usage** — track input/output tokens per request to catch prompt bloat regressions

**Dashboard example (Grafana / Datadog):**

    Panels:
      - p50/p95 total latency (time series)
      - Faithfulness score rolling 24h average (gauge)
      - Zero-hit rate (% time series)
      - Token usage per request (histogram)
      - Error rate by step (retrieval, rerank, generation)

------------------------------------------------------------------------

## 5. Debugging with Trace Replay

When a user reports a bad answer, you need to replay the exact request.

**Requirements for replay:** 1. Stored full trace (query, retrieved chunks at time of request, answer) 2. Pinned versions (embedding model, index snapshot, prompt version, LLM version) 3. A replay tool that re-runs generation given the stored chunks (not re-retrieval)

**Important:** Re-running retrieval against an updated index won’t reproduce the failure if the index has changed. Always log the chunk *content* not just the chunk IDs.

**Root-cause checklist:**

    Bad answer received
    ├── Was the relevant chunk retrieved?  → NO → Retrieval problem (embedding, top-K, filtering)
    ├── Was the relevant chunk in top-K but not used? → Context window / ordering problem
    ├── Chunk was used but answer is wrong → Generation / prompt problem
    └── Chunk was wrong → Corpus quality / chunking problem

------------------------------------------------------------------------

## 6. Alerting

Set alerts on the metrics most likely to surface real problems early.

| Alert                | Condition                                   | Severity |
|----------------------|---------------------------------------------|----------|
| High error rate      | Error rate \> 1% over 5 min                 | Critical |
| Latency spike        | p95 \> 5 s for 10 min                       | Warning  |
| Faithfulness drop    | Rolling 1h average \< 0.75                  | Warning  |
| Zero-hit surge       | Zero-hit rate \> 15% over 30 min            | Warning  |
| Token budget overrun | Mean prompt tokens \> 90% of context window | Info     |

**Avoid alert fatigue:** Only alert on actionable signals. Log everything else for post-hoc analysis.

------------------------------------------------------------------------

## Summary

    Every RAG request → structured log record (query + chunks + answer + latencies)
                     → trace spans (embed, search, rerank, generate)
                     → sampled quality score (LLM judge)
                     → aggregated dashboards + alerts

The goal is that any production failure is diagnosable within minutes using logs alone, without needing to reproduce it from scratch.

------------------------------------------------------------------------

Back to top
