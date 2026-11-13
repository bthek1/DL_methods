# Re-ranking for RAG — Everything You Need to Know (OSS Only)

Retrieval (dense/sparse/hybrid) is fast but imprecise — it uses approximate similarity over a large candidate pool. Re-ranking takes the top-k retrieval results and scores each one more carefully against the query using a cross-encoder. Much slower per chunk, but far more accurate.

Author

Benedict Thekkel

## Bi-encoder vs cross-encoder

| \- | Bi-encoder (retrieval) | Cross-encoder (re-ranking) |
|----|----|----|
| **How** | Embed query and doc independently | Score query+doc together |
| **Speed** | Fast — vectors precomputed | Slow — runs inference per pair |
| **Quality** | Good | Much better |
| **Scales to** | Millions of docs | Tens of docs |
| **Use for** | First-stage retrieval | Second-stage re-ranking |

Cross-encoders see the query and document together, so they understand the relationship between them rather than just measuring vector proximity.

------------------------------------------------------------------------

## OSS re-ranking models

### Best general-purpose

**`BAAI/bge-reranker-base`** — fast, strong, 512 token limit

    Size:        ~280MB
    Max tokens:  512 (query + doc combined)
    Quality:     Strong
    Speed:       Fast

**`BAAI/bge-reranker-large`** — best quality in BGE family

    Size:        ~1.3GB
    Max tokens:  512
    Quality:     Best
    Speed:       Slower

**`BAAI/bge-reranker-v2-m3`** — multilingual, longer context

    Size:        ~2.3GB
    Max tokens:  8192
    Quality:     Excellent
    Speed:       Slow

### Lightweight

**`cross-encoder/ms-marco-MiniLM-L-6-v2`** — tiny, fast, good baseline

    Size:        ~80MB
    Max tokens:  512
    Quality:     Good
    Speed:       Very fast

**`cross-encoder/ms-marco-MiniLM-L-12-v2`** — step up in quality

    Size:        ~130MB
    Max tokens:  512
    Quality:     Better
    Speed:       Fast

------------------------------------------------------------------------

## Production recommendation

> Use **`BAAI/bge-reranker-base`** as default — best speed/quality tradeoff. Use **`bge-reranker-v2-m3`** if you need long context or multilingual. Use **`ms-marco-MiniLM-L-6-v2`** if latency is critical.

------------------------------------------------------------------------

## Core setup

``` bash
pip install sentence-transformers
```

``` python
# reranker.py — singleton pattern, same as embeddings
from sentence_transformers import CrossEncoder
import torch

_reranker = None

def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(
            "BAAI/bge-reranker-base",
            device="cuda" if torch.cuda.is_available() else "cpu",
            max_length=512,
        )
    return _reranker


def rerank(query: str, chunks: list[str], top_k: int = 5) -> list[dict]:
    reranker = get_reranker()

    pairs  = [(query, chunk) for chunk in chunks]
    scores = reranker.predict(pairs, batch_size=32)

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        {"content": chunk, "rerank_score": float(score)}
        for chunk, score in ranked[:top_k]
    ]
```

------------------------------------------------------------------------

## Re-ranking with full chunk metadata

In prod you need to preserve IDs and metadata through the re-ranking step:

``` python
def rerank_chunks(
    query:   str,
    chunks:  list[dict],   # dicts with at minimum {"id": ..., "content": ...}
    top_k:   int = 5,
) -> list[dict]:
    reranker = get_reranker()

    pairs  = [(query, c["content"]) for c in chunks]
    scores = reranker.predict(pairs, batch_size=32)

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        {**chunk, "rerank_score": float(score)}
        for chunk, score in ranked[:top_k]
    ]
```

------------------------------------------------------------------------

## Truncation — critical for 512 token models

Most re-rankers have a 512 token limit across query + document combined. Silent truncation destroys scoring accuracy on long chunks.

``` python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

def truncate_for_reranker(
    query:     str,
    chunk:     str,
    max_tokens: int = 512,
    query_budget: int = 128,   # reserve this many tokens for the query
) -> str:
    query_tokens = enc.encode(query)[:query_budget]
    chunk_budget = max_tokens - len(query_tokens) - 4  # 4 for sep tokens
    chunk_tokens = enc.encode(chunk)[:chunk_budget]
    return enc.decode(chunk_tokens)


def rerank_with_truncation(
    query:  str,
    chunks: list[dict],
    top_k:  int = 5,
) -> list[dict]:
    reranker = get_reranker()

    truncated_contents = [
        truncate_for_reranker(query, c["content"])
        for c in chunks
    ]

    pairs  = list(zip([query] * len(chunks), truncated_contents))
    scores = reranker.predict(pairs, batch_size=32)

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        {**chunk, "rerank_score": float(score)}
        for chunk, score in ranked[:top_k]
    ]
```

------------------------------------------------------------------------

## Score thresholding

Filter out chunks below a minimum relevance score. Prevents feeding obviously irrelevant content to the LLM.

``` python
def rerank_with_threshold(
    query:          str,
    chunks:         list[dict],
    top_k:          int   = 5,
    min_score:      float = 0.0,   # cross-encoder scores are not normalised
                                   # bge-reranker outputs raw logits — tune empirically
                                   # typically -10 to +10 range
                                   # 0.0 is a reasonable starting threshold
) -> list[dict]:
    results = rerank_with_truncation(query, chunks, top_k=top_k)
    return [r for r in results if r["rerank_score"] >= min_score]
```

Tune `min_score` by logging scores in dev against known relevant/irrelevant chunks for your domain. Don’t hardcode without measuring.

------------------------------------------------------------------------

## Async re-ranking for parallel queries

``` python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=2)

async def rerank_async(
    query:  str,
    chunks: list[dict],
    top_k:  int = 5,
) -> list[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        lambda: rerank_with_truncation(query, chunks, top_k),
    )
```

------------------------------------------------------------------------

## Full retrieval + re-ranking pipeline

``` python
from .retrieval import retrieve, RetrievalConfig
from .reranker  import rerank_with_truncation

def retrieve_and_rerank(
    query:     str,
    tenant_id: str,
    top_k:     int = 5,
    retrieval_k: int = 20,   # retrieve more, re-rank down to top_k
) -> list[dict]:

    # Stage 1 — broad retrieval
    config  = RetrievalConfig(strategy="hybrid", top_k=retrieval_k)
    chunks  = retrieve(query, tenant_id, config)

    if not chunks:
        return []

    # Stage 2 — precise re-ranking
    chunk_dicts = [{"id": str(i), "content": c} for i, c in enumerate(chunks)]
    reranked    = rerank_with_truncation(query, chunk_dicts, top_k=top_k)

    return reranked
```

------------------------------------------------------------------------

## When to skip re-ranking

Re-ranking adds ~50–200ms latency depending on hardware and chunk count. Skip it when:

- Latency SLA is very tight and retrieval quality is already good enough
- `retrieval_k` is already small (≤5) — not enough candidates to meaningfully re-rank
- The query is a simple keyword lookup — BM25 already handled it well

Add a flag to your pipeline config:

``` python
@dataclass
class PipelineConfig:
    retrieval_k: int  = 20
    rerank:      bool = True
    rerank_k:    int  = 5
    strategy:    str  = "hybrid"
    compress:    bool = False


def run_pipeline(
    query:     str,
    tenant_id: str,
    config:    PipelineConfig = PipelineConfig(),
) -> list[str]:

    retrieval_config = RetrievalConfig(
        strategy=config.strategy,
        top_k=config.retrieval_k,
    )
    chunks = retrieve(query, tenant_id, retrieval_config)

    if config.rerank and len(chunks) > config.rerank_k:
        chunk_dicts = [{"id": str(i), "content": c} for i, c in enumerate(chunks)]
        reranked    = rerank_with_truncation(query, chunk_dicts, top_k=config.rerank_k)
        chunks      = [r["content"] for r in reranked]

    return chunks
```

------------------------------------------------------------------------

## Serving the re-ranker in prod

Same rules as the embedding model — load once, don’t reload per request:

For high throughput, serve via **`infinity-emb`** which supports cross-encoders:

``` bash
infinity_emb v2 \
  --model-id BAAI/bge-reranker-base \
  --port 7998
```

``` python
import httpx

def rerank_remote(
    query:  str,
    chunks: list[str],
    top_k:  int = 5,
) -> list[dict]:
    resp = httpx.post(
        "http://localhost:7998/rerank",
        json={
            "query":     query,
            "documents": chunks,
            "model":     "BAAI/bge-reranker-base",
        },
        timeout=30.0,
    )
    results = resp.json()["results"]
    # results: [{"index": 0, "relevance_score": 0.92}, ...]

    ranked = sorted(results, key=lambda x: x["relevance_score"], reverse=True)

    return [
        {
            "content":       chunks[r["index"]],
            "rerank_score":  r["relevance_score"],
        }
        for r in ranked[:top_k]
    ]
```

------------------------------------------------------------------------

## Common failure modes

| Problem | Fix |
|----|----|
| Re-ranker scores all chunks low | Check truncation — long chunks losing key content |
| Latency too high | Reduce `retrieval_k`, use smaller model, serve with infinity-emb |
| Good chunks scored poorly | Chunk too long — truncated before relevant content |
| No score threshold set | Irrelevant chunks still passed to LLM — add `min_score` filter |
| Model loaded per request | Singleton or remote service — never reload per query |
| GPU OOM on large batch | Reduce `batch_size` in `predict()` call |

------------------------------------------------------------------------

Back to top
