# Caching in RAG

Caching reduces latency and cost by avoiding redundant computation. RAG has three distinct caching layers: query-level, embedding-level, and document/chunk-level.

Author

Benedict Thekkel

## 1. Why Cache in RAG?

RAG pipelines have three expensive steps per request: 1. **Embedding** — encode the query via an embedding API (~10–30 ms, billable tokens) 2. **Generation** — call the LLM (~500–3000 ms, most expensive) 3. **Retrieval** — vector search (~5–50 ms, often the cheapest step)

For many applications, a significant fraction of queries are repeated or semantically very similar (FAQ-type questions, common lookups). Caching those saves real money and latency.

**Typical cache hit rates by domain:** - Internal knowledge-base chatbot: 30–60% (many repeated questions) - Customer support: 20–40% - General-purpose assistant: 5–15%

------------------------------------------------------------------------

## 2. Query-Level Caching (Answer Cache)

Cache the full answer for a query so the entire RAG pipeline is bypassed on a hit.

**Exact-match cache:**

``` python
import hashlib, redis

def rag_with_cache(query: str, tenant_id: str) -> str:
    cache_key = hashlib.sha256(f"{tenant_id}:{query}".encode()).hexdigest()
    
    cached = redis.get(cache_key)
    if cached:
        return cached.decode()
    
    answer = full_rag_pipeline(query, tenant_id)
    redis.setex(cache_key, 3600, answer)  # TTL: 1 hour
    return answer
```

**Limitations of exact-match:** - “What is our return policy?” and “What’s the return policy?” are different keys → low hit rate - Solution: **semantic cache** (see section 3)

**When to use exact-match:** high-traffic, predictable queries (pre-generated FAQ responses, report summaries).

------------------------------------------------------------------------

## 3. Semantic Cache

A semantic cache matches queries by meaning (embedding similarity) rather than exact text, dramatically increasing hit rate.

**How it works:** 1. Embed the incoming query 2. Search a “cache index” of previously answered queries 3. If cosine similarity \> threshold (e.g. 0.95), return the cached answer 4. Otherwise, run the full pipeline and store the result in the cache index

``` python
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_cache_lookup(query: str, threshold=0.95):
    query_vec = encoder.encode(query)
    results = cache_index.search(query_vec, top_k=1)
    
    if results and results[0].score >= threshold:
        return results[0].metadata["cached_answer"]
    return None
```

**Threshold tuning:** - High threshold (0.97+): conservative, fewer false hits, lower hit rate - Low threshold (0.90): aggressive, more hits but risk of returning wrong answers

**Tools:** GPTCache, LangChain `SemanticSimilarityExampleSelector`, custom Faiss/Qdrant cache index.

------------------------------------------------------------------------

## 4. Embedding Cache

Caching query embeddings avoids calling the embedding API for repeated or near-identical queries.

**Simple TTL-based embedding cache:**

``` python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=10_000)
def embed_query_cached(query: str) -> list[float]:
    return embedding_api.embed(query)
```

For multi-process / distributed systems, use Redis:

``` python
def embed_with_redis_cache(query: str) -> list[float]:
    key = f"embed:{hashlib.sha256(query.encode()).hexdigest()}"
    cached = redis.get(key)
    if cached:
        return json.loads(cached)
    vec = embedding_api.embed(query)
    redis.setex(key, 86400, json.dumps(vec))  # 24h TTL
    return vec
```

**Cost impact:** At \$0.02 per million tokens, embedding queries is cheap but adds ~20 ms latency. For high-frequency queries, caching eliminates this latency entirely.

------------------------------------------------------------------------

## 5. Document / Chunk Cache

Cache the processed representation of documents so re-indexing doesn’t re-embed unchanged content.

**Ingestion-time chunk cache:**

``` python
def ingest_with_chunk_cache(doc_id: str, content: str):
    checksum = sha256(content)
    
    # Check if embeddings already exist for this exact content
    cached_embeddings = chunk_embed_cache.get(checksum)
    if cached_embeddings:
        # Re-use stored embeddings, just upsert with updated metadata
        vector_index.upsert(cached_embeddings)
        return
    
    # Compute from scratch
    chunks = chunk(content)
    embeddings = embed_batch(chunks)
    chunk_embed_cache.set(checksum, embeddings, ttl=30_days)
    vector_index.upsert(embeddings)
```

**Benefit:** When a document is re-ingested (e.g., only metadata changed, not content), skip re-embedding. Saves significant cost for large corpora.

------------------------------------------------------------------------

## 6. Cache Invalidation

Caches must be invalidated when the underlying data changes.

**Invalidation triggers per layer:**

| Cache Layer | Invalidation Trigger |
|----|----|
| Answer cache | Source document updated → delete cached answers for affected queries |
| Semantic cache | Hard to invalidate selectively → use short TTL (minutes to hours) |
| Embedding cache | Query embeddings don’t expire (embeddings are deterministic) |
| Chunk embed cache | Content checksum changes → cache miss → recompute |

**Practical TTL defaults:**

| Layer             | Recommended TTL                              |
|-------------------|----------------------------------------------|
| Answer (exact)    | 1–24 hours (depends on doc update frequency) |
| Answer (semantic) | 15–60 minutes                                |
| Query embedding   | 24 hours                                     |
| Chunk embedding   | 30 days                                      |

**Access control and caching:** Never share cached answers across tenants. Cache keys must include `tenant_id` (and user role if fine-grained ACLs apply).

------------------------------------------------------------------------

## 7. Caching and Streaming

Answer caching is incompatible with streaming responses — you can’t cache a stream mid-flight.

**Solutions:** - **Buffer then cache**: collect the full stream into a string, cache it, then return it (lose streaming UX on first call but serve cached responses instantly) - **Cache at the retrieval level only**: cache the retrieved chunks, still stream the generation - **Async populate**: stream the response to the user, write to cache in the background after the stream completes

``` python
async def rag_stream_with_cache(query):
    cached = await cache.get(query)
    if cached:
        yield cached  # Return full cached answer as one chunk
        return

    full_answer = []
    async for token in llm_stream(query):
        full_answer.append(token)
        yield token

    # Cache asynchronously after streaming completes
    asyncio.create_task(cache.set(query, "".join(full_answer)))
```

------------------------------------------------------------------------

## Summary

| Cache Layer | What Is Cached | Saves | Key Challenge |
|----|----|----|----|
| Answer (exact) | Full answer keyed by query hash | LLM call + retrieval | Low hit rate |
| Answer (semantic) | Full answer keyed by similar query | LLM call + retrieval | Threshold tuning |
| Embedding | Query vector | Embedding API call | N/A (deterministic) |
| Chunk embedding | Document chunk vectors | Re-embedding on ingest | Invalidation on content change |

**Most impactful:** Semantic answer cache for high-traffic FAQ-like RAG systems. Start there before optimising other layers.

------------------------------------------------------------------------

Back to top
