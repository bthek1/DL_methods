# Retrieval for RAG — Everything You Need to Know (OSS Only)

Given a query, finds the most relevant chunks from your index. Quality here directly determines ceiling quality of the final answer — the LLM can’t reason over what retrieval didn’t surface.

Author

Benedict Thekkel

## Retrieval strategies

### 1. Dense retrieval (semantic)

Embed the query, find nearest vectors by cosine similarity. Catches semantic matches even with different wording.

    "patient recovery time" → matches "how long until discharge"

``` python
from .embeddings import embed
from .models import DocumentChunk
from pgvector.django import CosineDistance

def dense_retrieve(query: str, tenant_id: str, top_k: int = 10):
    q_vec = embed([
        f"Represent this sentence for searching relevant passages: {query}"
    ])[0]

    return list(
        DocumentChunk.objects
        .filter(tenant_id=tenant_id, index_status="indexed")
        .annotate(distance=CosineDistance("embedding", q_vec))
        .order_by("distance")
        [:top_k]
    )
```

------------------------------------------------------------------------

### 2. Sparse retrieval (BM25 / lexical)

Keyword-based scoring. Exact term matches, no embeddings. Catches rare terms, codes, IDs, and proper nouns that dense retrieval misses.

    "ICD-10 code M54.5" → dense may miss, BM25 nails it

**BM25 with `rank_bm25`:**

``` python
pip install rank-bm25
```

``` python
from rank_bm25 import BM25Okapi
import re

def tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", "", text.lower()).split()

# Build index (do this once, cache or rebuild on index update)
def build_bm25_index(chunks: list[str]) -> BM25Okapi:
    return BM25Okapi([tokenize(c) for c in chunks])

def bm25_retrieve(
    query: str,
    chunks: list[dict],
    bm25: BM25Okapi,
    top_k: int = 10,
) -> list[dict]:
    scores = bm25.get_scores(tokenize(query))
    top_indices = scores.argsort()[::-1][:top_k]
    return [
        {**chunks[i], "bm25_score": float(scores[i])}
        for i in top_indices
        if scores[i] > 0
    ]
```

**BM25 in Postgres with `pg_bm25` (Paradedb) — prod preferred:**

``` sql
-- Install extension
CREATE EXTENSION IF NOT EXISTS pg_bm25;

-- Create BM25 index on content
CALL paradedb.create_bm25(
    index_name => 'chunks_bm25',
    table_name => 'myapp_documentchunk',
    id_field   => 'id',
    text_fields => paradedb.field('content')
);
```

``` python
from django.db import connection

def bm25_retrieve_pg(query: str, tenant_id: str, top_k: int = 10):
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT id, content, paradedb.score(id) AS bm25_score
            FROM myapp_documentchunk
            WHERE tenant_id = %s
              AND index_status = 'indexed'
              AND content @@@ %s
            ORDER BY bm25_score DESC
            LIMIT %s
        """, [tenant_id, query, top_k])
        rows = cursor.fetchall()

    return [{"id": r[0], "content": r[1], "bm25_score": r[2]} for r in rows]
```

------------------------------------------------------------------------

### 3. Hybrid retrieval — production default

Combine dense + sparse. Dense handles semantics, sparse handles keywords. Together they cover each other’s blind spots.

**Reciprocal Rank Fusion (RRF)** — merges ranked lists without needing normalised scores:

``` python
def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    id_key: str = "id",
    k: int = 60,
) -> list[dict]:
    """
    k=60 is the standard default — dampens the impact of top ranks.
    Lower k = top ranks dominate more.
    """
    scores: dict = {}
    all_items: dict = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list):
            item_id = str(item[id_key])
            scores[item_id] = scores.get(item_id, 0) + 1 / (k + rank + 1)
            all_items[item_id] = item

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [
        {**all_items[id_], "rrf_score": scores[id_]}
        for id_ in sorted_ids
    ]


def hybrid_retrieve(
    query: str,
    tenant_id: str,
    chunks: list[dict],
    bm25: BM25Okapi,
    top_k: int = 10,
) -> list[dict]:
    dense_results  = dense_retrieve(query, tenant_id, top_k=top_k * 2)
    sparse_results = bm25_retrieve(query, chunks, bm25, top_k=top_k * 2)

    # Normalise to dicts with consistent id key
    dense_dicts  = [{"id": str(c.id), "content": c.content} for c in dense_results]
    sparse_dicts = [{"id": str(c["id"]), "content": c["content"]} for c in sparse_results]

    fused = reciprocal_rank_fusion([dense_dicts, sparse_dicts])
    return fused[:top_k]
```

------------------------------------------------------------------------

### 4. Metadata filtering

Always filter on tenant, status, and any available scoping field **before** vector search. Post-filtering kills recall.

``` python
def retrieve_with_filters(
    query: str,
    tenant_id: str,
    filters: dict,
    top_k: int = 10,
):
    q_vec = embed([
        f"Represent this sentence for searching relevant passages: {query}"
    ])[0]

    qs = DocumentChunk.objects.filter(
        tenant_id=tenant_id,
        index_status="indexed",
        **filters,        # e.g. {"source_type": "pdf", "document_id": some_id}
    )

    return list(
        qs
        .annotate(distance=CosineDistance("embedding", q_vec))
        .order_by("distance")
        [:top_k]
    )
```

------------------------------------------------------------------------

### 5. Multi-query retrieval

Generate multiple phrasings of the query, retrieve for each, fuse results. Increases recall on ambiguous or underspecified queries.

``` python
import httpx

def expand_query(query: str) -> list[str]:
    """Use a local LLM to generate query variants."""
    resp = httpx.post(
        "http://localhost:11434/api/generate",   # Ollama
        json={
            "model": "mistral",
            "prompt": f"""Generate 3 different search queries for the following question.
Return only the queries, one per line, no numbering or explanation.

Question: {query}""",
            "stream": False,
        }
    )
    variants = resp.json()["response"].strip().split("\n")
    return [query] + [v.strip() for v in variants if v.strip()]


def multi_query_retrieve(
    query: str,
    tenant_id: str,
    top_k: int = 10,
) -> list[dict]:
    queries   = expand_query(query)
    all_lists = []

    for q in queries:
        results = dense_retrieve(q, tenant_id, top_k=top_k)
        all_lists.append([{"id": str(c.id), "content": c.content} for c in results])

    fused = reciprocal_rank_fusion(all_lists)
    return fused[:top_k]
```

------------------------------------------------------------------------

### 6. HyDE (Hypothetical Document Embeddings)

Generate a fake answer to the query, embed that instead of the raw query. The fake answer lives in “answer space” which is closer to real document embeddings than question space.

``` python
def generate_hypothetical_doc(query: str) -> str:
    resp = httpx.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": f"""Write a short factual passage that would answer this question.
Write only the passage, no preamble.

Question: {query}""",
            "stream": False,
        }
    )
    return resp.json()["response"].strip()


def hyde_retrieve(
    query: str,
    tenant_id: str,
    top_k: int = 10,
) -> list[dict]:
    hypo_doc = generate_hypothetical_doc(query)

    # Embed the hypothetical doc — no instruction prefix, it's a document
    hypo_vec = embed([hypo_doc])[0]

    return list(
        DocumentChunk.objects
        .filter(tenant_id=tenant_id, index_status="indexed")
        .annotate(distance=CosineDistance("embedding", hypo_vec))
        .order_by("distance")
        [:top_k]
    )
```

------------------------------------------------------------------------

### 7. Contextual compression

After retrieving chunks, extract only the sentences relevant to the query. Reduces noise passed to the LLM.

``` python
def compress_chunks(query: str, chunks: list[str]) -> list[str]:
    compressed = []

    for chunk in chunks:
        resp = httpx.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": f"""Extract only the sentences from the passage below that are
relevant to the question. Return only the extracted text, nothing else.
If nothing is relevant, return an empty string.

Question: {query}

Passage:
{chunk}""",
                "stream": False,
            }
        )
        result = resp.json()["response"].strip()
        if result:
            compressed.append(result)

    return compressed
```

------------------------------------------------------------------------

## Putting it all together — production retrieval pipeline

``` python
from dataclasses import dataclass
from typing import Literal

@dataclass
class RetrievalConfig:
    strategy:    Literal["dense", "sparse", "hybrid", "hyde", "multi_query"] = "hybrid"
    top_k:       int  = 10
    compress:    bool = False
    filters:     dict = None


def retrieve(
    query:     str,
    tenant_id: str,
    config:    RetrievalConfig = RetrievalConfig(),
) -> list[str]:

    match config.strategy:
        case "dense":
            results = dense_retrieve(query, tenant_id, config.top_k)
            chunks  = [c.content for c in results]

        case "hybrid":
            all_chunks  = list(
                DocumentChunk.objects
                .filter(tenant_id=tenant_id, index_status="indexed")
                .values("id", "content")
            )
            bm25        = build_bm25_index([c["content"] for c in all_chunks])
            results     = hybrid_retrieve(query, tenant_id, all_chunks, bm25, config.top_k)
            chunks      = [r["content"] for r in results]

        case "hyde":
            results = hyde_retrieve(query, tenant_id, config.top_k)
            chunks  = [c.content for c in results]

        case "multi_query":
            results = multi_query_retrieve(query, tenant_id, config.top_k)
            chunks  = [r["content"] for r in results]

    if config.compress:
        chunks = compress_chunks(query, chunks)

    return chunks
```

------------------------------------------------------------------------

## top_k guidelines

| Scenario            | top_k                            |
|---------------------|----------------------------------|
| Precise factual Q&A | 3–5                              |
| General Q&A         | 5–10                             |
| Before re-ranking   | 20–50                            |
| Multi-query / HyDE  | 10–20 per query, fuse down to 10 |

Retrieve more than you need before re-ranking, then cut down. Cheap retrieval, expensive re-ranking.

------------------------------------------------------------------------

## Common failure modes

| Problem | Fix |
|----|----|
| Correct chunk not retrieved | Switch to hybrid, add multi-query, check chunk boundaries |
| Too many irrelevant chunks | Add metadata filters, lower top_k, add re-ranker |
| Rare terms / IDs not found | Add BM25 — dense embeddings blur rare tokens |
| Slow retrieval on filtered query | Add composite DB index on filter fields, increase `ef_search` |
| Same chunk retrieved multiple times | Deduplicate by `content_hash` after fusion |
| Recall drops on new documents | Check `index_status="indexed"` — new chunks may still be `pending` |

------------------------------------------------------------------------

Back to top
