# Indexing for RAG — Everything You Need to Know (OSS Only)

Takes your embedded chunks and organises them into a structure that makes approximate nearest-neighbour (ANN) search fast at query time. Without an index, every query would do a full linear scan across all vectors — fine at 10k chunks, unusable at 1M+.

Author

Benedict Thekkel

## Two layers of indexing

1.  **Vector index** — the ANN structure for similarity search
2.  **Metadata index** — standard DB indexes (B-tree, GIN) for filtering before/after vector search

Both matter. Missing metadata indexes kills performance on filtered queries just as much as missing vector indexes.

------------------------------------------------------------------------

## ANN algorithms

### HNSW (Hierarchical Navigable Small World)

Builds a multi-layer graph. Fast queries, high recall, large memory footprint. **Default choice for most prod workloads.**

    Query time:  O(log n)
    Build time:  Slow
    Memory:      High (graph stored in RAM)
    Recall:      Very high (~95-99%)

### IVF (Inverted File Index)

Clusters vectors into buckets, searches only nearby buckets. Lower memory, faster build, slightly lower recall.

    Query time:  Fast (searches subset of clusters)
    Build time:  Fast (needs training on sample data)
    Memory:      Lower than HNSW
    Recall:      Good (~90-95% with enough nprobe)

### Flat (brute force)

Exact search, no approximation. Only viable under ~100k vectors.

    Query time:  O(n) — linear scan
    Recall:      100% exact
    Memory:      Low (just the vectors)

------------------------------------------------------------------------

## OSS vector stores compared

| types | pgvector | Qdrant | Weaviate | Chroma | Milvus |
|----|----|----|----|----|----|
| **Best for** | Existing Postgres stack | Standalone prod | Built-in hybrid | Dev/prototyping | Massive scale |
| **HNSW** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **IVF** | ✓ | ✗ | ✗ | ✗ | ✓ |
| **Hybrid search** | Manual | ✓ | ✓ | ✗ | ✓ |
| **Metadata filtering** | SQL | ✓ | ✓ | Basic | ✓ |
| **Multi-tenancy** | SQL | Collections/payload | Multi-tenancy API | ✗ | Partitions |
| **Ops overhead** | None (already Postgres) | Low | Medium | None | High |

------------------------------------------------------------------------

## Production recommendation

> **pgvector if you’re already on Postgres.** Zero extra infra, SQL filtering, HNSW support since 0.5.0. Use **Qdrant** if you need a dedicated vector store with built-in hybrid search or more tuning control.

------------------------------------------------------------------------

## pgvector — full production setup

### Install

``` bash
pip install pgvector django psycopg2-binary
```

``` sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Django model

``` python
# models.py
from django.db import models
from pgvector.django import VectorField, HnswIndex, IvfflatIndex

class DocumentChunk(models.Model):
    document_id     = models.UUIDField(db_index=True)
    tenant_id       = models.UUIDField(db_index=True)
    content         = models.TextField()
    embedding       = VectorField(dimensions=768)
    embedding_model = models.CharField(max_length=100)
    chunk_strategy  = models.CharField(max_length=100)
    index_status    = models.CharField(
        max_length=20,
        default="pending",
        choices=[
            ("pending", "Pending"),
            ("indexed", "Indexed"),
            ("stale",   "Stale"),
            ("failed",  "Failed"),
        ],
        db_index=True,
    )
    token_count  = models.IntegerField()
    source_path  = models.TextField()
    page_number  = models.IntegerField(null=True)
    content_hash = models.CharField(max_length=64, db_index=True)
    created_at   = models.DateTimeField(auto_now_add=True)
    updated_at   = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            # Vector index — HNSW for prod
            HnswIndex(
                name="chunk_embedding_hnsw",
                fields=["embedding"],
                m=16,                    # neighbours per layer — 16 is default, increase for higher recall
                ef_construction=64,      # build-time search width — higher = better index, slower build
                opclasses=["vector_cosine_ops"],
            ),
            # Metadata indexes for filtered queries
            models.Index(fields=["tenant_id", "index_status"]),
            models.Index(fields=["document_id"]),
        ]
```

### Migration

``` python
# migrations/0001_initial.py
from django.db import migrations
from pgvector.django import VectorExtension

class Migration(migrations.Migration):
    operations = [
        VectorExtension(),
        # ... rest of migration
    ]
```

### HNSW tuning params

``` python
HnswIndex(
    m=16,               # edges per node per layer
                        # 8–16 for most workloads
                        # 32–64 for very high recall requirements
                        # higher = more memory, slower build, better recall

    ef_construction=64, # beam width during build
                        # 64–128 for prod
                        # higher = slower build, better index quality
                        # no effect on query speed
)
```

At query time, control recall vs speed tradeoff:

``` python
from django.db import connection

# Set per-session before querying
with connection.cursor() as cursor:
    cursor.execute("SET hnsw.ef_search = 100")
    # ef_search: how many candidates to explore at query time
    # default 40 — increase for higher recall at cost of latency
    # 100–200 for prod quality
```

### Bulk indexing

``` python
from pgvector.django import CosineDistance
from .models import DocumentChunk
from .embeddings import embed

def bulk_index(chunks: list[dict]):
    objs = []
    texts = [c["content"] for c in chunks]
    embeddings = embed(texts)  # your embed() singleton from previous section

    for chunk, vec in zip(chunks, embeddings):
        objs.append(DocumentChunk(
            document_id     = chunk["document_id"],
            tenant_id       = chunk["tenant_id"],
            content         = chunk["content"],
            embedding       = vec,
            embedding_model = "bge-base-en-v1.5",
            chunk_strategy  = "recursive_512_50",
            token_count     = chunk["token_count"],
            source_path     = chunk["source_path"],
            content_hash    = chunk["content_hash"],
            index_status    = "indexed",
        ))

    DocumentChunk.objects.bulk_create(
        objs,
        batch_size=200,
        update_conflicts=True,                           # upsert
        update_fields=["embedding", "content", "content_hash", "index_status", "updated_at"],
        unique_fields=["document_id", "chunk_index"],   # requires unique_together on model
    )
```

### Querying with metadata pre-filter

``` python
def retrieve(
    query: str,
    tenant_id: str,
    top_k: int = 5,
    source_path: str | None = None,
) -> list[DocumentChunk]:
    q_vec = embed([f"Represent this sentence for searching relevant passages: {query}"])[0]

    qs = (
        DocumentChunk.objects
        .filter(tenant_id=tenant_id, index_status="indexed")  # metadata pre-filter
        .annotate(distance=CosineDistance("embedding", q_vec))
        .order_by("distance")
        [:top_k]
    )

    if source_path:
        qs = qs.filter(source_path=source_path)

    return list(qs)
```

------------------------------------------------------------------------

## Qdrant — production setup

### Install + run

``` bash
pip install qdrant-client
docker run -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
```

### Create collection with HNSW

``` python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, HnswConfigDiff,
    OptimizersConfigDiff, PayloadSchemaType,
)

client = QdrantClient(host="localhost", port=6333)

client.create_collection(
    collection_name="document_chunks",
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE,
    ),
    hnsw_config=HnswConfigDiff(
        m=16,
        ef_construct=64,
        full_scan_threshold=10_000,  # fall back to flat below this count
    ),
    optimizers_config=OptimizersConfigDiff(
        indexing_threshold=20_000,   # build HNSW index after N vectors
    ),
)

# Create payload indexes for fast filtering
client.create_payload_index(
    collection_name="document_chunks",
    field_name="tenant_id",
    field_schema=PayloadSchemaType.KEYWORD,
)

client.create_payload_index(
    collection_name="document_chunks",
    field_name="index_status",
    field_schema=PayloadSchemaType.KEYWORD,
)
```

### Bulk upsert

``` python
from qdrant_client.models import PointStruct
import uuid

def bulk_index_qdrant(chunks: list[dict]):
    texts = [c["content"] for c in chunks]
    embeddings = embed(texts)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={
                "document_id":     str(chunk["document_id"]),
                "tenant_id":       str(chunk["tenant_id"]),
                "content":         chunk["content"],
                "source_path":     chunk["source_path"],
                "chunk_strategy":  "recursive_512_50",
                "embedding_model": "bge-base-en-v1.5",
                "index_status":    "indexed",
            }
        )
        for chunk, vec in zip(chunks, embeddings)
    ]

    client.upsert(
        collection_name="document_chunks",
        points=points,
        wait=True,  # wait for indexing to complete — set False for async
    )
```

### Querying with filter

``` python
from qdrant_client.models import Filter, FieldCondition, MatchValue

def retrieve_qdrant(query: str, tenant_id: str, top_k: int = 5):
    q_vec = embed([f"Represent this sentence for searching relevant passages: {query}"])[0]

    results = client.search(
        collection_name="document_chunks",
        query_vector=q_vec,
        query_filter=Filter(
            must=[
                FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
                FieldCondition(key="index_status", match=MatchValue(value="indexed")),
            ]
        ),
        limit=top_k,
        with_payload=True,
    )

    return [r.payload for r in results]
```

------------------------------------------------------------------------

## Index lifecycle management

Track every chunk’s index state and handle re-indexing on document change:

``` python
import hashlib

def compute_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()

def mark_stale_on_update(document_id: str, new_chunks: list[dict]):
    """Mark existing chunks stale if content has changed."""
    existing = DocumentChunk.objects.filter(document_id=document_id)
    new_hashes = {c["chunk_index"]: compute_hash(c["content"]) for c in new_chunks}

    to_stale = [
        c.id for c in existing
        if new_hashes.get(c.chunk_index) != c.content_hash
    ]

    DocumentChunk.objects.filter(id__in=to_stale).update(index_status="stale")

def reindex_stale(batch_size: int = 100):
    """Celery task — pick up stale chunks and re-embed."""
    stale = DocumentChunk.objects.filter(index_status="stale")[:batch_size]
    if not stale:
        return

    texts = [c.content for c in stale]
    embeddings = embed(texts)

    for chunk, vec in zip(stale, embeddings):
        chunk.embedding       = vec
        chunk.embedding_model = "bge-base-en-v1.5"
        chunk.index_status    = "indexed"
        chunk.content_hash    = compute_hash(chunk.content)

    DocumentChunk.objects.bulk_update(
        stale,
        ["embedding", "embedding_model", "index_status", "content_hash", "updated_at"],
    )
```

------------------------------------------------------------------------

## Common failure modes

| Problem | Fix |
|----|----|
| Full scan instead of HNSW | Check `EXPLAIN` — index only used above `min_rows` threshold in pgvector |
| Low recall on filtered queries | Pre-filter reduces candidate pool — increase `ef_search` or `top_k` |
| Index not built yet | pgvector defers HNSW build until row count threshold — insert dummy rows in dev or set threshold lower |
| Stale vectors serving wrong results | Always filter `index_status="indexed"` in queries |
| Dimension mismatch on model upgrade | New model → new `VectorField(dimensions=N)` column, dual-write during migration |
| Slow bulk insert | Use `bulk_create` with `batch_size=200`, disable autocommit, insert embeddings as lists not numpy arrays |

------------------------------------------------------------------------

Back to top
