# Embedding for RAG

Convert text into dense vectors that capture semantic meaning. Similar meaning = vectors close together in space. Retrieval = find vectors closest to the query vector.

Author

Benedict Thekkel

## Key dimensions to evaluate a model on

| Dimension | What it means |
|----|----|
| **Dimensionality** | Vector size — higher = more expressive, more storage |
| **Max tokens** | Hard limit on input length — chunks exceeding this get truncated silently |
| **Speed** | Tokens/sec on your hardware |
| **Quality** | MTEB benchmark score |
| **Language support** | Multilingual or English-only |

------------------------------------------------------------------------

## Recommended OSS models

### Best general-purpose default

**`BAAI/bge-small-en-v1.5`** — fast, small, punches above its weight

    Dimensions: 384
    Max tokens: 512
    MTEB score: ~63
    Size: ~130MB

**`BAAI/bge-base-en-v1.5`** — step up in quality, still fast

    Dimensions: 768
    Max tokens: 512
    MTEB score: ~64.2
    Size: ~440MB

**`BAAI/bge-large-en-v1.5`** — best quality in the BGE family

    Dimensions: 1024
    Max tokens: 512
    MTEB score: ~64.6
    Size: ~1.3GB

### Best for production quality

**`BAAI/bge-m3`** — multilingual, long context, hybrid dense+sparse in one model

    Dimensions: 1024
    Max tokens: 8192
    MTEB score: ~65+
    Size: ~2.3GB

### Lightweight / edge

**`sentence-transformers/all-MiniLM-L6-v2`** — very fast, widely used baseline

    Dimensions: 384
    Max tokens: 256
    MTEB score: ~56.3
    Size: ~90MB

### Long document embedding

**`nomic-ai/nomic-embed-text-v1.5`** — 8192 token context, Apache 2.0 licensed

    Dimensions: 768
    Max tokens: 8192
    MTEB score: ~62.4
    Size: ~275MB

------------------------------------------------------------------------

## Production recommendation

> Use **`BAAI/bge-base-en-v1.5`** as your default. Upgrade to **`bge-m3`** if you need long context or multilingual. Use **`bge-small`** if latency or memory is constrained.

------------------------------------------------------------------------

## Core setup with `sentence-transformers`

``` python
pip install sentence-transformers
```

``` python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Single
embedding = model.encode("What is the recovery time after ACL surgery?")

# Batch — always prefer batch over looping
embeddings = model.encode(
    ["chunk one text", "chunk two text", "chunk three text"],
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True,  # required for cosine similarity with BGE
)
```

------------------------------------------------------------------------

## BGE-specific: instruction prefix

BGE models expect a prefix on **queries** (not documents):

``` python
query = "Represent this sentence for searching relevant passages: What is the recovery time?"

query_embedding = model.encode(query, normalize_embeddings=True)

# Documents — no prefix
doc_embeddings = model.encode(chunks, normalize_embeddings=True, batch_size=32)
```

------------------------------------------------------------------------

## Batched indexing pipeline

``` python
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Generator

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def batch(iterable: list, size: int) -> Generator:
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

def embed_chunks(chunks: list[dict], batch_size: int = 64) -> list[dict]:
    texts = [c["content"] for c in chunks]
    all_embeddings = []

    for text_batch in batch(texts, batch_size):
        vecs = model.encode(
            text_batch,
            normalize_embeddings=True,
            batch_size=batch_size,
        )
        all_embeddings.extend(vecs.tolist())

    for chunk, vec in zip(chunks, all_embeddings):
        chunk["embedding"] = vec

    return chunks
```

------------------------------------------------------------------------

## GPU acceleration

``` python
import torch
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)

# On GPU, batch_size can go much higher
embeddings = model.encode(texts, batch_size=256, normalize_embeddings=True)
```

------------------------------------------------------------------------

## Storing in pgvector (Django stack)

``` python
# migration
from django.db import migrations
import pgvector.django

class Migration(migrations.Migration):
    operations = [
        migrations.RunSQL("CREATE EXTENSION IF NOT EXISTS vector"),
        pgvector.django.VectorExtension(),
    ]
```

``` python
# models.py
from django.db import models
from pgvector.django import VectorField

class DocumentChunk(models.Model):
    document_id = models.UUIDField()
    content = models.TextField()
    embedding = VectorField(dimensions=768)  # match your model
    embedding_model = models.CharField(max_length=100)
    chunk_strategy = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
```

``` python
# indexing
from pgvector.django import CosineDistance

def index_chunks(chunks: list[dict]):
    embedded = embed_chunks(chunks)
    objs = [
        DocumentChunk(
            document_id=c["document_id"],
            content=c["content"],
            embedding=c["embedding"],
            embedding_model="bge-base-en-v1.5",
            chunk_strategy="recursive_512_50",
        )
        for c in embedded
    ]
    DocumentChunk.objects.bulk_create(objs, batch_size=200)

# querying
def retrieve(query: str, top_k: int = 5):
    q_vec = model.encode(
        f"Represent this sentence for searching relevant passages: {query}",
        normalize_embeddings=True,
    ).tolist()

    return (
        DocumentChunk.objects
        .annotate(distance=CosineDistance("embedding", q_vec))
        .order_by("distance")[:top_k]
    )
```

------------------------------------------------------------------------

## Model versioning — critical for prod

When you update the embedding model, all existing vectors are incompatible. You need to track this:

``` python
EMBEDDING_MODEL_VERSION = "bge-base-en-v1.5"

# On index, always write the model name to the chunk row.
# On model upgrade, mark all existing chunks stale and re-index:

DocumentChunk.objects.exclude(
    embedding_model=EMBEDDING_MODEL_VERSION
).update(index_status="stale")
```

Then re-index stale chunks in a background Celery task — never block a request on re-indexing.

------------------------------------------------------------------------

## Serving the model in prod

Don’t load the model per-request. Load once at startup:

``` python
# embeddings.py — module-level singleton
from sentence_transformers import SentenceTransformer
import torch

_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(
            "BAAI/bge-base-en-v1.5",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    return _model

def embed(texts: list[str]) -> list[list[float]]:
    m = get_model()
    return m.encode(texts, normalize_embeddings=True, batch_size=64).tolist()
```

For high-throughput prod, run the model as a separate microservice with **`infinity-emb`**:

``` bash
pip install infinity-emb
infinity_emb v2 --model-id BAAI/bge-base-en-v1.5 --port 7997
```

``` python
import httpx

def embed_remote(texts: list[str]) -> list[list[float]]:
    resp = httpx.post(
        "http://localhost:7997/embeddings",
        json={"input": texts, "model": "BAAI/bge-base-en-v1.5"},
    )
    return [d["embedding"] for d in resp.json()["data"]]
```

------------------------------------------------------------------------

## Common failure modes

| Problem | Fix |
|----|----|
| Silent truncation | Check `token_count` before embedding — warn or split if over model max |
| Wrong similarity scores | Ensure `normalize_embeddings=True` for cosine similarity |
| Query/doc mismatch | Always apply BGE instruction prefix to queries only |
| Model loaded per request | Singleton pattern or separate embedding service |
| Stale vectors after model upgrade | Track `embedding_model` per chunk, re-index on change |
| OOM on large batch | Reduce `batch_size`, use generator batching |

------------------------------------------------------------------------

Back to top
