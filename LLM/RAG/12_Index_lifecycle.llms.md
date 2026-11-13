# Index Lifecycle Management

A vector index is not static. Documents are added, updated, and deleted over time. Index lifecycle management covers incremental updates, stale chunk detection, and when and how to re-index.

Author

Benedict Thekkel

## 1. The Three Operations

Every index lifecycle strategy reduces to three operations:

| Operation | Trigger | Action |
|----|----|----|
| **Add** | New document ingested | Chunk → embed → upsert chunks into index |
| **Update** | Existing document changed | Delete old chunks by doc ID → re-chunk → embed → insert |
| **Delete** | Document removed or expired | Delete all chunks associated with that doc ID |

**Key design requirement:** each chunk must store its parent document ID as metadata so you can efficiently delete/update all chunks for a given document.

``` python
# Metadata on each chunk
{
  "chunk_id": "doc42_chunk3",
  "doc_id": "doc42",
  "source_url": "https://...",
  "last_modified": "2024-11-01T10:00:00Z",
  "checksum": "sha256:abc123..."
}
```

------------------------------------------------------------------------

## 2. Incremental Ingestion Pipeline

A full re-index on every change is expensive. Incremental pipelines process only changed documents.

**Pattern: change detection via checksum or `last_modified`**

``` python
def sync_document(doc_id: str, content: str, last_modified: str):
    existing = metadata_store.get(doc_id)

    new_checksum = sha256(content)

    if existing and existing["checksum"] == new_checksum:
        return  # No change — skip

    # Delete old chunks
    vector_index.delete(filter={"doc_id": doc_id})

    # Re-chunk, embed, and insert
    chunks = chunk(content)
    embeddings = embed_batch(chunks)
    vector_index.upsert([
        {"id": f"{doc_id}_chunk{i}", "vector": emb, "metadata": {"doc_id": doc_id, ...}}
        for i, emb in enumerate(embeddings)
    ])

    # Update metadata store
    metadata_store.set(doc_id, {"checksum": new_checksum, "last_modified": last_modified})
```

**Source connectors with built-in change detection:** Confluence, Notion, SharePoint, S3 event notifications, database CDC (Debezium).

------------------------------------------------------------------------

## 3. Stale Chunk Detection

Chunks become stale when the source document changes but the index hasn’t been updated.

**Detection strategies:**

| Strategy | How | Latency |
|----|----|----|
| **Periodic crawl** | Scheduler re-fetches all source URLs, compares checksums | Hours to days |
| **Webhook / event-driven** | Source system pushes change notifications | Near real-time |
| **TTL-based** | Chunks expire after N days and must be re-ingested | Guaranteed freshness bound |
| **Retrieval-time validation** | On retrieval, fetch source and verify chunk still exists | Real-time but adds latency |

**Best practice:** combine event-driven for high-change sources with periodic crawl as a safety net for missed events.

------------------------------------------------------------------------

## 4. Re-Indexing Triggers

A full re-index (re-embed and re-insert all documents) is expensive but sometimes unavoidable.

**When to trigger a full re-index:**

| Trigger | Reason |
|----|----|
| Embedding model upgrade | New model produces incompatible vector space |
| Chunking strategy change | Old chunks don’t reflect new boundaries |
| Metadata schema change | Old chunks are missing new required fields |
| Vector DB migration | Moving to a different database or collection |
| Corpus quality audit | Found widespread errors (broken PDFs, encoding issues) |

**Zero-downtime re-index pattern:**

    1. Create a new shadow index (new_index_v2)
    2. Ingest all documents into new_index_v2 while new_index_v1 serves traffic
    3. Once ingestion completes, run evaluation to confirm new index quality ≥ old
    4. Atomic alias swap: alias "production" → new_index_v2
    5. Delete old index after TTL (give time to roll back if needed)

------------------------------------------------------------------------

## 5. Index Versioning

Treat your index like a software artifact: version it so you can roll back.

**Versioning approaches:**

| Approach | Description |
|----|----|
| **Aliased collections** | Pinecone/Weaviate support named aliases; point alias at a versioned collection |
| **Namespace versioning** | Use namespaces (e.g., `corpus_v3`) and route queries to the correct one |
| **Snapshot backup** | Periodic export of index state to S3 for disaster recovery |

**Metadata to track per version:**

``` json
{
  "index_version": "v3",
  "embedding_model": "text-embedding-3-large",
  "chunk_strategy": "recursive_512_64",
  "created_at": "2024-11-01",
  "doc_count": 12450,
  "chunk_count": 89320
}
```

------------------------------------------------------------------------

## 6. Deletion and Retention Policies

Uncontrolled growth bloats index size and degrades search performance.

**Retention policy types:**

| Policy | Use Case |
|----|----|
| **TTL expiry** | News, support tickets — auto-delete after N days |
| **Source-driven** | Delete when source document is deleted (event or crawl) |
| **Hard delete** | GDPR / right to erasure — delete by doc ID immediately |
| **Soft delete** | Mark as deleted in metadata; exclude via filter; physically purge in batch |

**GDPR compliance:** vector databases store embeddings, which may be considered personal data if the source is personal. Ensure hard delete is possible by doc ID and test it.

**Index compaction:** Most vector DBs accumulate tombstoned vectors. Periodically trigger compaction or vacuum to reclaim space.

------------------------------------------------------------------------

## Summary

    Document change detected
            │
            ▼
       Has checksum changed? ──NO──→ Skip
            │YES
            ▼
     Delete old chunks (by doc_id)
            │
            ▼
     Chunk → Embed → Insert new chunks
            │
            ▼
     Update metadata store (checksum, last_modified)

**Rule of thumb:** Store doc ID on every chunk. Without it, update and delete operations become full index scans.

------------------------------------------------------------------------

Back to top
