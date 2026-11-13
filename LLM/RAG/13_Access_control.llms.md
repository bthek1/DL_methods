# Access Control in RAG

Access control ensures users can only retrieve and receive answers based on documents they are authorised to see. The cardinal rule: filter *before* vector search, never after.

Author

Benedict Thekkel

## 1. The Cardinal Rule: Pre-Filter, Not Post-Filter

> **IMPORTANT:**
>
> **Never retrieve and then filter.** If you retrieve top-K chunks and then discard the ones a user cannot see, you reveal the *existence* of those documents (information leakage) and you waste vector search capacity. Always pass the access filter *into* the vector search so unauthorized chunks are never retrieved.

**Wrong pattern:**

``` python
# BAD — retrieves everything, then filters
chunks = index.search(query_vector, top_k=20)
allowed_chunks = [c for c in chunks if user_can_access(c)]  # Leaky!
```

**Correct pattern:**

``` python
# GOOD — filter is applied inside the vector search
chunks = index.search(
    query_vector,
    top_k=5,
    filter={"tenant_id": user.tenant_id}  # Pre-filter
)
```

------------------------------------------------------------------------

## 2. Multi-Tenancy Patterns

In SaaS applications with multiple tenants, each tenant must be isolated.

**Option A: Separate index per tenant**

| Pro | Con |
|----|----|
| Perfect isolation | High operational overhead for many tenants |
| No risk of cross-tenant leakage | Resource waste for small tenants |
| Simple queries (no filter needed) | Index creation latency on tenant onboarding |

Best for: high-security environments with a small number of large tenants.

**Option B: Shared index with tenant_id metadata filter**

| Pro | Con |
|----|----|
| Single index to manage | Must ensure filter is always applied |
| Low overhead per tenant | Some ANN indexes degrade with high-cardinality filters |
| Easy tenant onboarding | Misapplied filter = data breach |

Best for: large numbers of small tenants (typical SaaS).

**Option C: Namespace / partition per tenant** (supported by Pinecone, Weaviate, Qdrant)

- Each tenant gets a logical partition within the same physical index
- Queries are scoped to a namespace at the API level — no per-query filter needed
- Best balance of isolation and operational simplicity

------------------------------------------------------------------------

## 3. Document-Level ACLs

Beyond tenant-level isolation, individual documents may have fine-grained permissions (e.g., only certain roles or users can see a document).

**Approach: store ACL metadata on each chunk**

``` python
# Chunk metadata
{
  "doc_id": "contract_2024_acme",
  "tenant_id": "acme_corp",
  "allowed_roles": ["legal", "finance"],   # Role-based
  "allowed_users": ["user_789"],           # User-specific
  "classification": "confidential"
}
```

**At query time:**

``` python
user_roles = get_user_roles(user_id)  # e.g. ["finance"]

chunks = index.search(
    query_vector,
    top_k=5,
    filter={
        "tenant_id": user.tenant_id,
        "allowed_roles": {"$in": user_roles}
    }
)
```

**Limitation:** fine-grained ACL filters with many OR conditions can slow ANN search. Benchmark your vector DB’s filter performance under realistic ACL complexity.

------------------------------------------------------------------------

## 4. ACL Synchronisation

ACLs in the vector index must stay in sync with the source system’s permissions.

**The stale ACL problem:** - A user loses access to a document in the source system (e.g., Confluence, SharePoint) - The vector index still has the old chunk with the old ACL → unauthorized access

**Sync strategies:**

| Strategy | Frequency | Approach |
|----|----|----|
| **Event-driven** | Real-time | Source system publishes permission-change events; update chunk metadata |
| **Periodic re-sync** | Hourly/daily | Crawl source ACLs, compare to index metadata, patch differences |
| **Re-ingest on change** | On document update | Always re-ingest full document (including ACL) when it changes |

**Recommendation:** Treat ACL changes as document updates — trigger a full re-ingest of affected documents so chunk metadata is always fresh.

------------------------------------------------------------------------

## 5. Query-Time Identity Injection

The user’s identity must be injected at query time from a trusted source — never from the user’s own request.

**Secure pattern:**

``` python
# Authentication middleware resolves identity BEFORE hitting RAG
def rag_endpoint(request):
    user = auth.verify_token(request.headers["Authorization"])
    # user.tenant_id and user.roles are resolved server-side
    
    result = rag_pipeline.query(
        query=request.body["question"],
        tenant_id=user.tenant_id,       # Server-side, trusted
        allowed_roles=user.roles        # Server-side, trusted
    )
    return result
```

> **WARNING:**
>
> Never accept `tenant_id` or `allowed_roles` directly from the user’s request body. These values must come from your authentication system.

------------------------------------------------------------------------

## 6. Audit Logging

Access-controlled systems need audit trails for compliance (SOC 2, ISO 27001, GDPR).

**What to log per RAG request:**

``` json
{
  "request_id": "uuid",
  "timestamp": "2024-11-01T12:00:00Z",
  "user_id": "user_789",
  "tenant_id": "acme_corp",
  "query_hash": "sha256:...",   // Hash — don't log raw PII queries
  "retrieved_doc_ids": ["doc42", "doc17"],
  "applied_filter": {"tenant_id": "acme_corp", "allowed_roles": ["finance"]}
}
```

**Key audit questions to be able to answer:** - What documents did user X access on date Y? - Did any user access document Z without authorization? - When was the last time ACLs were synced for tenant T?

------------------------------------------------------------------------

## Summary

| Layer         | Rule                                                     |
|---------------|----------------------------------------------------------|
| Multi-tenancy | Namespace or metadata filter — never shared namespace    |
| Document ACLs | Store on chunk metadata; filter inside ANN search        |
| Identity      | Inject from auth system server-side — never trust client |
| ACL sync      | Treat permission changes as document updates → re-ingest |
| Audit         | Log doc IDs accessed per user per request                |

**The single most important rule:** the access filter must be part of the vector search call, not applied to results afterward.

------------------------------------------------------------------------

Back to top
