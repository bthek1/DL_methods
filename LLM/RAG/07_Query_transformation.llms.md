# Query Transformation for RAG

Raw user queries are often ambiguous, underspecified, or phrased nothing like the documents they’re trying to find. Query transformation rewrites or expands the query before retrieval to maximise recall.

Author

Benedict Thekkel

## Info

    "what's the recovery thing after knee op?" 
    → transform 
    → ["post-operative knee rehabilitation protocol", 
       "ACL surgery recovery timeline", 
       "knee replacement physiotherapy exercises"]

------------------------------------------------------------------------

## Strategies

| Strategy | What it does | Best for |
|----|----|----|
| **Query expansion** | Generate multiple phrasings | Ambiguous / underspecified queries |
| **HyDE** | Generate hypothetical answer, embed that | Short queries, question→answer domain gap |
| **Step-back** | Rephrase to more general question | Narrow queries missing broader context |
| **Query decomposition** | Split into sub-questions | Complex multi-hop queries |
| **Query rewriting** | Clean and clarify the raw query | Noisy / conversational input |
| **Routing** | Classify query to correct index/source | Multi-index or multi-tenant setups |

------------------------------------------------------------------------

## Local LLM setup (Ollama)

All examples below use Ollama for OSS. Install once, use across all strategies.

``` bash
# install + pull a model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral        # good default
ollama pull llama3.2       # stronger, larger
ollama pull qwen2.5:3b     # fast + lightweight
```

``` python
# llm.py — shared client
import httpx
import json

OLLAMA_BASE = "http://localhost:11434"

def generate(prompt: str, model: str = "mistral") -> str:
    resp = httpx.post(
        f"{OLLAMA_BASE}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def generate_json(prompt: str, model: str = "mistral") -> dict | list:
    """Force JSON output — works reliably with mistral/llama3."""
    resp = httpx.post(
        f"{OLLAMA_BASE}/api/generate",
        json={
            "model":  model,
            "prompt": prompt,
            "stream": False,
            "format": "json",   # Ollama native JSON mode
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    return json.loads(resp.json()["response"])
```

------------------------------------------------------------------------

## 1. Query expansion

Generate N alternative phrasings. Retrieve for all, fuse with RRF. Best general-purpose transformation.

``` python
def expand_query(query: str, n: int = 3) -> list[str]:
    result = generate_json(f"""Generate {n} alternative search queries for the question below.
Each should approach the question from a different angle or use different terminology.
Return a JSON object with key "queries" containing a list of strings.
Do not include the original question.

Question: {query}""")

    variants = result.get("queries", [])
    return [query] + [v for v in variants if v.strip()]


# Example output:
# expand_query("recovery time after ACL surgery")
# → [
#     "recovery time after ACL surgery",
#     "ACL reconstruction rehabilitation duration",
#     "how long does ACL surgery recovery take",
#     "post-operative ACL physiotherapy timeline",
#   ]
```

Usage in retrieval:

``` python
from .retrieval import dense_retrieve
from .rrf import reciprocal_rank_fusion

def multi_query_retrieve(
    query:     str,
    tenant_id: str,
    top_k:     int = 10,
) -> list[dict]:
    queries   = expand_query(query, n=3)
    all_lists = []

    for q in queries:
        results = dense_retrieve(q, tenant_id, top_k=top_k)
        all_lists.append([
            {"id": str(c.id), "content": c.content}
            for c in results
        ])

    fused = reciprocal_rank_fusion(all_lists)
    return fused[:top_k]
```

------------------------------------------------------------------------

## 2. HyDE (Hypothetical Document Embeddings)

Generate a fake answer, embed it, retrieve on the answer vector. Works because real documents live in “answer space” — closer to a fake answer than a raw question.

    "what is ACL recovery time?" 
    → generate fake passage about ACL recovery 
    → embed fake passage 
    → retrieve real documents near that embedding

``` python
from .embeddings import embed
from .models import DocumentChunk
from pgvector.django import CosineDistance

def generate_hypothetical_doc(query: str) -> str:
    return generate(f"""Write a short factual passage (3-5 sentences) that directly 
answers the following question. Write only the passage itself — no preamble, 
no "here is" or "this passage" — just the content.

Question: {query}""")


def hyde_retrieve(
    query:     str,
    tenant_id: str,
    top_k:     int = 10,
) -> list[dict]:
    hypo_doc = generate_hypothetical_doc(query)

    # Embed as a document — no instruction prefix
    hypo_vec = embed([hypo_doc])[0]

    results = list(
        DocumentChunk.objects
        .filter(tenant_id=tenant_id, index_status="indexed")
        .annotate(distance=CosineDistance("embedding", hypo_vec))
        .order_by("distance")
        [:top_k]
    )

    return [{"id": str(c.id), "content": c.content} for c in results]
```

**When HyDE helps most:** - Short, vague queries where the question form is very different from document form - FAQ or knowledge-base style docs where answers are declarative passages

**When HyDE hurts:** - Factual queries where the hypothetical doc hallucinates wrong details - The fake doc confidently embeds in the wrong direction

**Hedge by combining with standard dense retrieval:**

``` python
def hyde_plus_dense_retrieve(
    query:     str,
    tenant_id: str,
    top_k:     int = 10,
) -> list[dict]:
    hyde_results  = hyde_retrieve(query, tenant_id, top_k=top_k)
    dense_results = dense_retrieve(query, tenant_id, top_k=top_k)
    dense_dicts   = [{"id": str(c.id), "content": c.content} for c in dense_results]

    fused = reciprocal_rank_fusion([hyde_results, dense_dicts])
    return fused[:top_k]
```

------------------------------------------------------------------------

## 3. Step-back prompting

Rephrase a narrow or specific query to a broader question that captures the context needed to answer it. Then retrieve on both the original and step-back query.

    "what exercises should I do 3 weeks post ACL surgery?"
    → step back →
    "what is the standard ACL surgery rehabilitation protocol?"

``` python
def step_back(query: str) -> str:
    return generate(f"""Rewrite the following question as a broader, more general question 
that would help find background knowledge needed to answer it.
Return only the rewritten question, nothing else.

Original question: {query}""")


def step_back_retrieve(
    query:     str,
    tenant_id: str,
    top_k:     int = 10,
) -> list[dict]:
    broader_query = step_back(query)

    original_results = dense_retrieve(query, tenant_id, top_k=top_k)
    broader_results  = dense_retrieve(broader_query, tenant_id, top_k=top_k)

    original_dicts = [{"id": str(c.id), "content": c.content} for c in original_results]
    broader_dicts  = [{"id": str(c.id), "content": c.content} for c in broader_results]

    fused = reciprocal_rank_fusion([original_dicts, broader_dicts])
    return fused[:top_k]
```

------------------------------------------------------------------------

## 4. Query decomposition

Break a complex multi-part query into atomic sub-questions. Retrieve for each independently, merge results. Essential for multi-hop reasoning.

    "what are the differences between ACL and MCL recovery 
     and which has better outcomes with physio?"
    → decompose →
    [
      "ACL surgery recovery timeline and protocol",
      "MCL injury recovery timeline and protocol", 
      "physiotherapy outcomes for ACL vs MCL injuries",
    ]

``` python
def decompose_query(query: str) -> list[str]:
    result = generate_json(f"""Break the following question into simpler, independent 
sub-questions that can each be answered separately.
Return a JSON object with key "sub_questions" containing a list of strings.
If the question is already simple, return it as a single item list.

Question: {query}""")

    return result.get("sub_questions", [query])


def decomposed_retrieve(
    query:     str,
    tenant_id: str,
    top_k:     int = 10,
) -> list[dict]:
    sub_questions = decompose_query(query)
    all_lists     = []

    for sub_q in sub_questions:
        results = dense_retrieve(sub_q, tenant_id, top_k=top_k)
        all_lists.append([
            {"id": str(c.id), "content": c.content}
            for c in results
        ])

    fused = reciprocal_rank_fusion(all_lists)
    return fused[:top_k]
```

------------------------------------------------------------------------

## 5. Query rewriting

Clean conversational or noisy input into a crisp retrieval query. Essential when input comes from a chat UI where users type casually.

    "uhh yeah like i was asking about the knee thing, 
     what did you say about how long it takes again?"
    → rewrite →
    "knee surgery recovery duration"

``` python
def rewrite_query(
    query:   str,
    history: list[dict] | None = None,  # [{"role": "user"|"assistant", "content": ...}]
) -> str:
    history_str = ""
    if history:
        history_str = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in history[-4:]   # last 2 turns
        )
        history_str = f"Conversation history:\n{history_str}\n\n"

    return generate(f"""{history_str}Rewrite the following user query into a clear, 
concise search query suitable for a document retrieval system.
Remove filler words, resolve pronouns using conversation history if provided,
and use precise terminology.
Return only the rewritten query, nothing else.

User query: {query}""")
```

------------------------------------------------------------------------

## 6. Query routing

Classify the query and route to the appropriate index, collection, or retrieval strategy. Critical for multi-tenant or multi-source setups.

``` python
from dataclasses import dataclass
from typing import Literal

@dataclass
class RouteDecision:
    index:    str                              # which index/collection to search
    strategy: Literal["dense", "hybrid", "bm25"]
    reason:   str


AVAILABLE_INDEXES = {
    "clinical_notes":    "Patient clinical notes and assessments",
    "questionnaires":    "Patient questionnaire responses and scores",
    "knowledge_base":    "General healthcare and rehabilitation knowledge",
    "appointment_notes": "Appointment summaries and practitioner notes",
}


def route_query(query: str) -> RouteDecision:
    index_descriptions = "\n".join(
        f"- {k}: {v}" for k, v in AVAILABLE_INDEXES.items()
    )

    result = generate_json(f"""Given the following query and available indexes, 
decide which index to search and which retrieval strategy to use.

Available indexes:
{index_descriptions}

Retrieval strategies:
- dense: semantic similarity search, best for conceptual questions
- hybrid: dense + keyword search, best for most queries  
- bm25: keyword only, best for exact terms, IDs, codes

Return a JSON object with keys:
- "index": one of the index names above
- "strategy": one of dense, hybrid, bm25
- "reason": brief explanation

Query: {query}""")

    return RouteDecision(
        index    = result.get("index", "knowledge_base"),
        strategy = result.get("strategy", "hybrid"),
        reason   = result.get("reason", ""),
    )


def routed_retrieve(
    query:     str,
    tenant_id: str,
    top_k:     int = 10,
) -> list[dict]:
    decision = route_query(query)

    # Use decision.index to filter, decision.strategy to pick retrieval method
    filters = {"source_type": decision.index}

    config = RetrievalConfig(
        strategy=decision.strategy,
        top_k=top_k,
        filters=filters,
    )
    return retrieve(query, tenant_id, config)
```

------------------------------------------------------------------------

## Full transformation pipeline

Compose strategies based on query characteristics:

``` python
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class TransformConfig:
    rewrite:     bool = True                          # always clean input first
    strategy:    Literal[
        "expand", "hyde", "step_back",
        "decompose", "direct"
    ]            = "expand"                           # default expansion
    use_routing: bool = False


def transform_and_retrieve(
    query:     str,
    tenant_id: str,
    history:   list[dict] | None = None,
    top_k:     int = 10,
    config:    TransformConfig = TransformConfig(),
) -> list[dict]:

    # Step 1 — always rewrite noisy input first
    clean_query = rewrite_query(query, history) if config.rewrite else query

    # Step 2 — optionally route to correct index
    if config.use_routing:
        return routed_retrieve(clean_query, tenant_id, top_k)

    # Step 3 — apply transformation strategy
    match config.strategy:
        case "expand":
            return multi_query_retrieve(clean_query, tenant_id, top_k)

        case "hyde":
            return hyde_plus_dense_retrieve(clean_query, tenant_id, top_k)

        case "step_back":
            return step_back_retrieve(clean_query, tenant_id, top_k)

        case "decompose":
            return decomposed_retrieve(clean_query, tenant_id, top_k)

        case "direct":
            results = dense_retrieve(clean_query, tenant_id, top_k)
            return [{"id": str(c.id), "content": c.content} for c in results]
```

------------------------------------------------------------------------

## Strategy selection guide

| Query type                    | Recommended strategy                |
|-------------------------------|-------------------------------------|
| Short, clear factual question | HyDE or direct                      |
| Vague / conversational        | Rewrite → expand                    |
| Multi-part / complex          | Decompose                           |
| Narrow / overly specific      | Step-back                           |
| Multi-index setup             | Route first                         |
| General default               | Rewrite → expand → hybrid retrieval |

------------------------------------------------------------------------

## Latency cost per strategy

| Strategy            | Extra LLM calls | Typical added latency |
|---------------------|-----------------|-----------------------|
| Rewrite             | 1               | ~300ms                |
| Expand (3 variants) | 1               | ~500ms + 3× retrieval |
| HyDE                | 1               | ~500ms                |
| Step-back           | 1               | ~300ms + 2× retrieval |
| Decompose           | 1               | ~500ms + N× retrieval |
| Route               | 1               | ~300ms                |

All use local Ollama — latency depends on your hardware. On GPU, add ~50–100ms per call. On CPU, multiply by 5–10×.

------------------------------------------------------------------------

## Common failure modes

| Problem | Fix |
|----|----|
| Expansion generates near-identical variants | Use higher temperature, stronger prompt instruction |
| HyDE hallucinates wrong facts into embedding | Combine with standard dense, don’t use HyDE alone |
| Decomposition produces too many sub-questions | Cap at 4–5, limit top_k per sub-question |
| Step-back too broad, retrieves irrelevant context | Fuse with original query results via RRF |
| Routing misclassifies query | Add few-shot examples to routing prompt |
| Rewrite loses important specifics | Include conversation history, prompt to preserve key terms |
| LLM call adds too much latency | Cache transformed queries by hash, use smaller/faster model |

------------------------------------------------------------------------

Back to top
