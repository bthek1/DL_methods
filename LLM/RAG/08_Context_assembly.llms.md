# Context Assembly for RAG

Takes the re-ranked chunks and constructs the final context block that gets injected into the LLM prompt. Retrieval finds the right chunks — context assembly determines how they’re presented to the LLM. Poor assembly wastes tokens, confuses the model, or causes it to ignore good chunks.

Author

Benedict Thekkel

## The core problem

You have N chunks. The LLM has a finite context window. You need to:

- Fit within the token budget
- Order chunks for maximum comprehension
- Avoid redundancy
- Preserve enough metadata for citation
- Frame the context so the LLM actually uses it

All of these interact with each other.

------------------------------------------------------------------------

## Token budgeting

Always calculate your token budget explicitly before assembly. Never assume you have room.

``` python
# budget.py
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

# Token costs for common OSS models (context windows)
MODEL_CONTEXT_WINDOWS = {
    "mistral":         8_192,
    "llama3.2":       128_000,
    "llama3.2:3b":    128_000,
    "qwen2.5:3b":      32_768,
    "qwen2.5:14b":    128_000,
    "deepseek-r1:8b": 128_000,
    "gemma2:9b":        8_192,
}

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def count_tokens_batch(texts: list[str]) -> list[int]:
    return [count_tokens(t) for t in texts]


@dataclass
class TokenBudget:
    total:        int   # model context window
    system:       int   # reserved for system prompt
    query:        int   # reserved for user query
    output:       int   # reserved for generated response
    overhead:     int = 64   # separator tokens, formatting

    @property
    def context_budget(self) -> int:
        return self.total - self.system - self.query - self.output - self.overhead


def build_budget(
    model:       str,
    system_prompt: str,
    query:       str,
    max_output:  int = 1024,
) -> TokenBudget:
    return TokenBudget(
        total   = MODEL_CONTEXT_WINDOWS.get(model, 8_192),
        system  = count_tokens(system_prompt),
        query   = count_tokens(query),
        output  = max_output,
    )
```

------------------------------------------------------------------------

## Chunk selection within budget

Greedily pack the highest-ranked chunks that fit:

``` python
def select_chunks_within_budget(
    chunks:  list[dict],    # ordered by relevance (best first)
    budget:  int,           # available tokens for context
    min_chunk_tokens: int = 20,
) -> list[dict]:
    selected   = []
    used_tokens = 0

    for chunk in chunks:
        tokens = count_tokens(chunk["content"])

        if tokens < min_chunk_tokens:
            continue  # skip degenerate chunks

        if used_tokens + tokens > budget:
            # Try to fit a truncated version if we have meaningful space left
            remaining = budget - used_tokens
            if remaining > min_chunk_tokens * 3:
                truncated = enc.decode(
                    enc.encode(chunk["content"])[:remaining - 4]
                )
                selected.append({**chunk, "content": truncated, "truncated": True})
            break

        selected.append({**chunk, "truncated": False})
        used_tokens += tokens

    return selected
```

------------------------------------------------------------------------

## Chunk ordering

Where you place chunks in the context window affects how much the LLM attends to them. Research shows LLMs have a **“lost in the middle”** problem — they favour content at the start and end of context.

    [most relevant] [less relevant] [less relevant] [second most relevant]

``` python
def order_chunks_lost_in_middle(chunks: list[dict]) -> list[dict]:
    """
    Place highest-relevance chunks at start and end.
    Bury lower-relevance chunks in the middle.
    Assumes chunks are sorted best-first going in.
    """
    if len(chunks) <= 2:
        return chunks

    result = []
    left, right = 0, len(chunks) - 1
    turn = "left"

    sorted_chunks = sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)

    # Interleave — best at edges
    positioned = [None] * len(sorted_chunks)
    l, r = 0, len(sorted_chunks) - 1

    for i, chunk in enumerate(sorted_chunks):
        if i % 2 == 0:
            positioned[l] = chunk
            l += 1
        else:
            positioned[r] = chunk
            r -= 1

    return positioned


def order_chunks_chronological(chunks: list[dict]) -> list[dict]:
    """Order by source page/chunk index — preserves document narrative flow."""
    return sorted(chunks, key=lambda x: (
        x.get("source_path", ""),
        x.get("page_number", 0),
        x.get("chunk_index", 0),
    ))


def order_chunks_by_relevance(chunks: list[dict]) -> list[dict]:
    """Simple best-first — good default when not worried about lost-in-middle."""
    return sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)
```

------------------------------------------------------------------------

## Deduplication

Re-ranking and multi-query retrieval often surface the same chunk from different queries. Deduplicate before assembly.

``` python
def deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """Exact dedup by content_hash."""
    seen   = set()
    result = []

    for chunk in chunks:
        key = chunk.get("content_hash") or chunk["content"][:64]
        if key not in seen:
            seen.add(key)
            result.append(chunk)

    return result


def deduplicate_fuzzy(chunks: list[dict], threshold: float = 0.9) -> list[dict]:
    """
    Near-duplicate removal using token overlap (Jaccard similarity).
    Slower — use only when exact dedup misses redundant paraphrases.
    """
    def jaccard(a: str, b: str) -> float:
        set_a = set(a.lower().split())
        set_b = set(b.lower().split())
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    result = []
    for chunk in chunks:
        is_dup = any(
            jaccard(chunk["content"], kept["content"]) >= threshold
            for kept in result
        )
        if not is_dup:
            result.append(chunk)

    return result
```

------------------------------------------------------------------------

## Context formatting

How you wrap chunks in the prompt matters. The LLM needs clear boundaries and enough metadata to cite sources.

### Minimal format

``` python
def format_context_minimal(chunks: list[dict]) -> str:
    return "\n\n".join(c["content"] for c in chunks)
```

### Numbered with source

``` python
def format_context_numbered(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source_path", "unknown")
        page   = chunk.get("page_number")
        header = f"[{i}] Source: {source}"
        if page:
            header += f", page {page}"
        parts.append(f"{header}\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)
```

### XML tagged (best for instruction-following models)

``` python
def format_context_xml(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        attrs = f'index="{i}"'
        if "source_path" in chunk:
            attrs += f' source="{chunk["source_path"]}"'
        if "page_number" in chunk and chunk["page_number"]:
            attrs += f' page="{chunk["page_number"]}"'
        if "rerank_score" in chunk:
            attrs += f' score="{chunk["rerank_score"]:.3f}"'
        parts.append(f"<chunk {attrs}>\n{chunk['content']}\n</chunk>")
    return "\n\n".join(parts)
```

### Grouped by source document

``` python
from itertools import groupby

def format_context_grouped(chunks: list[dict]) -> str:
    sorted_chunks = sorted(chunks, key=lambda x: x.get("source_path", ""))
    parts = []

    for source, group in groupby(sorted_chunks, key=lambda x: x.get("source_path", "unknown")):
        group_chunks  = list(group)
        group_content = "\n\n".join(c["content"] for c in group_chunks)
        parts.append(f"## {source}\n\n{group_content}")

    return "\n\n---\n\n".join(parts)
```

------------------------------------------------------------------------

## Prompt templates

The system prompt framing determines whether the LLM uses the context or ignores it.

``` python
# prompts.py

GROUNDED_QA_SYSTEM = """You are a helpful assistant. Answer the user's question using \
ONLY the information provided in the context below.

Rules:
- If the context does not contain enough information to answer, say so explicitly.
- Do not use any knowledge outside the provided context.
- Cite the source chunk index [1], [2] etc. when making specific claims.
- Keep your answer concise and directly responsive to the question."""


GROUNDED_QA_SYSTEM_STRICT = """You are a precise assistant. You must answer using ONLY \
the provided context. 

If the answer is not contained in the context, respond with exactly:
"I don't have enough information in the provided context to answer this question."

Do not infer, speculate, or use outside knowledge under any circumstances."""


GROUNDED_QA_SYSTEM_SOFT = """You are a helpful assistant with access to a knowledge base. \
Use the provided context as your primary source. You may use general knowledge to \
clarify or connect ideas, but always prefer and cite the provided context."""


def build_prompt(
    query:       str,
    context:     str,
    system:      str = GROUNDED_QA_SYSTEM,
    extra_instructions: str = "",
) -> list[dict]:
    """Returns Ollama-compatible messages list."""
    user_content = f"""Context:
{context}

---

Question: {query}"""

    if extra_instructions:
        user_content += f"\n\n{extra_instructions}"

    return [
        {"role": "system",    "content": system},
        {"role": "user",      "content": user_content},
    ]
```

------------------------------------------------------------------------

## Contextual compression

After selection, strip irrelevant sentences from each chunk before assembly. Reduces noise, fits more chunks in budget.

``` python
from .llm import generate

def compress_chunk(query: str, chunk: str) -> str | None:
    result = generate(f"""Extract only the sentences from the passage below that are \
directly relevant to answering the question.
Return only the extracted sentences, no explanation or preamble.
If nothing is relevant, return the single word: IRRELEVANT

Question: {query}

Passage:
{chunk}""")

    if result.strip().upper() == "IRRELEVANT":
        return None
    return result.strip()


def compress_chunks(query: str, chunks: list[dict]) -> list[dict]:
    compressed = []
    for chunk in chunks:
        result = compress_chunk(query, chunk["content"])
        if result:
            compressed.append({**chunk, "content": result, "compressed": True})
    return compressed
```

Compression adds one LLM call per chunk — only use when context budget is tight or chunk quality is low.

------------------------------------------------------------------------

## Citation mapping

Track which chunks back which claims. Build a lookup so you can return sources with the answer.

``` python
from dataclasses import dataclass, field

@dataclass
class AssembledContext:
    context_str:   str
    chunks:        list[dict]
    token_count:   int
    citation_map:  dict[int, dict]   # index → chunk metadata


def assemble_with_citations(chunks: list[dict]) -> AssembledContext:
    citation_map = {}
    parts        = []

    for i, chunk in enumerate(chunks, 1):
        citation_map[i] = {
            "source_path":  chunk.get("source_path"),
            "page_number":  chunk.get("page_number"),
            "document_id":  chunk.get("document_id"),
            "chunk_id":     chunk.get("id"),
        }
        parts.append(f"[{i}] {chunk['content']}")

    context_str = "\n\n---\n\n".join(parts)

    return AssembledContext(
        context_str  = context_str,
        chunks       = chunks,
        token_count  = count_tokens(context_str),
        citation_map = citation_map,
    )
```

------------------------------------------------------------------------

## Full assembly pipeline

``` python
from dataclasses import dataclass
from typing import Literal

@dataclass
class AssemblyConfig:
    ordering:    Literal["relevance", "chronological", "lost_in_middle"] = "lost_in_middle"
    format:      Literal["minimal", "numbered", "xml", "grouped"]        = "xml"
    dedupe:      bool  = True
    compress:    bool  = False
    max_output:  int   = 1024
    model:       str   = "mistral"
    system:      str   = GROUNDED_QA_SYSTEM


def assemble_context(
    query:   str,
    chunks:  list[dict],
    config:  AssemblyConfig = AssemblyConfig(),
) -> AssembledContext:

    # 1. Deduplicate
    if config.dedupe:
        chunks = deduplicate_chunks(chunks)

    # 2. Compress (optional — adds latency)
    if config.compress:
        chunks = compress_chunks(query, chunks)

    # 3. Calculate token budget
    budget = build_budget(config.model, config.system, query, config.max_output)

    # 4. Select chunks within budget
    chunks = select_chunks_within_budget(chunks, budget.context_budget)

    # 5. Order
    match config.ordering:
        case "relevance":
            chunks = order_chunks_by_relevance(chunks)
        case "chronological":
            chunks = order_chunks_chronological(chunks)
        case "lost_in_middle":
            chunks = order_chunks_lost_in_middle(chunks)

    # 6. Format
    match config.format:
        case "minimal":
            context_str = format_context_minimal(chunks)
        case "numbered":
            context_str = format_context_numbered(chunks)
        case "xml":
            context_str = format_context_xml(chunks)
        case "grouped":
            context_str = format_context_grouped(chunks)

    return AssembledContext(
        context_str  = context_str,
        chunks       = chunks,
        token_count  = count_tokens(context_str),
        citation_map = {i+1: c for i, c in enumerate(chunks)},
    )
```

------------------------------------------------------------------------

## Recommendations by use case

| Use case                    | Format   | Ordering       | Compress |
|-----------------------------|----------|----------------|----------|
| Factual Q&A                 | XML      | lost_in_middle | No       |
| Long document summarisation | grouped  | chronological  | Yes      |
| Citation-heavy outputs      | numbered | relevance      | No       |
| Conversational / chat       | minimal  | relevance      | No       |
| Tight context budget        | xml      | lost_in_middle | Yes      |

------------------------------------------------------------------------

## Common failure modes

| Problem | Fix |
|----|----|
| LLM ignores context, uses own knowledge | Stricter system prompt, reduce chunk count, XML format |
| Context truncated mid-sentence | Token budget not accounting for system + query overhead |
| Duplicate content inflates context | Always deduplicate before budget selection |
| Citations don’t map back to sources | Build `citation_map` before formatting, preserve IDs throughout |
| Lost-in-middle degrading answers | Reorder — put most relevant at start and end |
| Compression hallucinating new content | Use extractive prompt strictly — “extract only”, never “summarise” |
| Answer fabricated despite grounding prompt | Add `GROUNDED_QA_SYSTEM_STRICT`, lower temperature |

------------------------------------------------------------------------

Back to top
