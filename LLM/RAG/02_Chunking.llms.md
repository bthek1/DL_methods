# Chunking for RAG

Splits documents into pieces small enough to embed meaningfully and fit in context, but large enough to carry useful information. It’s the single biggest lever on retrieval quality.

Author

Benedict Thekkel

## Core tradeoff

| Small chunks                     | Large chunks                    |
|----------------------------------|---------------------------------|
| Precise retrieval                | More context per chunk          |
| Less noise passed to LLM         | Fewer chunks to manage          |
| May lose surrounding context     | May retrieve irrelevant content |
| Higher chunk count, more storage | Lower recall on specific facts  |

Sweet spot in practice: **256–512 tokens with overlap**.

------------------------------------------------------------------------

## Strategies

### 1. Fixed-size with overlap

Split every N tokens, slide forward by N-overlap. Simple, predictable, works well as a baseline.

``` python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    length_function=len,  # swap for tiktoken for token-accurate splits
)

chunks = splitter.split_text(document_text)
```

Token-accurate version:

``` python
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

enc = tiktoken.get_encoding("cl100k_base")

def token_len(text):
    return len(enc.encode(text))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    length_function=token_len,
)
```

------------------------------------------------------------------------

### 2. Recursive character splitting

Tries to split on `\n\n`, then `\n`, then , then characters — respects natural text boundaries. This is the **recommended default**.

``` python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=512,
    chunk_overlap=50,
)

chunks = splitter.create_documents(
    texts=[document_text],
    metadatas=[{"source": "my_doc.pdf", "page": 1}]
)
```

------------------------------------------------------------------------

### 3. Semantic chunking

Embed every sentence, split where embedding similarity drops — i.e. where the topic changes. Higher quality, slower to build index.

``` python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # or "standard_deviation", "interquartile"
    breakpoint_threshold_amount=95,
)

chunks = splitter.split_text(document_text)
```

------------------------------------------------------------------------

### 4. Document-structure-aware splitting

Respect headings, sections, tables. Best for structured docs (Markdown, HTML, PDFs with structure).

``` python
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(markdown_text)
# Each chunk carries {"h1": "...", "h2": "..."} metadata automatically
```

For HTML:

``` python
from langchain_text_splitters import HTMLHeaderTextSplitter

splitter = HTMLHeaderTextSplitter(headers_to_split_on=[
    ("h1", "header_1"),
    ("h2", "header_2"),
])
chunks = splitter.split_text(html_content)
```

------------------------------------------------------------------------

### 5. Hierarchical / parent-child chunking

Index small child chunks for precise retrieval, but pass the larger parent chunk to the LLM for context. Best of both worlds.

``` python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Small chunks go into the vector index
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

# Large chunks are what gets returned to the LLM
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
store = InMemoryStore()  # swap for Redis/DB in prod

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(documents)

# Retrieves small chunks but returns parent context
results = retriever.invoke("your query")
```

------------------------------------------------------------------------

### 6. Agentic / late chunking

Embed the full document with a long-context model, then chunk the *embeddings* rather than the text. Preserves full document context in every chunk’s vector. Requires models that support it (e.g. `jina-embeddings-v3`).

Not widely available yet — worth watching but not standard prod practice.

------------------------------------------------------------------------

## Overlap — why it matters

Without overlap, a sentence split across a boundary loses context in both halves. Overlap of 10–15% of chunk size covers this.

``` python
# 512 token chunks, ~50 token overlap = ~10% overlap
# Good default. Don't go above 20% — wasteful and degrades precision.
chunk_size=512
chunk_overlap=50
```

------------------------------------------------------------------------

## Chunk size guidelines by use case

| Use case             | Recommended size                        |
|----------------------|-----------------------------------------|
| Q&A over dense docs  | 256–512 tokens                          |
| Summarisation        | 512–1024 tokens                         |
| Code                 | Full function / class (structure-aware) |
| Conversational / FAQ | 128–256 tokens                          |
| Long-form reports    | Hierarchical (200 child / 1000 parent)  |

------------------------------------------------------------------------

## What to store per chunk

``` python
{
    "id": "uuid",
    "document_id": "parent_doc_uuid",
    "chunk_index": 3,
    "content": "...",
    "embedding": [...],
    "content_hash": "sha256_of_content",
    "chunk_strategy": "recursive_512_50",
    "embedding_model": "text-embedding-3-small",
    "source_path": "uploads/report.pdf",
    "page_number": 2,
    "section_heading": "Results",
    "token_count": 487,
    "created_at": "...",
}
```

------------------------------------------------------------------------

## Production recommendations

**Use recursive character splitting as your default.** It handles most text types well without the overhead of semantic splitting.

**Add semantic chunking for high-value, unstructured long documents** where topic shifts matter — clinical notes, research papers, legal docs.

**Use hierarchical chunking when** retrieval precision is high but LLM answers feel incomplete or out of context.

**Always store `chunk_strategy` and `embedding_model`** — when you change either, you need to re-index everything and knowing which chunks are stale is essential.

**Re-index on document update**, not on a schedule. Use `content_hash` to detect actual changes and skip unchanged chunks.

``` python
import hashlib

def should_reindex(existing_hash: str, new_content: str) -> bool:
    new_hash = hashlib.sha256(new_content.encode()).hexdigest()
    return existing_hash != new_hash
```

**Validate chunk quality** before writing to the index — filter empty chunks, chunks under ~20 tokens, and chunks that are pure whitespace/boilerplate.

``` python
def is_valid_chunk(text: str, min_tokens: int = 20) -> bool:
    tokens = enc.encode(text)
    return len(tokens) >= min_tokens and text.strip()

chunks = [c for c in raw_chunks if is_valid_chunk(c)]
```

------------------------------------------------------------------------

Back to top
