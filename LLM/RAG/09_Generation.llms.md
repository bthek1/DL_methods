# Generation in RAG

Generation is the final stage where the LLM takes the retrieved context and produces an answer. Here’s everything about it:

Author

Benedict Thekkel

## 1. Prompt Construction

The core task is assembling a prompt that includes the retrieved documents alongside the user’s question.

**Basic structure:**

    System: You are a helpful assistant. Answer using only the provided context.

    Context:
    [Chunk 1]
    [Chunk 2]
    [Chunk 3]

    Question: {user_query}

    Answer:

**Key design decisions:** - Where to place context (before vs. after the question) - How to delimit chunks (XML tags, numbered sections, separators) - Whether to include chunk metadata (source, date, score) - Whether to instruct the model to say “I don’t know” if context is insufficient

------------------------------------------------------------------------

## 2. Context Window Management

LLMs have a fixed context window. With many retrieved chunks, you can exceed it.

**Strategies:** - **Truncation** — cut off chunks that exceed the limit (lossy) - **Contextual compression** — use an LLM or extractor to shorten each chunk to only the relevant sentences before injecting - **Map-reduce** — answer each chunk independently, then synthesize the answers - **Refine** — iteratively refine an answer by feeding one chunk at a time

------------------------------------------------------------------------

## 3. The Lost-in-the-Middle Problem

Research shows LLMs perform worse when relevant information is in the **middle** of a long context — they attend better to the beginning and end.

**Mitigations:** - Put the most relevant chunk first or last - Use re-ranking to surface the best chunk to a prominent position - Reduce the number of chunks passed to the prompt

------------------------------------------------------------------------

## 4. Answer Grounding

Ensuring the generated answer actually comes from the retrieved context, not the model’s parametric memory.

**Techniques:** - Explicit instruction: *“Only use the provided documents. Do not use prior knowledge.”* - Ask the model to quote or cite the source passage - Post-generation verification (check if claims exist in context) - Faithfulness scoring with RAGAS or TruLens

------------------------------------------------------------------------

## 5. Citation and Attribution

Telling the user *where* the answer came from builds trust and allows verification.

**Approaches:** - **Inline citations**: `The policy was updated in 2024 [Source: doc3.pdf, p.12]` - **Footnotes**: numbered references appended at the end - **Source cards**: return source metadata alongside the answer - Ask the LLM to output structured JSON with `answer` + `sources` fields

``` python
# Prompt snippet for citations
"After your answer, list the document IDs you used as: Sources: [id1, id2]"
```

------------------------------------------------------------------------

## 6. Handling Insufficient Context

When retrieved chunks don’t contain the answer:

- **Abstention**: instruct the model to say “I don’t have enough information”
- **Fallback to parametric knowledge**: allow the model to use its own knowledge but flag it
- **Confidence signaling**: ask the model to rate its own confidence
- **Re-retrieval**: trigger another retrieval pass with a rewritten query

------------------------------------------------------------------------

## 7. Structured vs. Free-form Output

RAG generation doesn’t always produce prose — sometimes you need structured output.

| Output Type       | Use Case                       |
|-------------------|--------------------------------|
| Free-form text    | Q&A, summarization             |
| JSON              | API responses, data extraction |
| Table             | Comparison queries             |
| Step-by-step list | How-to / procedural queries    |

Modern LLMs support **function calling / structured output** (e.g., OpenAI’s `response_format=json_schema`) to enforce output shape.

------------------------------------------------------------------------

## 8. Streaming

For UX responsiveness, the generated answer can be **streamed token-by-token** rather than waiting for the full response.

Most frameworks (LangChain, LlamaIndex) support `.stream()` or async generators:

``` python
for chunk in chain.stream({"question": query}):
    print(chunk, end="", flush=True)
```

------------------------------------------------------------------------

## 9. Multi-turn / Conversational RAG

In chat applications, generation must account for conversation history.

**Challenges:** - Prior turns contain context that affects what to retrieve - The full history can overflow the context window

**Solutions:** - **Condense question**: use the LLM to rewrite the latest question as a standalone query before retrieval - **Memory summarization**: compress old turns into a summary - **Message windowing**: only keep the last N turns

    Standalone question rewrite prompt:
    "Given the chat history and latest question, rewrite the question 
    as a standalone question that can be understood without the history."

------------------------------------------------------------------------

## 10. Prompt Injection Risks

Retrieved documents may contain **adversarial text** that attempts to hijack the LLM’s behavior (e.g., a document containing *“Ignore all previous instructions…”*).

**Mitigations:** - Sanitize or escape retrieved content before injecting - Use XML/delimiters to clearly separate context from instructions - Apply input/output guardrails (e.g., LlamaGuard, Nemo Guardrails)

------------------------------------------------------------------------

## Summary Flow

    User Query
        │
        ▼
    [Retriever] → Top-K Chunks
        │
        ▼
    [Context Window Manager] → Compressed / Ordered Chunks
        │
        ▼
    [Prompt Builder] → System + Context + Question
        │
        ▼
    [LLM] → Raw Answer
        │
        ▼
    [Post-processor] → Citations, Structured Output, Faithfulness Check
        │
        ▼
    Final Answer → User

------------------------------------------------------------------------

Back to top
