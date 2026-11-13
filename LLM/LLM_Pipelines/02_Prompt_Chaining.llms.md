# Prompt Chaining

A workflow pattern where a task is decomposed into a fixed sequence of LLM calls, each processing the output of the previous step.

Author

Benedict Thekkel

## 1. What is Prompt Chaining?

**Prompt chaining** decomposes a complex task into a sequence of smaller LLM calls, where each call processes the output of the previous one. Optional **gate checks** (programmatic validations) can be inserted between steps to catch errors early before they propagate.

    Input
      │
      ▼
    ┌────────┐     ┌───────┐     ┌────────┐     ┌───────┐     ┌────────┐
    │  LLM₁  │────▶│ Gate? │────▶│  LLM₂  │────▶│ Gate? │────▶│  LLM₃  │────▶ Output
    └────────┘     └───────┘     └────────┘     └───────┘     └────────┘
                     (fail → stop/retry)          (fail → stop/retry)

The trade-off: **higher latency** (sequential calls) in exchange for **higher accuracy** (each call is a simpler, more focused task).

------------------------------------------------------------------------

## 2. When to Use Prompt Chaining

**Good fit when:** - The task cleanly decomposes into fixed, sequential subtasks - Each subtask is simpler than the full task - You want to catch and correct errors early (gates) - You need to iterate/evaluate each step independently

**Not ideal when:** - Subtasks are independent (use Parallelization instead) - The number of steps can’t be predicted upfront (use Orchestrator-Workers) - A single well-crafted prompt with CoT already works

**Examples:** - Generate marketing copy → translate into multiple languages - Write document outline → validate outline → write full document - Extract data → validate schema → transform to output format - Meeting transcript → extract action items → check consistency → write summary

------------------------------------------------------------------------

## 3. Implementation Pattern

``` python
from openai import OpenAI

client = OpenAI()

def llm_call(prompt: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    return response.choices[0].message.content


def gate_check(text: str, criteria: str) -> bool:
    """Programmatic or LLM-based validation between steps."""
    result = llm_call(
        f"Does this text meet the criteria? Respond only YES or NO.\n\nCriteria: {criteria}\n\nText: {text}"
    )
    return result.strip().upper().startswith("YES")


def document_chain(transcript: str) -> str:
    """3-step chain: extract → validate → summarise."""

    # Step 1: Extract key items
    extraction = llm_call(
        f"Extract all action items and owners from this transcript as a bulleted list:\n\n{transcript}",
        system="You are a precise meeting analyst."
    )

    # Gate: confirm extraction found items
    if not gate_check(extraction, "Contains at least one action item with an owner"):
        return "No actionable items found in transcript."

    # Step 2: Validate consistency
    validated = llm_call(
        f"Check each action item below against the original transcript for accuracy. "
        f"Remove any that are not explicitly mentioned.\n\nItems:\n{extraction}\n\nTranscript:\n{transcript}"
    )

    # Step 3: Write summary
    summary = llm_call(
        f"Write a concise executive summary (3-5 sentences) based on these verified action items:\n\n{validated}"
    )

    return summary
```

------------------------------------------------------------------------

## 4. AlphaCodium Example — A Real-World Win

By switching from a **single prompt** to a **multi-step chain**, AlphaCodium increased GPT-4 accuracy on CodeContests from **19% → 44%** (pass@5).

Their 6-step chain: 1. Reflect on the problem statement 2. Reason on the public test cases 3. Generate possible solutions 4. Rank solutions 5. Generate synthetic tests 6. Iterate on solutions against public + synthetic tests

Each step is a focused LLM call doing one thing well — the classic single-responsibility principle applied to prompts.

------------------------------------------------------------------------

## 5. Best Practices

- **One thing per prompt**: Each step should have a single, clear objective. Avoid “God prompts” that try to do everything.
- **Structured intermediate outputs**: Use JSON/XML between steps to make parsing reliable and reduce errors.
- **Insert gates at decision points**: Fail fast rather than propagating bad intermediate results through expensive steps.
- **Eval each step independently**: Breaking the chain makes it easier to identify which step is failing.
- **Start simple**: A 2-step chain is better than a monolithic prompt. Don’t jump straight to 8 steps.
- **Watch latency**: Each sequential call adds wall-clock time. If steps are independent, consider Parallelization.

------------------------------------------------------------------------

Back to top
