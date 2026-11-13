# Parallelization

A workflow pattern where multiple LLM calls run concurrently — either splitting a task into independent subtasks (Sectioning) or running the same task multiple times to achieve consensus (Voting).

Author

Benedict Thekkel

## 1. What is Parallelization?

**Parallelization** has two distinct variants:

### Sectioning — Split and conquer

Break a task into **independent subtasks** that run concurrently, then aggregate results.

             ┌──────────────▶ LLM Worker A ─────────────┐
             │                                           │
    Input ───┼──────────────▶ LLM Worker B ─────────────┼──▶ Aggregator ──▶ Output
             │                                           │
             └──────────────▶ LLM Worker C ─────────────┘

### Voting — Run multiple times, pick the best

Run the **same task multiple times** independently, then aggregate results via majority vote or synthesis.

             ┌──────────────▶ LLM Run 1 ─────────────┐
             │                                        │
    Input ───┼──────────────▶ LLM Run 2 ─────────────┼──▶ Vote / Aggregate ──▶ Final Output
             │                                        │
             └──────────────▶ LLM Run 3 ─────────────┘

------------------------------------------------------------------------

## 2. When to Use Parallelization

**Sectioning — good fit when:** - Subtasks are genuinely independent (one doesn’t affect another) - Each aspect of a complex task benefits from dedicated, focused attention - Latency matters and subtasks can be run concurrently

**Voting — good fit when:** - High accuracy is critical and multiple independent attempts improve confidence - The task has stochastic variability you want to reduce - You need to detect adversarial content or edge cases

**Examples — Sectioning:** - Run a guardrails check (LLM A) in parallel with generating the core response (LLM B) - Evaluate multiple dimensions of an LLM response simultaneously (accuracy, tone, relevance, safety)

**Examples — Voting:** - Code vulnerability review: 3 different prompts each scan for different vulnerability classes; flag if any finds a problem - Content moderation: Multiple evaluators assess different risk dimensions; require a threshold of votes to flag

------------------------------------------------------------------------

## 3. Implementation Pattern

``` python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()


async def llm_call_async(prompt: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    return response.choices[0].message.content


# ── Sectioning Example ─────────────────────────────────────────────────────

async def review_code_sectioned(code: str) -> dict:
    """Evaluate code across 3 dimensions in parallel."""
    security_task = llm_call_async(
        f"Review this code for security vulnerabilities only:\n\n{code}",
        system="You are a security expert. Focus only on security issues."
    )
    performance_task = llm_call_async(
        f"Review this code for performance issues only:\n\n{code}",
        system="You are a performance engineer. Focus only on performance."
    )
    style_task = llm_call_async(
        f"Review this code for style and maintainability only:\n\n{code}",
        system="You are a senior engineer. Focus only on code quality."
    )

    security, performance, style = await asyncio.gather(
        security_task, performance_task, style_task
    )
    return {"security": security, "performance": performance, "style": style}


# ── Voting Example ──────────────────────────────────────────────────────────

async def classify_with_voting(text: str, n_votes: int = 3) -> str:
    """Run classification n times and return the majority vote."""
    tasks = [
        llm_call_async(
            f"Is this content safe or unsafe? Respond with only 'safe' or 'unsafe'.\n\n{text}"
        )
        for _ in range(n_votes)
    ]
    votes = await asyncio.gather(*tasks)
    normalised = [v.strip().lower() for v in votes]
    return max(set(normalised), key=normalised.count)  # majority vote


# Usage
results = asyncio.run(review_code_sectioned("def get_user(id): return db.query(f'SELECT * FROM users WHERE id={id}')"))
verdict = asyncio.run(classify_with_voting("Buy cheap meds online no prescription"))
```

------------------------------------------------------------------------

## 4. Guardrail + Response in Parallel

A common production pattern: run the guardrail check **concurrently** with the main response generation. If the guardrail flags the input, discard the main response.

``` python
async def safe_respond(user_input: str) -> str:
    guardrail_task = llm_call_async(
        f"Does this input contain harmful, illegal, or inappropriate content? "
        f"Reply only YES or NO.\n\nInput: {user_input}"
    )
    response_task = llm_call_async(
        user_input,
        system="You are a helpful assistant."
    )

    guardrail_result, response = await asyncio.gather(guardrail_task, response_task)

    if guardrail_result.strip().upper().startswith("YES"):
        return "I'm unable to help with that request."
    return response
```

This adds **zero latency overhead** compared to running guardrails sequentially.

------------------------------------------------------------------------

## 5. Best Practices

- **Use `asyncio.gather` for true concurrency**: Parallel HTTP calls with async I/O, not threads.
- **Set a concurrency limit**: Use `asyncio.Semaphore` to avoid hitting API rate limits.
- **Design subtasks to be truly independent**: If Task B needs Task A’s output, it’s a chain, not parallelization.
- **For voting, use odd numbers**: 3 or 5 votes avoid ties in binary decisions.
- **Aggregate thoughtfully**: For complex outputs, a synthesis LLM call is better than naive concatenation.
- **Cost consideration**: N parallel calls cost N times as much. Verify the quality improvement justifies it.

------------------------------------------------------------------------

Back to top
