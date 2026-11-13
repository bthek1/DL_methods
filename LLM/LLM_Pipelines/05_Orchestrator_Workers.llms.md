# Orchestrator-Workers

A workflow pattern where a central orchestrator LLM dynamically plans tasks, delegates them to specialised worker LLMs, and synthesises the results — handling subtask structures that cannot be predicted upfront.

Author

Benedict Thekkel

## 1. What is Orchestrator-Workers?

In the **orchestrator-workers** pattern, a central LLM acts as a **project manager**: it analyses the incoming task, decides what subtasks are needed, delegates them to specialised worker LLMs, and then synthesises all results into a final output.

                        ┌─────────────────────────────────────────────┐
                        │               ORCHESTRATOR                   │
                        │   Analyses task → creates dynamic plan       │
                        └────────────┬──────────────────┬─────────────┘
                                     │  delegates       │
                        ┌────────────▼──┐         ┌─────▼────────────┐
                        │   Worker A    │         │    Worker B       │
                        │ (specialised) │         │  (specialised)    │
                        └────────────┬──┘         └─────┬────────────┘
                                     │  results         │
                        ┌────────────▼──────────────────▼─────────────┐
                        │               SYNTHESISER                    │
                        │       Combines worker outputs                │
                        └─────────────────────────────────────────────┘
                                             │
                                           Output

**Key difference from Parallelization**: subtasks are **not pre-defined**. The orchestrator determines them dynamically based on the specific input.

------------------------------------------------------------------------

## 2. When to Use Orchestrator-Workers

**Good fit when:** - You cannot predict the required subtasks upfront (they depend on the input) - Different subtasks need different specialised prompts/tools - The task is too complex for a single LLM call - You want a human-readable plan before execution

**Examples:** - **Multi-file code changes**: The orchestrator reads the codebase, decides which files need changing, delegates each file edit to a worker - **Research tasks**: Orchestrator decides which sources to search, delegates queries to search workers, synthesises findings - **Document generation**: Orchestrator creates an outline, assigns each section to a specialised writer worker - **Data pipeline building**: Orchestrator analyses requirements, assigns schema design / ETL / testing to different workers

**Comparison with Parallelization:**

|                    | Parallelization          | Orchestrator-Workers      |
|--------------------|--------------------------|---------------------------|
| Subtask definition | Fixed upfront            | Dynamically determined    |
| Flexibility        | Low                      | High                      |
| Predictability     | High                     | Lower                     |
| Best for           | Known, independent tasks | Open-ended, complex tasks |

------------------------------------------------------------------------

## 3. Implementation Pattern

``` python
import json
from openai import OpenAI

client = OpenAI()


def llm_call(prompt: str, system: str = "", model: str = "gpt-4o") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


def orchestrate(task: str) -> str:
    """
    Orchestrator-Workers pattern:
    1. Orchestrator creates a dynamic plan
    2. Workers execute each subtask
    3. Synthesiser combines all results
    """

    # ── Step 1: Orchestrator plans ─────────────────────────────────────────
    plan_json = llm_call(
        f"Break down this task into subtasks. Return JSON array of objects with "
        f"'subtask_id' (int), 'description' (str), 'worker_type' (str).\n\nTask: {task}",
        system="You are a project planning expert. Output only valid JSON."
    )
    subtasks = json.loads(plan_json)

    # ── Step 2: Workers execute ────────────────────────────────────────────
    worker_results = []
    for subtask in subtasks:
        system_prompt = WORKER_SYSTEMS.get(
            subtask["worker_type"],
            "You are a helpful expert. Complete the task precisely."
        )
        result = llm_call(subtask["description"], system=system_prompt)
        worker_results.append({
            "subtask_id": subtask["subtask_id"],
            "description": subtask["description"],
            "result": result
        })

    # ── Step 3: Synthesiser combines ───────────────────────────────────────
    results_text = "\n\n".join(
        f"[Subtask {r['subtask_id']}: {r['description']}]\n{r['result']}"
        for r in worker_results
    )
    final = llm_call(
        f"Synthesise these subtask results into a coherent final response for the original task.\n"
        f"Original task: {task}\n\nSubtask Results:\n{results_text}",
        system="You are a skilled synthesiser. Combine results into a unified, coherent response."
    )
    return final


WORKER_SYSTEMS = {
    "researcher": "You are a research expert. Provide factual, well-sourced information.",
    "writer": "You are a skilled technical writer. Write clearly and concisely.",
    "analyst": "You are a data analyst. Provide quantitative insights and reasoning.",
    "coder": "You are a senior software engineer. Write clean, production-quality code.",
}


# Usage
result = orchestrate(
    "Create a technical blog post about RAG pipelines including: "
    "an overview, code examples, and a comparison of vector databases."
)
```

------------------------------------------------------------------------

## 4. Managing Junior Engineers — An Analogy

The orchestrator-workers pattern mirrors how a senior engineer manages a team:

| Engineering Team | Orchestrator-Workers |
|----|----|
| Senior engineer analyses requirements | Orchestrator analyses the task |
| Creates a sprint plan with clear tickets | Generates a structured plan with subtasks |
| Assigns tickets to engineers by speciality | Delegates subtasks to specialist workers |
| Reviews and integrates PRs | Synthesiser combines worker outputs |

> **Key insight**: The most successful orchestrator builders have experience managing junior engineers. Give workers clear goals and concrete instructions — not vague open-ended directions.

------------------------------------------------------------------------

## 5. Best Practices

- **Output structured plans**: Have the orchestrator produce JSON/structured plans that are easy to parse and log.
- **Validate the plan before executing**: Check that the plan is sensible before spinning up workers (a gate).
- **Use worker specialisation**: Different system prompts for different worker types dramatically improve quality.
- **Parallelise workers when possible**: If worker subtasks are independent, run them concurrently (combine with Parallelization pattern).
- **Cap the number of subtasks**: Add a `max_subtasks` guard to prevent runaway planning.
- **Log each worker’s input/output**: Debugging is much harder when you can’t see what each worker received and returned.
- **Consider deterministic plans**: Pre-define plans for common task types and use the orchestrator only to select between them (more reliable, easier to test).

------------------------------------------------------------------------

Back to top
