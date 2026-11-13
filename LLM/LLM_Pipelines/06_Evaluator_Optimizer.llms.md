# Evaluator-Optimizer

A workflow pattern where one LLM generates an output and another evaluates it, looping until the output meets a quality threshold — analogous to a human writer and an editor working together.

Author

Benedict Thekkel

## 1. What is the Evaluator-Optimizer Pattern?

In the **evaluator-optimizer** pattern, two LLMs work in a feedback loop: - **Generator** produces an output (translation, code, summary, plan) - **Evaluator** critiques the output against defined criteria and provides structured feedback - The loop continues until the evaluator is satisfied or a max iteration limit is reached

    Task ──▶ Generator ──▶ Output
                  ▲            │
                  │            ▼
               Feedback    Evaluator ──▶ PASS ──▶ Final Output
                  │            │
                  └────────────┘ (loop while FAIL)

This mirrors the iterative writing process: draft → editorial feedback → revise → repeat.

------------------------------------------------------------------------

## 2. When to Use Evaluator-Optimizer

**Two signs it’s a good fit:** 1. LLM responses can be demonstrably improved when given specific feedback 2. The LLM can itself provide that feedback (or you have clear, automatable criteria)

**Good fit when:** - You have clear, articulable quality criteria - Iterative refinement adds measurable value - A single-pass output is consistently below the quality bar

**Examples:** - **Literary translation**: Nuances the translator misses can be caught by a separate evaluator with cultural knowledge - **Complex search**: The evaluator decides whether the search results are comprehensive enough, triggering more searches - **Code generation**: Evaluator runs tests and feeds back which test cases failed - **Technical writing**: Evaluator checks accuracy, clarity, completeness and provides structured feedback

**Not ideal when:** - Quality criteria are vague or subjective (the evaluator will oscillate) - First-pass quality is already good enough - Latency budget doesn’t allow multiple round trips

------------------------------------------------------------------------

## 3. Implementation Pattern

``` python
from openai import OpenAI
from dataclasses import dataclass
from typing import Optional

client = OpenAI()


@dataclass
class EvalResult:
    passed: bool
    score: int           # 1-10
    feedback: str        # Specific, actionable critique


def llm_call(prompt: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )
    return response.choices[0].message.content


def generate(task: str, feedback: Optional[str] = None) -> str:
    """Generator: produce or revise based on feedback."""
    if feedback:
        prompt = f"Revise your previous output based on this feedback:\n\n{feedback}\n\nOriginal task: {task}"
    else:
        prompt = task
    return llm_call(prompt, system="You are an expert writer. Produce high-quality output.")


def evaluate(task: str, output: str, criteria: list[str]) -> EvalResult:
    """Evaluator: assess output against criteria and provide feedback."""
    criteria_text = "\n".join(f"- {c}" for c in criteria)
    response = llm_call(
        f"""Evaluate this output against the criteria below.
Score from 1-10 and provide specific actionable feedback.
Output PASS if score >= 8, else FAIL.

Criteria:
{criteria_text}

Task: {task}

Output to evaluate:
{output}

Respond in this format:
VERDICT: PASS or FAIL
SCORE: <1-10>
FEEDBACK: <specific actionable feedback>""",
        system="You are a rigorous quality evaluator."
    )

    lines = response.strip().split("\n")
    verdict = "PASS" in lines[0].upper()
    score = int(lines[1].split(":")[1].strip())
    feedback = "\n".join(lines[2:]).replace("FEEDBACK:", "").strip()
    return EvalResult(passed=verdict, score=score, feedback=feedback)


def evaluator_optimizer(task: str, criteria: list[str], max_iterations: int = 3) -> str:
    """Run the generator-evaluator loop until quality is met or max iterations reached."""
    output = generate(task)

    for iteration in range(max_iterations):
        eval_result = evaluate(task, output, criteria)
        print(f"Iteration {iteration + 1}: score={eval_result.score}, passed={eval_result.passed}")

        if eval_result.passed:
            print(f"Quality threshold met after {iteration + 1} iteration(s).")
            return output

        # Revise based on feedback
        output = generate(task, feedback=eval_result.feedback)

    print(f"Reached max iterations ({max_iterations}). Returning best output.")
    return output


# Usage
result = evaluator_optimizer(
    task="Translate this to French preserving technical tone: 'The kernel panic was triggered by a null pointer dereference in the interrupt handler.'",
    criteria=[
        "Technically accurate translation of all computer science terms",
        "Preserves the formal, technical register of the original",
        "Natural French phrasing (not a literal word-for-word translation)",
    ],
)
```

------------------------------------------------------------------------

## 4. Convergence and Stopping Criteria

The loop needs explicit stopping conditions to avoid infinite refinement:

| Stopping Condition            | Implementation                             |
|-------------------------------|--------------------------------------------|
| Quality threshold met         | `if eval_result.score >= threshold: break` |
| Max iterations reached        | `for i in range(max_iterations)`           |
| No improvement between rounds | Compare scores across iterations           |
| Execution-based pass          | `if all_tests_pass(output): break`         |

**Detecting oscillation**: If score goes 6 → 7 → 6 → 7, the evaluator and generator are stuck. Cap iterations and return the best-scored output.

``` python
best_output = output
best_score = 0

for iteration in range(max_iterations):
    eval_result = evaluate(task, output, criteria)
    if eval_result.score > best_score:
        best_score = eval_result.score
        best_output = output
    if eval_result.passed:
        return output
    output = generate(task, feedback=eval_result.feedback)

return best_output  # Return best seen, even if threshold not met
```

------------------------------------------------------------------------

## 5. Best Practices

- **Make criteria explicit and measurable**: Vague criteria like ‘good quality’ lead to inconsistent evaluations. Use specific, binary-checkable criteria.
- **Use a stronger model as evaluator**: The evaluator can be a larger/smarter model than the generator if budget allows.
- **Provide Chain-of-Thought in the evaluator**: Ask it to explain its reasoning before giving a verdict. This reduces noise and catches reasoning errors.
- **Cap iterations aggressively**: 2-3 iterations is usually sufficient. Diminishing returns set in quickly.
- **Log every iteration**: Generator input, evaluator feedback, and score — critical for debugging quality plateaus.
- **Use execution-based evaluation when possible**: For code, running tests is more reliable than LLM-as-judge scoring.
- **Separate evaluator from generator**: Never use the same LLM call for both — the generator will rationalise its own output.

------------------------------------------------------------------------

Back to top
