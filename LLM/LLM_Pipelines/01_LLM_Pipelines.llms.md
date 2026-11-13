# LLM Pipeline Designs

A comprehensive guide to the core architectural patterns for building LLM-powered systems, from simple prompt calls to autonomous agents.

Author

Benedict Thekkel

## 1. What are LLM Pipelines?

An **LLM pipeline** is the architectural pattern that defines how one or more LLM calls are composed together — with tools, memory, retrieval, and control flow — to accomplish a task.

There is an important architectural distinction between two categories:

| Category | Definition |
|----|----|
| **Workflows** | LLMs and tools orchestrated through **predefined code paths**. Deterministic, predictable. |
| **Agents** | LLMs **dynamically direct** their own processes and tool usage. Flexible, autonomous. |

Most real-world systems use workflows. Agents are reserved for open-ended tasks where the number of steps cannot be predicted upfront.

------------------------------------------------------------------------

## 2. The Seven Core Patterns

| \# | Pattern | Complexity | Use When |
|----|----|----|----|
| 0 | **Augmented LLM** | Minimal | Single call + retrieval/tools is enough |
| 1 | **Prompt Chaining** | Low | Task decomposes into fixed sequential steps |
| 2 | **Routing** | Low | Different input types need different handling |
| 3 | **Parallelization** | Medium | Subtasks are independent, or need consensus |
| 4 | **Orchestrator-Workers** | Medium-High | Subtasks can’t be predicted upfront |
| 5 | **Evaluator-Optimizer** | Medium-High | Iterative refinement with clear quality criteria |
| 6 | **Autonomous Agents** | High | Open-ended tasks, environment feedback loop |

> **Key principle**: Start with the simplest approach. Add complexity only when it demonstrably improves outcomes. Each level costs roughly **10× more effort** than the previous.

------------------------------------------------------------------------

## 3. Building Block: The Augmented LLM

Every pattern builds on this foundation — a single LLM enhanced with three augmentations:

    ┌─────────────────────────────────────────┐
    │              Augmented LLM              │
    │                                         │
    │  ┌──────────┐  ┌─────────┐  ┌────────┐ │
    │  │ Retrieval│  │  Tools  │  │ Memory │ │
    │  └──────────┘  └─────────┘  └────────┘ │
    │                    │                    │
    │              ┌─────▼─────┐              │
    │              │    LLM    │              │
    │              └───────────┘              │
    └─────────────────────────────────────────┘

- **Retrieval**: Vector search (RAG) to inject relevant context
- **Tools**: Functions the LLM can call (APIs, code execution, DB queries)
- **Memory**: Short-term (conversation history) and long-term (vector store, key-value)

For many applications, optimising a single LLM call with good retrieval and in-context examples is sufficient.

------------------------------------------------------------------------

## 4. Decision Guide: Which Pattern to Use?

    Can a single LLM call with good retrieval solve it?
      YES → Use Augmented LLM (stop here)
      NO ↓

    Does the task decompose into fixed, sequential subtasks?
      YES → Prompt Chaining
      NO ↓

    Do different input types need different handling?
      YES → Routing
      NO ↓

    Are subtasks independent (can run at the same time)?
      YES → Parallelization
      NO ↓

    Do you have clear quality criteria and iterative refinement helps?
      YES → Evaluator-Optimizer
      NO ↓

    Can you predict all subtasks upfront?
      NO → Orchestrator-Workers
      YES (but complex) → Orchestrator-Workers

    Is the problem truly open-ended with unpredictable steps + environment feedback?
      YES → Autonomous Agent (use sparingly)

------------------------------------------------------------------------

## 5. Sub-notebooks in This Section

| Notebook | Pattern | Key Concept |
|----|----|----|
| [02 - Prompt Chaining](../../LLM/LLM_Pipelines/prompt_chaining.llms.md) | Sequential steps | Each LLM call feeds the next |
| [03 - Routing](../../LLM/LLM_Pipelines/routing.llms.md) | Input classification | Direct inputs to specialised handlers |
| [04 - Parallelization](../../LLM/LLM_Pipelines/parallelization.llms.md) | Concurrent execution | Sectioning + Voting |
| [05 - Orchestrator-Workers](../../LLM/LLM_Pipelines/orchestrator_workers.llms.md) | Dynamic task delegation | Central planner + specialist workers |
| [06 - Evaluator-Optimizer](../../LLM/LLM_Pipelines/evaluator_optimizer.llms.md) | Iterative refinement | Generator ⇄ Evaluator loop |
| [07 - Autonomous Agents](../../LLM/LLM_Pipelines/autonomous_agents.llms.md) | Self-directed execution | LLM controls its own tool use loop |

------------------------------------------------------------------------

## 6. References

- [Anthropic — Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) (Dec 2024)
- [Applied LLMs — What We’ve Learned From a Year of Building with LLMs](https://applied-llms.org/) (Jun 2024)
- [O’Reilly — What We Learned from a Year of Building with LLMs](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms/) (May 2024)

------------------------------------------------------------------------

Back to top
