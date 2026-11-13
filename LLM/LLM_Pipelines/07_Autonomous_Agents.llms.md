# Autonomous Agents

The highest-complexity LLM pattern: an LLM that dynamically directs its own tool use in an open-ended loop, planning and acting until a task is complete — or a stopping condition is reached.

Author

Benedict Thekkel

## 1. What are Autonomous Agents?

An **autonomous agent** is an LLM that controls its own reasoning loop: it observes the environment, decides which tool to call, acts, observes the result, and continues until the task is done or a stopping condition is triggered.

    Human Goal
         │
         ▼
    ┌─────────────────────────────────────────────────┐
    │                    AGENT LOOP                    │
    │                                                  │
    │  ┌──────────┐   plan/act   ┌────────────────┐    │
    │  │   LLM    │─────────────▶│  Tool Calls    │    │
    │  │(Reasoner)│              │ (search, code, │    │
    │  └──────────┘              │  file, APIs)   │    │
    │       ▲                    └───────┬────────┘    │
    │       │     observe result         │             │
    │       └────────────────────────────┘             │
    │                                                  │
    │  Exit when: task complete / max steps / error    │
    └─────────────────────────────────────────────────┘
         │
         ▼
    Final Output (+ optional human review)

**Key distinction**: Unlike workflows (where the control flow is fixed in code), the agent **decides its own next step** at each iteration based on what it observes.

------------------------------------------------------------------------

## 2. When to Use Autonomous Agents

**Use agents when:** - The problem is genuinely open-ended with an unpredictable number of steps - The environment provides reliable ground-truth feedback (test results, API responses, file reads) - The task cannot be decomposed into a fixed workflow upfront - You can tolerate higher latency and cost - You can sandbox the environment and add guardrails

**Avoid agents when:** - A simpler workflow pattern would work (it usually does) - Low latency is required - Errors compound quickly (each step’s failure probability multiplies) - The environment has irreversible side effects (deleting data, sending emails, charging customers)

**Proven examples in production:** - **Coding agents**: Resolve GitHub issues by reading/writing code files, running tests, iterating (SWE-bench) - **Computer use**: Navigate a browser/desktop to complete tasks - **Research agents**: Search the web, synthesise findings, decide when to stop searching - **Customer support**: Access order history, issue refunds, update tickets — in a conversational loop

------------------------------------------------------------------------

## 3. The ReAct Pattern — Core Agent Loop

**ReAct** (Reasoning + Acting) is the standard agent loop: the LLM alternates between reasoning about what to do and taking an action:

    Thought: I need to find the Python docs for asyncio.
    Action: web_search("asyncio.gather documentation Python 3.12")
    Observation: [search results returned]

    Thought: The docs show gather accepts coroutines. Let me check the example.
    Action: web_search("asyncio.gather example return values")
    Observation: [more results]

    Thought: I have enough information to answer.
    Action: finish(answer="asyncio.gather runs coroutines concurrently and returns their results as a list...")

Each `Thought → Action → Observation` cycle is one step of the agent loop.

------------------------------------------------------------------------

## 4. Implementation Pattern

``` python
import json
from openai import OpenAI

client = OpenAI()

# ── Tool definitions ────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Returns a list of relevant snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Returns file content as a string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file"}
                },
                "required": ["path"],
            },
        },
    },
]


def dispatch_tool(name: str, arguments: dict) -> str:
    """Execute a tool call and return the result as a string."""
    if name == "web_search":
        return web_search(arguments["query"])   # your search implementation
    elif name == "read_file":
        with open(arguments["path"]) as f:
            return f.read()
    return f"Unknown tool: {name}"


# ── Agent loop ──────────────────────────────────────────────────────────────

def run_agent(goal: str, max_steps: int = 10) -> str:
    """Run an autonomous agent until the task is complete or max_steps is reached."""
    messages = [
        {"role": "system", "content": "You are an autonomous assistant. Use tools to complete tasks. Think step by step."},
        {"role": "user", "content": goal},
    ]

    for step in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message
        messages.append(message)

        # No tool call → agent is done
        if not message.tool_calls:
            return message.content

        # Execute each tool call and feed results back
        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            result = dispatch_tool(tool_call.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    return "Max steps reached without completing the task."


# Usage
answer = run_agent("Research the top 3 vector databases used in production RAG systems in 2025 and compare their performance characteristics.")
```

------------------------------------------------------------------------

## 5. Human-in-the-Loop Checkpoints

For high-stakes actions, add **human approval checkpoints** before the agent proceeds:

``` python
REQUIRE_APPROVAL = {"delete_file", "send_email", "charge_customer", "deploy_code"}

def dispatch_tool_with_approval(name: str, arguments: dict) -> str:
    if name in REQUIRE_APPROVAL:
        approval = input(f"Agent wants to call '{name}' with args {arguments}. Approve? [y/N]: ")
        if approval.lower() != "y":
            return f"Action '{name}' was rejected by the user."
    return dispatch_tool(name, arguments)
```

This is the **human-centaur model**: AI handles the heavy lifting, humans retain control over irreversible actions.

------------------------------------------------------------------------

## 6. Error Compounding — The Key Risk

The probability of completing a multi-step task successfully **decreases exponentially** with the number of steps:

$$P(\text{success}) = p^n$$

Where $p$ is the per-step success rate and $n$ is the number of steps.

| Steps | 99% per-step | 95% per-step | 90% per-step |
|-------|--------------|--------------|--------------|
| 5     | 95%          | 77%          | 59%          |
| 10    | 90%          | 60%          | 35%          |
| 20    | 82%          | 36%          | 12%          |

This is why **tool design matters enormously**: well-documented, hard-to-misuse tools dramatically improve the per-step success rate.

------------------------------------------------------------------------

## 7. Best Practices

### Agent Design

- **Maintain simplicity**: Resist over-engineering. A well-designed agent is just an LLM using tools in a loop.
- **Show planning steps**: Expose the agent’s reasoning (`Thought:` traces) for debugging and user trust.
- **Set explicit stopping conditions**: `max_steps`, time budget, or a `finish` tool the LLM must call.

### Tool Design (Agent-Computer Interface)

- **Design tools like APIs for a junior developer**: Clear descriptions, explicit parameter types, example usage in the docstring.
- **Use absolute paths, not relative**: Avoids errors when the agent changes directories.
- **Prefer idempotent tools**: Tools that can be safely retried without side effects.
- **Poka-yoke your tools**: Make it hard to call a tool incorrectly (e.g., require explicit confirmation parameters for destructive operations).

### Safety and Reliability

- **Sandbox the environment**: Agents should not be able to affect systems outside their intended scope.
- **Require approval for irreversible actions**: Delete, send, deploy — always confirm.
- **Log every step**: `[step N] Tool: X | Args: Y | Result: Z` — essential for post-hoc debugging.
- **Test in sandboxed environments extensively** before giving access to production systems.

------------------------------------------------------------------------

Back to top
