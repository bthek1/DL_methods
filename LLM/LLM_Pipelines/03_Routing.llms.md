# Routing

A workflow pattern that classifies an input and directs it to a specialised downstream handler — enabling separation of concerns and optimised per-type prompts.

Author

Benedict Thekkel

## 1. What is Routing?

**Routing** classifies the input first, then sends it to the best-suited sub-pipeline for that category. Without routing, a single catch-all prompt is forced to handle every input type, which degrades performance for each.

                             ┌──────────────────────────┐
                        ┌───▶│  Handler A (e.g. refunds) │
                        │    └──────────────────────────┘
    Input ──▶ Classifier─┤
                        │    ┌──────────────────────────┐
                        ├───▶│  Handler B (tech support) │
                        │    └──────────────────────────┘
                        │
                        │    ┌──────────────────────────┐
                        └───▶│  Handler C (general FAQ)  │
                             └──────────────────────────┘

The classifier can be: - An **LLM call** (flexible, handles nuance) - A **traditional classifier** (faster, cheaper for well-defined categories) - A **keyword/rule system** (deterministic, ultra-fast for simple cases)

------------------------------------------------------------------------

## 2. When to Use Routing

**Good fit when:** - There are distinct input categories that are genuinely better handled differently - Optimising one category’s prompt hurts another category’s performance - You want to route easy queries to cheaper/faster models and hard ones to more capable models - Input classification can be done accurately (LLM or traditional classifier)

**Examples:** - **Customer service**: General questions → FAQ bot; Refunds → billing pipeline; Technical issues → escalation pipeline - **Model cost optimisation**: Simple/common questions → Claude Haiku; Complex/rare questions → Claude Sonnet - **Document processing**: Invoices → invoice extractor; Contracts → contract analyser; Emails → email responder - **Code assistance**: Bug reports → debugging prompt; Feature requests → design prompt; Questions → explanation prompt

------------------------------------------------------------------------

## 3. Implementation Pattern

``` python
from openai import OpenAI
from enum import Enum

client = OpenAI()

class QueryType(str, Enum):
    REFUND = "refund"
    TECHNICAL = "technical"
    GENERAL = "general"


def classify_query(query: str) -> QueryType:
    """Use a fast model to classify the incoming query."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # cheap + fast for classification
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the customer query into exactly one category: "
                    "'refund', 'technical', or 'general'. Respond with only the category name."
                ),
            },
            {"role": "user", "content": query},
        ],
    )
    raw = response.choices[0].message.content.strip().lower()
    return QueryType(raw)


SYSTEM_PROMPTS = {
    QueryType.REFUND: (
        "You are a billing specialist. Help the customer with their refund request. "
        "Be empathetic, explain the refund policy clearly, and collect any required information."
    ),
    QueryType.TECHNICAL: (
        "You are a senior technical support engineer. Diagnose the issue step by step. "
        "Ask clarifying questions and provide actionable solutions."
    ),
    QueryType.GENERAL: (
        "You are a helpful customer service agent. Answer general questions concisely "
        "and direct the customer to the right resource if needed."
    ),
}


def route_and_respond(query: str) -> str:
    """Classify the query and route to the appropriate specialised handler."""
    query_type = classify_query(query)
    system_prompt = SYSTEM_PROMPTS[query_type]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content


# Usage
answer = route_and_respond("I was charged twice for my subscription last month.")
```

------------------------------------------------------------------------

## 4. Cost-Optimised Routing Pattern

Route by **query complexity** to balance cost vs quality:

``` python
def route_by_complexity(query: str) -> str:
    """Use a cheap model for easy queries, expensive model for hard ones."""
    complexity = classify_complexity(query)  # 'simple' or 'complex'

    model = "claude-haiku-4-5" if complexity == "simple" else "claude-sonnet-4-5"

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}],
    )
    return response.choices[0].message.content
```

This pattern can reduce inference costs by 60-80% while maintaining quality on complex queries.

------------------------------------------------------------------------

## 5. Best Practices

- **Keep categories mutually exclusive**: Ambiguous boundaries cause misclassification. Define categories carefully.
- **Add a catch-all fallback**: Always handle `unknown` or low-confidence classifications gracefully.
- **Validate classifier accuracy first**: Before building all downstream handlers, confirm the classifier achieves \>90% accuracy on a representative sample.
- **Log classification decisions**: Debugging misroutes is much easier with a trace of what category was assigned.
- **Prefer traditional classifiers for high-throughput paths**: A fine-tuned BERT classifier is 100× cheaper and faster than an LLM for classification — use LLM classification only when categories are nuanced or frequently changing.
- **Use confidence thresholds**: If classifier confidence is low, route to a human or a more capable model.

------------------------------------------------------------------------

Back to top
