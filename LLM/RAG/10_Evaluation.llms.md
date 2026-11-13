# Evaluation in RAG

Evaluation measures how well your RAG pipeline retrieves relevant context and generates faithful, accurate answers. It covers metric choice, test set construction, and regression tracking.

Author

Benedict Thekkel

## 1. Why RAG Evaluation Is Hard

RAG has two components that can each fail independently — the retriever and the generator. A correct-sounding answer may be unfaithful to the retrieved context; a relevant chunk may still not be used correctly.

**Three failure modes:** - Retriever returns wrong/irrelevant chunks (retrieval failure) - Generator ignores or misuses the correct chunk (generation failure) - Correct chunk exists in the corpus but isn’t retrieved (coverage failure)

**Evaluation must cover all three layers separately and end-to-end.**

------------------------------------------------------------------------

## 2. Retrieval Metrics

These metrics evaluate how good the retrieved chunks are, independent of generation.

| Metric | What It Measures |
|----|----|
| **Precision@K** | Fraction of top-K retrieved chunks that are relevant |
| **Recall@K** | Fraction of all relevant chunks that appear in top-K |
| **MRR** (Mean Reciprocal Rank) | How high the first relevant chunk is ranked |
| **NDCG** (Normalized Discounted Cumulative Gain) | Ranks weighted by position and relevance grade |
| **Context Relevance** (RAGAS) | LLM-judged: are the retrieved chunks relevant to the question? |

**Practical minimum:** track Recall@K (did we retrieve the chunk that contains the answer?) and Context Relevance.

------------------------------------------------------------------------

## 3. Generation Metrics

These metrics evaluate how good the generated answer is given the retrieved context.

| Metric | What It Measures |
|----|----|
| **Faithfulness** (RAGAS) | Is every claim in the answer supported by the retrieved context? |
| **Answer Relevance** (RAGAS) | Does the answer actually address the question asked? |
| **Answer Correctness** | Is the answer factually correct against a ground truth? |
| **Completeness** | Does the answer cover all aspects of the question? |
| **ROUGE / BLEU** | Lexical overlap with a reference answer (weak signal for open-ended QA) |

**Key insight:** Faithfulness and Answer Relevance are the two most actionable RAGAS metrics. A low faithfulness score points to generation failure; a low answer relevance score often points to retrieval failure.

------------------------------------------------------------------------

## 4. RAGAS Framework

RAGAS (Retrieval Augmented Generation Assessment) is the most widely used framework for RAG evaluation. It computes LLM-judged metrics without requiring human-labelled answers for every question.

**Core RAGAS metrics:**

| Metric | Formula Intuition |
|----|----|
| **Faithfulness** | `# claims in answer supported by context / # total claims in answer` |
| **Answer Relevance** | Reverse-engineer questions from the answer; cosine similarity to original question |
| **Context Precision** | Proportion of retrieved chunks that are relevant |
| **Context Recall** | Proportion of ground-truth statements covered by retrieved chunks |

``` python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

data = {
    "question": ["What is the capital of France?"],
    "answer": ["Paris is the capital of France."],
    "contexts": [["France is a country in Europe. Its capital is Paris."]],
    "ground_truth": ["Paris"],
}

result = evaluate(Dataset.from_dict(data), metrics=[faithfulness, answer_relevancy])
print(result)
```

------------------------------------------------------------------------

## 5. Building a Test Set

A good evaluation test set is the foundation of trustworthy metrics.

**Construction approaches:**

| Approach | Description | Cost |
|----|----|----|
| **Manual curation** | Human experts write Q&A pairs | High quality, expensive |
| **Synthetic generation** | LLM generates (question, answer, context) triples from your corpus | Fast, scalable |
| **User query mining** | Sample real production queries and label answers | Reflects actual use |
| **Adversarial** | Craft hard questions: multi-hop, negations, out-of-scope | Tests edge cases |

**RAGAS TestsetGenerator:**

``` python
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

generator = TestsetGenerator.with_openai()
testset = generator.generate_with_langchain_docs(
    documents,
    test_size=50,
    distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
)
```

**Rule of thumb:** 100–200 diverse questions is enough to track meaningful regressions for most production pipelines.

------------------------------------------------------------------------

## 6. Offline vs. Online Evaluation

| Mode | When | What |
|----|----|----|
| **Offline (batch)** | Pre-deployment / CI | Run test set through pipeline, compute metrics, compare to baseline |
| **Online (production)** | Post-deployment | Sample live traffic, score with LLM judge, track dashboards |

**Offline evaluation checklist:** - Pin all versions (embedding model, retriever config, LLM, prompt template) - Log raw inputs/outputs alongside metrics for debugging - Set a minimum acceptable threshold per metric (e.g., Faithfulness ≥ 0.85) - Fail CI if any metric regresses beyond a tolerance (e.g., ≥ 5% relative drop)

**Online sampling:** - Don’t score 100% of traffic — sample 5–10% to control LLM judge cost - Use stratified sampling to cover rare query types

------------------------------------------------------------------------

## 7. LLM-as-Judge

Using a strong LLM (e.g., GPT-4o) to judge outputs is now standard because it scales better than human annotation and correlates well with human judgment.

**Prompt pattern:**

    You are an expert evaluator. Given the question, retrieved context, and answer below,
    rate the faithfulness of the answer on a scale of 1–5.

    Question: {question}
    Context: {context}
    Answer: {answer}

    Faithfulness score (1=completely unfaithful, 5=fully supported): 

**Pitfalls:** - Judge LLM can be biased toward longer or more confident-sounding answers - Self-consistency: same judge returns different scores on re-runs → average over 3 runs - Use a different model as judge than as generator to avoid self-preference bias

------------------------------------------------------------------------

## 8. Regression Tracking

Treat evaluation like a software test suite: every pipeline change should be validated against the baseline before shipping.

**Workflow:**

    1. Commit pipeline change (new embedding model, chunk size, prompt, etc.)
    2. Run eval suite on pinned test set
    3. Compare metric deltas to baseline
    4. If any metric drops > threshold → block merge
    5. If all metrics pass → update baseline, merge

**Tools:** - **MLflow / W&B** — log eval runs, compare experiments, track metric history - **RAGAS CI integration** — run `ragas evaluate` in GitHub Actions - **LangSmith / Phoenix** — built-in dataset + eval runner with regression dashboards

**Baseline storage:**

``` json
{
  "version": "2024-11-01",
  "faithfulness": 0.91,
  "answer_relevancy": 0.87,
  "context_recall": 0.79
}
```

------------------------------------------------------------------------

## 9. End-to-End vs. Component Evaluation

| Level | What to Measure | When to Use |
|----|----|----|
| **Component: Retriever** | Recall@K, Precision@K | Optimising chunk size, embedding model, top-K |
| **Component: Reranker** | NDCG, precision delta vs. no reranker | Evaluating reranker effectiveness |
| **Component: Generator** | Faithfulness, Answer Relevance | Optimising prompts, models |
| **End-to-end** | Answer Correctness, user satisfaction | Validating overall pipeline |

**Tip:** Start with end-to-end. If overall quality is poor, drill into components to isolate the bottleneck.

------------------------------------------------------------------------

## Summary

| Metric             | Scope      | Requires ground truth? |
|--------------------|------------|------------------------|
| Context Relevance  | Retriever  | No (LLM judge)         |
| Faithfulness       | Generator  | No (LLM judge)         |
| Answer Relevance   | Generator  | No (LLM judge)         |
| Context Recall     | Retriever  | Yes                    |
| Answer Correctness | End-to-end | Yes                    |

The **no-reference** metrics (Faithfulness, Answer Relevance, Context Relevance) are your day-to-day production monitors. The **reference-based** metrics are for periodic deep-dives with curated test sets.

------------------------------------------------------------------------

Back to top
