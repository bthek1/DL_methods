# RAG

RAG

Author

Benedict Thekkel

## **1. What is Retrieval-Augmented Generation (RAG)?**

Retrieval-Augmented Generation (**RAG**) is a method that enhances **Large Language Models (LLMs)** by integrating **external knowledge retrieval** before generating responses. Unlike standard LLMs that rely solely on pre-trained knowledge, RAG **retrieves relevant documents** from a knowledge base and **injects them into the prompt** before inference.

### **Key Features of RAG:**

- **Improves factual accuracy** by pulling real-time or domain-specific knowledge.
- **Reduces hallucinations** by grounding the model in reliable sources.
- **Enhances performance on domain-specific tasks** such as finance, healthcare, and legal analysis.
- **Eliminates the need for full fine-tuning** by dynamically incorporating external information.

------------------------------------------------------------------------

### Key subsections / decision areas in production RAG

**Chunking** — size, overlap, strategy (fixed, semantic, hierarchical)

**Embedding** — model choice, dimensionality, batching, versioning

**Indexing** — vector DB choice, ANN algorithm (HNSW vs IVF), index params

**Retrieval** — dense vs sparse vs hybrid, top-k, filtering

**Re-ranking** — cross-encoder, when to apply, latency tradeoff

**Query transformation** — expansion, HyDE, step-back

**Context assembly** — ordering, compression, token budgeting

**Generation** — prompt template, grounding instructions, citation format

**Evaluation** — RAGAS metrics, test set construction, regression tracking

**Observability** — logging query + chunks + answer together, latency tracing

**Index lifecycle** — incremental updates, stale detection, re-indexing triggers

**Access control** — tenant filtering before vector search, never post-filter

**Caching** — query-level, embedding-level, document-level

## **2. How RAG Works**

### **Step-by-Step Process**

1.  **User Query**: The user inputs a question or request.
2.  **Retrieval**: A search engine or vector database retrieves the most relevant documents from a knowledge source.
3.  **Augmentation**: The retrieved documents are inserted into the model’s prompt.
4.  **Generation**: The LLM processes both the query and retrieved information to generate a response.

### **Architecture Overview**

Below is a high-level architecture of RAG:

    User Query → Embedding Model → Vector Database → Top-K Document Retrieval → Prompt Augmentation → LLM → Response

### **Diagram: Basic RAG Workflow**

    +-------------+        +----------------+        +--------------------+        +-------------+
    | User Query  | ---->  |  Document Index| ---->  |  LLM Augmentation  | ---->  |  AI Response |
    +-------------+        +----------------+        +--------------------+        +-------------+

------------------------------------------------------------------------

## **3. Comparing RAG with Traditional LLM Methods**

| Feature                  | Standard LLMs | Fine-Tuned LLMs     | RAG     |
|--------------------------|---------------|---------------------|---------|
| **External Knowledge**   | ❌ No         | ✅ Limited          | ✅ Yes  |
| **Memory Efficiency**    | ✅ Yes        | ❌ No (New Weights) | ✅ Yes  |
| **Real-Time Updates**    | ❌ No         | ❌ No               | ✅ Yes  |
| **Accuracy Improvement** | ❌ Limited    | ✅ Yes              | ✅ Yes  |
| **Scalability**          | ✅ High       | ❌ Costly           | ✅ High |

------------------------------------------------------------------------

## **4. Implementing RAG in Python**

A basic **RAG system** consists of: - **LLM (e.g., OpenAI GPT, Mistral, LLaMA)** - **Vector database (e.g., FAISS, Pinecone, ChromaDB)** - **Embedding model (e.g., SentenceTransformers, OpenAI embeddings)**

### **Step 1: Install Dependencies**

``` bash
pip install langchain faiss-cpu openai sentence-transformers
```

### **Step 2: Load a Knowledge Base**

``` python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load documents
loader = TextLoader("data.txt")
documents = loader.load()

# Split into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
```

### **Step 3: Generate Embeddings and Store in FAISS**

``` python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert documents to embeddings
doc_texts = [doc.page_content for doc in docs]
doc_embeddings = embedding_model.encode(doc_texts)

# Store embeddings in FAISS
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))
```

### **Step 4: Retrieve Relevant Documents**

``` python
def retrieve_documents(query, k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    retrieved_texts = [doc_texts[i] for i in indices[0]]
    return "\n".join(retrieved_texts)

query = "Explain neural networks."
retrieved_docs = retrieve_documents(query)
print(retrieved_docs)
```

### **Step 5: Pass Retrieved Data to LLM**

``` python
import openai

def generate_rag_response(query):
    retrieved_docs = retrieve_documents(query)
    
    prompt = f"Use the following retrieved documents to answer:\n\n{retrieved_docs}\n\nUser: {query}\nAssistant:"
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are an AI assistant."},
                  {"role": "user", "content": prompt}]
    )
    
    return response["choices"][0]["message"]["content"]

# Example query
response = generate_rag_response("What is deep learning?")
print(response)
```

------------------------------------------------------------------------

## **5. Evaluating RAG Performance**

To measure the effectiveness of RAG, compare: - **Retrieval Accuracy**: How relevant are the retrieved documents? - **Response Quality**: Does the LLM provide accurate answers based on the retrieval? - **Latency**: Is retrieval slowing down response generation?

### **Performance Metrics**

| Metric               | Description                                         |
|----------------------|-----------------------------------------------------|
| **Recall@K**         | Percentage of correct documents retrieved           |
| **BLEU Score**       | Measures text similarity to ground truth            |
| **Response Latency** | Measures time taken to retrieve + generate response |

### **Graph: Accuracy Improvement Using RAG**

This graph shows accuracy improvement with RAG compared to a standalone LLM.

    Accuracy (%)
    │
    │   90 ──  RAG-based LLM
    │
    │   75 ──  Fine-tuned LLM
    │
    │   60 ──  Standard LLM
    │
    └───────────────────
            LLM Type

------------------------------------------------------------------------

## **6. Scaling RAG for Large Applications**

For production-scale RAG systems: 1. **Use Distributed Vector Databases** (Pinecone, Weaviate) instead of FAISS. 2. **Pre-filter Documents** to improve retrieval speed. 3. **Optimize Context Window** to avoid overloading the LLM. 4. **Use Hybrid Search** (BM25 + Embeddings) for better recall.

------------------------------------------------------------------------

## **7. Challenges and Considerations**

| Challenge | Solution |
|----|----|
| **Latency in Retrieval** | Optimize vector search, use caching |
| **Memory Consumption** | Use compressed embeddings, distributed storage |
| **Data Drift** | Regularly update knowledge base |
| **Hallucination Despite RAG** | Filter retrieved documents to ensure factual consistency |

------------------------------------------------------------------------

## **8. Advanced RAG Techniques**

### **Multi-Stage RAG**

Instead of a single retrieval step, **multi-stage RAG** refines retrieval by applying **re-ranking algorithms**.

### **Graph-Based RAG**

Instead of keyword matching, **knowledge graphs** can retrieve structured facts more accurately.

### **Agent-Based RAG**

Combines RAG with **autonomous AI agents** that perform multiple reasoning steps before generating a response.

------------------------------------------------------------------------

## **9. Real-World Use Cases**

| Industry | Application |
|----|----|
| **Healthcare** | Medical chatbots retrieving up-to-date research papers |
| **Legal** | AI-powered legal document search and Q&A |
| **Finance** | Market analysis by retrieving real-time reports |
| **Customer Support** | AI assistants providing support from internal documentation |

------------------------------------------------------------------------

## **10. RAG Paradigm Evolution: Naive → Advanced → Modular**

The RAG research landscape has evolved through three distinct paradigms (Gao et al., 2023):

### Naive RAG

The original pattern: **Index → Retrieve → Generate**. One retrieval step using the raw user query; straightforward but limited.

**Weaknesses:** Low retrieval precision, hallucinations when retrieved chunks are irrelevant, no ability to adapt when the first retrieval fails.

------------------------------------------------------------------------

### Advanced RAG

Adds **pre-retrieval** and **post-retrieval** processing steps around the same core pipeline.

| Stage | Techniques |
|----|----|
| Pre-retrieval | Query rewriting, HyDE, step-back prompting, query expansion |
| Indexing | Hierarchical chunks, metadata enrichment, sliding window |
| Post-retrieval | Re-ranking, context compression, lost-in-the-middle mitigations |

Covered in detail across notebooks 02–08 in this series.

------------------------------------------------------------------------

### Modular RAG

A **flexible, composable architecture** where modules can be swapped, added, or reordered. Goes beyond the fixed retrieve-then-generate chain.

New patterns enabled: - **Iterative retrieval** — retrieve → generate → retrieve → … (ITER-RETGEN) - **Recursive retrieval** — progressively refine queries (IRCoT, ToC) - **Adaptive retrieval** — LLM decides *when* to retrieve (FLARE, Self-RAG) - **Tool use** — retrieval is one of many tools an agent can call (Agentic RAG) - **Knowledge graphs** — structured retrieval alongside vector search (Graph RAG) - **Multi-modal** — retrieve images, audio, video alongside text

This series follows the Modular RAG philosophy — each notebook covers one replaceable module.

------------------------------------------------------------------------

| Paradigm | Structure | Flexibility | Best for |
|----|----|----|----|
| Naive RAG | Fixed: Index→Retrieve→Generate | Low | Simple Q&A, prototypes |
| Advanced RAG | Fixed + pre/post steps | Medium | Production text search |
| Modular RAG | Composable | High | Complex enterprise systems |

------------------------------------------------------------------------

## **11. Summary**

- **RAG enhances LLMs by integrating real-time information retrieval**.
- **It prevents hallucinations and improves factual accuracy**.
- **Using vector databases like FAISS or Pinecone enables fast retrieval**.
- **Hybrid search and multi-stage retrieval can further optimize results**.
- **Scaling RAG requires optimizing retrieval efficiency and memory usage**.

By leveraging RAG, **LLMs can become more accurate, reliable, and adaptable without expensive fine-tuning**.

------------------------------------------------------------------------

Back to top
