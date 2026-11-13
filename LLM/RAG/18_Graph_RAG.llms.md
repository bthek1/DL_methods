# Graph RAG

Graph RAG combines knowledge graphs with vector retrieval to capture structured relationships between entities — enabling multi-hop reasoning and relationship-aware answers that flat vector search cannot provide.

Author

Benedict Thekkel

## 1. Why Flat Vector Search Is Not Enough

Vector search retrieves semantically similar passages but ignores the relationships *between* entities. It struggles with:

| Query type | Problem with vector-only RAG |
|----|----|
| *“How does drug X affect patients with condition Y who also take drug Z?”* | Three-hop reasoning required across entities |
| *“What are all subsidiaries of Company A?”* | Requires traversing a corporate hierarchy |
| *“Find all papers that cite both Author X and Author Y”* | Structural graph query, not a semantic one |
| *“What are the downstream consequences of policy change P?”* | Causal chain traversal |

Knowledge graphs represent entities as nodes and relationships as labelled directed edges, enabling exact and multi-hop queries that vector similarity simply cannot answer.

------------------------------------------------------------------------

## 2. Knowledge Graph Basics

A knowledge graph is a set of **triples**: `(subject, predicate, object)`.

    (Aspirin,     inhibits,          COX-2 enzyme)
    (COX-2 enzyme, involved_in,       inflammation_pathway)
    (ibuprofen,   also_inhibits,     COX-2 enzyme)
    (Aspirin,     contraindicated_with, warfarin)

**Graph terminology:** - **Node** — an entity (drug, person, concept, document) - **Edge** — a directed, labelled relationship - **Subgraph** — a connected subset of nodes and edges - **Community** — a cluster of densely connected nodes

**Storage options:** - Graph databases: Neo4j, Amazon Neptune, TigerGraph - In-memory: NetworkX (Python), small-scale prototyping - Hybrid: store graph structure in Neo4j, embeddings in a vector DB, then join at query time

------------------------------------------------------------------------

## 3. Building a Knowledge Graph from Documents

### Step 1: Entity and relation extraction

``` python
EXTRACTION_PROMPT = """
Extract all entities and relationships from this text as a list of triples.
Format: (subject | predicate | object)
Only extract explicit facts — do not infer.

Text: {text}
"""

# Example output:
# (Pfizer | manufactures | Paxlovid)
# (Paxlovid | approved_for | COVID-19 treatment)
# (Paxlovid | contraindicated_with | strong CYP3A inhibitors)
```

### Step 2: Entity resolution (deduplication)

Different mentions of the same entity must be merged:

``` python
# "USA", "United States", "U.S." → canonical entity "United States"
# Use fuzzy matching + embedding similarity to identify co-referents
from rapidfuzz import fuzz

def are_same_entity(a: str, b: str, threshold=85) -> bool:
    return fuzz.ratio(a.lower(), b.lower()) > threshold
```

### Step 3: Load into graph database

``` python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

def add_triple(tx, subject, predicate, obj, source_doc):
    tx.run(
        "MERGE (s:Entity {name: $subject}) "
        "MERGE (o:Entity {name: $object}) "
        "MERGE (s)-[r:RELATION {type: $predicate, source: $source}]->(o)",
        subject=subject, predicate=predicate, object=obj, source=source_doc
    )

with driver.session() as session:
    for triple in extracted_triples:
        session.execute_write(add_triple, *triple, source_doc="doc_id")
```

------------------------------------------------------------------------

## 4. Retrieval Strategies

### 4.1 Entity-anchored retrieval

1.  Extract entities from the query
2.  Look them up in the graph
3.  Traverse N hops from those anchor nodes
4.  Return the subgraph as context

``` python
def graph_retrieve(query: str, hops: int = 2) -> list[dict]:
    # Step 1: Extract entities from query
    entities = extract_entities(query)  # e.g. ["Aspirin", "warfarin"]
    
    # Step 2: Find anchor nodes
    # Step 3: Traverse N hops
    cypher = """
    MATCH (start:Entity)-[r*1..{hops}]-(related:Entity)
    WHERE start.name IN $entities
    RETURN start, r, related
    LIMIT 50
    """.format(hops=hops)
    
    return session.run(cypher, entities=entities).data()
```

### 4.2 Hybrid: vector + graph

Vector search finds semantically relevant nodes; graph traversal explores their neighbourhood:

``` python
def hybrid_retrieve(query: str) -> list[str]:
    # 1. Vector search to find relevant entity nodes
    seed_entities = vector_store.search(query, top_k=5)
    
    # 2. Expand via graph traversal
    subgraph = graph.expand(seed_entities, hops=2)
    
    # 3. Convert subgraph triples to natural language
    return triples_to_text(subgraph)
```

### 4.3 Community-based summarisation (Microsoft GraphRAG)

Group graph nodes into communities (using Leiden or Louvain algorithm). Pre-generate a summary for each community. At query time, retrieve the most relevant community summaries:

    - Faster for broad queries ("tell me about the pharmaceutical industry")
    - Higher cost at index time (summaries must be generated per community)
    - Enables global reasoning across the entire corpus

------------------------------------------------------------------------

## 5. From Subgraph to LLM Context

Raw graph triples are not great LLM input. Convert to a readable format first:

``` python
def triples_to_text(triples: list[tuple]) -> str:
    """Convert (subject, predicate, object) triples to readable sentences."""
    lines = []
    for s, p, o in triples:
        # Simple template
        pred_readable = p.replace("_", " ")
        lines.append(f"{s} {pred_readable} {o}.")
    return "\n".join(lines)

# Or use an LLM to summarise the subgraph:
SUBGRAPH_PROMPT = """
Summarise the following knowledge graph facts in 2-3 sentences,
focusing on what is relevant to the question: {question}

Facts:
{triples}
"""
```

**Hybrid context pattern:** Combine graph-derived facts with retrieved text chunks:

    System: Answer using both the structured facts and the document excerpts below.

    [Graph facts]
    Aspirin inhibits COX-2 enzyme.
    Aspirin is contraindicated with warfarin.

    [Document excerpts]
    ...relevant passage from clinical guidelines...

------------------------------------------------------------------------

## 6. Trade-offs and When to Use Graph RAG

| Dimension | Vector-only RAG | Graph RAG |
|----|----|----|
| Index build cost | Low | High (entity extraction + graph construction) |
| Maintenance | Easy (re-embed changed docs) | Hard (graph updates, deduplication) |
| Multi-hop queries | Poor | Excellent |
| Semantic similarity | Excellent | Moderate (entity matching) |
| Relationship queries | Poor | Excellent |
| Global corpus queries | Poor | Good (community summaries) |

**Use Graph RAG when:** - Your domain has rich entity relationships (healthcare, finance, legal, knowledge bases) - Users ask multi-hop questions spanning many documents - You need to answer structural queries (“all X connected to Y”) - You need explainable reasoning paths (trace the graph edge by edge)

**Avoid when:** - Domain is flat (FAQs, simple Q&A) - High document velocity (graph maintenance is expensive) - Low latency is critical (graph queries + triple-to-text conversion adds time)

**Key tools:** Microsoft GraphRAG library, LlamaIndex `KnowledgeGraphIndex`, LangChain Neo4j integration, `graphrag` (Microsoft OSS), G-Retriever.

------------------------------------------------------------------------

Back to top
