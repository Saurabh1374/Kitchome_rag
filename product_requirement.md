# Product Requirement Document (PRD): Kitchome Multi-Domain RAG Application

## 1. Executive Summary
**Kitchome RAG** is a domain-aware Retrieval-Augmented Generation system designed for kitchen, appliance, culinary, and home management knowledge. Because vector data spans across distinct domains (e.g., cooking recipes, appliance user manuals, interior design/decor, maintenance guides), searching across a unified unpartitioned vector space yields domain noise and degraded retrieval quality.

To solve this, Kitchome RAG implements a two-stage retrieval strategy:
1. **Namespace Resolution (`skills.ms`)**: Querying `skills.ms` (Skill & Index Namespace Router) first to identify the exact index namespace(s) matching the query context.
2. **Domain-Targeted Vector Search**: Executing similarity search strictly within the determined index namespace(s).

---

## 2. Problem Statement & Architecture Vision

### The Problem
Traditional RAG architectures execute vector search across a monolith index or indiscriminately across all namespaces. For broad domain platforms (such as Kitchen & Home), searching a recipe for "air fryer wings" might retrieve irrelevant chunks from an "air fryer user manual troubleshooting guide".

### The Solution: `skills.ms` Router Architecture
```
                     +---------------------------------------+
                     |              User Query               |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |    skills.ms (Namespace Router)       |
                     |  - Domain classification              |
                     |  - Index namespace mapping            |
                     +---------------------------------------+
                                         |
                       Target Namespace: "recipes_culinary"
                                         v
                     +---------------------------------------+
                     |  Vector Store (Target Namespace Only) |
                     |  - Dense embedding search             |
                     |  - Filtered top-k retrieval           |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |     Context Reranking & LLM RAG       |
                     +---------------------------------------+
```

---

## 3. Product Features & Requirements

### 3.1. `skills.ms` Namespace Registry & Routing
- **Skill Mapping**: Centralized registry mapping domain skills and capabilities to vector database index namespaces (e.g., `recipes_culinary`, `appliances_troubleshooting`, `home_decor_design`, `cleaning_maintenance`).
- **Namespace Resolution Engine**: Takes user query input, determines domain intent and required skills, and outputs designated target namespace(s) along with fallback strategies.
- **Dynamic Skill Expansion**: Supports registering new domain skills and vector namespaces without downtime.

### 3.2. Data Ingestion Pipeline & Progressive Summarization
- **Multi-Format Document Parsing**: Support for PDF, Markdown, HTML, JSON, and raw text files.
- **Progressive Streaming Summarization & 700+150 Token Purge**:
  - As chunking proceeds, chunk summaries are continuously appended to a running summary buffer.
  - When the running summary reaches **700 tokens** (+ **150 token buffer** allowance = 850 tokens), the verbose buffer is **purged** and compressed into a tight high-level summary representation.
  - Continues progressively until EOF.
- **Dynamic `skills.ms` Meta Store Update (<1500 words)**:
  - Upon reaching EOF, generates a final overall document summary under **1500 words** and saves it directly into the runtime `skills.ms` properties file (`skills_registry.json`).
- **Semantic & Adaptive Chunking**: Configurable ~700-token chunk sizes.
- **Partitioned Index Loading**: Automated upserting into designated vector store namespaces according to `skills.ms` mappings.

### 3.3. Retrieval & Generation Pipeline
- **Namespace-Scoped Search**: Enforce vector similarity search within single or multi-namespace bounds identified by `skills.ms`.
- **Hybrid / Dense Search**: Support dense vector search with metadata filtering.
- **Re-Ranking & Context Synthesis**: Optional cross-encoder re-ranking for top retrieved candidates before prompting LLM.
- **Grounded Generation**: Citation-backed LLM responses with source document provenance.

---

## 4. Technical Stack
- **Language**: Python 3.10+
- **Vector Database**: **`pgvector` (PostgreSQL with `pgvector` extension)** (`src/vector_store/pgvector_store.py`), supporting native `skills.ms` namespace indexing via PostgreSQL JSONB & HNSW/Cosine distance vectors, with automatic local fallback.
- **Embedding Framework**: Native dense feature projection engine with optional `sentence-transformers` / `openai` support
- **Skill Router (`skills.ms`)**: Intent classifier & metadata namespace registry (`src/skills_ms/`)
- **Data Validation & Pipeline**: `pydantic`, `pgvector`, `psycopg2-binary`, `numpy`, `pytest`

---

## 5. Implementation Milestones

| Milestone | Target Deliverable | Status |
| :--- | :--- | :--- |
| **Milestone 1** | PRD & Architecture Plan | **Completed** |
| **Milestone 2** | `skills.ms` Namespace Registry & Routing Service | **Completed** |
| **Milestone 3** | Multi-Domain Ingestion Pipeline | **Completed** |
| **Milestone 4** | PostgreSQL `pgvector` Database Integration | **Completed** |
| **Milestone 5** | RAG Query Engine & Evaluation | **Next** |

---

## 6. Verification & Quality Acceptance Criteria
- **Namespace Isolation Test**: Querying "How to fix error code E3 on instant pot" routes to `appliances_troubleshooting` namespace, while "How to make instant pot chili" routes to `recipes_culinary`.
- **Ingestion Validation**: Ingestion pipeline processes benchmark documents and creates correctly metadata-tagged chunks in designated namespaces.
- **Accuracy**: RAG answers maintain high relevance score without cross-domain hallucinations.
