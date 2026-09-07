# Kitchome Enterprise Multi-Domain RAG

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL pgvector](https://img.shields.io/badge/pgvector-supported-336791.svg)](https://github.com/pgvector/pgvector)
[![Architecture](https://img.shields.io/badge/Architecture-Ensemble%20Namespace%20RAG-orange.svg)]()
[![Security](https://img.shields.io/badge/Security-RBAC%20%2B%20ABAC-green.svg)]()

**Kitchome RAG** is a domain-aware, enterprise-grade Retrieval-Augmented Generation system. Unlike traditional RAG implementations that execute similarity search across an unpartitioned monolithic vector database, Kitchome partitions knowledge into target **Index Namespaces** (`skills.ms`). 

It combines **two-channel ensemble routing**, **dual-resolution retrieval** (progressive document summaries + body chunks), **headless multi-tier authorization (RBAC + ABAC)**, and an automated **Strategy E evaluation telemetry loop**.

---

## 1. Combined Retrieval & Generation Architecture

```mermaid
flowchart TD
    UserQuery([User Query + Auth Context]) --> AuthGuard[1. Headless AuthN / AuthZ Guard]
    
    subgraph Security ["Security Layer (RBAC + ABAC)"]
        AuthGuard --> TokenParse[Validate Token / Claims]
        TokenParse --> TierScope[Extract User Tier & Allowed Namespaces]
    end

    TierScope --> EnsembleRouter[2. skills.ms Ensemble Router]

    subgraph RoutingEngine ["Ensemble Routing Engine"]
        EnsembleRouter --> LexicalChannel[Channel 1: Lexical Scorer\n- Word boundary regex\n- Mutual exclusion & damping]
        EnsembleRouter --> SemanticChannel[Channel 2: Semantic Prototype\n- Cosine sim to exemplar centroids\n- Zero extra model latency]
        LexicalChannel --> Fusion[Score Fusion: α·S_sem + 1-α·S_lex]
        SemanticChannel --> Fusion
        Fusion --> MarginCheck{Confidence Margin\nΔ = S1 - S2 ≥ 0.20?}
        MarginCheck -- High Confidence --> SingleNS[Single Namespace]
        MarginCheck -- Ambiguous / Tie --> MultiNS[Top-2 Namespaces]
    end

    SingleNS --> ScopeIntersection[Intersect with User's Allowed Namespaces]
    MultiNS --> ScopeIntersection

    ScopeIntersection --> VectorSearch[3. Vector Store Search with SQL ABAC Predicates]

    subgraph VectorDB ["pgvector Partitioned Namespaces"]
        VectorSearch --> DualResolution[Dual-Resolution Search\nSummary Chunks + Body Chunks]
        DualResolution --> SQLFilter["WHERE namespace IN (...) AND metadata @> ABAC_Claims"]
    end

    SQLFilter --> FallbackGuard{4. Strategy E Check\nTop Similarity ≥ 0.40?}
    
    subgraph StrategyE ["Fallback & Telemetry Loop"]
        FallbackGuard -- Yes --> RetrievedChunks[Candidate Chunks]
        FallbackGuard -- Low Relevance --> TriggerFallback[Expand to Secondary Namespaces]
        TriggerFallback --> TelemetryLog[Log Telemetry: Precision Metric]
        TriggerFallback --> RetrievedChunks
        FallbackGuard -- Yes --> TelemetryLog
    end

    RetrievedChunks --> PromptBuilder[5. Parent-Context Prompt Augmentation]
    PromptBuilder --> Synthesis[6. Grounded Generation with Citations]
    Synthesis --> FinalOutput([Final Response with Source Provenance])
```

---

## 2. High-Level Functional Specifications

### 2.1. Headless Authentication & Authorization (RBAC + ABAC)
Enforces multi-tier security at the vector database level (defense-in-depth), preventing unauthorized chunks from ever entering LLM memory:

* **RBAC Multi-Tier Access Matrix**:
  * **`free`**: Access to `recipes_culinary` and `general_home`.
  * **`premium`**: Adds `appliances_troubleshooting`, `cleaning_maintenance`, and `home_decor_design`.
  * **`scholar`**: Adds `academia_research` and scientific journals.
  * **`enterprise`**: All public namespaces plus **private tenant partitions** (`tenant_<id>_*`).
* **ABAC Database Predicates**:
  Dynamic SQL/JSONB predicates filter by `tenant_id`, `access_tier`, and `clearance_level`:
  ```sql
  WHERE namespace = ANY(:authorized_namespaces)
    AND (metadata->>'tenant_id' = :tenant_id OR metadata->>'tenant_id' = 'global')
    AND (metadata->>'access_tier' = ANY(:user_tier_levels))
    AND (CAST(metadata->>'clearance_level' AS INT) <= :clearance)
  ```

---

### 2.2. Two-Channel Ensemble Router (`skills.ms`)

The router eliminates the **"Fatal Routing Miss"** hazard (where a query is locked into the wrong namespace with zero recall) by combining deterministic lexical rules with lightweight semantic prototype classification:

```text
                            User Query
                                |
                                v
               +----------------------------------+
               | 1. High-Precision Lexical Rules  |  (Explicit tags, exact domain triggers)
               +----------------------------------+
                                | (Uncertain / Ambiguous)
                                v
               +----------------------------------+
               | 2. Semantic Prototype Router     |  (Lightweight Cosine Similarity against
               |    (Reuses existing Embedder)    |   domain exemplar embeddings; <5ms)
               +----------------------------------+
                                |
                     Confidence Margin Check
                     (Top 1 Score - Top 2 Score)
                               / \
                    High Margin   Low Margin / Tie
                        /             \
                       v               v
             Single-Namespace     Multi-Namespace Search
               Vector Search       (Top-2 namespaces with
                                    cross-reranking)
```

#### How It Works:
1. **Tier 1: High-Precision Lexical Rules ($S_{lex}$)**:
   * Evaluates exact domain triggers, file path patterns, and token-boundary regexes (`\b(recipe|cook|error|stain)\b`).
   * Applies mutual-exclusion damping (e.g., strong culinary verbs suppress appliance/cleaning scores by 80%) to prevent false triggers like *"clean the chicken"* from routing to cleaning.
2. **Tier 2: Semantic Prototype Router ($S_{sem}$)**:
   * When queries are synthetically complex, colloquial, or lack explicit keywords, the query is embedded once via `EmbeddingEngine`.
   * Computes cosine similarity against pre-computed centroid vectors of curated **Exemplar Utterances** for each domain ($<2\text{ ms}$ compute, zero external classifier models).
3. **Confidence Margin Check ($\Delta = S_1 - S_2$)**:
   * **High Margin ($\Delta \ge 0.20$)**: Clear domain winner $\rightarrow$ routes strictly to a **Single Namespace** to optimize latency and eliminate cross-domain noise.
   * **Low Margin / Ambiguity ($\Delta < 0.20$)**: Boundary ambiguity or multi-domain intent (e.g., *"How to clean air fryer basket without peeling coating"*) $\rightarrow$ triggers **Multi-Namespace Search** across Top-2 candidate namespaces, with downstream cross-reranking to guarantee zero recall loss.

---

### 2.3. Dual-Resolution Vector Storage
Every ingested document is indexed in two complementary resolutions:
* **Summary Chunk (`is_summary: true`)**: The progressive document summary (<1500 tokens). Responds directly to high-level, holistic questions (*"Summarize the safety rules for this appliance"*).
* **Body Chunks (`is_summary: false`)**: Detailed 700-token chunks. Responds to specific factoid inquiries (*"Error code E3 replacement part"*).
* **Parent-Context Injection**: When body chunks are retrieved, the parent document summary is injected as a header in the prompt, giving the LLM complete situational awareness.

---

### 2.4. Strategy E Fallback Loop & Telemetry
* **Relevance Guard**: If the top retrieved chunk similarity is below threshold $\gamma$ (e.g., $0.40$), the engine automatically broadens the search across adjacent namespaces.
* **Evaluation Telemetry**: Every query emits a `RetrievalTelemetry` event recording confidence margins, similarity scores, and fallback triggers:
  $$\text{Routing Precision} = 1 - \frac{\text{Fallback Events}}{\text{Total Queries}}$$
  This provides an automated audit trail to discover missing exemplars or out-of-domain knowledge gaps.

---

### 2.5. Event-Driven Controller-Worker Ingestion Architecture

For enterprise-scale documents (e.g., 500-page user guides), synchronous processing triggers HTTP gateway timeouts (504s) and risks memory exhaustion. The system implements a **Database-Backed Event-Driven Controller-Worker Architecture** that slashes processing time to 3–5 minutes without requiring external message brokers (like Redis or Celery).

```mermaid
flowchart TD
    subgraph Intake ["1. Document Intake (Non-Blocking SLA < 10ms)"]
        FileDrop[Channel A: Folder Drop\ndata/appliances/blender.md]
        APIUpload[Channel B: REST API\nPOST /api/v1/ingest]
    end

    subgraph CursorGate ["2. Cursor as Gatekeeper (Idempotency & Versioning)"]
        FileDrop --> ComputeHash[Compute SHA-256 Hash]
        APIUpload --> ComputeHash
        ComputeHash --> CheckCursor{Check ingestion_cursor\nHash matches ACTIVE doc?}
        CheckCursor -- "YES (Unchanged)" --> SkipIngest([Bypass & Skip\n0 Compute Wasted])
        CheckCursor -- "NO (New or Modified)" --> GetVersion[Read previous version from Cursor\nSet new_version = current + 1]
    end

    subgraph ControllerQueue ["3. Controller & DB-Backed Queue (Zero Redis/Celery Overhead)"]
        GetVersion --> CreateJob["INSERT INTO ingestion_queue\nchunk_job_id = cjob_123, status = 'PENDING'\nINSERT INTO ingestion_chunk_jobs\nexpected_chunks = N, status = 'PENDING'"]
        CreateJob --> HTTPResp([HTTP 202 Accepted\nReturns job_id in < 10ms])
        CreateJob --> WorkerClaim["Worker Claims Job\n(FOR UPDATE SKIP LOCKED - Pessimistic Lock)"]
    end

    subgraph WorkerCompute ["4. Worker Execution (O(1) Streaming & Dynamic Lease Renewal)"]
        WorkerClaim --> S1[Stage 1: PARSING\nText Extract via Context Managers]
        S1 --> S2[Stage 2: SUMMARIZATION\nProgressive Buffer Purge at 850 Tokens]
        S2 --> S3["Stage 3: CHUNKING\nStream mini-batches (32 chunks)\nUpdate ingestion_chunk_jobs (chunked_count = N)"]
        S3 --> S4["Stage 4: EMBEDDING\nBatch Vectorization\nHeartbeat: renew crash_recovery_timeout_at"]
        
        S1 -.->|Fail| ErrorLog[Record failed_stage & reason]
        S2 -.->|Fail| ErrorLog
        S3 -.->|Fail| ErrorLog
        S4 -.->|Fail| ErrorLog
        ErrorLog --> RetryCheck{retry_count < 3?}
        RetryCheck -- Yes --> Requeue[Re-queue with Backoff]
        RetryCheck -- No --> Exhausted[FAILED_RETRY_EXHAUSTED\nHalt & Await Manual Retry]
        Requeue --> CreateJob
    end

    subgraph RecoverySweeper ["5. Worker Registry & Crash Recovery Sweeper"]
        WorkerHeartbeat[Worker Registry: ingestion_workers\nTracks liveness every 15-30s]
        Sweeper[Background Sweeper Loop:\nDetects NOW() > crash_recovery_timeout_at\nAND worker DEAD]
        Sweeper --> CheckChunkTable{Chunk Job status == 'CHUNKED'?}
        CheckChunkTable -- Yes --> RapidResume[Skip re-chunking!\nResume directly from embedding]
        CheckChunkTable -- No --> FullRetry[Re-queue to PENDING]
    end

    subgraph Stage5Commit ["6. Stage 5: Atomic State Flip (Zero Downtime)"]
        S4 --> AtomicTx["BEGIN TRANSACTION (< 10ms);\n1. UPDATE kitchome_chunks SET is_latest = false WHERE doc_family = :fam AND job_id != :job_id;\n2. UPDATE kitchome_chunks SET is_latest = true WHERE metadata.job_id = :job_id;\n3. Advance ingestion_cursor (v2 is ACTIVE, v1 is ARCHIVED);\n4. UPDATE ingestion_queue SET status = 'COMPLETED';\n5. Purge ephemeral staging rows in ingestion_job_chunks;\nCOMMIT;"]
    end

    subgraph Storage ["7. Vector Storage & Cursor State"]
        AtomicTx --> VectorStore[("pgvector Store\n- v1 Chunks: is_latest = false (Archived)\n- v2 Chunks: is_latest = true (Active)")]
        AtomicTx --> CursorTable[("ingestion_cursor\nDocument Catalog")]
    end

    subgraph QuashFlow ["8. User-Driven Quash Flow (Zero Clutter)"]
        UserQuash[User: POST /documents/v1/quash] --> DoQuash["1. DELETE FROM kitchome_chunks WHERE document_id = :v1;\n2. UPDATE ingestion_cursor SET status = 'QUASHED';"]
        DoQuash --> VectorStore
        DoQuash --> CursorTable
    end

    style CursorGate fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style ControllerQueue fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    style Stage5Commit fill:#e8f8f5,stroke:#27ae60,stroke-width:2px
    style RecoverySweeper fill:#f1f8e9,stroke:#558b2f,stroke-width:2px
    style QuashFlow fill:#fbe9e7,stroke:#c62828,stroke-width:2px
```

#### Why DB-Backed Async Queue (`SKIP LOCKED`) Over Celery & Redis:
1. **Zero External Infrastructure**: Celery requires deploying, monitoring, and maintaining an external Redis or RabbitMQ cluster. Our architecture reuses PostgreSQL with zero extra services.
2. **ACID Transaction Atomicity**: In Celery, task state (Redis) and vector state (Postgres) live in two separate systems that can desynchronize. In our system, the job completion, cursor advance, and vector write occur in **one atomic database transaction**.
3. **Pessimistic Non-Blocking Locks**: Workers claim jobs via `SELECT ... FOR UPDATE SKIP LOCKED`, guaranteeing that multiple concurrent workers never process the same file, with zero thread contention or deadlocks.

---

### 2.6. Multi-Table Schema & Document Lifecycle

```
[ Table 1: ingestion_cursor (Permanent) ]              [ Table 2: ingestion_queue (Orchestration) ]
- document_id (PK)                                     - job_id (PK)
- file_path (UNIQUE)                                   - document_id (FK)
- content_hash (SHA-256)                               - doc_family & version
- version (e.g. 1, 2)                                  - chunk_job_id (FK to chunk jobs)
- is_latest (bool)                                     - status: PENDING | PROCESSING | COMPLETED |
- status: ACTIVE | ARCHIVED | QUASHED                               FAILED | FAILED_RETRY_EXHAUSTED
- chunks_count                                         - current_stage & failed_stage
- created_at & updated_at                              - worker_id & crash_recovery_timeout_at

[ Table 3: ingestion_chunk_jobs (Chunk Ledger) ]       [ Table 4: ingestion_workers (Registry & Health) ]
- chunk_job_id (PK)                                    - worker_id (PK)
- job_id (FK to queue)                                 - hostname & pid
- expected_chunks (estimated upfront)                  - status: IDLE | BUSY | OFFLINE | DEAD
- chunked_count (progress counter)                     - current_job_id
- chunk_ids (JSON UUID array)                          - last_heartbeat_at
- status: PENDING | CHUNKING | CHUNKED |               - tasks_completed
          EMBEDDING | COMPLETED | FAILED
```

#### The 3-Tier Document Lifecycle:
1. **`ACTIVE` (`is_latest = true`)**: Live chunks in `pgvector`. Target of all default RAG queries.
2. **`ARCHIVED` (`is_latest = false`)**: Chunks remain in `pgvector` but are ignored by standard searches, **eradicating the "KNN Duplicate Swarm" problem**. Only retrieved if a user explicitly requests a dated version.
3. **`QUASHED` (Purged from Vector DB)**: When a user deactivates an old version, chunks are physically purged (`DELETE FROM kitchome_chunks WHERE doc_id = ...`) to free HNSW index memory, while `ingestion_cursor` preserves an immutable audit record.

#### Decoupled Two Worker Pools & Swarm Embedding (Model 1):
* **Worker Pool 1 (`ChunkerWorker`)**: Dedicated parsing, summarization, and chunking. Streams mini-batches ($O(1)$ memory) into the ephemeral staging ledger `ingestion_job_chunks` (`is_embedded = 0`), locks the `actual_total_chunks` ground truth into `ingestion_chunk_jobs`, and transitions the document to `READY_FOR_EMBED`.
* **Worker Pool 2 (`EmbedderWorker` Swarm)**: Horizontally scalable GPU/compute workers that concurrently claim 32-chunk batches via `SKIP LOCKED` (`is_embedded = 0` $\rightarrow$ `is_embedded = 2`). Computes embeddings and executes a clean `INSERT INTO kitchome_chunks` with `embedding NOT NULL` and `is_latest = false` (Model 1: zero MVCC dead-tuple bloat, zero nullable embeddings).
* **Ground-Truth Validation & Barrier Synchronization**: Overcomes intake heuristic estimation discrepancies (`expected_chunks`). When an embedder completes a batch, it increments `embedded_count`. When `embedded_count == actual_total_chunks` and 0 unfinished chunks remain in staging, the final finisher executes **Stage 5 Atomic State Flip** (`is_latest = true`, commits the cursor version, purges `ingestion_job_chunks`, and marks document `COMPLETED`).
* **Unified Mode (`IngestionWorker`)**: Operates seamlessly in single-node/dev environments by orchestrating `ChunkerWorker` and `EmbedderWorker` internally to completion.

#### Worker Health Check & Dynamic Crash Recovery:
* **Active Worker Pool**: `get_available_workers(timeout_seconds=60)` scans `ingestion_workers`, automatically flagging nodes that missed heartbeats as `DEAD`.
* **Dynamic Lease Extension**: For heavy 500-page files taking minutes, the worker continuously renews `crash_recovery_timeout_at` during mini-batch embedding, ensuring active workers are never killed prematurely.
* **Rapid Crash Recovery**: The sweeper detects expired leases and dead workers. If the chunk job is already marked `status = 'CHUNKED'`, the recovering worker skips text splitting and resumes directly from embedding without repeating work. Expired batch leases (`is_embedded = 2`) are automatically reclaimed by healthy swarm workers.
* **Granular Failure Diagnostics & Manual Re-Ingestion**: Explicit stage recording (`FAILED_PARSING`, `FAILED_SUMMARIZATION`, `FAILED_CHUNKING`, `FAILED_EMBEDDING`, `FAILED_DATABASE`). When retries reach 3/3, status halts at `FAILED_RETRY_EXHAUSTED`. Operators trigger `manual_retry(job_id)` to reset retries and re-queue.

---

## 3. Project Structure

```text
Kitchome_rag/
├── config.py                   # Configuration and environment settings
├── project_details.md          # Comprehensive architectural specification
├── product_requirement.md      # Product requirement document & milestones
├── skills_registry.json        # Canonical domain definitions & curated exemplars
├── data/                       # Domain document storage (recipes, manuals, decor)
├── src/
│   ├── auth/                   # Headless AuthN / AuthZ Module
│   │   ├── context.py          # UserContext and UserTier models
│   │   ├── rbac.py             # RBAC tier-to-namespace mappings
│   │   └── abac.py             # ABAC policy evaluator & DB filter generator
│   ├── skills_ms/              # skills.ms Namespace & Routing Service
│   │   ├── registry.py         # Skill registry & exemplar loader
│   │   └── router.py           # Two-channel ensemble router (Lexical + Semantic)
│   ├── ingestion/              # Event-Driven Ingestion Service (Producer / Worker)
│   │   ├── cursor.py           # Permanent Document Catalog & Idempotency Gatekeeper
│   │   ├── queue.py            # DB-Backed Async Job Queue (SKIP LOCKED)
│   │   ├── controller.py       # Ingestion Controller (intake, fast 202 Accepted)
│   │   ├── worker.py           # 5-Stage Worker (Map-reduce, batch embed, atomic flip)
│   │   ├── loader.py           # Multi-format parser with domain inference
│   │   ├── chunker.py          # Text chunker (~700 tokens)
│   │   ├── summarizer.py       # Progressive & Map-Reduce summarizer
│   │   ├── embedder.py         # Dense feature projection engine (384-dim)
│   │   └── pipeline.py         # Pipeline coordinator
│   ├── vector_store/           # Storage Layer
│   │   ├── base.py             # In-memory partitioned vector store with ABAC
│   │   └── pgvector_store.py   # PostgreSQL pgvector with JSONB ABAC & is_latest filters
│   └── rag/                    # Guarded RAG Service & Query Tool
│       ├── engine.py           # Guarded RAG query orchestrator
│       ├── telemetry.py        # Strategy E evaluation telemetry logger
│       └── generator.py        # Grounded citation synthesizer
└── tests/
    ├── test_ingestion.py       # Ingestion & progressive summarization tests
    ├── test_ensemble_router.py # Lexical, semantic prototype & margin tests
    ├── test_auth_rbac_abac.py  # User tier & tenant isolation tests
    ├── test_rag_engine_telemetry.py # End-to-end RAG & Strategy E tests
    └── test_ingestion_controller_worker.py # Controller-worker & cursor tests
```

---

## 4. Quickstart & Verification

### Running the Test Suite
```bash
pytest tests/ -v
```

### Example Usage (Headless Guarded RAG Tool)

```python
from src.auth.context import UserContext, UserTier
from src.rag.engine import RAGQueryEngine

# Initialize the guarded query engine
rag_tool = RAGQueryEngine()

# Example: Scholar tier querying academic or appliance knowledge
user = UserContext(
    user_id="usr_101",
    tier=UserTier.SCHOLAR,
    tenant_id="global",
    clearance_level=2
)

response = rag_tool.query(
    query_text="How do I clean and prep salmon for pan searing?",
    user_context=user
)

print("Target Namespaces:", response["resolved_namespaces"])
print("Routing Margin:", response["confidence_margin"])
print("Answer:", response["answer"])
print("Citations:", response["citations"])
```
