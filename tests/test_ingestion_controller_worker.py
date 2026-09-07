import os
import time
import tempfile
import pytest
from typing import Dict, Any

from src.ingestion.cursor import IngestionCursorManager, IngestionCursorRecord
from src.ingestion.queue import IngestionQueueManager
from src.ingestion.controller import IngestionController
from src.ingestion.worker import IngestionWorker, ChunkerWorker, EmbedderWorker
from src.ingestion.chunker import TextChunker
from src.vector_store.base import NamespaceVectorStore
from src.skills_ms.router import SkillsRouter

@pytest.fixture
def temp_env():
    with tempfile.TemporaryDirectory() as tmpdir:
        cursor_db = os.path.join(tmpdir, "test_cursor.db")
        queue_db = os.path.join(tmpdir, "test_queue.db")
        data_root = os.path.join(tmpdir, "data")
        os.makedirs(data_root, exist_ok=True)

        cursor_mgr = IngestionCursorManager(db_path=cursor_db)
        queue_mgr = IngestionQueueManager(db_path=queue_db)
        vector_store = NamespaceVectorStore()
        router = SkillsRouter()

        controller = IngestionController(
            cursor_manager=cursor_mgr,
            queue_manager=queue_mgr,
            vector_store=vector_store,
            router=router,
            data_root=data_root
        )

        worker = IngestionWorker(
            worker_id="test_worker_1",
            cursor_manager=cursor_mgr,
            queue_manager=queue_mgr,
            vector_store=vector_store,
            router=router,
            lease_duration_seconds=60.0
        )

        yield {
            "tmpdir": tmpdir,
            "cursor_mgr": cursor_mgr,
            "queue_mgr": queue_mgr,
            "vector_store": vector_store,
            "router": router,
            "controller": controller,
            "worker": worker,
            "data_root": data_root
        }

def test_worker_registry_and_health_check(temp_env):
    """Test worker registration, liveness tracking, and automatic dead worker detection."""
    queue_mgr: IngestionQueueManager = temp_env["queue_mgr"]
    
    # Worker 1 already registered in fixture
    queue_mgr.register_worker("worker_healthy", hostname="node-1", pid=1001)
    queue_mgr.register_worker("worker_stale", hostname="node-2", pid=1002)

    # Both workers should be active
    available = queue_mgr.get_available_workers(heartbeat_timeout_seconds=60.0)
    worker_ids = [w["worker_id"] for w in available]
    assert "worker_healthy" in worker_ids
    assert "worker_stale" in worker_ids

    # Simulate worker_stale missing heartbeats (> 60s ago)
    now = time.time()
    with queue_mgr._get_connection() as conn:
        conn.execute("UPDATE ingestion_workers SET last_heartbeat_at = ? WHERE worker_id = 'worker_stale';", (now - 120,))
        conn.commit()

    # Query available workers: stale worker should now be marked DEAD and excluded from pool
    available_after = queue_mgr.get_available_workers(heartbeat_timeout_seconds=60.0)
    active_ids = [w["worker_id"] for w in available_after]
    assert "worker_healthy" in active_ids
    assert "worker_stale" not in active_ids

    # Direct health check
    health_healthy = queue_mgr.check_worker_health("worker_healthy")
    assert health_healthy["healthy"] is True

    health_stale = queue_mgr.check_worker_health("worker_stale")
    assert health_stale["healthy"] is False
    assert health_stale["status"] == "DEAD"

def test_dynamic_crash_recovery_timestamp_extension(temp_env):
    """Test that workers actively renew their lease timestamp to prevent premature sweeper kills."""
    queue_mgr: IngestionQueueManager = temp_env["queue_mgr"]
    controller: IngestionController = temp_env["controller"]

    content = "# Blender Safety Manual\nAlways disconnect power before cleaning blade assembly."
    res = controller.intake_document(content=content, title="Blender Manual", declared_domain="appliances")
    job_id = res["job_id"]

    # Claim job
    job = queue_mgr.claim_next_job("test_worker_1", lease_duration_seconds=60.0)
    assert job is not None
    initial_timeout = job.crash_recovery_timeout_at

    # Worker works and extends lease
    time.sleep(0.05)
    renewed = queue_mgr.heartbeat_job(job_id, "test_worker_1", extend_seconds=180.0)
    assert renewed is True

    updated_job = queue_mgr.get_job(job_id)
    assert updated_job.crash_recovery_timeout_at > initial_timeout

def test_crash_recovery_sweeper_and_rapid_resume(temp_env):
    """Test that dead worker jobs are reclaimed and resume fast using chunk table status."""
    queue_mgr: IngestionQueueManager = temp_env["queue_mgr"]
    controller: IngestionController = temp_env["controller"]
    worker: IngestionWorker = temp_env["worker"]

    content = "# Deep Cleaning Guide\nUse warm water and non-abrasive sponge on enameled surfaces.\n\nNever submerge heating coils."
    res = controller.intake_document(content=content, title="Cleaning Guide", declared_domain="cleaning")
    job_id = res["job_id"]
    chunk_job_id = res["chunk_job_id"]

    # Claim by worker_dead
    queue_mgr.register_worker("worker_dead", hostname="crash-node", pid=9999)
    job = queue_mgr.claim_next_job("worker_dead", lease_duration_seconds=10.0)
    assert job is not None

    # Simulate worker_dead completing chunking before crashing
    queue_mgr.update_chunk_progress(
        chunk_job_id=chunk_job_id,
        chunked_count=5,
        chunk_ids=["c1", "c2", "c3", "c4", "c5"],
        expected_chunks=5,
        status="CHUNKED"
    )

    # Simulate crash: lease expires in the past and worker stops heartbeating
    now = time.time()
    with queue_mgr._get_connection() as conn:
        conn.execute("UPDATE ingestion_queue SET crash_recovery_timeout_at = ? WHERE job_id = ?;", (now - 50, job_id))
        conn.execute("UPDATE ingestion_workers SET last_heartbeat_at = ? WHERE worker_id = 'worker_dead';", (now - 120,))
        conn.commit()

    # Sweeper runs
    recovered = queue_mgr.recover_crashed_jobs(now=now)
    assert job_id in recovered

    reclaimed_job = queue_mgr.get_job(job_id)
    assert reclaimed_job.status == "PENDING"
    assert reclaimed_job.retry_count == 1
    assert reclaimed_job.worker_id is None

    # Healthy worker claims and executes
    exec_res = worker.process_next_job()
    assert exec_res is not None
    assert exec_res["status"] == "COMPLETED"

def test_controller_fast_intake_and_cursor_idempotency(temp_env):
    """Test non-blocking intake and zero-compute bypass on identical content."""
    controller: IngestionController = temp_env["controller"]
    worker: IngestionWorker = temp_env["worker"]

    content = "# Air Fryer Quickstart\nPreheat basket at 380F for 3 minutes before placing food."
    
    # 1. First intake -> returns ACCEPTED 202
    res1 = controller.intake_document(content=content, title="Air Fryer Guide", declared_domain="appliances")
    assert res1["status"] == "ACCEPTED"
    assert res1["http_status"] == 202
    assert res1["version"] == 1
    assert "job_id" in res1
    assert "chunk_job_id" in res1

    # Execute job to completion
    worker.process_next_job()

    # 2. Second intake with exact same content -> Cursor Gatekeeper bypasses!
    res2 = controller.intake_document(content=content, title="Air Fryer Guide", declared_domain="appliances")
    assert res2["status"] == "SKIPPED"
    assert res2["reason"] == "identical_hash"
    assert res2["version"] == 1

def test_granular_failure_diagnostics_and_retry_exhaustion(temp_env):
    """Test that stage failures are recorded and halt at FAILED_RETRY_EXHAUSTED after 3 attempts."""
    queue_mgr: IngestionQueueManager = temp_env["queue_mgr"]
    controller: IngestionController = temp_env["controller"]
    worker: IngestionWorker = temp_env["worker"]

    # Intake pointing to nonexistent file to force PARSING failure
    res = controller.intake_document(content="Temporary dummy text", title="Fail Doc", declared_domain="appliances")
    job_id = res["job_id"]

    # Corrupt file_path to force stage failure
    with queue_mgr._get_connection() as conn:
        conn.execute("UPDATE ingestion_queue SET file_path = '/invalid/nonexistent_path.md' WHERE job_id = ?;", (job_id,))
        conn.commit()

    # Attempt 1: Fails
    r1 = worker.process_next_job()
    assert r1["status"] == "FAILED"
    assert r1["failed_stage"] == "PARSING"
    j1 = queue_mgr.get_job(job_id)
    assert j1.status == "FAILED"
    assert j1.retry_count == 1

    # Reset to PENDING to simulate retry loop
    with queue_mgr._get_connection() as conn:
        conn.execute("UPDATE ingestion_queue SET status = 'PENDING' WHERE job_id = ?;", (job_id,))
        conn.commit()

    # Attempt 2: Fails
    worker.process_next_job()
    j2 = queue_mgr.get_job(job_id)
    assert j2.retry_count == 2

    # Reset to PENDING for final retry
    with queue_mgr._get_connection() as conn:
        conn.execute("UPDATE ingestion_queue SET status = 'PENDING' WHERE job_id = ?;", (job_id,))
        conn.commit()

    # Attempt 3: Retry limit reached -> transitions to FAILED_RETRY_EXHAUSTED
    worker.process_next_job()
    j3 = queue_mgr.get_job(job_id)
    assert j3.status == "FAILED_RETRY_EXHAUSTED"
    assert j3.retry_count == 3
    assert j3.failed_stage == "PARSING"

def test_manual_reingestion_resets_retries(temp_env):
    """Test operator manual retry resetting retries to 0 and purging staged chunks."""
    queue_mgr: IngestionQueueManager = temp_env["queue_mgr"]
    controller: IngestionController = temp_env["controller"]

    content = "# Manual Retry Recipe\nAdd olive oil and saute garlic for 2 minutes."
    res = controller.intake_document(content=content, title="Garlic Recipe", declared_domain="recipes")
    job_id = res["job_id"]

    # Simulate job stuck in FAILED_RETRY_EXHAUSTED
    with queue_mgr._get_connection() as conn:
        conn.execute(
            "UPDATE ingestion_queue SET status = 'FAILED_RETRY_EXHAUSTED', retry_count = 3, failed_stage = 'EMBEDDING' WHERE job_id = ?;",
            (job_id,)
        )
        conn.commit()

    # Trigger manual retry
    retry_res = controller.manual_retry(job_id)
    assert retry_res["status"] == "RE_QUEUED"
    assert retry_res["retry_count"] == 0

    reloaded = queue_mgr.get_job(job_id)
    assert reloaded.status == "PENDING"
    assert reloaded.retry_count == 0
    assert reloaded.failed_stage is None

def test_stage_5_atomic_state_flip_and_quash(temp_env):
    """Test version state flip (is_latest true/false) and user-driven quash physically deleting vectors."""
    controller: IngestionController = temp_env["controller"]
    worker: IngestionWorker = temp_env["worker"]
    vector_store: NamespaceVectorStore = temp_env["vector_store"]
    cursor_mgr: IngestionCursorManager = temp_env["cursor_mgr"]

    # 1. Ingest Version 1
    content_v1 = "# Pasta Carbonara v1\nUse cream and bacon."
    res_v1 = controller.intake_document(content=content_v1, title="Carbonara", declared_domain="recipes")
    doc_id_v1 = res_v1["document_id"]
    worker.process_next_job()

    # Check v1 chunks in vector store
    v1_results = vector_store.search(
        query_vector=[0.1] * 384,
        namespace="recipes_culinary",
        top_k=10,
        is_latest=True
    )
    assert len(v1_results) > 0
    assert v1_results[0]["document_id"] == doc_id_v1
    assert v1_results[0]["metadata"]["is_latest"] is True

    # 2. Ingest Version 2 (Authentic update)
    content_v2 = "# Pasta Carbonara v2 Authentic\nNever use cream. Use guanciale and pecorino romano with egg yolks."
    res_v2 = controller.intake_document(content=content_v2, title="Carbonara", declared_domain="recipes")
    doc_id_v2 = res_v2["document_id"]
    assert res_v2["version"] == 2
    worker.process_next_job()

    # Search with is_latest=True: ONLY v2 should be returned (v1 is archived)
    latest_results = vector_store.search(
        query_vector=[0.1] * 384,
        namespace="recipes_culinary",
        top_k=10,
        is_latest=True
    )
    latest_doc_ids = {r["document_id"] for r in latest_results}
    assert doc_id_v2 in latest_doc_ids
    assert doc_id_v1 not in latest_doc_ids

    # Search with is_latest=False: v1 should be visible in archived search
    archived_results = vector_store.search(
        query_vector=[0.1] * 384,
        namespace="recipes_culinary",
        top_k=10,
        is_latest=False
    )
    archived_doc_ids = {r["document_id"] for r in archived_results}
    assert doc_id_v1 in archived_doc_ids

    # 3. Quash Version 1 (User deactivates old version)
    quash_res = controller.quash_document_version(doc_id_v1)
    assert quash_res["status"] == "QUASHED"
    assert quash_res["chunks_deleted"] > 0

    # Verify v1 is completely gone from vector store (even in archived search)
    after_quash = vector_store.search(
        query_vector=[0.1] * 384,
        namespace="recipes_culinary",
        top_k=10,
        is_latest=False
    )
    after_quash_doc_ids = {r["document_id"] for r in after_quash}
    assert doc_id_v1 not in after_quash_doc_ids

    # Cursor still retains immutable audit record marked QUASHED
    v1_cursor = cursor_mgr.get_by_path(res_v1["file_path"])
    # List all records for Carbonara family
    family_records = cursor_mgr.list_records(doc_family="carbonara")
    v1_record = next((r for r in family_records if r.version == 1), None)
    assert v1_record is not None
    assert v1_record.status == "QUASHED"
    assert v1_record.is_latest is False

def test_two_worker_pool_chunking_handoff_and_staging_ledger(temp_env):
    """
    Test Worker Pool 1 (ChunkerWorker):
    - Consumes PENDING job.
    - Runs Stages 1-3.
    - Streams chunks into staging ledger (is_embedded = 0).
    - Locks actual_total_chunks ground truth.
    - Transitions queue to READY_FOR_EMBED with worker_id = NULL.
    """
    queue_mgr: IngestionQueueManager = temp_env["queue_mgr"]
    controller: IngestionController = temp_env["controller"]
    router: SkillsRouter = temp_env["router"]

    content = (
        "# Espresso Machine Maintenance\n"
        "Descaling procedure requires 50% white vinegar and 50% distilled water.\n\n"
        "Run two full brew cycles through the group head.\n\n"
        "Flush with three full reservoirs of clean water before preparing coffee.\n\n"
        "Wipe the steam wand after every milk frothing session."
    )
    res = controller.intake_document(content=content, title="Espresso Maintenance", declared_domain="appliances")
    job_id = res["job_id"]
    chunk_job_id = res["chunk_job_id"]

    # Initialize Pool 1 ChunkerWorker
    chunker = ChunkerWorker(
        worker_id="chunker_pool1_1",
        queue_manager=queue_mgr,
        router=router
    )

    chunk_res = chunker.process_next_job()
    assert chunk_res is not None
    assert chunk_res["status"] == "READY_FOR_EMBED"
    assert chunk_res["actual_total_chunks"] > 0
    actual_chunks = chunk_res["actual_total_chunks"]

    # Verify queue status
    queue_job = queue_mgr.get_job(job_id)
    assert queue_job.status == "READY_FOR_EMBED"
    assert queue_job.current_stage == "EMBEDDING"
    assert queue_job.worker_id is None  # Released for Swarm Embedders

    # Verify chunk job status & actual_total_chunks locked
    chunk_job = queue_mgr.get_chunk_job(chunk_job_id)
    assert chunk_job.status == "READY_FOR_EMBED"
    assert chunk_job.actual_total_chunks == actual_chunks
    assert chunk_job.chunked_count == actual_chunks
    assert chunk_job.embedded_count == 0

    # Verify staging ledger in ingestion_job_chunks
    with queue_mgr._get_connection() as conn:
        cur = conn.execute("SELECT COUNT(*) as cnt FROM ingestion_job_chunks WHERE job_id = ? AND is_embedded = 0;", (job_id,))
        staged_count = cur.fetchone()["cnt"]
        assert staged_count == actual_chunks

    # Verify ChunkerWorker is marked IDLE
    worker_health = queue_mgr.check_worker_health("chunker_pool1_1")
    assert worker_health["status"] == "IDLE"

def test_embedder_swarm_concurrency_and_barrier_synchronization(temp_env):
    """
    Test Worker Pool 2 (EmbedderWorker Swarm):
    - Concurrent embedder workers claiming batches via SKIP LOCKED.
    - Disjoint claims (zero duplicate processing).
    - Barrier synchronization triggering Stage 5 on final batch.
    """
    queue_mgr: IngestionQueueManager = temp_env["queue_mgr"]
    controller: IngestionController = temp_env["controller"]
    vector_store: NamespaceVectorStore = temp_env["vector_store"]
    cursor_mgr: IngestionCursorManager = temp_env["cursor_mgr"]
    router: SkillsRouter = temp_env["router"]

    # 4 distinct paragraphs to generate at least 4 chunks
    content = (
        "Paragraph One: Induction cooktops use electromagnetic fields to heat cookware directly.\n\n"
        "Paragraph Two: Cast iron and magnetic stainless steel cookware are fully compatible.\n\n"
        "Paragraph Three: Copper and aluminum pans require an interface adapter disc.\n\n"
        "Paragraph Four: Keep digital displays clean and free of grease splatters."
    )
    res = controller.intake_document(content=content, title="Induction Guide", declared_domain="appliances")
    job_id = res["job_id"]

    # Run ChunkerWorker to prepare staging ledger with multi-chunk generation
    custom_chunker = TextChunker(chunk_size=80, chunk_overlap=10)
    chunker = ChunkerWorker(
        worker_id="chunker_worker_lead",
        queue_manager=queue_mgr,
        router=router,
        chunker=custom_chunker
    )
    chunker_res = chunker.process_next_job()
    assert chunker_res["status"] == "READY_FOR_EMBED"
    total_chunks = chunker_res["actual_total_chunks"]
    assert total_chunks >= 3

    # Instantiate two concurrent embedder workers in Pool 2 Swarm
    embedder_1 = EmbedderWorker(
        worker_id="embedder_swarm_1",
        cursor_manager=cursor_mgr,
        queue_manager=queue_mgr,
        vector_store=vector_store,
        batch_size=1
    )
    embedder_2 = EmbedderWorker(
        worker_id="embedder_swarm_2",
        cursor_manager=cursor_mgr,
        queue_manager=queue_mgr,
        vector_store=vector_store,
        batch_size=1
    )

    # Claim batch concurrently
    batch_1 = queue_mgr.claim_next_chunk_batch("embedder_swarm_1", batch_size=1)
    batch_2 = queue_mgr.claim_next_chunk_batch("embedder_swarm_2", batch_size=1)

    assert batch_1 is not None
    assert batch_2 is not None

    # Assert disjoint claims (no duplicate chunk claimed)
    ids_1 = {c["chunk_id"] for c in batch_1["chunks"]}
    ids_2 = {c["chunk_id"] for c in batch_2["chunks"]}
    assert ids_1.isdisjoint(ids_2)

    # Embedder 1 completes batch 1 -> barrier not complete
    res_1 = embedder_1.execute_batch(batch_1)
    assert res_1["status"] == "BATCH_COMPLETED"
    assert res_1["barrier_resolved"] is False

    # Embedder 2 completes batch 2
    res_2 = embedder_2.execute_batch(batch_2)

    # Finish remaining batches until barrier resolves
    final_res = None
    while True:
        b = embedder_1.process_next_batch()
        if not b:
            b = embedder_2.process_next_batch()
        if not b:
            break
        final_res = b
        if b.get("barrier_resolved"):
            break

    assert final_res is not None
    assert final_res["barrier_resolved"] is True
    assert final_res["status"] == "COMPLETED"

    # Verify queue status is COMPLETED
    completed_job = queue_mgr.get_job(job_id)
    assert completed_job.status == "COMPLETED"

    # Verify cursor record
    cursor_record = cursor_mgr.get_by_path(completed_job.file_path)
    assert cursor_record is not None
    assert cursor_record.is_latest is True
    assert cursor_record.chunks_count == total_chunks

def test_model_1_zero_leak_and_staging_ledger_purge(temp_env):
    """
    Test Model 1 Clean Vector Insertion:
    - Clean INSERT into vector store with embedding NOT NULL.
    - Ephemeral staging ledger ingestion_job_chunks is completely purged upon completion.
    """
    queue_mgr: IngestionQueueManager = temp_env["queue_mgr"]
    controller: IngestionController = temp_env["controller"]
    vector_store: NamespaceVectorStore = temp_env["vector_store"]
    cursor_mgr: IngestionCursorManager = temp_env["cursor_mgr"]
    router: SkillsRouter = temp_env["router"]

    content = "# Sourdough Starter\nFeed equal parts flour and water every 24 hours at room temperature."
    res = controller.intake_document(content=content, title="Sourdough Guide", declared_domain="recipes")
    job_id = res["job_id"]

    # Chunker stages chunks
    chunker = ChunkerWorker(worker_id="test_chunker_m1", queue_manager=queue_mgr, router=router)
    chunker.process_next_job()

    # Verify staged before embedding
    with queue_mgr._get_connection() as conn:
        cur = conn.execute("SELECT COUNT(*) as count FROM ingestion_job_chunks WHERE job_id = ?;", (job_id,))
        assert cur.fetchone()["count"] > 0

    # Embedder processes all batches to completion
    embedder = EmbedderWorker(
        worker_id="test_embedder_m1",
        cursor_manager=cursor_mgr,
        queue_manager=queue_mgr,
        vector_store=vector_store,
        batch_size=32
    )
    b_res = embedder.process_next_batch()
    assert b_res is not None
    assert b_res["barrier_resolved"] is True

    # Check vector store: all chunks have non-null embeddings and is_latest=True
    search_res = vector_store.search(
        query_vector=[0.05] * 384,
        namespace="recipes_culinary",
        top_k=10,
        is_latest=True
    )
    assert len(search_res) > 0
    for chunk in search_res:
        assert chunk["metadata"]["is_latest"] is True

    # Verify ephemeral staging ledger was purged (Model 1 zero-leak)
    with queue_mgr._get_connection() as conn:
        cur = conn.execute("SELECT COUNT(*) as count FROM ingestion_job_chunks WHERE job_id = ?;", (job_id,))
        assert cur.fetchone()["count"] == 0

def test_ground_truth_actual_total_chunks_discrepancy_validation(temp_env):
    """
    Test actual_total_chunks ground truth overcoming intake heuristic discrepancies.
    Simulate expected_chunks = 100 on intake, but Chunker generates 2 chunks.
    Barrier MUST synchronize on actual_total_chunks = 2, not 100.
    """
    queue_mgr: IngestionQueueManager = temp_env["queue_mgr"]
    controller: IngestionController = temp_env["controller"]
    vector_store: NamespaceVectorStore = temp_env["vector_store"]
    cursor_mgr: IngestionCursorManager = temp_env["cursor_mgr"]
    router: SkillsRouter = temp_env["router"]

    # Short document (produces 1 or 2 chunks)
    content = "Quick tip: Salt boiling water generously before adding pasta.\n\nCook al dente."
    res = controller.intake_document(content=content, title="Salting Pasta", declared_domain="recipes")
    job_id = res["job_id"]
    chunk_job_id = res["chunk_job_id"]

    # Artificially alter expected_chunks to 100 to simulate discrepancy
    with queue_mgr._get_connection() as conn:
        conn.execute("UPDATE ingestion_chunk_jobs SET expected_chunks = 100 WHERE chunk_job_id = ?;", (chunk_job_id,))
        conn.commit()

    # Chunker runs and sets actual_total_chunks based on true output
    chunker = ChunkerWorker(worker_id="chunker_discrepancy", queue_manager=queue_mgr, router=router)
    c_res = chunker.process_next_job()
    actual_chunks = c_res["actual_total_chunks"]
    assert actual_chunks < 10

    # Verify chunk_job record has actual_total_chunks != 100
    cj = queue_mgr.get_chunk_job(chunk_job_id)
    assert cj.actual_total_chunks == actual_chunks
    assert cj.expected_chunks == 100  # Initial heuristic preserved for audit

    # Embedder processes
    embedder = EmbedderWorker(
        worker_id="embedder_discrepancy",
        cursor_manager=cursor_mgr,
        queue_manager=queue_mgr,
        vector_store=vector_store,
        batch_size=32
    )
    final_res = embedder.process_next_batch()

    # Barrier resolves immediately on actual_total_chunks, does not hang waiting for 100!
    assert final_res is not None
    assert final_res["barrier_resolved"] is True
    assert final_res["status"] == "COMPLETED"

    # Queue job is COMPLETED
    job = queue_mgr.get_job(job_id)
    assert job.status == "COMPLETED"

def test_strict_phased_guard_blocks_embedder_during_chunking(temp_env):
    """
    Verify the strict phased SQL guard:
    Even if chunks are already staged into ingestion_job_chunks,
    embedder workers CANNOT claim any batches while chunking is in progress
    (q.status == 'PROCESSING', current_stage == 'CHUNKING', actual_total_chunks IS NULL).
    Only once lock_chunking_complete() sets READY_FOR_EMBED and actual_total_chunks is the embedder unblocked.
    """
    queue_mgr: IngestionQueueManager = temp_env["queue_mgr"]
    controller: IngestionController = temp_env["controller"]

    content = "Batch 1 content.\n\nBatch 2 content.\n\nBatch 3 content."
    res = controller.intake_document(content=content, title="Guard Test", declared_domain="appliances")
    job_id = res["job_id"]
    chunk_job_id = res["chunk_job_id"]

    # 1. Chunker claims the job -> status becomes PROCESSING, stage PARSING
    job = queue_mgr.claim_next_chunking_job("chunker_active")
    assert job is not None
    assert job.job_id == job_id
    queue_mgr.update_stage(job_id, "CHUNKING")

    # 2. Stage some chunks into ingestion_job_chunks before chunking finishes
    from src.ingestion.chunker import TextChunk
    staged_chunks = [
        TextChunk(
            chunk_id="chk_test_1",
            document_id=job.document_id,
            namespace="appliances",
            chunk_index=0,
            text="Batch 1 content.",
            metadata={"file_path": job.file_path}
        )
    ]
    queue_mgr.stage_chunks(job_id, staged_chunks)

    # 3. An embedder attempts to claim a batch -> MUST BE BLOCKED (returns None)
    blocked_claim = queue_mgr.claim_next_chunk_batch("embedder_eager", batch_size=32)
    assert blocked_claim is None  # Guard successfully prevented premature spillover!

    # 4. Now Chunker finishes all chunking and locks ground truth
    queue_mgr.lock_chunking_complete(
        chunk_job_id=chunk_job_id,
        job_id=job_id,
        actual_total_chunks=1,
        chunk_ids=["chk_test_1"],
        worker_id="chunker_active"
    )

    # 5. Embedder attempts to claim again -> MUST SUCCEED now that handoff is official
    allowed_claim = queue_mgr.claim_next_chunk_batch("embedder_eager", batch_size=32)
    assert allowed_claim is not None
    assert allowed_claim["job_id"] == job_id
    assert len(allowed_claim["chunks"]) == 1


def test_thread_local_connection_pooling_reuse(temp_env):
    """Verify thread-local connection reuse within the same thread and isolation across threads."""
    import threading
    queue_mgr: IngestionQueueManager = temp_env["queue_mgr"]
    cursor_mgr: IngestionCursorManager = temp_env["cursor_mgr"]

    # 1. Multiple calls in main thread should return the exact same connection object
    with queue_mgr._get_connection() as conn1, queue_mgr._get_connection() as conn2:
        assert conn1 is conn2

    with cursor_mgr._get_connection() as cur_conn1, cursor_mgr._get_connection() as cur_conn2:
        assert cur_conn1 is cur_conn2

    # 2. Spawn a separate thread and verify it gets a distinct connection instance
    other_conn = []
    def thread_target():
        with queue_mgr._get_connection() as c:
            other_conn.append(c)
        queue_mgr.close()

    t = threading.Thread(target=thread_target)
    t.start()
    t.join()

    assert len(other_conn) == 1
    assert other_conn[0] is not conn1

    # 3. Closing connection in main thread resets thread-local
    queue_mgr.close()
    with queue_mgr._get_connection() as conn3:
        assert conn3 is not conn1


def test_worker_run_forever_bounded_execution(temp_env):
    """Test that workers' run_forever loop terminates cleanly when max_iterations is reached."""
    queue_mgr: IngestionQueueManager = temp_env["queue_mgr"]
    cursor_mgr: IngestionCursorManager = temp_env["cursor_mgr"]
    vector_store: NamespaceVectorStore = temp_env["vector_store"]
    router: SkillsRouter = temp_env["router"]

    chunker_worker = ChunkerWorker(
        worker_id="test_chunker_bounded",
        queue_manager=queue_mgr,
        router=router
    )
    # Run for exactly 2 iterations
    start_time = time.time()
    chunker_worker.run_forever(poll_interval=0.01, max_iterations=2)
    elapsed = time.time() - start_time
    assert elapsed < 2.0
    assert chunker_worker._is_running is False

    embedder_worker = EmbedderWorker(
        worker_id="test_embedder_bounded",
        cursor_manager=cursor_mgr,
        queue_manager=queue_mgr,
        vector_store=vector_store
    )
    start_time = time.time()
    embedder_worker.run_forever(poll_interval=0.01, max_iterations=2)
    elapsed = time.time() - start_time
    assert elapsed < 2.0
    assert embedder_worker._is_running is False


def test_worker_supervisor_thread_mode(temp_env):
    """Test WorkerSupervisor managing worker lifecycle and clean shutdown in thread mode."""
    from src.ingestion.run_workers import WorkerSupervisor

    supervisor = WorkerSupervisor(
        role="all",
        mode="thread",
        chunker_concurrency=1,
        embedder_concurrency=1,
        poll_interval=0.01,
        max_iterations=3
    )

    supervisor.start()
    assert len(supervisor.workers) == 2

    # Allow iterations to finish and stop
    time.sleep(0.2)
    supervisor.stop(timeout=2.0)

    for w in supervisor.workers:
        assert not w.is_alive()


