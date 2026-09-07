import os
import json
import sqlite3
import time
import uuid
import socket
import threading
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class IngestionQueueJob(BaseModel):
    job_id: str
    document_id: str
    file_path: str
    doc_family: str
    version: int = 1
    declared_domain: Optional[str] = None
    job_type: str = "INGEST"
    status: str = "PENDING"  # PENDING | PROCESSING | COMPLETED | FAILED | FAILED_RETRY_EXHAUSTED
    current_stage: Optional[str] = None  # PARSING | SUMMARIZATION | CHUNKING | EMBEDDING | DATABASE
    chunk_job_id: Optional[str] = None
    worker_id: Optional[str] = None
    crash_recovery_timeout_at: Optional[float] = None
    last_heartbeat_at: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    failed_stage: Optional[str] = None
    error_reason: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

class IngestionChunkJob(BaseModel):
    chunk_job_id: str
    job_id: str
    expected_chunks: int = 0
    actual_total_chunks: Optional[int] = None
    chunked_count: int = 0
    embedded_count: int = 0
    chunk_ids: List[str] = Field(default_factory=list)
    status: str = "PENDING"  # PENDING | CHUNKING | CHUNKED | READY_FOR_EMBED | EMBEDDING | COMPLETED | FAILED
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

class IngestionWorkerRecord(BaseModel):
    worker_id: str
    hostname: str
    pid: int
    status: str = "IDLE"  # IDLE | BUSY | OFFLINE | DEAD
    current_job_id: Optional[str] = None
    last_heartbeat_at: float = Field(default_factory=time.time)
    registered_at: float = Field(default_factory=time.time)
    tasks_completed: int = 0

class IngestionQueueManager:
    """
    Database-Backed Ingestion Queue, Worker Registry, and Chunk Job Ledger.
    Provides ACID transaction guarantees with optimistic/pessimistic non-blocking claims,
    dynamic lease renewal, worker health checking, and crash recovery.
    """
    def __init__(self, db_path: str = "data/ingestion_queue.db"):
        self.db_path = db_path
        self._local = threading.local()
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            self._local.conn = conn
        with self._local.conn:
            yield self._local.conn

    def close(self):
        """Cleanly closes the thread-local SQLite connection."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                self._local.conn.close()
            except Exception:
                pass
            self._local.conn = None

    def _init_db(self):
        with self._get_connection() as conn:
            # 1. Main Queue Table (Orchestration only)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_queue (
                    job_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    doc_family TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    declared_domain TEXT,
                    job_type TEXT DEFAULT 'INGEST',
                    status TEXT DEFAULT 'PENDING',
                    current_stage TEXT,
                    chunk_job_id TEXT,
                    worker_id TEXT,
                    crash_recovery_timeout_at REAL,
                    last_heartbeat_at REAL,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    failed_stage TEXT,
                    error_reason TEXT,
                    created_at REAL,
                    started_at REAL,
                    completed_at REAL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_queue_status ON ingestion_queue (status, created_at);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_queue_worker ON ingestion_queue (worker_id);")

            # 2. Chunk Job Table (Ledger tracking expected vs actual chunks ground truth)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_chunk_jobs (
                    chunk_job_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    expected_chunks INTEGER DEFAULT 0,
                    actual_total_chunks INTEGER DEFAULT NULL,
                    chunked_count INTEGER DEFAULT 0,
                    embedded_count INTEGER DEFAULT 0,
                    chunk_ids TEXT DEFAULT '[]',
                    status TEXT DEFAULT 'PENDING',
                    created_at REAL,
                    updated_at REAL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_jobs_job ON ingestion_chunk_jobs (job_id);")

            # 3. Ephemeral In-Flight Chunk Staging Ledger (for Swarm Embedding)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_job_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    is_embedded INTEGER DEFAULT 0,
                    worker_id TEXT,
                    lease_expires_at REAL,
                    created_at REAL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_job_chunks_embed ON ingestion_job_chunks (job_id, is_embedded, chunk_index);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_job_chunks_claim ON ingestion_job_chunks (is_embedded, lease_expires_at);")

            # 4. Worker Registry & Liveness Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_workers (
                    worker_id TEXT PRIMARY KEY,
                    hostname TEXT NOT NULL,
                    pid INTEGER NOT NULL,
                    status TEXT DEFAULT 'IDLE',
                    current_job_id TEXT,
                    last_heartbeat_at REAL,
                    registered_at REAL,
                    tasks_completed INTEGER DEFAULT 0
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_workers_status ON ingestion_workers (status, last_heartbeat_at);")

            # 5. Dead Letter Queue Table (DLQ for poison-pill chunk batches)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_dlq (
                    dlq_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    start_offset INTEGER NOT NULL,
                    end_offset INTEGER NOT NULL,
                    chunk_ids TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    error_reason TEXT NOT NULL,
                    exception_trace TEXT,
                    retry_count INTEGER NOT NULL,
                    quarantined_at REAL NOT NULL,
                    status TEXT DEFAULT 'QUARANTINED'
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dlq_job ON ingestion_dlq (job_id, status);")
            conn.commit()

    # ==========================================
    # Queue Operations
    # ==========================================

    def enqueue_job(
        self,
        document_id: str,
        file_path: str,
        doc_family: str,
        version: int = 1,
        declared_domain: Optional[str] = None,
        job_type: str = "INGEST",
        expected_chunks: int = 0
    ) -> IngestionQueueJob:
        now = time.time()
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        chunk_job_id = f"cjob_{uuid.uuid4().hex[:12]}"

        with self._get_connection() as conn:
            # 1. Insert chunk job entry
            conn.execute(
                """
                INSERT INTO ingestion_chunk_jobs (
                    chunk_job_id, job_id, expected_chunks, chunked_count,
                    chunk_ids, status, created_at, updated_at
                ) VALUES (?, ?, ?, 0, '[]', 'PENDING', ?, ?);
                """,
                (chunk_job_id, job_id, expected_chunks, now, now)
            )

            # 2. Insert main queue job entry
            conn.execute(
                """
                INSERT INTO ingestion_queue (
                    job_id, document_id, file_path, doc_family, version,
                    declared_domain, job_type, status, current_stage,
                    chunk_job_id, worker_id, crash_recovery_timeout_at,
                    last_heartbeat_at, retry_count, max_retries,
                    created_at, started_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', 'PARSING', ?, NULL, NULL, NULL, 0, 3, ?, NULL, NULL);
                """,
                (job_id, document_id, file_path, doc_family, version, declared_domain, job_type, chunk_job_id, now)
            )
            conn.commit()

        return self.get_job(job_id)

    def get_job(self, job_id: str) -> Optional[IngestionQueueJob]:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM ingestion_queue WHERE job_id = ? LIMIT 1;", (job_id,))
            row = cur.fetchone()
            if not row:
                return None
            return IngestionQueueJob(
                job_id=row["job_id"],
                document_id=row["document_id"],
                file_path=row["file_path"],
                doc_family=row["doc_family"],
                version=row["version"],
                declared_domain=row["declared_domain"],
                job_type=row["job_type"],
                status=row["status"],
                current_stage=row["current_stage"],
                chunk_job_id=row["chunk_job_id"],
                worker_id=row["worker_id"],
                crash_recovery_timeout_at=row["crash_recovery_timeout_at"],
                last_heartbeat_at=row["last_heartbeat_at"],
                retry_count=row["retry_count"],
                max_retries=row["max_retries"],
                failed_stage=row["failed_stage"],
                error_reason=row["error_reason"],
                created_at=row["created_at"],
                started_at=row["started_at"],
                completed_at=row["completed_at"]
            )

    def update_job_file_path(self, job_id: str, new_file_path: str) -> None:
        """Updates the target file_path for a job (e.g. after raw input is parsed into markdown)."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE ingestion_queue SET file_path = ? WHERE job_id = ?;",
                (new_file_path, job_id)
            )
            conn.commit()

    def claim_next_job(self, worker_id: str, lease_duration_seconds: float = 120.0) -> Optional[IngestionQueueJob]:
        now = time.time()
        timeout_at = now + lease_duration_seconds
        with self._get_connection() as conn:
            # Atomic claim
            cur = conn.execute(
                """
                SELECT job_id FROM ingestion_queue
                WHERE status = 'PENDING'
                ORDER BY created_at ASC
                LIMIT 1;
                """
            )
            row = cur.fetchone()
            if not row:
                return None

            job_id = row["job_id"]
            cur_update = conn.execute(
                """
                UPDATE ingestion_queue
                SET status = 'PROCESSING',
                    worker_id = ?,
                    started_at = ?,
                    last_heartbeat_at = ?,
                    crash_recovery_timeout_at = ?
                WHERE job_id = ? AND status = 'PENDING';
                """,
                (worker_id, now, now, timeout_at, job_id)
            )
            if cur_update.rowcount == 0:
                return None  # Claimed by another worker

            # Update worker status to BUSY
            conn.execute(
                """
                UPDATE ingestion_workers
                SET status = 'BUSY', current_job_id = ?, last_heartbeat_at = ?
                WHERE worker_id = ?;
                """,
                (job_id, now, worker_id)
            )
            conn.commit()

        return self.get_job(job_id)

    def claim_next_chunking_job(self, worker_id: str, lease_duration_seconds: float = 120.0) -> Optional[IngestionQueueJob]:
        """Claims next PENDING document job for ChunkerWorker (Pool 1)."""
        now = time.time()
        timeout_at = now + lease_duration_seconds
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                SELECT job_id FROM ingestion_queue
                WHERE status = 'PENDING'
                ORDER BY created_at ASC
                LIMIT 1;
                """
            )
            row = cur.fetchone()
            if not row:
                return None

            job_id = row["job_id"]
            cur_update = conn.execute(
                """
                UPDATE ingestion_queue
                SET status = 'PROCESSING',
                    current_stage = 'PARSING',
                    worker_id = ?,
                    started_at = ?,
                    last_heartbeat_at = ?,
                    crash_recovery_timeout_at = ?
                WHERE job_id = ? AND status = 'PENDING';
                """,
                (worker_id, now, now, timeout_at, job_id)
            )
            if cur_update.rowcount == 0:
                return None

            conn.execute(
                """
                UPDATE ingestion_workers
                SET status = 'BUSY', current_job_id = ?, last_heartbeat_at = ?
                WHERE worker_id = ?;
                """,
                (job_id, now, worker_id)
            )
            conn.commit()

        return self.get_job(job_id)

    def lock_chunking_complete(
        self,
        chunk_job_id: str,
        job_id: str,
        actual_total_chunks: int,
        chunk_ids: List[str],
        worker_id: Optional[str] = None
    ):
        """
        Locks actual_total_chunks ground truth and transitions job to READY_FOR_EMBED.
        Handoff point from ChunkerWorker (Pool 1) to EmbedderWorker Swarm (Pool 2).
        """
        now = time.time()
        chunk_ids_json = json.dumps(chunk_ids)
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE ingestion_chunk_jobs
                SET actual_total_chunks = ?,
                    chunked_count = ?,
                    chunk_ids = ?,
                    status = 'READY_FOR_EMBED',
                    updated_at = ?
                WHERE chunk_job_id = ? OR job_id = ?;
                """,
                (actual_total_chunks, len(chunk_ids), chunk_ids_json, now, chunk_job_id, job_id)
            )
            conn.execute(
                """
                UPDATE ingestion_queue
                SET status = 'READY_FOR_EMBED',
                    current_stage = 'EMBEDDING',
                    worker_id = NULL,
                    last_heartbeat_at = ?
                WHERE job_id = ?;
                """,
                (now, job_id)
            )
            if worker_id:
                conn.execute(
                    """
                    UPDATE ingestion_workers
                    SET status = 'IDLE', current_job_id = NULL,
                        tasks_completed = tasks_completed + 1, last_heartbeat_at = ?
                    WHERE worker_id = ?;
                    """,
                    (now, worker_id)
                )
            conn.commit()

    def heartbeat_job(self, job_id: str, worker_id: str, extend_seconds: float = 120.0) -> bool:
        now = time.time()
        new_timeout = now + extend_seconds
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                UPDATE ingestion_queue
                SET last_heartbeat_at = ?,
                    crash_recovery_timeout_at = ?
                WHERE job_id = ? AND worker_id = ? AND status = 'PROCESSING';
                """,
                (now, new_timeout, job_id, worker_id)
            )
            # Update worker heartbeat
            conn.execute(
                """
                UPDATE ingestion_workers
                SET last_heartbeat_at = ?
                WHERE worker_id = ?;
                """,
                (now, worker_id)
            )
            conn.commit()
            return cur.rowcount > 0

    def update_stage(self, job_id: str, stage: str):
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE ingestion_queue SET current_stage = ?, last_heartbeat_at = ? WHERE job_id = ?;",
                (stage, now, job_id)
            )
            conn.commit()

    def complete_job(self, job_id: str, worker_id: str, status: str = "COMPLETED"):
        now = time.time()
        with self._get_connection() as conn:
            # 1. Complete queue job
            cur = conn.execute(
                """
                UPDATE ingestion_queue
                SET status = ?, completed_at = ?, last_heartbeat_at = ?, worker_id = ?
                WHERE job_id = ?;
                """,
                (status, now, now, worker_id, job_id)
            )
            # 2. Complete chunk job
            conn.execute(
                "UPDATE ingestion_chunk_jobs SET status = ?, updated_at = ? WHERE job_id = ?;",
                (status, now, job_id)
            )
            # 3. Purge ephemeral staged chunks to prevent duplicate storage
            conn.execute("DELETE FROM ingestion_job_chunks WHERE job_id = ?;", (job_id,))

            # 4. Set worker back to IDLE
            conn.execute(
                """
                UPDATE ingestion_workers
                SET status = 'IDLE', current_job_id = NULL,
                    tasks_completed = tasks_completed + 1, last_heartbeat_at = ?
                WHERE worker_id = ?;
                """,
                (now, worker_id)
            )
            conn.commit()

    def fail_job(self, job_id: str, worker_id: str, stage: str, error_reason: str):
        now = time.time()
        with self._get_connection() as conn:
            cur = conn.execute("SELECT retry_count, max_retries FROM ingestion_queue WHERE job_id = ?;", (job_id,))
            row = cur.fetchone()
            if not row:
                return

            new_retries = row["retry_count"] + 1
            max_retries = row["max_retries"]

            if new_retries >= max_retries:
                status = "FAILED_RETRY_EXHAUSTED"
            else:
                status = "FAILED"

            conn.execute(
                """
                UPDATE ingestion_queue
                SET status = ?,
                    failed_stage = ?,
                    error_reason = ?,
                    retry_count = ?,
                    worker_id = NULL,
                    last_heartbeat_at = ?
                WHERE job_id = ?;
                """,
                (status, stage, error_reason, new_retries, now, job_id)
            )

            conn.execute(
                "UPDATE ingestion_chunk_jobs SET status = 'FAILED', updated_at = ? WHERE job_id = ?;",
                (now, job_id)
            )

            # Set worker to IDLE
            conn.execute(
                """
                UPDATE ingestion_workers
                SET status = 'IDLE', current_job_id = NULL, last_heartbeat_at = ?
                WHERE worker_id = ?;
                """,
                (now, worker_id)
            )
            conn.commit()

    def manual_retry_job(self, job_id: str) -> Optional[IngestionQueueJob]:
        now = time.time()
        with self._get_connection() as conn:
            # Purge any partial staged chunks for clean restart
            conn.execute("DELETE FROM ingestion_job_chunks WHERE job_id = ?;", (job_id,))
            conn.execute(
                """
                UPDATE ingestion_queue
                SET status = 'PENDING',
                    retry_count = 0,
                    failed_stage = NULL,
                    error_reason = NULL,
                    worker_id = NULL,
                    started_at = NULL,
                    completed_at = NULL,
                    current_stage = 'PARSING',
                    created_at = ?
                WHERE job_id = ?;
                """,
                (now, job_id)
            )
            conn.execute(
                """
                UPDATE ingestion_chunk_jobs
                SET status = 'PENDING', chunked_count = 0, chunk_ids = '[]', updated_at = ?
                WHERE job_id = ?;
                """,
                (now, job_id)
            )
            conn.commit()
        return self.get_job(job_id)

    def recover_crashed_jobs(self, now: Optional[float] = None) -> List[str]:
        """
        Sweeper for dead worker jobs whose crash_recovery_timeout_at has expired.
        """
        now = now or time.time()
        recovered_job_ids = []
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                SELECT job_id, worker_id, retry_count, max_retries, current_stage
                FROM ingestion_queue
                WHERE status = 'PROCESSING' AND crash_recovery_timeout_at < ?;
                """,
                (now,)
            )
            expired_jobs = cur.fetchall()

            for job in expired_jobs:
                job_id = job["job_id"]
                new_retries = job["retry_count"] + 1
                max_retries = job["max_retries"]

                # Mark worker as DEAD if registered
                if job["worker_id"]:
                    conn.execute(
                        "UPDATE ingestion_workers SET status = 'DEAD', current_job_id = NULL WHERE worker_id = ?;",
                        (job["worker_id"],)
                    )

                if new_retries >= max_retries:
                    conn.execute(
                        """
                        UPDATE ingestion_queue
                        SET status = 'FAILED_RETRY_EXHAUSTED',
                            failed_stage = current_stage,
                            error_reason = 'Worker crash: lease expired without heartbeat',
                            retry_count = ?,
                            worker_id = NULL,
                            last_heartbeat_at = ?
                        WHERE job_id = ?;
                        """,
                        (new_retries, now, job_id)
                    )
                else:
                    conn.execute(
                        """
                        UPDATE ingestion_queue
                        SET status = 'PENDING',
                            current_stage = 'CRASH_RECOVERED',
                            error_reason = 'Worker crashed; re-queued automatically',
                            retry_count = ?,
                            worker_id = NULL,
                            last_heartbeat_at = ?
                        WHERE job_id = ?;
                        """,
                        (new_retries, now, job_id)
                    )
                recovered_job_ids.append(job_id)

            # Also sweep expired chunk batches in staging table
            conn.execute(
                """
                UPDATE ingestion_job_chunks
                SET is_embedded = 0, worker_id = NULL, lease_expires_at = NULL
                WHERE is_embedded = 2 AND lease_expires_at < ?;
                """,
                (now,)
            )

            conn.commit()
        return recovered_job_ids

    # ==========================================
    # Chunk Ledger & Staging Operations
    # ==========================================

    def stage_chunks(self, job_id: str, chunks: List[Any]):
        now = time.time()
        with self._get_connection() as conn:
            for c in chunks:
                meta_str = json.dumps(c.metadata if hasattr(c, "metadata") else {})
                conn.execute(
                    """
                    INSERT OR REPLACE INTO ingestion_job_chunks (
                        chunk_id, job_id, chunk_index, text, metadata, is_embedded, created_at
                    ) VALUES (?, ?, ?, ?, ?, 0, ?);
                    """,
                    (c.chunk_id, job_id, c.chunk_index, c.text, meta_str, now)
                )
            conn.commit()

    def get_unembedded_chunks(self, job_id: str, limit: int = 32) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                SELECT chunk_id, job_id, chunk_index, text, metadata
                FROM ingestion_job_chunks
                WHERE job_id = ? AND is_embedded = 0
                ORDER BY chunk_index ASC
                LIMIT ?;
                """,
                (job_id, limit)
            )
            rows = cur.fetchall()
            results = []
            for r in rows:
                results.append({
                    "chunk_id": r["chunk_id"],
                    "job_id": r["job_id"],
                    "chunk_index": r["chunk_index"],
                    "text": r["text"],
                    "metadata": json.loads(r["metadata"])
                })
            return results

    def mark_chunks_embedded(self, job_id: str, chunk_ids: List[str]):
        if not chunk_ids:
            return
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE ingestion_job_chunks SET is_embedded = 1 WHERE job_id = ? AND chunk_id IN ({placeholders});",
                [job_id] + chunk_ids
            )
            conn.commit()

    def claim_next_chunk_batch(
        self,
        worker_id: str,
        batch_size: int = 32,
        lease_seconds: float = 60.0
    ) -> Optional[Dict[str, Any]]:
        """
        Swarm Embedding Claim: Atomically claims up to batch_size unembedded chunks
        for an active document using BEGIN IMMEDIATE write serialization and offset tracking.
        """
        now = time.time()
        lease_expires = now + lease_seconds
        with self._get_connection() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE;")
            except sqlite3.OperationalError:
                pass

            # Find an active document with pending chunks
            cur = conn.execute(
                """
                SELECT q.job_id, q.doc_family, q.document_id, q.version, q.declared_domain, cj.actual_total_chunks
                FROM ingestion_queue q
                JOIN ingestion_chunk_jobs cj ON q.job_id = cj.job_id
                JOIN ingestion_job_chunks jc ON q.job_id = jc.job_id
                WHERE (q.status = 'READY_FOR_EMBED' OR (q.status = 'PROCESSING' AND q.current_stage = 'EMBEDDING'))
                  AND cj.actual_total_chunks IS NOT NULL
                  AND (jc.is_embedded = 0 OR (jc.is_embedded = 2 AND jc.lease_expires_at < ?))
                GROUP BY q.job_id
                ORDER BY q.created_at ASC
                LIMIT 1;
                """,
                (now,)
            )
            job_row = cur.fetchone()
            if not job_row:
                return None

            job_id = job_row["job_id"]

            # Advance queue status to PROCESSING
            conn.execute(
                "UPDATE ingestion_queue SET status = 'PROCESSING', current_stage = 'EMBEDDING', last_heartbeat_at = ? WHERE job_id = ? AND status = 'READY_FOR_EMBED';",
                (now, job_id)
            )

            # Claim batch ordered strictly by monotonic chunk_index
            cur_chunks = conn.execute(
                """
                SELECT chunk_id, chunk_index, text, metadata
                FROM ingestion_job_chunks
                WHERE job_id = ?
                  AND (is_embedded = 0 OR (is_embedded = 2 AND lease_expires_at < ?))
                ORDER BY chunk_index ASC
                LIMIT ?;
                """,
                (job_id, now, batch_size)
            )
            chunk_rows = cur_chunks.fetchall()
            if not chunk_rows:
                return None

            claimed_ids = [r["chunk_id"] for r in chunk_rows]
            placeholders = ",".join("?" for _ in claimed_ids)

            conn.execute(
                f"""
                UPDATE ingestion_job_chunks
                SET is_embedded = 2,
                    worker_id = ?,
                    lease_expires_at = ?
                WHERE chunk_id IN ({placeholders});
                """,
                [worker_id, lease_expires] + claimed_ids
            )

            conn.execute(
                "UPDATE ingestion_workers SET status = 'BUSY', current_job_id = ?, last_heartbeat_at = ? WHERE worker_id = ?;",
                (job_id, now, worker_id)
            )
            conn.commit()

            chunk_indices = [r["chunk_index"] for r in chunk_rows]
            start_offset = min(chunk_indices) if chunk_indices else 0
            end_offset = max(chunk_indices) if chunk_indices else 0

            return {
                "job_id": job_id,
                "doc_family": job_row["doc_family"],
                "document_id": job_row["document_id"],
                "version": job_row["version"],
                "declared_domain": job_row["declared_domain"],
                "actual_total_chunks": job_row["actual_total_chunks"],
                "start_offset": start_offset,
                "end_offset": end_offset,
                "chunks": [
                    {
                        "chunk_id": r["chunk_id"],
                        "chunk_index": r["chunk_index"],
                        "text": r["text"],
                        "metadata": json.loads(r["metadata"])
                    }
                    for r in chunk_rows
                ]
            }

    def complete_chunk_batch(
        self,
        job_id: str,
        worker_id: str,
        chunk_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Marks batch chunks embedded, increments embedded_count, and checks barrier completeness.
        Returns is_barrier_complete = True if all chunks match actual_total_chunks.
        """
        now = time.time()
        if not chunk_ids:
            return {"is_barrier_complete": False, "job_id": job_id, "embedded_count": 0, "actual_total_chunks": 0}

        placeholders = ",".join("?" for _ in chunk_ids)
        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE ingestion_job_chunks SET is_embedded = 1, worker_id = ?, lease_expires_at = NULL WHERE chunk_id IN ({placeholders});",
                [worker_id] + chunk_ids
            )

            conn.execute(
                "UPDATE ingestion_chunk_jobs SET embedded_count = embedded_count + ?, updated_at = ? WHERE job_id = ?;",
                (len(chunk_ids), now, job_id)
            )

            # Check ground truth barrier
            cur = conn.execute(
                "SELECT actual_total_chunks, embedded_count FROM ingestion_chunk_jobs WHERE job_id = ? LIMIT 1;",
                (job_id,)
            )
            cj = cur.fetchone()
            actual_total = cj["actual_total_chunks"] if cj else None
            embedded = cj["embedded_count"] if cj else 0

            # Double check staging table remaining count (excluding completed and quarantined)
            cur_unfin = conn.execute(
                "SELECT COUNT(*) as unfin FROM ingestion_job_chunks WHERE job_id = ? AND is_embedded NOT IN (1, -1);",
                (job_id,)
            )
            unfin = cur_unfin.fetchone()["unfin"]

            cur_dlq = conn.execute(
                "SELECT COUNT(*) as dlq_count FROM ingestion_job_chunks WHERE job_id = ? AND is_embedded = -1;",
                (job_id,)
            )
            dlq_count = cur_dlq.fetchone()["dlq_count"]

            is_barrier_complete = (actual_total is not None and (embedded + dlq_count) >= actual_total and unfin == 0)

            conn.execute(
                "UPDATE ingestion_workers SET status = 'IDLE', current_job_id = NULL, last_heartbeat_at = ? WHERE worker_id = ?;",
                (now, worker_id)
            )
            conn.commit()

            return {
                "is_barrier_complete": is_barrier_complete,
                "is_degraded": (dlq_count > 0),
                "job_id": job_id,
                "embedded_count": embedded,
                "actual_total_chunks": actual_total,
                "quarantined_chunks": dlq_count,
                "unfinished_chunks": unfin
            }

    def release_chunk_batch(self, chunk_ids: List[str]) -> None:
        """
        Releases the lease on an in-progress batch of chunks so they can be re-claimed.
        """
        if not chunk_ids:
            return
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE ingestion_job_chunks SET is_embedded = 0, worker_id = NULL, lease_expires_at = NULL WHERE chunk_id IN ({placeholders}) AND is_embedded = 2;",
                chunk_ids
            )
            conn.commit()

    def quarantine_chunk_batch(
        self,
        job_id: str,
        document_id: str,
        start_offset: int,
        end_offset: int,
        chunk_ids: List[str],
        payload: Dict[str, Any],
        error_reason: str,
        exception_trace: Optional[str] = None,
        retry_count: int = 3,
        worker_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Quarantines a poisoned chunk batch into ingestion_dlq, marks chunks is_embedded = -1,
        and checks if remaining non-poison chunks resolve the document barrier.
        """
        now = time.time()
        dlq_id = f"dlq_{uuid.uuid4().hex[:8]}"
        placeholders = ",".join("?" for _ in chunk_ids)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_dlq (
                    dlq_id, job_id, document_id, start_offset, end_offset,
                    chunk_ids, payload, error_reason, exception_trace,
                    retry_count, quarantined_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'QUARANTINED');
                """,
                (
                    dlq_id,
                    job_id,
                    document_id,
                    start_offset,
                    end_offset,
                    json.dumps(chunk_ids),
                    json.dumps(payload),
                    error_reason,
                    exception_trace or "",
                    retry_count,
                    now
                )
            )

            # Mark chunks as quarantined (-1) so normal embedders skip them
            if chunk_ids:
                conn.execute(
                    f"UPDATE ingestion_job_chunks SET is_embedded = -1, lease_expires_at = NULL WHERE chunk_id IN ({placeholders});",
                    chunk_ids
                )

            # Check if all chunks are now resolved (either embedded or quarantined)
            cur = conn.execute(
                "SELECT actual_total_chunks, embedded_count FROM ingestion_chunk_jobs WHERE job_id = ? LIMIT 1;",
                (job_id,)
            )
            cj = cur.fetchone()
            actual_total = cj["actual_total_chunks"] if cj else None
            embedded = cj["embedded_count"] if cj else 0

            cur_unfin = conn.execute(
                "SELECT COUNT(*) as unfin FROM ingestion_job_chunks WHERE job_id = ? AND is_embedded NOT IN (1, -1);",
                (job_id,)
            )
            unfin = cur_unfin.fetchone()["unfin"]

            cur_dlq = conn.execute(
                "SELECT COUNT(*) as dlq_count FROM ingestion_job_chunks WHERE job_id = ? AND is_embedded = -1;",
                (job_id,)
            )
            dlq_count = cur_dlq.fetchone()["dlq_count"]

            is_barrier_complete = (actual_total is not None and (embedded + dlq_count) >= actual_total and unfin == 0)

            if worker_id:
                conn.execute(
                    "UPDATE ingestion_workers SET status = 'IDLE', current_job_id = NULL, last_heartbeat_at = ? WHERE worker_id = ?;",
                    (now, worker_id)
                )

            conn.commit()

            return {
                "dlq_id": dlq_id,
                "job_id": job_id,
                "quarantined_count": len(chunk_ids),
                "is_barrier_complete": is_barrier_complete,
                "is_degraded": True,
                "unfinished_chunks": unfin,
                "embedded_count": embedded,
                "actual_total_chunks": actual_total
            }

    def list_dlq_records(
        self,
        job_id: Optional[str] = None,
        status: str = "QUARANTINED"
    ) -> List[Dict[str, Any]]:
        """Lists records from the Dead Letter Queue."""
        with self._get_connection() as conn:
            if job_id:
                cur = conn.execute(
                    "SELECT * FROM ingestion_dlq WHERE job_id = ? AND status = ? ORDER BY quarantined_at DESC;",
                    (job_id, status)
                )
            else:
                cur = conn.execute(
                    "SELECT * FROM ingestion_dlq WHERE status = ? ORDER BY quarantined_at DESC;",
                    (status,)
                )
            rows = cur.fetchall()
            return [
                {
                    "dlq_id": r["dlq_id"],
                    "job_id": r["job_id"],
                    "document_id": r["document_id"],
                    "start_offset": r["start_offset"],
                    "end_offset": r["end_offset"],
                    "chunk_ids": json.loads(r["chunk_ids"]),
                    "payload": json.loads(r["payload"]),
                    "error_reason": r["error_reason"],
                    "exception_trace": r["exception_trace"],
                    "retry_count": r["retry_count"],
                    "quarantined_at": r["quarantined_at"],
                    "status": r["status"]
                }
                for r in rows
            ]

    def replay_dlq_batch(self, dlq_id: str) -> bool:
        """
        Replays a quarantined batch from ingestion_dlq by resetting chunk is_embedded = 0.
        """
        now = time.time()
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM ingestion_dlq WHERE dlq_id = ? AND status = 'QUARANTINED' LIMIT 1;", (dlq_id,))
            row = cur.fetchone()
            if not row:
                return False

            chunk_ids = json.loads(row["chunk_ids"])
            job_id = row["job_id"]
            placeholders = ",".join("?" for _ in chunk_ids)

            # Reset chunks to pending (is_embedded = 0)
            conn.execute(
                f"UPDATE ingestion_job_chunks SET is_embedded = 0, lease_expires_at = NULL WHERE chunk_id IN ({placeholders});",
                chunk_ids
            )

            # Mark DLQ record as REPLAYED
            conn.execute(
                "UPDATE ingestion_dlq SET status = 'REPLAYED' WHERE dlq_id = ?;",
                (dlq_id,)
            )

            # Reopen queue job if it had completed as degraded or failed
            conn.execute(
                """
                UPDATE ingestion_queue
                SET status = 'PROCESSING', current_stage = 'EMBEDDING', last_heartbeat_at = ?
                WHERE job_id = ? AND status IN ('ACTIVE_DEGRADED', 'FAILED', 'COMPLETED');
                """,
                (now, job_id)
            )

            conn.commit()
            return True

    def update_chunk_progress(
        self,
        chunk_job_id: str,
        chunked_count: int,
        chunk_ids: List[str],
        expected_chunks: Optional[int] = None,
        status: Optional[str] = None
    ):
        now = time.time()
        chunk_ids_json = json.dumps(chunk_ids)
        with self._get_connection() as conn:
            if expected_chunks is not None and status is not None:
                conn.execute(
                    """
                    UPDATE ingestion_chunk_jobs
                    SET chunked_count = ?, chunk_ids = ?, expected_chunks = ?, status = ?, updated_at = ?
                    WHERE chunk_job_id = ?;
                    """,
                    (chunked_count, chunk_ids_json, expected_chunks, status, now, chunk_job_id)
                )
            elif status is not None:
                conn.execute(
                    """
                    UPDATE ingestion_chunk_jobs
                    SET chunked_count = ?, chunk_ids = ?, status = ?, updated_at = ?
                    WHERE chunk_job_id = ?;
                    """,
                    (chunked_count, chunk_ids_json, status, now, chunk_job_id)
                )
            else:
                conn.execute(
                    """
                    UPDATE ingestion_chunk_jobs
                    SET chunked_count = ?, chunk_ids = ?, updated_at = ?
                    WHERE chunk_job_id = ?;
                    """,
                    (chunked_count, chunk_ids_json, now, chunk_job_id)
                )
            conn.commit()

    def get_chunk_job(self, chunk_job_id: str) -> Optional[IngestionChunkJob]:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM ingestion_chunk_jobs WHERE chunk_job_id = ? LIMIT 1;", (chunk_job_id,))
            row = cur.fetchone()
            if not row:
                return None
            return IngestionChunkJob(
                chunk_job_id=row["chunk_job_id"],
                job_id=row["job_id"],
                expected_chunks=row["expected_chunks"],
                actual_total_chunks=row["actual_total_chunks"] if "actual_total_chunks" in row.keys() else None,
                chunked_count=row["chunked_count"],
                embedded_count=row["embedded_count"] if "embedded_count" in row.keys() else 0,
                chunk_ids=json.loads(row["chunk_ids"]),
                status=row["status"],
                created_at=row["created_at"],
                updated_at=row["updated_at"]
            )

    # ==========================================
    # Worker Registry & Health Check
    # ==========================================

    def register_worker(self, worker_id: str, hostname: Optional[str] = None, pid: Optional[int] = None):
        now = time.time()
        hostname = hostname or socket.gethostname()
        pid = pid or os.getpid()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO ingestion_workers (
                    worker_id, hostname, pid, status, current_job_id,
                    last_heartbeat_at, registered_at, tasks_completed
                ) VALUES (?, ?, ?, 'IDLE', NULL, ?, ?, 0);
                """,
                (worker_id, hostname, pid, now, now)
            )
            conn.commit()

    def worker_heartbeat(self, worker_id: str, status: Optional[str] = None, current_job_id: Optional[str] = None):
        now = time.time()
        with self._get_connection() as conn:
            if status and current_job_id is not None:
                conn.execute(
                    "UPDATE ingestion_workers SET last_heartbeat_at = ?, status = ?, current_job_id = ? WHERE worker_id = ?;",
                    (now, status, current_job_id, worker_id)
                )
            elif status:
                conn.execute(
                    "UPDATE ingestion_workers SET last_heartbeat_at = ?, status = ? WHERE worker_id = ?;",
                    (now, status, worker_id)
                )
            else:
                conn.execute(
                    "UPDATE ingestion_workers SET last_heartbeat_at = ? WHERE worker_id = ?;",
                    (now, worker_id)
                )
            conn.commit()

    def unregister_worker(self, worker_id: str):
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE ingestion_workers SET status = 'OFFLINE', current_job_id = NULL, last_heartbeat_at = ? WHERE worker_id = ?;",
                (now, worker_id)
            )
            conn.commit()

    def get_available_workers(self, heartbeat_timeout_seconds: float = 60.0) -> List[Dict[str, Any]]:
        now = time.time()
        stale_cutoff = now - heartbeat_timeout_seconds
        with self._get_connection() as conn:
            # First, flag any worker that missed heartbeats as DEAD
            conn.execute(
                "UPDATE ingestion_workers SET status = 'DEAD' WHERE status IN ('IDLE', 'BUSY') AND last_heartbeat_at < ?;",
                (stale_cutoff,)
            )
            conn.commit()

            cur = conn.execute("SELECT * FROM ingestion_workers WHERE status IN ('IDLE', 'BUSY');")
            rows = cur.fetchall()
            return [
                {
                    "worker_id": r["worker_id"],
                    "hostname": r["hostname"],
                    "pid": r["pid"],
                    "status": r["status"],
                    "current_job_id": r["current_job_id"],
                    "last_heartbeat_age": round(now - r["last_heartbeat_at"], 2),
                    "tasks_completed": r["tasks_completed"]
                }
                for r in rows
            ]

    def check_worker_health(self, worker_id: str, heartbeat_timeout_seconds: float = 60.0) -> Dict[str, Any]:
        now = time.time()
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM ingestion_workers WHERE worker_id = ? LIMIT 1;", (worker_id,))
            row = cur.fetchone()
            if not row:
                return {"worker_id": worker_id, "status": "UNKNOWN", "healthy": False}

            age = now - row["last_heartbeat_at"]
            if row["status"] == "OFFLINE":
                return {"worker_id": worker_id, "status": "OFFLINE", "healthy": False, "heartbeat_age": round(age, 2)}
            if age > heartbeat_timeout_seconds or row["status"] == "DEAD":
                return {"worker_id": worker_id, "status": "DEAD", "healthy": False, "heartbeat_age": round(age, 2)}

            return {
                "worker_id": worker_id,
                "status": row["status"],
                "healthy": True,
                "current_job_id": row["current_job_id"],
                "heartbeat_age": round(age, 2),
                "tasks_completed": row["tasks_completed"]
            }
