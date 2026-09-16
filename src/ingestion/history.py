import os
import time
import json
import sqlite3
import logging
import threading
import uuid
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from contextlib import contextmanager

from .loader import RawDocument

logger = logging.getLogger("kitchome.ingestion.history")

class IngestionHistoryRecord(BaseModel):
    job_id: str
    document_id: str
    title: str
    user_id: str
    tenant_id: str
    file_path: Optional[str] = None
    content_preview: Optional[str] = None
    content_hash: Optional[str] = None
    content: Optional[str] = ""
    declared_domain: Optional[str] = None
    access_tier: str = "free"
    clearance_level: int = 1
    status: str = "PENDING"  # PENDING | IN_PROGRESS | COMPLETED | FAILED
    chunks_ingested: int = 0
    namespace: Optional[str] = None
    summary: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

class IngestionHistoryManager:
    """
    Thread-safe SQLite-backed manager tracking document ingestion lifecycle,
    per-user history, failure diagnostics, and manual retries without memory bloat.
    """
    def __init__(self, db_path: str = "data/ingestion_history.db"):
        self.db_path = db_path
        self._local = threading.local()
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()
        logger.debug("IngestionHistoryManager initialized at db_path='%s'", os.path.abspath(db_path))

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
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                self._local.conn.close()
            except Exception as e:
                logger.debug("Error closing connection: %s", e)
            self._local.conn = None

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_history (
                    job_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    file_path TEXT,
                    content_preview TEXT,
                    content_hash TEXT,
                    content TEXT,
                    declared_domain TEXT,
                    access_tier TEXT DEFAULT 'free',
                    clearance_level INTEGER DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    chunks_ingested INTEGER DEFAULT 0,
                    namespace TEXT,
                    summary TEXT,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    metadata_json TEXT DEFAULT '{}',
                    created_at REAL,
                    updated_at REAL
                );
            """)

            # Dynamic migration for existing databases
            cols = {col["name"] for col in conn.execute("PRAGMA table_info(ingestion_history);").fetchall()}
            if "file_path" not in cols:
                conn.execute("ALTER TABLE ingestion_history ADD COLUMN file_path TEXT;")
            if "content_preview" not in cols:
                conn.execute("ALTER TABLE ingestion_history ADD COLUMN content_preview TEXT;")
            if "content_hash" not in cols:
                conn.execute("ALTER TABLE ingestion_history ADD COLUMN content_hash TEXT;")

            conn.execute("CREATE INDEX IF NOT EXISTS idx_hist_tenant_user ON ingestion_history (tenant_id, user_id, created_at DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hist_status ON ingestion_history (status);")
            conn.commit()

    def _row_to_record(self, row: sqlite3.Row) -> IngestionHistoryRecord:
        meta = {}
        try:
            raw_meta = row["metadata_json"]
            if raw_meta:
                meta = json.loads(raw_meta)
        except Exception:
            pass

        return IngestionHistoryRecord(
            job_id=row["job_id"],
            document_id=row["document_id"],
            title=row["title"],
            user_id=row["user_id"],
            tenant_id=row["tenant_id"],
            file_path=row["file_path"] if "file_path" in row.keys() else None,
            content_preview=row["content_preview"] if "content_preview" in row.keys() else None,
            content_hash=row["content_hash"] if "content_hash" in row.keys() else None,
            content=row["content"] if "content" in row.keys() and row["content"] else "",
            declared_domain=row["declared_domain"],
            access_tier=row["access_tier"],
            clearance_level=row["clearance_level"],
            status=row["status"],
            chunks_ingested=row["chunks_ingested"],
            namespace=row["namespace"],
            summary=row["summary"],
            error_message=row["error_message"],
            retry_count=row["retry_count"],
            metadata=meta,
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )

    def record_start(
        self,
        document_id: str,
        title: str,
        user_id: str,
        tenant_id: str,
        file_path: Optional[str] = None,
        content_preview: Optional[str] = None,
        content_hash: Optional[str] = None,
        content: Optional[str] = None,
        clearance_level: int = 1,
        access_tier: str = "free",
        declared_domain: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None
    ) -> IngestionHistoryRecord:
        jid = job_id or f"ing_{uuid.uuid4().hex[:12]}"
        now = time.time()
        meta_json = json.dumps(metadata or {})

        preview = content_preview
        if preview is None and content:
            preview = content[:250].strip()

        stored_content = content or ""

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_history (
                    job_id, document_id, title, user_id, tenant_id,
                    file_path, content_preview, content_hash, content,
                    declared_domain, access_tier, clearance_level, status,
                    chunks_ingested, retry_count, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'IN_PROGRESS', 0, 0, ?, ?, ?);
                """,
                (
                    jid, document_id, title, user_id, tenant_id,
                    file_path, preview, content_hash, stored_content,
                    declared_domain, access_tier, clearance_level, meta_json, now, now
                )
            )
            conn.commit()

        logger.info("Recorded ingestion start: job_id='%s', doc='%s', user='%s', tenant='%s', file='%s'", jid, document_id, user_id, tenant_id, file_path or 'none')
        return self.get_job(jid)  # type: ignore

    def record_success(
        self,
        job_id: str,
        namespace: str,
        chunks_ingested: int,
        summary: Optional[str] = None
    ) -> Optional[IngestionHistoryRecord]:
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE ingestion_history
                SET status = 'COMPLETED',
                    namespace = ?,
                    chunks_ingested = ?,
                    summary = ?,
                    error_message = NULL,
                    updated_at = ?
                WHERE job_id = ?;
                """,
                (namespace, chunks_ingested, summary, now, job_id)
            )
            conn.commit()

        logger.info("Recorded ingestion success: job_id='%s', chunks=%d, ns='%s'", job_id, chunks_ingested, namespace)
        return self.get_job(job_id)

    def record_failure(
        self,
        job_id: str,
        error_message: str
    ) -> Optional[IngestionHistoryRecord]:
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE ingestion_history
                SET status = 'FAILED',
                    error_message = ?,
                    updated_at = ?
                WHERE job_id = ?;
                """,
                (error_message, now, job_id)
            )
            conn.commit()

        logger.warning("Recorded ingestion failure: job_id='%s', reason='%s'", job_id, error_message)
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> Optional[IngestionHistoryRecord]:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM ingestion_history WHERE job_id = ?;", (job_id,))
            row = cur.fetchone()
            return self._row_to_record(row) if row else None

    def get_history(
        self,
        tenant_id: str,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[IngestionHistoryRecord]:
        query = "SELECT * FROM ingestion_history WHERE tenant_id = ?"
        params: List[Any] = [tenant_id]

        if user_id is not None:
            query += " AND user_id = ?"
            params.append(user_id)

        if status is not None:
            query += " AND status = ?"
            params.append(status.upper())

        query += " ORDER BY created_at DESC LIMIT ?;"
        params.append(limit)

        with self._get_connection() as conn:
            cur = conn.execute(query, params)
            return [self._row_to_record(r) for r in cur.fetchall()]

    def retry_job(
        self,
        job_id: str,
        caller_user_id: str,
        is_admin: bool = False,
        pipeline: Any = None
    ) -> Dict[str, Any]:
        """
        Retries a failed ingestion job.
        Strictly enforces user-ownership: only the original owner or an admin can retry.
        """
        record = self.get_job(job_id)
        if not record:
            raise ValueError(f"Ingestion job '{job_id}' not found.")

        if not is_admin and record.user_id != caller_user_id:
            raise PermissionError(
                f"Access Denied: Ingestion job '{job_id}' was created by user '{record.user_id}'. "
                f"You do not have permission to retry it."
            )

        new_retry_count = record.retry_count + 1
        now = time.time()

        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE ingestion_history
                SET status = 'IN_PROGRESS',
                    retry_count = ?,
                    updated_at = ?
                WHERE job_id = ?;
                """,
                (new_retry_count, now, job_id)
            )
            conn.commit()

        # Re-execute via IngestionPipeline if provided
        if pipeline is not None:
            try:
                if record.file_path and os.path.exists(record.file_path) and hasattr(pipeline, "ingest_file"):
                    ingest_res = pipeline.ingest_file(
                        file_path=record.file_path,
                        document_id=record.document_id,
                        title=record.title,
                        tenant_id=record.tenant_id,
                        clearance_level=record.clearance_level,
                        access_tier=record.access_tier,
                        declared_domain=record.declared_domain,
                        metadata=record.metadata
                    )
                else:
                    raw_doc = RawDocument(
                        document_id=record.document_id,
                        title=record.title,
                        source_path=record.file_path or f"retry://{record.document_id}",
                        content=record.content or "",
                        declared_domain=record.declared_domain,
                        access_tier=record.access_tier,
                        tenant_id=record.tenant_id,
                        clearance_level=record.clearance_level,
                        metadata=record.metadata
                    )
                    ingest_res = pipeline.ingest_document(raw_doc)

                self.record_success(
                    job_id=job_id,
                    namespace=ingest_res.get("namespace", "general_home"),
                    chunks_ingested=ingest_res.get("chunks_ingested", 0),
                    summary=ingest_res.get("summary")
                )
                return {
                    "status": "SUCCESS",
                    "job_id": job_id,
                    "document_id": record.document_id,
                    "retry_count": new_retry_count,
                    "namespace": ingest_res.get("namespace"),
                    "chunks_ingested": ingest_res.get("chunks_ingested", 0),
                    "message": f"Job '{job_id}' retried successfully (attempt #{new_retry_count})."
                }
            except Exception as e:
                err_msg = str(e)
                self.record_failure(job_id=job_id, error_message=err_msg)
                return {
                    "status": "FAILED",
                    "job_id": job_id,
                    "document_id": record.document_id,
                    "retry_count": new_retry_count,
                    "error_message": err_msg,
                    "message": f"Job '{job_id}' retry failed: {err_msg}"
                }

        return {
            "status": "RE_QUEUED",
            "job_id": job_id,
            "document_id": record.document_id,
            "retry_count": new_retry_count,
            "message": f"Job '{job_id}' marked for retry."
        }

# Global singleton
local_ingestion_history_manager = IngestionHistoryManager()
