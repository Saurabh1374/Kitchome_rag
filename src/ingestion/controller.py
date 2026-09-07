import os
import math
import hashlib
import uuid
from typing import Optional, Dict, Any

from .cursor import IngestionCursorManager
from .queue import IngestionQueueManager
from ..vector_store.base import NamespaceVectorStore
from ..skills_ms.router import SkillsRouter

class IngestionController:
    """
    Decoupled Ingestion Controller (Producer / Intake Service).
    Handles non-blocking intake (<10ms SLA), cursor idempotency checks,
    canonical file storage, and quash operations.
    """
    def __init__(
        self,
        cursor_manager: Optional[IngestionCursorManager] = None,
        queue_manager: Optional[IngestionQueueManager] = None,
        vector_store: Optional[NamespaceVectorStore] = None,
        router: Optional[SkillsRouter] = None,
        data_root: str = "data"
    ):
        self.cursor_mgr = cursor_manager or IngestionCursorManager()
        self.queue_mgr = queue_manager or IngestionQueueManager()
        self.vector_store = vector_store or NamespaceVectorStore()
        self.router = router or SkillsRouter()
        self.data_root = data_root

    def intake_document(
        self,
        file_path: Optional[str] = None,
        content: Optional[str] = None,
        title: Optional[str] = None,
        doc_family: Optional[str] = None,
        declared_domain: Optional[str] = None,
        access_tier: str = "free",
        tenant_id: str = "global",
        clearance_level: int = 1
    ) -> Dict[str, Any]:
        """
        Fast non-blocking document intake (<10ms SLA).
        """
        if not file_path and not content:
            raise ValueError("Either file_path or content must be provided.")

        # 1. Resolve content and file_path
        if file_path and not content:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

        if not declared_domain and file_path:
            parent_dir = os.path.basename(os.path.dirname(file_path))
            if parent_dir and parent_dir not in ("data", ".", ""):
                declared_domain = parent_dir
        declared_domain = declared_domain or "general_home"

        if not doc_family:
            if title:
                doc_family = title.lower().strip().replace(" ", "_")
            elif file_path:
                doc_family = os.path.splitext(os.path.basename(file_path))[0]
            else:
                doc_family = "doc"

        # 2. Canonical file persistence if content passed directly
        if not file_path:
            filename = f"{doc_family}.md"
            target_dir = os.path.join(self.data_root, declared_domain)
            os.makedirs(target_dir, exist_ok=True)
            file_path = os.path.join(target_dir, filename)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

        # 3. Compute SHA-256 Hash
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # 4. Point 1: Cursor Gatekeeper Check (Idempotency)
        existing_cursor = self.cursor_mgr.get_by_hash(content_hash)
        if existing_cursor and existing_cursor.status == "ACTIVE":
            return {
                "status": "SKIPPED",
                "reason": "identical_hash",
                "document_id": existing_cursor.document_id,
                "doc_family": existing_cursor.doc_family,
                "version": existing_cursor.version,
                "file_path": existing_cursor.file_path,
                "message": "Content hash matches active document version. 0 compute wasted."
            }

        # 5. Point 2: Versioning & Family grouping
        next_version = self.cursor_mgr.get_next_version(doc_family)
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_family}_v{next_version}"))

        # 6. Estimate Expected Chunks (~700 tokens / ~2800 chars)
        expected_chunks = max(1, math.ceil(len(content) / 2800))

        # 7. Non-Blocking Enqueue
        job = self.queue_mgr.enqueue_job(
            document_id=doc_id,
            file_path=file_path,
            doc_family=doc_family,
            version=next_version,
            declared_domain=declared_domain,
            job_type="UPDATE" if next_version > 1 else "INGEST",
            expected_chunks=expected_chunks
        )

        return {
            "status": "ACCEPTED",
            "http_status": 202,
            "job_id": job.job_id,
            "chunk_job_id": job.chunk_job_id,
            "document_id": doc_id,
            "doc_family": doc_family,
            "version": next_version,
            "expected_chunks": expected_chunks,
            "file_path": file_path,
            "message": f"Document queued successfully as version {next_version}."
        }

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        job = self.queue_mgr.get_job(job_id)
        if not job:
            return {"status": "NOT_FOUND", "job_id": job_id}

        result = {
            "job_id": job.job_id,
            "document_id": job.document_id,
            "file_path": job.file_path,
            "doc_family": job.doc_family,
            "version": job.version,
            "status": job.status,
            "current_stage": job.current_stage,
            "worker_id": job.worker_id,
            "retry_count": job.retry_count,
            "error_reason": job.error_reason,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at
        }

        if job.chunk_job_id:
            chunk_job = self.queue_mgr.get_chunk_job(job.chunk_job_id)
            if chunk_job:
                result["chunk_job"] = {
                    "chunk_job_id": chunk_job.chunk_job_id,
                    "expected_chunks": chunk_job.expected_chunks,
                    "chunked_count": chunk_job.chunked_count,
                    "status": chunk_job.status
                }
                denom = max(1, chunk_job.expected_chunks)
                result["progress_percentage"] = round((chunk_job.chunked_count / denom) * 100.0, 1)

        return result

    def get_worker_health_status(self, timeout_seconds: float = 60.0) -> Dict[str, Any]:
        available = self.queue_mgr.get_available_workers(heartbeat_timeout_seconds=timeout_seconds)
        idle_count = sum(1 for w in available if w["status"] == "IDLE")
        busy_count = sum(1 for w in available if w["status"] == "BUSY")
        return {
            "total_available_workers": len(available),
            "idle_workers": idle_count,
            "busy_workers": busy_count,
            "workers": available
        }

    def manual_retry(self, job_id: str) -> Dict[str, Any]:
        # 1. Clean up any uncommitted staged chunks in vector store
        self.vector_store.delete_chunks_by_job_id(job_id)
        # 2. Reset queue job to PENDING
        job = self.queue_mgr.manual_retry_job(job_id)
        if not job:
            return {"status": "NOT_FOUND", "job_id": job_id}
        return {
            "status": "RE_QUEUED",
            "job_id": job.job_id,
            "retry_count": job.retry_count,
            "message": "Job retried manually. Retries reset to 0 and staged chunks purged."
        }

    def quash_document_version(self, document_id: str) -> Dict[str, Any]:
        """
        User-driven Quash: Physically deletes chunks from vector store, marks cursor QUASHED.
        """
        chunks_deleted = self.vector_store.delete_chunks_by_document_id(document_id)
        quashed = self.cursor_mgr.quash_version(document_id)
        return {
            "status": "QUASHED" if quashed else "NOT_FOUND",
            "document_id": document_id,
            "chunks_deleted": chunks_deleted,
            "message": f"Document version quashed. {chunks_deleted} vector chunks physically deleted."
        }
