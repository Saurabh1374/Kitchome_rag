import os
import socket
import time
import uuid
import math
import hashlib
import logging
import traceback
from typing import Optional, List, Dict, Any, Union

from config import config as app_config
from .loader import DocumentLoader, RawDocument
from .chunker import TextChunker, TextChunk
from .summarizer import DocumentSummarizer
from .embedder import BaseEmbedder, get_embedder, EmbeddingEngine
from .cursor import IngestionCursorManager, IngestionCursorRecord
from .queue import IngestionQueueManager, IngestionQueueJob
from .parser import LocalDocumentParser
from ..vector_store.base import NamespaceVectorStore, VectorChunk
from ..skills_ms.router import SkillsRouter

logger = logging.getLogger(__name__)

def hashlib_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

class ChunkerWorker:
    """
    Worker Pool 1: Parsing, Summarization, and Chunking.
    Consumes PENDING document jobs from ingestion_queue, stages chunks into
    ingestion_job_chunks, locks actual_total_chunks ground truth, and handoffs to
    Worker Pool 2 by transitioning the document to READY_FOR_EMBED.
    """
    def __init__(
        self,
        worker_id: Optional[str] = None,
        queue_manager: Optional[IngestionQueueManager] = None,
        cursor_manager: Optional[IngestionCursorManager] = None,
        router: Optional[SkillsRouter] = None,
        chunker: Optional[TextChunker] = None,
        summarizer: Optional[DocumentSummarizer] = None,
        parser: Optional[LocalDocumentParser] = None,
        lease_duration_seconds: float = 120.0
    ):
        self.worker_id = worker_id or f"chunker_{socket.gethostname()}_{os.getpid()}_{uuid.uuid4().hex[:4]}"
        self.queue_mgr = queue_manager or IngestionQueueManager()
        self.cursor_mgr = cursor_manager or IngestionCursorManager()
        self.router = router or SkillsRouter()
        self.chunker = chunker or TextChunker()
        self.summarizer = summarizer or DocumentSummarizer()
        self.parser = parser or LocalDocumentParser()
        self.lease_duration_seconds = lease_duration_seconds

        # Register worker in cluster registry
        self.queue_mgr.register_worker(self.worker_id)
        self._is_running = True

    def heartbeat(self, current_job_id: Optional[str] = None):
        if current_job_id:
            self.queue_mgr.heartbeat_job(current_job_id, self.worker_id, extend_seconds=self.lease_duration_seconds)
        else:
            self.queue_mgr.worker_heartbeat(self.worker_id)

    def process_next_job(self) -> Optional[Dict[str, Any]]:
        """Claims and executes a single chunking job. Returns execution summary or None if queue empty."""
        job = self.queue_mgr.claim_next_chunking_job(self.worker_id, lease_duration_seconds=self.lease_duration_seconds)
        if not job:
            return None
        return self.execute_job(job)

    def execute_job(self, job: IngestionQueueJob) -> Dict[str, Any]:
        stage = "PARSING"
        try:
            # Stage 1: Parsing
            stage = "PARSING"
            self.queue_mgr.update_stage(job.job_id, "PARSING")
            self.heartbeat(job.job_id)

            if not os.path.exists(job.file_path):
                raise FileNotFoundError(f"Target document file missing: {job.file_path}")

            # Parse input file into clean Markdown and save under <dir>/processed/<filename>.md
            doc = self.parser.parse_and_save(job.file_path, declared_domain=job.declared_domain)
            doc.document_id = job.document_id

            # If output markdown is in a new processed path, synchronize queue job file_path
            if doc.source_path != job.file_path:
                self.queue_mgr.update_job_file_path(job.job_id, doc.source_path)
                job.file_path = doc.source_path

            # Record intermediate parsed state in IngestionCursor with status = 'PARSED'
            parsed_content_hash = hashlib_sha256(doc.content)
            cursor_record = IngestionCursorRecord(
                document_id=doc.document_id,
                file_path=doc.source_path,
                content_hash=parsed_content_hash,
                doc_family=job.doc_family,
                version=job.version,
                is_latest=False,
                status="PARSED",
                declared_domain=doc.declared_domain,
                access_tier=doc.access_tier,
                tenant_id=doc.tenant_id,
                clearance_level=doc.clearance_level
            )
            self.cursor_mgr.record_parsed(cursor_record)

            target_namespace = self.router.resolve_namespace_for_document(
                file_path=doc.source_path,
                content=doc.content,
                declared_domain=doc.declared_domain
            )

            # Stage 2: Summarization
            stage = "SUMMARIZATION"
            self.queue_mgr.update_stage(job.job_id, "SUMMARIZATION")
            self.heartbeat(job.job_id)

            doc_summary, extracted_keywords = self.summarizer.process_and_summarize_document(
                document_title=doc.title,
                text=doc.content
            )

            # Update skills.ms registry
            self.router.registry.update_dynamic_summary(
                namespace=target_namespace,
                document_title=doc.title,
                summary=doc_summary,
                extra_keywords=extracted_keywords
            )

            # Stage 3: Chunking
            stage = "CHUNKING"
            self.queue_mgr.update_stage(job.job_id, "CHUNKING")
            self.heartbeat(job.job_id)

            all_chunk_ids: List[str] = []
            content_hash = hashlib_sha256(doc.content)

            # Stream chunks in bounded mini-batches (32 chunks)
            for chunk_batch in self.chunker.chunk_document_stream(doc, namespace=target_namespace, batch_size=32):
                for c in chunk_batch:
                    c.metadata["doc_family"] = job.doc_family
                    c.metadata["job_id"] = job.job_id
                    c.metadata["version"] = job.version
                    c.metadata["target_namespace"] = target_namespace
                    c.metadata["doc_summary"] = doc_summary
                    c.metadata["document_title"] = doc.title
                    c.metadata["access_tier"] = doc.access_tier
                    c.metadata["tenant_id"] = doc.tenant_id
                    c.metadata["clearance_level"] = doc.clearance_level
                    c.metadata["file_path"] = job.file_path
                    c.metadata["content_hash"] = content_hash

                self.queue_mgr.stage_chunks(job.job_id, chunk_batch)
                all_chunk_ids.extend([c.chunk_id for c in chunk_batch])
                self.heartbeat(job.job_id)

            # Lock actual_total_chunks ground truth & transition to READY_FOR_EMBED
            chunk_job_id = job.chunk_job_id or f"chunk_job_{job.job_id}"
            self.queue_mgr.lock_chunking_complete(
                chunk_job_id=chunk_job_id,
                job_id=job.job_id,
                actual_total_chunks=len(all_chunk_ids),
                chunk_ids=all_chunk_ids,
                worker_id=self.worker_id
            )

            return {
                "status": "READY_FOR_EMBED",
                "job_id": job.job_id,
                "chunk_job_id": chunk_job_id,
                "document_id": job.document_id,
                "namespace": target_namespace,
                "actual_total_chunks": len(all_chunk_ids)
            }

        except Exception as e:
            logger.error(f"ChunkerWorker {self.worker_id} failed at stage {stage} on job {job.job_id}: {e}", exc_info=True)
            self.queue_mgr.fail_job(
                job_id=job.job_id,
                worker_id=self.worker_id,
                stage=stage,
                error_reason=str(e)
            )
            return {
                "status": "FAILED",
                "job_id": job.job_id,
                "failed_stage": stage,
                "error": str(e)
            }

    def run_forever(
        self,
        poll_interval: float = 1.0,
        max_iterations: Optional[int] = None,
        stop_event: Optional[Any] = None
    ):
        """Continuously polls the queue for PENDING chunking jobs until stopped."""
        logger.info(f"ChunkerWorker {self.worker_id} started daemon loop.")
        self._is_running = True
        iterations = 0
        try:
            while self._is_running and not (stop_event and stop_event.is_set()):
                if max_iterations is not None and iterations >= max_iterations:
                    break
                self.heartbeat()
                res = self.process_next_job()
                iterations += 1
                if not res:
                    time.sleep(poll_interval)
        except KeyboardInterrupt:
            logger.info(f"Shutdown requested for ChunkerWorker {self.worker_id}")
        finally:
            self.stop()

    def stop(self):
        self._is_running = False
        self.queue_mgr.unregister_worker(self.worker_id)
        if hasattr(self.queue_mgr, "close"):
            self.queue_mgr.close()


class EmbedderWorker:
    """
    Worker Pool 2: Swarm Embedding & Barrier Synchronization.
    Concurrently claims mini-batches of staged chunks via contiguous offset windows,
    computes vector embeddings, executes 5-point post-completion validation,
    inserts clean vector rows (Model 1: embedding NOT NULL), and synchronizes
    on the actual_total_chunks barrier. Quarantines persistent batch failures to DLQ.
    The final swarm worker to resolve the barrier executes Stage 5 (Atomic State Flip).
    """
    def __init__(
        self,
        worker_id: Optional[str] = None,
        cursor_manager: Optional[IngestionCursorManager] = None,
        queue_manager: Optional[IngestionQueueManager] = None,
        vector_store: Optional[NamespaceVectorStore] = None,
        embedder: Optional[Union[BaseEmbedder, EmbeddingEngine]] = None,
        batch_size: Optional[int] = None,
        lease_seconds: float = 60.0,
        index_summary_chunk: bool = False,
        max_batch_retries: Optional[int] = None
    ):
        self.worker_id = worker_id or f"embedder_{socket.gethostname()}_{os.getpid()}_{uuid.uuid4().hex[:4]}"
        self.cursor_mgr = cursor_manager or IngestionCursorManager()
        self.queue_mgr = queue_manager or IngestionQueueManager()
        self.vector_store = vector_store or NamespaceVectorStore()
        self.embedder = embedder or get_embedder()
        self.batch_size = batch_size if batch_size is not None else app_config.embedder.batch_size
        self.lease_seconds = lease_seconds
        self.index_summary_chunk = index_summary_chunk
        self.max_batch_retries = max_batch_retries if max_batch_retries is not None else app_config.embedder.max_batch_retries
        self._batch_fail_counts: Dict[str, int] = {}

        # Register worker in cluster registry
        self.queue_mgr.register_worker(self.worker_id)
        self._is_running = True

    def heartbeat(self, current_job_id: Optional[str] = None):
        if current_job_id:
            self.queue_mgr.heartbeat_job(current_job_id, self.worker_id, extend_seconds=self.lease_seconds)
        else:
            self.queue_mgr.worker_heartbeat(self.worker_id)

    def _validate_batch_embeddings(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: Any,
        expected_dim: int,
        check_norm: bool = True
    ) -> None:
        """
        Post-completion 5-point validation gate:
        1. Cardinality check: len(embeddings) == len(chunks)
        2. Dimensionality check: len(vec) == expected_dim
        3. Finiteness check: no NaN, Inf, or -Inf values
        4. Non-zero energy check: norm > 1e-6
        5. Unit normalization check: abs(norm - 1.0) < 1e-2 (if check_norm is True)
        """
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embedding validation failed [Cardinality]: received {len(embeddings)} embeddings "
                f"for {len(chunks)} chunks."
            )

        for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            chunk_id = chunk.get("chunk_id", f"idx_{i}")

            if len(vec) != expected_dim:
                raise ValueError(
                    f"Embedding validation failed [Dimension] for chunk {chunk_id}: "
                    f"expected dimension {expected_dim}, got {len(vec)}."
                )

            norm_sq = 0.0
            for val in vec:
                if not math.isfinite(val) or math.isnan(val):
                    raise ValueError(
                        f"Embedding validation failed [Non-Finite] for chunk {chunk_id}: "
                        f"detected NaN or infinite component."
                    )
                norm_sq += val * val

            norm = math.sqrt(norm_sq)

            if norm < 1e-6:
                raise ValueError(
                    f"Embedding validation failed [Zero-Vector] for chunk {chunk_id}: "
                    f"vector norm is zero or near-zero ({norm:.8f})."
                )

            if check_norm:
                if abs(norm - 1.0) > 1e-2:
                    raise ValueError(
                        f"Embedding validation failed [Normalization] for chunk {chunk_id}: "
                        f"vector norm is {norm:.6f} (expected ~1.0)."
                    )

    def _resolve_barrier(
        self,
        batch: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        barrier_res: Dict[str, Any],
        is_degraded: bool = False
    ) -> Dict[str, Any]:
        job_id = batch["job_id"]
        stage = "DATABASE"
        self.queue_mgr.update_stage(job_id, "DATABASE")

        # Flip previous versions of this doc_family
        self.vector_store.flip_is_latest(
            doc_family=batch["doc_family"],
            new_active_job_id=job_id,
            new_document_id=batch["document_id"]
        )

        # Optional summary chunk
        first_meta = chunks[0]["metadata"] if chunks else {}
        doc_summary = first_meta.get("doc_summary", "")
        if self.index_summary_chunk and doc_summary:
            summary_emb = self.embedder.embed_text(doc_summary)
            self.vector_store.upsert_chunks([
                VectorChunk(
                    chunk_id=f"summary_{batch['document_id']}",
                    document_id=batch["document_id"],
                    namespace=first_meta.get("target_namespace", "general"),
                    text=doc_summary,
                    metadata={
                        "is_summary": True,
                        "chunk_type": "document_summary",
                        "document_title": first_meta.get("document_title", ""),
                        "doc_family": batch["doc_family"],
                        "job_id": job_id,
                        "version": batch["version"],
                        "is_latest": True,
                        "access_tier": first_meta.get("access_tier", "free"),
                        "tenant_id": first_meta.get("tenant_id", "global"),
                        "clearance_level": first_meta.get("clearance_level", 1),
                        "doc_summary": doc_summary
                    },
                    embedding=summary_emb
                )
            ])

        # Advance Cursor
        cursor_status = "ACTIVE_DEGRADED" if is_degraded else "ACTIVE"
        cursor_record = IngestionCursorRecord(
            document_id=batch["document_id"],
            file_path=first_meta.get("file_path", ""),
            content_hash=first_meta.get("content_hash", ""),
            doc_family=batch["doc_family"],
            version=batch["version"],
            is_latest=True,
            status=cursor_status,
            declared_domain=batch.get("declared_domain"),
            resolved_namespace=first_meta.get("target_namespace", "general"),
            access_tier=first_meta.get("access_tier", "free"),
            tenant_id=first_meta.get("tenant_id", "global"),
            clearance_level=first_meta.get("clearance_level", 1),
            chunks_count=barrier_res.get("actual_total_chunks") or len(chunks)
        )
        self.cursor_mgr.commit_version(cursor_record)

        # Complete Queue Job & purge ephemeral staging table
        queue_status = "COMPLETED_DEGRADED" if is_degraded else "COMPLETED"
        self.queue_mgr.complete_job(job_id, self.worker_id, status=queue_status)

        return {
            "status": queue_status,
            "barrier_resolved": True,
            "is_degraded": is_degraded,
            "job_id": job_id,
            "document_id": batch["document_id"],
            "version": batch["version"],
            "chunks_in_batch": len(chunks),
            "actual_total_chunks": barrier_res.get("actual_total_chunks")
        }

    def process_next_batch(self) -> Optional[Dict[str, Any]]:
        """
        Claims and executes a single mini-batch from the swarm staging ledger.
        Returns batch execution summary or None if no pending batches.
        """
        batch = self.queue_mgr.claim_next_chunk_batch(
            worker_id=self.worker_id,
            batch_size=self.batch_size,
            lease_seconds=self.lease_seconds
        )
        if not batch or not batch.get("chunks"):
            return None
        return self.execute_batch(batch)

    def execute_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        job_id = batch["job_id"]
        stage = "EMBEDDING"
        chunks = batch["chunks"]
        chunk_ids = [c["chunk_id"] for c in chunks]
        start_offset = batch.get("start_offset", 0)
        end_offset = batch.get("end_offset", 0)
        batch_key = f"{job_id}:{start_offset}"

        try:
            self.heartbeat(job_id)
            texts = [c["text"] for c in chunks]

            # 1. Compute Embeddings
            embeddings = self.embedder.embed_batch(texts)

            # 2. 5-point Post-Completion Validation Gate
            check_norm = getattr(self.embedder, "normalize", getattr(app_config.embedder, "normalize", True))
            self._validate_batch_embeddings(
                chunks=chunks,
                embeddings=embeddings,
                expected_dim=self.embedder.dimension,
                check_norm=check_norm
            )

            # 3. Stage clean vector chunks (Model 1: embedding NOT NULL, is_latest=False)
            vector_chunks = []
            for c, emb in zip(chunks, embeddings):
                meta = c["metadata"].copy()
                meta["doc_family"] = batch["doc_family"]
                meta["job_id"] = job_id
                meta["version"] = batch["version"]
                meta["is_latest"] = False  # Staged until Stage 5 barrier resolves
                meta["is_summary"] = False

                vector_chunks.append(VectorChunk(
                    chunk_id=c["chunk_id"],
                    document_id=batch["document_id"],
                    namespace=meta.get("target_namespace", "general"),
                    text=c["text"],
                    metadata=meta,
                    embedding=emb
                ))

            self.vector_store.upsert_chunks(vector_chunks)

            # 4. Complete chunk batch & check barrier
            barrier_res = self.queue_mgr.complete_chunk_batch(
                job_id=job_id,
                worker_id=self.worker_id,
                chunk_ids=chunk_ids
            )
            self._batch_fail_counts.pop(batch_key, None)

            # 5. Check if this worker resolves the barrier
            if barrier_res["is_barrier_complete"]:
                is_degraded = barrier_res.get("is_degraded", False)
                return self._resolve_barrier(batch, chunks, barrier_res, is_degraded=is_degraded)
            else:
                return {
                    "status": "BATCH_COMPLETED",
                    "barrier_resolved": False,
                    "job_id": job_id,
                    "document_id": batch["document_id"],
                    "chunks_in_batch": len(chunks),
                    "start_offset": start_offset,
                    "end_offset": end_offset,
                    "embedded_count": barrier_res["embedded_count"],
                    "actual_total_chunks": barrier_res["actual_total_chunks"]
                }

        except Exception as e:
            current_fails = self._batch_fail_counts.get(batch_key, 0) + 1
            self._batch_fail_counts[batch_key] = current_fails
            trace_str = traceback.format_exc()
            logger.error(
                f"EmbedderWorker {self.worker_id} error processing batch [{start_offset}..{end_offset}] "
                f"for job {job_id} (attempt {current_fails}/{self.max_batch_retries}): {e}",
                exc_info=True
            )

            if current_fails >= self.max_batch_retries:
                logger.warning(
                    f"EmbedderWorker {self.worker_id} quarantining batch [{start_offset}..{end_offset}] "
                    f"to DLQ for job {job_id} after {current_fails} failures: {e}"
                )
                dlq_res = self.queue_mgr.quarantine_chunk_batch(
                    job_id=job_id,
                    document_id=batch["document_id"],
                    start_offset=start_offset,
                    end_offset=end_offset,
                    chunk_ids=chunk_ids,
                    payload=batch,
                    error_reason=str(e),
                    exception_trace=trace_str,
                    retry_count=current_fails,
                    worker_id=self.worker_id
                )
                self._batch_fail_counts.pop(batch_key, None)

                if dlq_res.get("is_barrier_complete"):
                    return self._resolve_barrier(batch, chunks, dlq_res, is_degraded=True)

                return {
                    "status": "QUARANTINED",
                    "dlq_id": dlq_res.get("dlq_id"),
                    "job_id": job_id,
                    "document_id": batch["document_id"],
                    "quarantined_count": len(chunk_ids),
                    "error": str(e)
                }
            else:
                self.queue_mgr.release_chunk_batch(chunk_ids)
                return {
                    "status": "RETRY",
                    "job_id": job_id,
                    "attempt": current_fails,
                    "max_retries": self.max_batch_retries,
                    "error": str(e)
                }

    def run_forever(
        self,
        poll_interval: float = 1.0,
        max_iterations: Optional[int] = None,
        stop_event: Optional[Any] = None
    ):
        """Continuously polls for available chunk batches to embed until stopped."""
        logger.info(f"EmbedderWorker {self.worker_id} started daemon loop.")
        self._is_running = True
        iterations = 0
        try:
            while self._is_running and not (stop_event and stop_event.is_set()):
                if max_iterations is not None and iterations >= max_iterations:
                    break
                self.heartbeat()
                res = self.process_next_batch()
                iterations += 1
                if not res:
                    time.sleep(poll_interval)
        except KeyboardInterrupt:
            logger.info(f"Shutdown requested for EmbedderWorker {self.worker_id}")
        finally:
            self.stop()

    def stop(self):
        self._is_running = False
        self.queue_mgr.unregister_worker(self.worker_id)
        if hasattr(self.queue_mgr, "close"):
            self.queue_mgr.close()
        if hasattr(self.cursor_mgr, "close"):
            self.cursor_mgr.close()


class IngestionWorker:
    """
    Unified Ingestion Worker.
    Orchestrates the complete 5-stage ingestion pipeline by delegating to
    ChunkerWorker (Pool 1) and EmbedderWorker (Pool 2 Swarm).
    Provides seamless single-process operation while leveraging Model 1 staging.
    """
    def __init__(
        self,
        worker_id: Optional[str] = None,
        cursor_manager: Optional[IngestionCursorManager] = None,
        queue_manager: Optional[IngestionQueueManager] = None,
        vector_store: Optional[NamespaceVectorStore] = None,
        router: Optional[SkillsRouter] = None,
        chunker: Optional[TextChunker] = None,
        summarizer: Optional[DocumentSummarizer] = None,
        embedder: Optional[Union[BaseEmbedder, EmbeddingEngine]] = None,
        lease_duration_seconds: float = 120.0,
        index_summary_chunk: bool = False,
        parser: Optional[LocalDocumentParser] = None
    ):
        self.worker_id = worker_id or f"worker_{socket.gethostname()}_{os.getpid()}_{uuid.uuid4().hex[:4]}"
        self.cursor_mgr = cursor_manager or IngestionCursorManager()
        self.queue_mgr = queue_manager or IngestionQueueManager()
        self.vector_store = vector_store or NamespaceVectorStore()
        self.router = router or SkillsRouter()
        self.chunker = chunker or TextChunker()
        self.summarizer = summarizer or DocumentSummarizer()
        self.embedder = embedder or get_embedder()
        self.parser = parser or LocalDocumentParser()
        self.lease_duration_seconds = lease_duration_seconds
        self.index_summary_chunk = index_summary_chunk

        # Underlying specialized workers sharing the worker_id for single-node deployment
        self.chunker_worker = ChunkerWorker(
            worker_id=self.worker_id,
            queue_manager=self.queue_mgr,
            cursor_manager=self.cursor_mgr,
            router=self.router,
            chunker=self.chunker,
            summarizer=self.summarizer,
            parser=self.parser,
            lease_duration_seconds=self.lease_duration_seconds
        )
        self.embedder_worker = EmbedderWorker(
            worker_id=self.worker_id,
            cursor_manager=self.cursor_mgr,
            queue_manager=self.queue_mgr,
            vector_store=self.vector_store,
            embedder=self.embedder,
            batch_size=32,
            lease_seconds=self.lease_duration_seconds,
            index_summary_chunk=self.index_summary_chunk
        )

        # Register worker in registry
        self.queue_mgr.register_worker(self.worker_id)
        self._is_running = True

    def heartbeat(self, current_job_id: Optional[str] = None):
        if current_job_id:
            self.queue_mgr.heartbeat_job(current_job_id, self.worker_id, extend_seconds=self.lease_duration_seconds)
        else:
            self.queue_mgr.worker_heartbeat(self.worker_id)

    def run_maintenance(self) -> List[str]:
        """Runs sweeper to recover any crashed jobs across the cluster."""
        return self.queue_mgr.recover_crashed_jobs()

    def process_next_job(self) -> Optional[Dict[str, Any]]:
        """
        Executes a job to completion in a single-daemon setting:
        1. Claims and executes chunking (Stages 1-3).
        2. Loops through and embeds all batches until the barrier completes (Stages 4-5).
        If no pending chunking job exists, executes any available swarm embedding batch.
        """
        chunk_res = self.chunker_worker.process_next_job()
        if chunk_res:
            if chunk_res.get("status") == "FAILED":
                return chunk_res

            # Process all embedding batches for this document until barrier resolves
            last_res = None
            while True:
                batch_res = self.embedder_worker.process_next_batch()
                if not batch_res:
                    break
                last_res = batch_res
                if batch_res.get("barrier_resolved"):
                    return batch_res
            return last_res or chunk_res

        # If no chunking job claimed, process any available embedding batch
        batch_res = self.embedder_worker.process_next_batch()
        if batch_res:
            return batch_res

        return None

    def execute_job(self, job: IngestionQueueJob) -> Dict[str, Any]:
        """Executes chunking and embedding to completion for a single specific job."""
        chunk_res = self.chunker_worker.execute_job(job)
        if chunk_res.get("status") == "FAILED":
            return chunk_res

        last_res = None
        while True:
            batch_res = self.embedder_worker.process_next_batch()
            if not batch_res:
                break
            last_res = batch_res
            if batch_res.get("barrier_resolved"):
                return batch_res
        return last_res or chunk_res

    def run_forever(
        self,
        poll_interval: float = 1.0,
        max_iterations: Optional[int] = None,
        stop_event: Optional[Any] = None
    ):
        """Continuously runs the unified pipeline (chunking + embedding) until stopped."""
        logger.info(f"IngestionWorker {self.worker_id} started daemon loop.")
        self._is_running = True
        iterations = 0
        try:
            while self._is_running and not (stop_event and stop_event.is_set()):
                if max_iterations is not None and iterations >= max_iterations:
                    break
                self.heartbeat()
                res = self.process_next_job()
                iterations += 1
                if not res:
                    time.sleep(poll_interval)
        except KeyboardInterrupt:
            logger.info(f"Shutdown requested for IngestionWorker {self.worker_id}")
        finally:
            self.stop()

    def stop(self):
        self._is_running = False
        self.chunker_worker.stop()
        self.embedder_worker.stop()
        self.queue_mgr.unregister_worker(self.worker_id)
        if hasattr(self.queue_mgr, "close"):
            self.queue_mgr.close()
        if hasattr(self.cursor_mgr, "close"):
            self.cursor_mgr.close()
