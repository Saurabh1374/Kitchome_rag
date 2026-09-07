import os
import math
import json
import time
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO

from config import EmbedderConfig
from src.ingestion.embedder import (
    BaseEmbedder,
    HashEmbedder,
    HuggingFaceEmbedder,
    OllamaEmbedder,
    OpenAIEmbedder,
    FastEmbedder,
    get_embedder,
    EmbeddingEngine,
    BatchEmbeddingResult
)
from src.ingestion.cursor import IngestionCursorManager, IngestionCursorRecord
from src.ingestion.queue import IngestionQueueManager
from src.ingestion.controller import IngestionController
from src.ingestion.worker import IngestionWorker, EmbedderWorker
from src.ingestion.chunker import TextChunk
from src.ingestion.run_workers import WorkerSupervisor
from src.vector_store.base import NamespaceVectorStore
from src.skills_ms.router import SkillsRouter


@pytest.fixture
def embed_env():
    with tempfile.TemporaryDirectory() as tmpdir:
        cursor_db = os.path.join(tmpdir, "test_cursor.db")
        queue_db = os.path.join(tmpdir, "test_queue.db")
        data_root = os.path.join(tmpdir, "data")
        os.makedirs(data_root, exist_ok=True)

        cursor_mgr = IngestionCursorManager(db_path=cursor_db)
        queue_mgr = IngestionQueueManager(db_path=queue_db)
        vector_store = NamespaceVectorStore()
        router = SkillsRouter()

        yield {
            "tmpdir": tmpdir,
            "cursor_mgr": cursor_mgr,
            "queue_mgr": queue_mgr,
            "vector_store": vector_store,
            "router": router,
            "data_root": data_root
        }


# =====================================================================
# 1. Decoupled Compute Engine Tests
# =====================================================================

def test_hash_embedder_dimensions_and_normalization():
    """Verify HashEmbedder supports arbitrary dimensions, unit normalization, and telemetry."""
    for dim in (128, 384, 768):
        embedder = HashEmbedder(dimension=dim, normalize=True)
        assert embedder.dimension == dim
        assert embedder.provider_name == "hash"

        vec = embedder.embed_text("Microwave convection oven heating cycle")
        assert len(vec) == dim
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-4

        batch_res = embedder.embed_batch([
            "Induction cooktop burner coils",
            "Refrigerator dual compressor temperature sensor"
        ])
        assert isinstance(batch_res, BatchEmbeddingResult)
        assert len(batch_res) == 2
        assert batch_res.dimension == dim
        assert batch_res.provider == "hash"
        assert batch_res.latency_ms >= 0.0

        for b_vec in batch_res:
            assert len(b_vec) == dim
            b_norm = math.sqrt(sum(x * x for x in b_vec))
            assert abs(b_norm - 1.0) < 1e-4


def test_huggingface_embedder_lazy_import_and_api_mode():
    """Verify HuggingFaceEmbedder lazy imports locally and handles serverless API mode."""
    # Local mode without sentence_transformers raises helpful ImportError
    with patch.dict("sys.modules", {"sentence_transformers": None}):
        hf_local = HuggingFaceEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2", is_api=False)
        with pytest.raises(ImportError, match="sentence-transformers"):
            hf_local.embed_text("test input")

    # API mode with mocked HTTP response
    mock_embeddings = [[0.1, 0.2, 0.3, 0.4]]
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(mock_embeddings).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
        hf_api = HuggingFaceEmbedder(
            model_name="BAAI/bge-small-en-v1.5",
            api_key="hf_test_token_12345",
            dimension=4,
            is_api=True,
            normalize=True
        )
        assert hf_api.provider_name == "huggingface_api"
        batch_res = hf_api.embed_batch(["Preheat oven to 375F"])
        assert len(batch_res) == 1
        assert len(batch_res[0]) == 4
        # Verify normalization was applied
        norm = math.sqrt(sum(x * x for x in batch_res[0]))
        assert abs(norm - 1.0) < 1e-4

        # Verify Authorization header was sent
        req_arg = mock_urlopen.call_args[0][0]
        assert req_arg.get_header("Authorization") == "Bearer hf_test_token_12345"


def test_get_embedder_factory_switching():
    """Verify factory instantiates correct providers with overrides."""
    e_hash = get_embedder(provider="hash", dimension=256)
    assert isinstance(e_hash, HashEmbedder)
    assert e_hash.dimension == 256

    e_ollama = get_embedder(provider="ollama", model_name="nomic-embed-text", base_url="http://localhost:11434")
    assert isinstance(e_ollama, OllamaEmbedder)
    assert e_ollama.model_name == "nomic-embed-text"

    e_openai = get_embedder(provider="openai", api_key="sk-test", dimension=1536)
    assert isinstance(e_openai, OpenAIEmbedder)
    assert e_openai.dimension == 1536

    # Unknown provider falls back gracefully to HashEmbedder
    e_fallback = get_embedder(provider="nonexistent_provider")
    assert isinstance(e_fallback, HashEmbedder)


# =====================================================================
# 2. Post-Completion 5-Point Validation Gate Tests
# =====================================================================

def test_post_completion_validation_gate(embed_env):
    """Verify all 5 checks in _validate_batch_embeddings."""
    worker = EmbedderWorker(
        worker_id="test_val_worker",
        cursor_manager=embed_env["cursor_mgr"],
        queue_manager=embed_env["queue_mgr"],
        vector_store=embed_env["vector_store"],
        embedder=HashEmbedder(dimension=4, normalize=True)
    )

    chunks = [
        {"chunk_id": "c1", "text": "Chunk one"},
        {"chunk_id": "c2", "text": "Chunk two"}
    ]

    # 1. Cardinality check failure
    with pytest.raises(ValueError, match=r"\[Cardinality\]"):
        worker._validate_batch_embeddings(chunks, [[0.5, 0.5, 0.5, 0.5]], expected_dim=4)

    # 2. Dimensionality check failure
    with pytest.raises(ValueError, match=r"\[Dimension\]"):
        worker._validate_batch_embeddings(
            chunks,
            [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],  # len 3 instead of 4
            expected_dim=4
        )

    # 3. Finiteness check failure (NaN and Inf)
    with pytest.raises(ValueError, match=r"\[Non-Finite\]"):
        worker._validate_batch_embeddings(
            chunks,
            [[0.5, 0.5, float("nan"), 0.5], [0.5, 0.5, 0.5, 0.5]],
            expected_dim=4
        )

    with pytest.raises(ValueError, match=r"\[Non-Finite\]"):
        worker._validate_batch_embeddings(
            chunks,
            [[0.5, 0.5, float("inf"), 0.5], [0.5, 0.5, 0.5, 0.5]],
            expected_dim=4
        )

    # 4. Zero-Vector energy check failure
    with pytest.raises(ValueError, match=r"\[Zero-Vector\]"):
        worker._validate_batch_embeddings(
            chunks,
            [[0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]],
            expected_dim=4
        )

    # 5. Unit Normalization check failure
    with pytest.raises(ValueError, match=r"\[Normalization\]"):
        worker._validate_batch_embeddings(
            chunks,
            [[5.0, 5.0, 5.0, 5.0], [5.0, 5.0, 5.0, 5.0]],
            expected_dim=4,
            check_norm=True
        )

    # Success: properly normalized 4D unit vectors
    unit_val = 0.5  # sqrt(4 * 0.25) = 1.0
    worker._validate_batch_embeddings(
        chunks,
        [[unit_val, unit_val, unit_val, unit_val], [unit_val, unit_val, unit_val, unit_val]],
        expected_dim=4,
        check_norm=True
    )


# =====================================================================
# 3. Offset Tracking & Multi-Worker Swarming Tests
# =====================================================================

def test_multi_worker_contiguous_offset_windows(embed_env):
    """
    Verify two concurrent workers claiming from the same job_id receive
    strictly contiguous, disjoint offset windows [0..31] and [32..63].
    """
    queue_mgr: IngestionQueueManager = embed_env["queue_mgr"]

    # Enqueue a job and stage 64 chunks
    job = queue_mgr.enqueue_job(
        file_path="manuals/dishwasher_sh60.pdf",
        doc_family="bosch_sh60",
        document_id="doc_sh60_v1",
        version=1,
        declared_domain="cooking"
    )
    # Transition to READY_FOR_EMBED
    queue_mgr.claim_next_chunking_job("chunker_1")
    queue_mgr.update_stage(job.job_id, "CHUNKING")

    staged_chunks = [
        TextChunk(
            chunk_id=f"sh60_c_{i:03d}",
            document_id=job.document_id,
            namespace="cooking",
            text=f"Dishwasher cycle step {i} instructions.",
            chunk_index=i,
            total_chunks=64,
            metadata={"step": i, "target_namespace": "cooking"}
        )
        for i in range(64)
    ]
    queue_mgr.stage_chunks(job.job_id, staged_chunks)
    queue_mgr.lock_chunking_complete(
        chunk_job_id=job.chunk_job_id,
        job_id=job.job_id,
        actual_total_chunks=64,
        chunk_ids=[c.chunk_id for c in staged_chunks],
        worker_id="chunker_1"
    )

    # Worker 1 claims batch of 32
    batch_1 = queue_mgr.claim_next_chunk_batch(worker_id="embedder_worker_1", batch_size=32)
    assert batch_1 is not None
    assert batch_1["start_offset"] == 0
    assert batch_1["end_offset"] == 31
    assert len(batch_1["chunks"]) == 32
    assert batch_1["chunks"][0]["chunk_index"] == 0
    assert batch_1["chunks"][-1]["chunk_index"] == 31

    # Worker 2 claims batch of 32
    batch_2 = queue_mgr.claim_next_chunk_batch(worker_id="embedder_worker_2", batch_size=32)
    assert batch_2 is not None
    assert batch_2["start_offset"] == 32
    assert batch_2["end_offset"] == 63
    assert len(batch_2["chunks"]) == 32
    assert batch_2["chunks"][0]["chunk_index"] == 32
    assert batch_2["chunks"][-1]["chunk_index"] == 63

    # Disjoint offset assertion
    b1_ids = {c["chunk_id"] for c in batch_1["chunks"]}
    b2_ids = {c["chunk_id"] for c in batch_2["chunks"]}
    assert b1_ids.isdisjoint(b2_ids)

    # Both batches are executed
    emb_worker_1 = EmbedderWorker(
        worker_id="embedder_worker_1",
        cursor_manager=embed_env["cursor_mgr"],
        queue_manager=queue_mgr,
        vector_store=embed_env["vector_store"]
    )
    res_1 = emb_worker_1.execute_batch(batch_1)
    assert res_1["status"] == "BATCH_COMPLETED"
    assert res_1["barrier_resolved"] is False

    emb_worker_2 = EmbedderWorker(
        worker_id="embedder_worker_2",
        cursor_manager=embed_env["cursor_mgr"],
        queue_manager=queue_mgr,
        vector_store=embed_env["vector_store"]
    )
    res_2 = emb_worker_2.execute_batch(batch_2)
    assert res_2["status"] == "COMPLETED"
    assert res_2["barrier_resolved"] is True
    assert res_2["is_degraded"] is False


# =====================================================================
# 4. DLQ Quarantine, Self-Healing, and Replay Tests
# =====================================================================

class FlakyEmbedder(BaseEmbedder):
    """Mock embedder that fails on specific chunks to test retry and quarantine."""
    def __init__(self, dimension: int = 384):
        self._dim = dimension
        self._base = HashEmbedder(dimension=dimension, normalize=True)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def provider_name(self) -> str:
        return "flaky"

    def embed_text(self, text: str):
        return self._base.embed_text(text)

    def embed_batch(self, texts):
        res = self._base.embed_batch(texts)
        # If poison pill text is present, return zero vectors or NaNs
        embs = []
        for t, v in zip(texts, res.embeddings):
            if "POISON_PILL" in t:
                embs.append([float("nan")] * self._dim)
            else:
                embs.append(v)
        return BatchEmbeddingResult(embeddings=embs, provider="flaky", model="flaky-1", dimension=self._dim)


def test_poison_batch_quarantine_to_dlq_and_replay(embed_env):
    """
    Verify that an unrecoverable poison batch is retried up to max_batch_retries,
    quarantined into ingestion_dlq, the document finishes as ACTIVE_DEGRADED,
    and replay_dlq_batch re-enqueues it cleanly.
    """
    queue_mgr: IngestionQueueManager = embed_env["queue_mgr"]
    cursor_mgr: IngestionCursorManager = embed_env["cursor_mgr"]
    vector_store: NamespaceVectorStore = embed_env["vector_store"]

    job = queue_mgr.enqueue_job(
        file_path="manuals/refrigerator_rf99.pdf",
        doc_family="samsung_rf99",
        document_id="doc_rf99_v1",
        version=1,
        declared_domain="appliances"
    )
    queue_mgr.claim_next_chunking_job("chunker_1")
    queue_mgr.update_stage(job.job_id, "CHUNKING")

    # 10 chunks total:
    # batch 1 (indices 0..4): valid
    # batch 2 (indices 5..9): contains POISON_PILL
    staged_chunks = [
        TextChunk(
            chunk_id=f"rf99_c_{i:02d}",
            document_id=job.document_id,
            namespace="appliances",
            text=f"Refrigerator step {i} - POISON_PILL text" if i >= 5 else f"Refrigerator step {i} normal text",
            chunk_index=i,
            total_chunks=10,
            metadata={"step": i, "target_namespace": "appliances"}
        )
        for i in range(10)
    ]
    queue_mgr.stage_chunks(job.job_id, staged_chunks)
    queue_mgr.lock_chunking_complete(
        chunk_job_id=job.chunk_job_id,
        job_id=job.job_id,
        actual_total_chunks=10,
        chunk_ids=[c.chunk_id for c in staged_chunks],
        worker_id="chunker_1"
    )

    worker = EmbedderWorker(
        worker_id="dlq_test_worker",
        cursor_manager=cursor_mgr,
        queue_manager=queue_mgr,
        vector_store=vector_store,
        embedder=FlakyEmbedder(dimension=64),
        batch_size=5,
        max_batch_retries=2
    )

    # 1. Process batch 1: clean success
    res_b1 = worker.process_next_batch()
    assert res_b1["status"] == "BATCH_COMPLETED"
    assert res_b1["start_offset"] == 0
    assert res_b1["end_offset"] == 4

    # 2. Process batch 2: attempt 1 -> RETRY
    res_b2_att1 = worker.process_next_batch()
    assert res_b2_att1["status"] == "RETRY"
    assert res_b2_att1["attempt"] == 1

    # 3. Process batch 2: attempt 2 (max_batch_retries reached) -> QUARANTINED & COMPLETED_DEGRADED
    res_b2_att2 = worker.process_next_batch()
    assert res_b2_att2["status"] == "COMPLETED_DEGRADED"
    assert res_b2_att2["barrier_resolved"] is True
    assert res_b2_att2["is_degraded"] is True

    # 4. Verify DLQ record
    dlq_records = queue_mgr.list_dlq_records(job_id=job.job_id)
    assert len(dlq_records) == 1
    dlq = dlq_records[0]
    assert dlq["status"] == "QUARANTINED"
    assert dlq["start_offset"] == 5
    assert dlq["end_offset"] == 9
    assert len(dlq["chunk_ids"]) == 5
    assert "Non-Finite" in dlq["error_reason"]

    # 5. Verify Cursor Record status is ACTIVE_DEGRADED
    active_cursor = cursor_mgr.get_active_by_family("samsung_rf99")
    assert active_cursor is not None
    assert active_cursor.status == "ACTIVE_DEGRADED"
    assert active_cursor.document_id == "doc_rf99_v1"

    # 6. Verify replay_dlq_batch re-enqueues the quarantined chunks
    replayed = queue_mgr.replay_dlq_batch(dlq["dlq_id"])
    assert replayed is True

    # Check DLQ status updated
    updated_dlq = queue_mgr.list_dlq_records(job_id=job.job_id, status="REPLAYED")
    assert len(updated_dlq) == 1
    assert updated_dlq[0]["dlq_id"] == dlq["dlq_id"]


# =====================================================================
# 5. CLI & Supervisor Integration Tests
# =====================================================================

def test_cli_supervisor_embedder_overrides():
    """Verify WorkerSupervisor instantiates embedder with overrides in thread mode."""
    supervisor = WorkerSupervisor(
        role="embedder",
        mode="thread",
        embedder_concurrency=1,
        batch_size=16,
        poll_interval=0.1,
        max_iterations=1,
        embedder_overrides={"provider": "hash", "dimension": 128}
    )
    supervisor.start()
    supervisor.stop(timeout=2.0)
    assert len(supervisor.workers) == 1
