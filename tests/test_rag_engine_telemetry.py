import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.skills_ms.router import SkillsRouter
from src.vector_store.base import NamespaceVectorStore
from src.ingestion.loader import RawDocument
from src.ingestion.pipeline import IngestionPipeline
from src.auth.context import UserContext, UserTier
from src.rag.engine import RAGQueryEngine
from src.rag.telemetry import TelemetryTracker

def test_rag_query_engine_rbac_denial():
    router = SkillsRouter()
    vector_store = NamespaceVectorStore()
    engine = RAGQueryEngine(router=router, vector_store=vector_store)

    # Free user queries an appliance error code domain
    user_free = UserContext(user_id="u_free", tier=UserTier.FREE)
    res = engine.query("how to fix microwave error code E3 voltage", user_context=user_free)

    assert res["status"] == "ACCESS_DENIED"
    assert "appliances_troubleshooting" in res["predicted_namespaces"]
    assert len(res["authorized_namespaces"]) == 0
    assert "Access Denied" in res["answer"]

def test_rag_query_engine_dual_resolution_and_parent_context():
    router = SkillsRouter()
    vector_store = NamespaceVectorStore()
    tracker = TelemetryTracker()
    
    # Ingest document with dual-resolution summary chunk enabled
    pipeline = IngestionPipeline(
        router=router, 
        vector_store=vector_store,
        index_summary_chunk=True
    )

    doc_airfryer = RawDocument(
        document_id="doc_af_101",
        title="Instant Vortex Air Fryer Manual",
        source_path="/appliances/vortex_manual.md",
        content="The Instant Vortex Air Fryer Pro uses 1500W power. To clear error E3, turn off unit and let heating element cool for 20 minutes.",
        declared_domain="appliances_troubleshooting",
        access_tier="premium"
    )

    ingest_res = pipeline.ingest_document(doc_airfryer)
    assert ingest_res["chunks_ingested"] >= 2  # 1 body chunk + 1 summary chunk

    engine = RAGQueryEngine(
        router=router, 
        vector_store=vector_store, 
        telemetry=tracker,
        relevance_threshold=0.30
    )

    user_premium = UserContext(user_id="u_prem", tier=UserTier.PREMIUM)
    
    # Query appliance troubleshooting
    query = "Instant Vortex Air Fryer error code E3"
    result = engine.query(query, user_context=user_premium)

    assert result["status"] == "SUCCESS"
    assert "appliances_troubleshooting" in result["authorized_namespaces"]
    assert result["chunks_retrieved"] > 0
    assert len(result["citations"]) > 0
    assert "Instant Vortex Air Fryer Manual" in result["formatted_prompt"]
    assert "[Document Overview]:" in result["formatted_prompt"]

    # Telemetry verification
    events = tracker.get_events()
    assert len(events) == 1
    assert events[0].user_tier == "premium"
    assert tracker.get_routing_precision() >= 0.0

def test_rag_strategy_e_fallback_trigger_on_out_of_domain():
    router = SkillsRouter()
    vector_store = NamespaceVectorStore()
    tracker = TelemetryTracker()
    engine = RAGQueryEngine(
        router=router, 
        vector_store=vector_store, 
        telemetry=tracker,
        relevance_threshold=0.85 # High threshold to guarantee fallback trigger
    )

    user_premium = UserContext(user_id="u_prem", tier=UserTier.PREMIUM)
    res = engine.query("xyz obscure query with low similarity", user_context=user_premium)

    assert res["status"] == "SUCCESS"
    assert res["fallback_triggered"] is True
    
    events = tracker.get_events()
    assert len(events) == 1
    assert events[0].fallback_triggered is True
    assert tracker.get_routing_precision() == 0.0
