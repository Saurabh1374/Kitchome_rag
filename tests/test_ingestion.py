import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.skills_ms.registry import SkillNamespaceRegistry
from src.skills_ms.router import SkillsRouter
from src.ingestion.loader import RawDocument
from src.ingestion.pipeline import IngestionPipeline
from src.vector_store.base import NamespaceVectorStore

def test_skills_ms_router_namespace_resolution():
    router = SkillsRouter()
    
    # Query resolution tests
    assert router.resolve_namespaces_for_query("best crispy chicken recipe")[0] == "recipes_culinary"
    assert router.resolve_namespaces_for_query("air fryer error code E3 manual")[0] == "appliances_troubleshooting"
    assert router.resolve_namespaces_for_query("modern cabinet layout decor")[0] == "home_decor_design"
    assert router.resolve_namespaces_for_query("remove granite countertop stain")[0] == "cleaning_maintenance"

def test_ingestion_pipeline_namespace_isolation():
    router = SkillsRouter()
    vector_store = NamespaceVectorStore()
    pipeline = IngestionPipeline(router=router, vector_store=vector_store)

    doc_recipe = RawDocument(
        document_id="doc_1",
        title="Spicy Pasta Recipe",
        source_path="/data/recipes/pasta_recipe.md",
        content="Boil water and add olive oil, pasta, and parmesan cheese for 10 minutes."
    )

    doc_manual = RawDocument(
        document_id="doc_2",
        title="Blender User Manual",
        source_path="/data/appliances/blender_manual.md",
        content="Voltage 120V. If motor overheats with Error E1, reset thermal fuse."
    )

    res_recipe = pipeline.ingest_document(doc_recipe)
    res_manual = pipeline.ingest_document(doc_manual)

    assert res_recipe["namespace"] == "recipes_culinary"
    assert res_manual["namespace"] == "appliances_troubleshooting"

    # Verify namespace isolation in vector store
    recipe_chunks = vector_store.get_all_chunks_in_namespace("recipes_culinary")
    manual_chunks = vector_store.get_all_chunks_in_namespace("appliances_troubleshooting")

    assert len(recipe_chunks) == 1
    assert len(manual_chunks) == 1
    assert recipe_chunks[0].document_id == "doc_1"
    assert manual_chunks[0].document_id == "doc_2"

def test_vector_search_scoped_by_namespace():
    router = SkillsRouter()
    vector_store = NamespaceVectorStore()
    pipeline = IngestionPipeline(router=router, vector_store=vector_store)

    doc_recipe = RawDocument(
        document_id="doc_1",
        title="Chicken Air Fryer Recipe",
        source_path="/recipes/chicken.md",
        content="Air fryer chicken wings at 400 degrees with salt and paprika."
    )
    doc_manual = RawDocument(
        document_id="doc_2",
        title="Air Fryer Error Manual",
        source_path="/appliances/manual.md",
        content="Air fryer error E3 indicates heating element overheating."
    )

    pipeline.ingest_document(doc_recipe)
    pipeline.ingest_document(doc_manual)

    # Search for recipe query -> skills.ms routes to recipes_culinary
    recipe_query = "how to make air fryer chicken wings"
    target_ns = router.resolve_namespaces_for_query(recipe_query)[0]
    assert target_ns == "recipes_culinary"

    emb = pipeline.embedder.embed_text(recipe_query)
    results = vector_store.search(query_vector=emb, namespace=target_ns, top_k=2)

    assert len(results) == 1
    assert results[0]["metadata"]["document_title"] == "Chicken Air Fryer Recipe"

def test_pgvector_store_initialization_and_fallback():
    from src.vector_store.pgvector_store import PGVectorStore
    pg_store = PGVectorStore(host="localhost", port=5432, db_name="test_db")
    # Store should initialize smoothly either in PG connected or local fallback mode
    assert hasattr(pg_store, "is_connected")
    assert hasattr(pg_store, "upsert_chunks")
    assert hasattr(pg_store, "search")

def test_progressive_summarizer_purging_and_final_summary():
    from src.ingestion.summarizer import DocumentSummarizer
    summarizer = DocumentSummarizer(summary_threshold_tokens=50, buffer_allowance_tokens=20, max_final_summary_words=100)
    
    # Generate long synthetic document text
    paragraphs = [f"Section {i}: Important details about feature {i} and troubleshooting steps for item {i}." for i in range(25)]
    text = "\n\n".join(paragraphs)
    
    doc_summary, keywords = summarizer.process_and_summarize_document("Long Manual", text)
    
    assert "Overall Summary of Long Manual:" in doc_summary
    assert summarizer.count_words(doc_summary) <= 100
    assert len(keywords) > 0


def test_dynamic_unregistered_namespace_auto_creation(tmp_path):
    # Use temporary registry file to avoid polluting workspace JSON
    reg_file = str(tmp_path / "test_skills_registry.json")
    registry = SkillNamespaceRegistry(properties_file=reg_file)
    router = SkillsRouter(registry=registry)
    vector_store = NamespaceVectorStore()
    pipeline = IngestionPipeline(router=router, vector_store=vector_store)

    # Ingest document with brand-new unregistered declared_domain
    doc_smart = RawDocument(
        document_id="doc_smart_1",
        title="Smart Home Hub Setup",
        source_path="/data/smart_home/hub_guide.md",
        content="Configure Zigbee and Matter protocols for your smart switch and hub system.",
        declared_domain="smart_home_automation"
    )

    res = pipeline.ingest_document(doc_smart)

    # 1. Verify namespace returned is smart_home_automation
    assert res["namespace"] == "smart_home_automation"

    # 2. Verify new skill was dynamically registered in registry
    all_namespaces = registry.get_all_namespaces()
    assert "smart_home_automation" in all_namespaces

    # 3. Verify vector store indexed under the new namespace
    smart_chunks = vector_store.get_all_chunks_in_namespace("smart_home_automation")
    assert len(smart_chunks) == 1
    assert smart_chunks[0].document_id == "doc_smart_1"

    # 4. Verify router routes future queries to the new dynamic namespace using extracted keywords
    query_ns = router.resolve_namespaces_for_query("how to setup smart switch with zigbee")[0]
    assert query_ns == "smart_home_automation"



