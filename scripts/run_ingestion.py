import os
import sys

# Ensure src module is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import config
from src.skills_ms.router import SkillsRouter
from src.vector_store.pgvector_store import PGVectorStore
from src.ingestion.pipeline import IngestionPipeline

def main():
    print("=" * 70)
    print("🚀 Starting Kitchome RAG Ingestion Pipeline with skills.ms Routing & PGVector")
    print("=" * 70)

    # Initialize components with PGVector store
    router = SkillsRouter(default_namespace=config.skills_ms.default_namespace)
    vector_store = PGVectorStore(
        host=config.pgvector.host,
        port=config.pgvector.port,
        db_name=config.pgvector.db_name,
        user=config.pgvector.user,
        password=config.pgvector.password,
        table_name=config.pgvector.vector_table,
        embedding_dimension=config.ingestion.embedding_dimension,
        storage_path=config.vector_db_path
    )
    pipeline = IngestionPipeline(router=router, vector_store=vector_store)

    data_dir = config.data_dir
    print(f"📁 Ingesting documents from data directory: {data_dir}\n")

    # Ingest documents recursively across subdirectories
    results = pipeline.ingest_directory(data_dir)

    print("📊 Ingestion Summary per File:")
    for res in results:
        print(f"  • Document: {res['title']}")
        print(f"    - Target Namespace (via skills.ms): [{res['namespace']}]")
        print(f"    - Document Summary Tokens: {res['doc_summary_tokens']} (limit < 1500 tokens)")
        print(f"    - Doc Summary Preview: {res['doc_summary'][:140]}...")
        print(f"    - Chunks Ingested: {res['chunks_ingested']}\n")

    print("--- Vector Store Namespace Statistics ---")
    stats = vector_store.get_namespace_stats()
    for ns, count in stats.items():
        print(f"  📌 Namespace [{ns}]: {count} chunks indexed")

    print("\n✅ Ingestion Pipeline completed successfully!")
    print(f"💾 Vector Database saved to: {config.vector_db_path}")
    print(f"⚙️ Dynamic skills.ms properties registry loaded & updated at runtime: {config.skills_ms.registry_file}")

    # Demonstration of pre-search skills.ms namespace routing
    print("\n" + "=" * 70)
    print("🔍 Testing skills.ms Pre-Search Routing & Retrieval Demonstration")
    print("=" * 70)

    sample_queries = [
        "How do I crisp chicken wings in an air fryer?",
        "What does Error E3 mean on my air fryer and how to fix it?",
        "What is the recommended clearance around a kitchen island?",
        "How do I seal granite countertops to avoid lemon juice stains?"
    ]

    embedder = pipeline.embedder

    for query in sample_queries:
        # Step 1: Check skills.ms for correct index namespace
        target_namespaces = router.resolve_namespaces_for_query(query)
        target_ns = target_namespaces[0]
        
        print(f"\n❓ User Query: '{query}'")
        print(f"🎯 skills.ms Namespace Lookup Result: [{target_ns}]")

        # Step 2: Perform search strictly within target namespace
        query_vector = embedder.embed_text(query)
        matches = vector_store.search(query_vector=query_vector, namespace=target_ns, top_k=2)

        print(f"🔎 Retrived Chunks from Namespace [{target_ns}]:")
        for idx, match in enumerate(matches, 1):
            print(f"   [{idx}] Score: {match['similarity_score']} | Doc: {match['metadata']['document_title']}")
            print(f"       Text: {match['text'][:120]}...")

if __name__ == "__main__":
    main()
