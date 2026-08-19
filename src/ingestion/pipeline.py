from typing import List, Dict, Any, Optional
from ..skills_ms.router import SkillsRouter
from ..vector_store.base import NamespaceVectorStore, VectorChunk
from .loader import RawDocument, DocumentLoader
from .chunker import TextChunker
from .summarizer import DocumentSummarizer
from .embedder import EmbeddingEngine

class IngestionPipeline:
    """
    Kitchome RAG Ingestion Pipeline with dynamic skills.ms meta store updates.
    Summarizes documents page-by-page (purging 700-token raw blocks), creates doc summary (<1500 tokens),
    and updates skills.ms runtime properties file.
    """
    def __init__(
        self,
        router: Optional[SkillsRouter] = None,
        vector_store: Optional[NamespaceVectorStore] = None,
        chunker: Optional[TextChunker] = None,
        summarizer: Optional[DocumentSummarizer] = None,
        embedder: Optional[EmbeddingEngine] = None
    ):
        self.router = router or SkillsRouter()
        self.vector_store = vector_store or NamespaceVectorStore()
        self.chunker = chunker or TextChunker()
        self.summarizer = summarizer or DocumentSummarizer()
        self.embedder = embedder or EmbeddingEngine()

    def ingest_document(self, document: RawDocument) -> Dict[str, Any]:
        """
        Ingests a document:
        1. Resolves target namespace via skills.ms
        2. Summarizes 700-token blocks, purges raw blocks, generates <1500 token document summary
        3. Updates skills.ms registry & runtime properties file dynamically
        4. Chunks text, embeds, and indexes into target namespace in vector store
        """
        # 1. Check skills.ms for correct index namespace
        target_namespace = self.router.resolve_namespace_for_document(
            file_path=document.source_path,
            content=document.content,
            declared_domain=document.declared_domain
        )

        # 2. Summarize document page-after-page (700 token blocks purged) -> doc summary (<1500 tokens)
        doc_summary, extracted_keywords = self.summarizer.process_and_summarize_document(
            document_title=document.title,
            text=document.content
        )

        # 3. Update skills.ms meta store & save runtime properties file
        self.router.registry.update_dynamic_summary(
            namespace=target_namespace,
            document_title=document.title,
            summary=doc_summary,
            extra_keywords=extracted_keywords
        )

        # 4. Chunk document (~700 tokens) preserving namespace context
        text_chunks = self.chunker.chunk_document(document, namespace=target_namespace)
        if not text_chunks:
            return {
                "document_id": document.document_id, 
                "chunks_ingested": 0, 
                "namespace": target_namespace,
                "summary": doc_summary
            }

        # 5. Generate embeddings for chunks
        texts = [c.text for c in text_chunks]
        embeddings = self.embedder.embed_batch(texts)

        # 6. Construct VectorChunks for vector store with summary attached to metadata
        vector_chunks = []
        for tc, emb in zip(text_chunks, embeddings):
            tc.metadata["doc_summary"] = doc_summary
            vector_chunks.append(VectorChunk(
                chunk_id=tc.chunk_id,
                document_id=tc.document_id,
                namespace=target_namespace,
                text=tc.text,
                metadata=tc.metadata,
                embedding=emb
            ))

        # 7. Upsert into vector store under target namespace
        upsert_count = self.vector_store.upsert_chunks(vector_chunks)

        return {
            "document_id": document.document_id,
            "title": document.title,
            "namespace": target_namespace,
            "chunks_ingested": upsert_count,
            "doc_summary_tokens": self.summarizer.estimate_tokens(doc_summary),
            "doc_summary": doc_summary
        }

    def ingest_directory(self, dir_path: str, declared_domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Ingests all documents in a directory.
        """
        documents = DocumentLoader.load_directory(dir_path, declared_domain=declared_domain)
        results = []
        for doc in documents:
            res = self.ingest_document(doc)
            results.append(res)
        return results
