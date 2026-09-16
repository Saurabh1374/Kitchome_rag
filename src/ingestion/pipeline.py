from typing import List, Dict, Any, Optional
from ..skills_ms.router import SkillsRouter
from ..vector_store.base import NamespaceVectorStore, VectorChunk, DocumentSummaryRecord
from ..vector_store.factory import get_vector_store
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
        embedder: Optional[EmbeddingEngine] = None,
        index_summary_chunk: bool = False
    ):
        self.router = router or SkillsRouter()
        self.vector_store = vector_store or get_vector_store()
        self.chunker = chunker or TextChunker()
        self.summarizer = summarizer or DocumentSummarizer()
        self.embedder = embedder or EmbeddingEngine()
        self.index_summary_chunk = index_summary_chunk

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

        # 6. Construct VectorChunks for vector store with summary and security attached to metadata
        vector_chunks = []
        for tc, emb in zip(text_chunks, embeddings):
            tc.metadata["doc_summary"] = doc_summary
            tc.metadata["document_title"] = document.title
            tc.metadata["access_tier"] = document.access_tier
            tc.metadata["tenant_id"] = document.tenant_id
            tc.metadata["clearance_level"] = document.clearance_level
            tc.metadata["is_summary"] = False
            tc.metadata["chunk_type"] = "content"
            vector_chunks.append(VectorChunk(
                chunk_id=tc.chunk_id,
                document_id=tc.document_id,
                namespace=target_namespace,
                text=tc.text,
                metadata=tc.metadata,
                embedding=emb
            ))

        # 7. Dedicated Document Summary Table Upsert
        if doc_summary and hasattr(self.vector_store, "upsert_summary"):
            summary_emb = self.embedder.embed_text(doc_summary)
            self.vector_store.upsert_summary(DocumentSummaryRecord(
                document_id=document.document_id,
                namespace=target_namespace,
                document_title=document.title,
                summary_text=doc_summary,
                token_count=self.summarizer.estimate_tokens(doc_summary),
                embedding=summary_emb,
                access_tier=document.access_tier,
                tenant_id=document.tenant_id,
                clearance_level=document.clearance_level,
                metadata={"title": document.title}
            ))

        # Optional Dual-Resolution: Index document summary as a first-class chunk
        if self.index_summary_chunk and doc_summary:
            summary_emb = self.embedder.embed_text(doc_summary)
            vector_chunks.append(VectorChunk(
                chunk_id=f"summary_{document.document_id}",
                document_id=document.document_id,
                namespace=target_namespace,
                text=doc_summary,
                metadata={
                    "is_summary": True,
                    "chunk_type": "document_summary",
                    "document_title": document.title,
                    "access_tier": document.access_tier,
                    "tenant_id": document.tenant_id,
                    "clearance_level": document.clearance_level,
                    "doc_summary": doc_summary
                },
                embedding=summary_emb
            ))

        # 8. Upsert chunks into vector store under target namespace
        upsert_count = self.vector_store.upsert_chunks(vector_chunks)

        return {
            "document_id": document.document_id,
            "title": document.title,
            "namespace": target_namespace,
            "chunks_ingested": upsert_count,
            "doc_summary_tokens": self.summarizer.estimate_tokens(doc_summary),
            "doc_summary": doc_summary,
            "summary": doc_summary
        }

    def ingest_file(
        self,
        file_path: str,
        document_id: str,
        title: str,
        tenant_id: str = "global",
        clearance_level: int = 1,
        access_tier: str = "free",
        declared_domain: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Streaming file ingestion from a stored disk path.
        Parses format, summarizes using file stream, generates chunks lazily in batches,
        and flushes to vector store with bounded RAM.
        """
        import os
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Target document file missing: {file_path}")

        # 1. Preview text for namespace routing to avoid excessive reads
        preview_text = ""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            preview_text = f.read(4096)

        target_namespace = self.router.resolve_namespace_for_document(
            file_path=file_path,
            content=preview_text,
            declared_domain=declared_domain
        )

        # 2. Summarize document directly from file stream
        doc_summary, extracted_keywords = self.summarizer.process_and_summarize_file(
            document_title=title,
            file_path=file_path
        )

        # 3. Update skills.ms registry & runtime properties file dynamically
        self.router.registry.update_dynamic_summary(
            namespace=target_namespace,
            document_title=title,
            summary=doc_summary,
            extra_keywords=extracted_keywords
        )

        # 4. Construct RawDocument representation
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        raw_doc = RawDocument(
            document_id=document_id,
            title=title,
            source_path=file_path,
            content=content,
            declared_domain=declared_domain,
            access_tier=access_tier,
            tenant_id=tenant_id,
            clearance_level=clearance_level,
            metadata=metadata or {}
        )

        # 5. Chunk and embed in streaming micro-batches (32 chunks per batch)
        total_ingested = 0
        for batch_chunks in self.chunker.chunk_document_stream(raw_doc, namespace=target_namespace, batch_size=32):
            texts = [c.text for c in batch_chunks]
            embeddings = self.embedder.embed_batch(texts)
            vector_batch = []
            for tc, emb in zip(batch_chunks, embeddings):
                tc.metadata["doc_summary"] = doc_summary
                tc.metadata["document_title"] = title
                tc.metadata["access_tier"] = access_tier
                tc.metadata["tenant_id"] = tenant_id
                tc.metadata["clearance_level"] = clearance_level
                tc.metadata["is_summary"] = False
                tc.metadata["chunk_type"] = "content"
                vector_batch.append(VectorChunk(
                    chunk_id=tc.chunk_id,
                    document_id=tc.document_id,
                    namespace=target_namespace,
                    text=tc.text,
                    metadata=tc.metadata,
                    embedding=emb
                ))
            total_ingested += self.vector_store.upsert_chunks(vector_batch)

        # 6. Dedicated Document Summary Table Upsert
        if doc_summary and hasattr(self.vector_store, "upsert_summary"):
            summary_emb = self.embedder.embed_text(doc_summary)
            self.vector_store.upsert_summary(DocumentSummaryRecord(
                document_id=document_id,
                namespace=target_namespace,
                document_title=title,
                summary_text=doc_summary,
                token_count=self.summarizer.estimate_tokens(doc_summary),
                embedding=summary_emb,
                access_tier=access_tier,
                tenant_id=tenant_id,
                clearance_level=clearance_level,
                metadata={"title": title}
            ))

        return {
            "document_id": document_id,
            "title": title,
            "namespace": target_namespace,
            "chunks_ingested": total_ingested,
            "doc_summary_tokens": self.summarizer.estimate_tokens(doc_summary),
            "doc_summary": doc_summary,
            "summary": doc_summary,
            "file_path": file_path
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
