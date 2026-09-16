import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class VectorChunk(BaseModel):
    chunk_id: str
    document_id: str
    namespace: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: List[float] = Field(default_factory=list)

class DocumentSummaryRecord(BaseModel):
    document_id: str
    namespace: str
    document_title: str
    summary_text: str
    token_count: int = 0
    embedding: List[float] = Field(default_factory=list)
    access_tier: str = "free"
    tenant_id: str = "global"
    clearance_level: int = 1
    metadata: Dict[str, Any] = Field(default_factory=dict)

class NamespaceVectorStore:
    """
    Partitioned Vector Store where vectors belong to explicit namespaces (skills.ms index namespaces).
    Supports dedicated document summaries and ABAC metadata filtering.
    """
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        # Map: namespace -> List[VectorChunk]
        self._namespaces: Dict[str, List[VectorChunk]] = {}
        # Map: document_id -> DocumentSummaryRecord
        self._summaries: Dict[str, DocumentSummaryRecord] = {}
        if storage_path and os.path.exists(storage_path):
            self.load_from_file(storage_path)

    def upsert_chunks(self, chunks: List[VectorChunk]) -> int:
        count = 0
        for chunk in chunks:
            if chunk.namespace not in self._namespaces:
                self._namespaces[chunk.namespace] = []
            
            # Replace if chunk_id already exists in namespace
            existing_idx = next(
                (i for i, c in enumerate(self._namespaces[chunk.namespace]) if c.chunk_id == chunk.chunk_id), 
                None
            )
            if existing_idx is not None:
                self._namespaces[chunk.namespace][existing_idx] = chunk
            else:
                self._namespaces[chunk.namespace].append(chunk)
            count += 1
        
        if self.storage_path:
            self.save_to_file(self.storage_path)
            
        return count

    def search(
        self, 
        query_vector: List[float], 
        namespace: Any, 
        top_k: int = 3,
        abac_filter: Optional[Dict[str, Any]] = None,
        is_summary: Optional[bool] = None,
        is_latest: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Similarity search restricted to specified skills.ms namespace(s) with optional
        ABAC metadata filtering and dual-resolution summary filtering.
        """
        if isinstance(namespace, str):
            target_namespaces = [namespace]
        elif isinstance(namespace, (list, tuple, set)):
            target_namespaces = list(namespace)
        else:
            target_namespaces = [str(namespace)]

        candidate_chunks: List[VectorChunk] = []
        for ns in target_namespaces:
            if ns in self._namespaces:
                candidate_chunks.extend(self._namespaces[ns])

        if not candidate_chunks:
            return []

        query_arr = np.array(query_vector, dtype=np.float32)
        norm_q = np.linalg.norm(query_arr)

        results = []
        for chunk in candidate_chunks:
            # 1. Dual-resolution filter
            if is_summary is not None:
                chunk_is_summary = chunk.metadata.get("is_summary", False)
                if chunk_is_summary != is_summary:
                    continue

            # 2. ABAC Predicate filter
            if abac_filter:
                user_tenant = abac_filter.get("tenant_id", "global")
                chunk_tenant = chunk.metadata.get("tenant_id", "global")
                if chunk_tenant != "global" and chunk_tenant != user_tenant:
                    continue

                permitted_tiers = abac_filter.get("permitted_tiers")
                if permitted_tiers:
                    chunk_tier = chunk.metadata.get("access_tier", "free")
                    if chunk_tier not in permitted_tiers:
                        continue

                max_clearance = abac_filter.get("max_clearance")
                if max_clearance is not None:
                    chunk_clearance = int(chunk.metadata.get("clearance_level", 1))
                    if chunk_clearance > max_clearance:
                        continue

            # 3. is_latest filter (Default: True if chunk has is_latest in metadata, unless is_latest is explicitly None/False)
            if is_latest is not None:
                chunk_is_latest = chunk.metadata.get("is_latest", True)
                if chunk_is_latest != is_latest:
                    continue

            # 4. Cosine similarity
            chunk_arr = np.array(chunk.embedding, dtype=np.float32)
            norm_c = np.linalg.norm(chunk_arr)
            if norm_q == 0 or norm_c == 0:
                similarity = 0.0
            else:
                similarity = float(np.dot(query_arr, chunk_arr) / (norm_q * norm_c))

            results.append({
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "namespace": chunk.namespace,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "similarity_score": round(similarity, 4)
            })

        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]

    def upsert_summary(self, summary: DocumentSummaryRecord) -> str:
        """Stores or updates a dedicated document summary record."""
        self._summaries[summary.document_id] = summary
        if self.storage_path:
            self.save_to_file(self.storage_path)
        return summary.document_id

    def get_summary(self, document_id: str, abac_filter: Optional[Dict[str, Any]] = None) -> Optional[DocumentSummaryRecord]:
        """Retrieves a single parent document summary by document_id, applying ABAC permissions if specified."""
        summary = self._summaries.get(document_id)
        if not summary:
            return None
        if abac_filter:
            user_tenant = abac_filter.get("tenant_id", "global")
            if summary.tenant_id != "global" and summary.tenant_id != user_tenant:
                return None
            permitted_tiers = abac_filter.get("permitted_tiers")
            if permitted_tiers and summary.access_tier not in permitted_tiers:
                return None
            max_clearance = abac_filter.get("max_clearance")
            if max_clearance is not None and summary.clearance_level > max_clearance:
                return None
        return summary

    def get_summaries(self, document_ids: List[str], abac_filter: Optional[Dict[str, Any]] = None) -> Dict[str, DocumentSummaryRecord]:
        """Batch retrieves parent document summaries for a list of document IDs, applying ABAC permissions."""
        results = {}
        for doc_id in document_ids:
            if doc_id in self._summaries:
                summary = self.get_summary(doc_id, abac_filter=abac_filter)
                if summary:
                    results[doc_id] = summary
        return results

    def delete_summary(self, document_id: str) -> bool:
        """Deletes a document summary record by document_id."""
        if document_id in self._summaries:
            del self._summaries[document_id]
            if self.storage_path:
                self.save_to_file(self.storage_path)
            return True
        return False

    def search_summaries(
        self,
        query_vector: List[float],
        namespace: Any,
        top_k: int = 3,
        abac_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Searches dedicated document summaries using cosine similarity and ABAC filters."""
        if isinstance(namespace, str):
            target_namespaces = [namespace]
        elif isinstance(namespace, (list, set)):
            target_namespaces = list(namespace)
        else:
            target_namespaces = []

        query_arr = np.array(query_vector, dtype=np.float32)
        norm_q = np.linalg.norm(query_arr)

        candidates = [
            s for s in self._summaries.values()
            if not target_namespaces or s.namespace in target_namespaces
        ]

        # Apply ABAC filter
        if abac_filter:
            user_tenant = abac_filter.get("tenant_id", "global")
            permitted_tiers = abac_filter.get("permitted_tiers")
            max_clearance = abac_filter.get("max_clearance")
            filtered = []
            for s in candidates:
                if s.tenant_id != "global" and s.tenant_id != user_tenant:
                    continue
                if permitted_tiers and s.access_tier not in permitted_tiers:
                    continue
                if max_clearance is not None and s.clearance_level > max_clearance:
                    continue
                filtered.append(s)
            candidates = filtered

        results = []
        for s in candidates:
            if not s.embedding:
                continue
            s_arr = np.array(s.embedding, dtype=np.float32)
            norm_s = np.linalg.norm(s_arr)
            sim = 0.0 if (norm_q == 0 or norm_s == 0) else float(np.dot(query_arr, s_arr) / (norm_q * norm_s))
            results.append({
                "document_id": s.document_id,
                "namespace": s.namespace,
                "document_title": s.document_title,
                "summary_text": s.summary_text,
                "token_count": s.token_count,
                "similarity_score": round(sim, 4),
                "metadata": s.metadata
            })

        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]

    def delete_chunks_by_document_id(self, document_id: str) -> int:
        """
        Physically deletes all chunks matching document_id across all namespaces (for Quash).
        Also removes the parent summary if present.
        """
        deleted_count = 0
        for ns in list(self._namespaces.keys()):
            before_len = len(self._namespaces[ns])
            self._namespaces[ns] = [c for c in self._namespaces[ns] if c.document_id != document_id]
            deleted_count += (before_len - len(self._namespaces[ns]))
        
        if document_id in self._summaries:
            del self._summaries[document_id]

        if self.storage_path and deleted_count > 0:
            self.save_to_file(self.storage_path)
        return deleted_count

    def delete_chunks_by_job_id(self, job_id: str) -> int:
        """
        Physically deletes all uncommitted staged chunks matching job_id (for failure cleanup).
        """
        deleted_count = 0
        for ns in list(self._namespaces.keys()):
            before_len = len(self._namespaces[ns])
            self._namespaces[ns] = [c for c in self._namespaces[ns] if c.metadata.get("job_id") != job_id]
            deleted_count += (before_len - len(self._namespaces[ns]))
        if self.storage_path and deleted_count > 0:
            self.save_to_file(self.storage_path)
        return deleted_count

    def flip_is_latest(self, doc_family: str, new_active_job_id: Optional[str] = None, new_document_id: Optional[str] = None) -> int:
        """
        Atomically flips is_latest in memory for doc_family:
        Sets new active chunks to is_latest = True, older version chunks to is_latest = False.
        """
        flipped_count = 0
        for ns in self._namespaces.values():
            for chunk in ns:
                if chunk.metadata.get("doc_family") == doc_family:
                    is_new = (new_active_job_id and chunk.metadata.get("job_id") == new_active_job_id) or \
                             (new_document_id and chunk.document_id == new_document_id)
                    chunk.metadata["is_latest"] = bool(is_new)
                    flipped_count += 1
        if self.storage_path and flipped_count > 0:
            self.save_to_file(self.storage_path)
        return flipped_count

    def get_namespace_stats(self) -> Dict[str, int]:
        return {ns: len(chunks) for ns, chunks in self._namespaces.items()}

    def get_all_chunks_in_namespace(self, namespace: str) -> List[VectorChunk]:
        return self._namespaces.get(namespace, [])

    def save_to_file(self, file_path: str) -> None:
        serialized = {}
        for ns, chunks in self._namespaces.items():
            serialized[ns] = [chunk.model_dump() for chunk in chunks]
        if self._summaries:
            serialized["__document_summaries__"] = {k: v.model_dump() for k, v in self._summaries.items()}
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serialized, f, indent=2)

    def load_from_file(self, file_path: str) -> None:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._namespaces = {}
        self._summaries = {}
        for ns, chunks_data in data.items():
            if ns == "__document_summaries__":
                self._summaries = {k: DocumentSummaryRecord(**v) for k, v in chunks_data.items()}
            elif isinstance(chunks_data, list):
                self._namespaces[ns] = [VectorChunk(**c) for c in chunks_data]
