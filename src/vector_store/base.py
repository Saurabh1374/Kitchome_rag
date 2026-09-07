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

class NamespaceVectorStore:
    """
    Partitioned Vector Store where vectors belong to explicit namespaces (skills.ms index namespaces).
    """
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        # Map: namespace -> List[VectorChunk]
        self._namespaces: Dict[str, List[VectorChunk]] = {}
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

    def delete_chunks_by_document_id(self, document_id: str) -> int:
        """
        Physically deletes all chunks matching document_id across all namespaces (for Quash).
        """
        deleted_count = 0
        for ns in list(self._namespaces.keys()):
            before_len = len(self._namespaces[ns])
            self._namespaces[ns] = [c for c in self._namespaces[ns] if c.document_id != document_id]
            deleted_count += (before_len - len(self._namespaces[ns]))
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
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serialized, f, indent=2)

    def load_from_file(self, file_path: str) -> None:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._namespaces = {}
        for ns, chunks_data in data.items():
            self._namespaces[ns] = [VectorChunk(**c) for c in chunks_data]
