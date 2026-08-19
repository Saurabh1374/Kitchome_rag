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
        namespace: str, 
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Similarity search restricted to the specified skills.ms namespace.
        """
        if namespace not in self._namespaces or not self._namespaces[namespace]:
            return []

        chunks = self._namespaces[namespace]
        query_arr = np.array(query_vector, dtype=np.float32)
        
        # Calculate Cosine Similarities
        results = []
        for chunk in chunks:
            chunk_arr = np.array(chunk.embedding, dtype=np.float32)
            norm_q = np.linalg.norm(query_arr)
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
