import hashlib
import numpy as np
from typing import List

class EmbeddingEngine:
    """
    Embedding generator creating dense vector representations.
    Uses semantic hash-vector projection for zero-dependency execution, 
    with standard 384-dimension output.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed_text(self, text: str) -> List[float]:
        """
        Generates a normalized 384-dim dense vector embedding for text.
        """
        text_clean = text.lower().strip()
        tokens = text_clean.split()
        
        vector = np.zeros(self.dimension, dtype=np.float32)
        
        # Token feature mapping
        for token in tokens:
            # Deterministic feature index mapping based on md5 hash
            hash_val = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)
            idx = hash_val % self.dimension
            val = (hash_val % 100) / 100.0 - 0.5
            vector[idx] += val
            
            # Additional n-gram features for semantic proximity
            if len(token) >= 3:
                for i in range(len(token) - 2):
                    sub = token[i:i+3]
                    sub_hash = int(hashlib.md5(sub.encode('utf-8')).hexdigest(), 16)
                    sub_idx = sub_hash % self.dimension
                    vector[sub_idx] += 0.2

        # Normalize L2 norm
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]
