import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from .loader import RawDocument

class TextChunk(BaseModel):
    chunk_id: str
    document_id: str
    namespace: str
    chunk_index: int
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TextChunker:
    """
    Adaptive text chunking with metadata and skills.ms namespace preservation.
    """
    def __init__(self, chunk_size: int = 2800, chunk_overlap: int = 200):
        self.chunk_size = chunk_size  # ~700 tokens (approx 4 chars per token)
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: RawDocument, namespace: str) -> List[TextChunk]:
        content = document.content.strip()
        if not content:
            return []

        # Split into paragraphs/sections first
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        
        chunks_text: List[str] = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk = f"{current_chunk}\n\n{para}".strip()
            else:
                if current_chunk:
                    chunks_text.append(current_chunk)
                
                # If paragraph itself is larger than chunk_size, split by sentences/lines
                if len(para) > self.chunk_size:
                    sub_splits = self._split_large_text(para)
                    chunks_text.extend(sub_splits)
                    current_chunk = ""
                else:
                    current_chunk = para

        if current_chunk:
            chunks_text.append(current_chunk)

        result_chunks = []
        for idx, text in enumerate(chunks_text):
            chunk_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document.document_id}_chunk_{idx}"))
            
            chunk_metadata = {
                **document.metadata,
                "document_title": document.title,
                "source_path": document.source_path,
                "chunk_index": idx,
                "total_chunks": len(chunks_text),
                "declared_domain": document.declared_domain
            }

            result_chunks.append(TextChunk(
                chunk_id=chunk_uuid,
                document_id=document.document_id,
                namespace=namespace,
                chunk_index=idx,
                text=text,
                metadata=chunk_metadata
            ))

        return result_chunks

    def _split_large_text(self, text: str) -> List[str]:
        splits = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.chunk_size
            if end >= text_len:
                splits.append(text[start:])
                break
            
            # Find closest sentence break or space near end
            break_pos = text.rfind(". ", start, end)
            if break_pos == -1 or break_pos < start + (self.chunk_size // 2):
                break_pos = text.rfind(" ", start, end)
            
            if break_pos == -1 or break_pos <= start:
                break_pos = end

            splits.append(text[start:break_pos].strip())
            start = max(break_pos + 1, start + self.chunk_size - self.chunk_overlap)
            
        return [s for s in splits if s]
