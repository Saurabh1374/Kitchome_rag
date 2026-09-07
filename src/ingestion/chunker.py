import re
import uuid
from typing import List, Dict, Any, Optional, Iterator, Tuple
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
    Adaptive text chunking with sentence-aware sliding overlap, 
    hierarchical section breadcrumb tracking, and skills.ms namespace preservation.
    """
    def __init__(self, chunk_size: int = 2800, chunk_overlap: int = 200):
        self.chunk_size = chunk_size  # ~700 tokens (approx 4 chars per token)
        self.chunk_overlap = chunk_overlap

    def _extract_sentence_overlap(self, text: str, overlap_budget: int) -> str:
        """
        Extracts the maximum number of trailing complete sentences from text 
        whose combined length does not exceed overlap_budget characters.
        Snaps cleanly to sentence boundaries to prevent broken words or fragments.
        """
        if not text or overlap_budget <= 0:
            return ""

        # Split text into candidate sentences using fixed-width sentence terminator lookbehinds
        raw_sentences = [
            s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) 
            if s.strip()
        ]
        if not raw_sentences:
            return ""

        selected = []
        current_len = 0

        for s in reversed(raw_sentences):
            added_len = len(s) + (1 if selected else 0)
            if current_len + added_len <= overlap_budget:
                selected.append(s)
                current_len += added_len
            else:
                break

        if selected:
            return " ".join(reversed(selected)).strip()

        # Fallback: if the last sentence alone exceeds overlap_budget,
        # snap to the nearest word boundary in the tail
        tail = text[-overlap_budget:].strip()
        first_space = tail.find(" ")
        if first_space != -1 and first_space < len(tail) // 2:
            return tail[first_space + 1:].strip()
        return tail

    @staticmethod
    def _detect_heading(line: str) -> Optional[Tuple[int, str]]:
        """Detects markdown heading level and text from a line."""
        line = line.strip()
        match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            title = re.sub(r'[*_`]', '', title).strip()
            return level, title
        return None

    def _generate_chunks_lazy(self, document: RawDocument, namespace: str) -> Iterator[TextChunk]:
        content = document.content.strip()
        if not content:
            return

        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        current_chunk = ""
        chunk_idx = 0
        heading_stack: List[str] = []
        has_overlap = False
        overlap_char_count = 0

        def _get_breadcrumb_meta() -> Tuple[str, str, List[str]]:
            clean_headings = [h for h in heading_stack if h.lower() != document.title.lower()]
            parts = [document.title] + clean_headings if clean_headings else [document.title]
            breadcrumb_str = " > ".join(parts)
            section_heading = heading_stack[-1] if heading_stack else document.title
            return breadcrumb_str, section_heading, list(heading_stack)

        for para in paragraphs:
            h_info = self._detect_heading(para.split("\n")[0])
            # If this paragraph starts a new heading and current_chunk already has body content, flush current_chunk
            if h_info and current_chunk:
                lines = [l.strip() for l in current_chunk.split("\n") if l.strip()]
                has_body = any(not self._detect_heading(l) for l in lines)
                if has_body:
                    chunk_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document.document_id}_chunk_{chunk_idx}"))
                    b_str, s_head, h_list = _get_breadcrumb_meta()
                    yield TextChunk(
                        chunk_id=chunk_uuid,
                        document_id=document.document_id,
                        namespace=namespace,
                        chunk_index=chunk_idx,
                        text=current_chunk,
                        metadata={
                            **document.metadata,
                            "document_title": document.title,
                            "source_path": document.source_path,
                            "chunk_index": chunk_idx,
                            "declared_domain": document.declared_domain,
                            "breadcrumb": b_str,
                            "section_heading": s_head,
                            "heading_hierarchy": h_list,
                            "has_overlap": has_overlap,
                            "overlap_char_count": overlap_char_count
                        }
                    )
                    chunk_idx += 1
                    current_chunk = ""
                    has_overlap = False
                    overlap_char_count = 0

            # Update heading stack if para introduces a heading
            for line in para.split("\n"):
                h = self._detect_heading(line)
                if h:
                    level, h_title = h
                    heading_stack = heading_stack[:level - 1]
                    heading_stack.append(h_title)

            # Test if appending paragraph fits within chunk_size
            test_chunk = f"{current_chunk}\n\n{para}".strip() if current_chunk else para
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunk_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document.document_id}_chunk_{chunk_idx}"))
                    b_str, s_head, h_list = _get_breadcrumb_meta()
                    yield TextChunk(
                        chunk_id=chunk_uuid,
                        document_id=document.document_id,
                        namespace=namespace,
                        chunk_index=chunk_idx,
                        text=current_chunk,
                        metadata={
                            **document.metadata,
                            "document_title": document.title,
                            "source_path": document.source_path,
                            "chunk_index": chunk_idx,
                            "declared_domain": document.declared_domain,
                            "breadcrumb": b_str,
                            "section_heading": s_head,
                            "heading_hierarchy": h_list,
                            "has_overlap": has_overlap,
                            "overlap_char_count": overlap_char_count
                        }
                    )
                    chunk_idx += 1

                    # Extract sentence-aware overlap seed for the next chunk (only if not crossing heading)
                    if not h_info:
                        overlap_seed = self._extract_sentence_overlap(current_chunk, self.chunk_overlap)
                    else:
                        overlap_seed = ""

                    if overlap_seed:
                        has_overlap = True
                        overlap_char_count = len(overlap_seed)
                    else:
                        has_overlap = False
                        overlap_char_count = 0
                else:
                    overlap_seed = ""
                    has_overlap = False
                    overlap_char_count = 0

                # Handle oversized paragraph (> chunk_size)
                if len(para) > self.chunk_size:
                    for sub, sub_overlap, sub_char_count in self._split_large_text(para):
                        chunk_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document.document_id}_chunk_{chunk_idx}"))
                        b_str, s_head, h_list = _get_breadcrumb_meta()
                        yield TextChunk(
                            chunk_id=chunk_uuid,
                            document_id=document.document_id,
                            namespace=namespace,
                            chunk_index=chunk_idx,
                            text=sub,
                            metadata={
                                **document.metadata,
                                "document_title": document.title,
                                "source_path": document.source_path,
                                "chunk_index": chunk_idx,
                                "declared_domain": document.declared_domain,
                                "breadcrumb": b_str,
                                "section_heading": s_head,
                                "heading_hierarchy": h_list,
                                "has_overlap": sub_overlap,
                                "overlap_char_count": sub_char_count
                            }
                        )
                        chunk_idx += 1
                    current_chunk = ""
                    has_overlap = False
                    overlap_char_count = 0
                else:
                    if overlap_seed and not para.startswith(overlap_seed):
                        current_chunk = f"{overlap_seed}\n\n{para}".strip()
                    else:
                        current_chunk = para

        if current_chunk:
            chunk_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document.document_id}_chunk_{chunk_idx}"))
            b_str, s_head, h_list = _get_breadcrumb_meta()
            yield TextChunk(
                chunk_id=chunk_uuid,
                document_id=document.document_id,
                namespace=namespace,
                chunk_index=chunk_idx,
                text=current_chunk,
                metadata={
                    **document.metadata,
                    "document_title": document.title,
                    "source_path": document.source_path,
                    "chunk_index": chunk_idx,
                    "declared_domain": document.declared_domain,
                    "breadcrumb": b_str,
                    "section_heading": s_head,
                    "heading_hierarchy": h_list,
                    "has_overlap": has_overlap,
                    "overlap_char_count": overlap_char_count
                }
            )

    def chunk_document(self, document: RawDocument, namespace: str) -> List[TextChunk]:
        return list(self._generate_chunks_lazy(document, namespace))

    def chunk_document_stream(
        self, 
        document: RawDocument, 
        namespace: str, 
        batch_size: int = 32
    ) -> Iterator[List[TextChunk]]:
        """
        Streams chunks in bounded mini-batches (default 32 chunks) lazily to guarantee true O(1) memory.
        """
        batch: List[TextChunk] = []
        for chunk in self._generate_chunks_lazy(document, namespace):
            batch.append(chunk)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _split_large_text(self, text: str) -> List[Tuple[str, bool, int]]:
        """
        Splits an oversized paragraph into sentence-aware sub-chunks with overlap.
        Returns a list of tuples: (sub_chunk_text, has_overlap, overlap_char_count)
        """
        splits: List[Tuple[str, bool, int]] = []
        start = 0
        text_len = len(text)
        is_overlap = False
        overlap_len = 0
        
        while start < text_len:
            end = start + self.chunk_size
            if end >= text_len:
                sub = text[start:].strip()
                if sub:
                    splits.append((sub, is_overlap, overlap_len))
                break
            
            # Find closest sentence break near end
            break_pos = -1
            search_region = text[start:end]
            for m in re.finditer(r'(?<=[.!?])\s+', search_region):
                candidate = start + m.end()
                if candidate >= start + (self.chunk_size // 3):
                    break_pos = candidate
            
            if break_pos == -1 or break_pos < start + (self.chunk_size // 3):
                break_pos = text.rfind(" ", start, end)
            
            if break_pos == -1 or break_pos <= start:
                break_pos = end

            sub = text[start:break_pos].strip()
            if sub:
                splits.append((sub, is_overlap, overlap_len))
                overlap_seed = self._extract_sentence_overlap(sub, self.chunk_overlap)
                if overlap_seed:
                    seed_pos = text.find(overlap_seed, start, break_pos + 1)
                    if seed_pos != -1 and seed_pos > start:
                        start = seed_pos
                        is_overlap = True
                        overlap_len = len(overlap_seed)
                        continue

            start = break_pos
            is_overlap = False
            overlap_len = 0
            
        return splits
