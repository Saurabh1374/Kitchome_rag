import os
import re
from typing import List, Tuple

class DocumentSummarizer:
    """
    Progressive Streaming Document Summarizer:
    1. Summarizes incoming document chunks sequentially.
    2. Appends chunk summary to running summary buffer.
    3. When running summary reaches 700 tokens + 150 token buffer (850 tokens total),
       PURGES the verbose summary and compresses it into a compact high-level representation.
    4. Repeats until EOF.
    5. Generates final overall summary under 1500 words/tokens for skills.ms meta store.
    """
    def __init__(
        self, 
        summary_threshold_tokens: int = 700, 
        buffer_allowance_tokens: int = 150, 
        max_final_summary_words: int = 1500
    ):
        self.summary_threshold_tokens = summary_threshold_tokens
        self.buffer_allowance_tokens = buffer_allowance_tokens
        self.purge_trigger_tokens = summary_threshold_tokens + buffer_allowance_tokens  # 850 tokens
        self.max_final_summary_words = max_final_summary_words

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimates token count (~4 characters per token average or word count).
        """
        words = len(text.split())
        chars = len(text)
        return max(words, chars // 4)

    @staticmethod
    def count_words(text: str) -> int:
        return len(text.split())

    def process_and_summarize_file(self, document_title: str, file_path: str) -> Tuple[str, List[str]]:
        """
        Executes progressive streaming summarization directly from a file path.
        Reads line-by-line / paragraph-by-paragraph to avoid loading entire document into memory.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document file not found: {file_path}")

        running_summary_buffer = f"Document Title: {document_title}\n"
        current_para: List[str] = []
        has_content = False

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    if current_para:
                        para_text = "\n".join(current_para).strip()
                        current_para = []
                        if para_text:
                            has_content = True
                            chunk_summary = self._summarize_single_chunk(para_text)
                            running_summary_buffer += f"\n- {chunk_summary}"
                            if self.estimate_tokens(running_summary_buffer) >= self.purge_trigger_tokens:
                                running_summary_buffer = self._purge_and_compact_buffer(running_summary_buffer)
                else:
                    current_para.append(stripped)

            if current_para:
                para_text = "\n".join(current_para).strip()
                if para_text:
                    has_content = True
                    chunk_summary = self._summarize_single_chunk(para_text)
                    running_summary_buffer += f"\n- {chunk_summary}"
                    if self.estimate_tokens(running_summary_buffer) >= self.purge_trigger_tokens:
                        running_summary_buffer = self._purge_and_compact_buffer(running_summary_buffer)

        if not has_content:
            return f"Summary of {document_title}: Empty document", []

        final_summary = self._create_final_skills_summary(document_title, running_summary_buffer)
        extracted_keywords = self._extract_keywords(final_summary)
        return final_summary, extracted_keywords

    def process_and_summarize_document(self, document_title: str, text: str) -> Tuple[str, List[str]]:
        """
        Executes progressive chunk summarization, 700+150 token buffer purging,
        and EOF final summary (<1500 words) for skills.ms.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return f"Summary of {document_title}: Empty document", []

        running_summary_buffer = f"Document Title: {document_title}\n"
        
        # 1. Progressive streaming summarization as chunking proceeds
        for idx, para in enumerate(paragraphs, 1):
            chunk_summary = self._summarize_single_chunk(para)
            running_summary_buffer += f"\n- {chunk_summary}"

            # Check if running summary reached 700 tokens + 150 token buffer (850 tokens)
            current_tokens = self.estimate_tokens(running_summary_buffer)
            if current_tokens >= self.purge_trigger_tokens:
                # PURGE verbose buffer and create a compact high-level summary (~350 tokens)
                running_summary_buffer = self._purge_and_compact_buffer(running_summary_buffer)

        # 2. EOF reached: Create final overall summary under 1500 words
        final_summary = self._create_final_skills_summary(document_title, running_summary_buffer)
        
        # 3. Extract keywords for skills.ms meta store
        extracted_keywords = self._extract_keywords(final_summary)

        return final_summary, extracted_keywords

    def _summarize_single_chunk(self, chunk_text: str) -> str:
        """
        Extracts key point / heading from a single chunk.
        """
        lines = [line.strip() for line in chunk_text.split("\n") if line.strip()]
        key_lines = []
        for line in lines:
            if line.startswith("#") or line.startswith("-") or line.startswith("*") or ":" in line:
                key_lines.append(line.lstrip("#*- ").strip())
            elif len(line) > 20:
                key_lines.append(line[:100])

        if not key_lines:
            key_lines = lines[:2]
            
        return " | ".join(key_lines[:4])

    def _purge_and_compact_buffer(self, buffer_text: str) -> str:
        """
        PURGES the accumulated summary buffer (which hit 700+150=850 tokens)
        and compresses it into a tight, compact representation (~300-350 tokens).
        """
        lines = [l.strip() for l in buffer_text.split("\n") if l.strip()]
        
        # Filter high-priority headings and core facts
        compact_points = []
        for line in lines:
            if line.startswith("Document Title:") or "Error" in line or "Recipe" in line or "Guide" in line or "Specs" in line:
                compact_points.append(line)
            elif ":" in line:
                compact_points.append(line)

        if len(compact_points) < 3:
            compact_points = lines[:6]

        # Rebuild purged compact summary
        compact_summary = "COMPACT PURGED SUMMARY:\n" + "\n".join(compact_points[:10])
        return compact_summary

    def _create_final_skills_summary(self, title: str, running_buffer: str) -> str:
        """
        Final EOF overall summary guaranteed under 1500 words for skills.ms meta store.
        """
        words = self.count_words(running_buffer)
        
        if words <= self.max_final_summary_words:
            return f"Overall Summary of {title}:\n" + running_buffer.strip()

        # If exceeding 1500 words, condense to fit within 1500 words
        word_list = running_buffer.split()
        condensed_words = word_list[:self.max_final_summary_words - 20]
        return f"Overall Summary of {title}:\n" + " ".join(condensed_words) + "..."

    def _extract_keywords(self, text: str) -> List[str]:
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        stopwords = {
            "this", "that", "with", "from", "have", "been", "where", "what", "when", 
            "your", "into", "their", "more", "also", "using", "summary", "about", "which",
            "overall", "compact", "purged", "document", "title"
        }
        filtered = [w for w in words if w not in stopwords]
        
        freq = {}
        for w in filtered:
            freq[w] = freq.get(w, 0) + 1
        
        sorted_keywords = sorted(freq.keys(), key=lambda k: freq[k], reverse=True)
        return sorted_keywords[:15]
