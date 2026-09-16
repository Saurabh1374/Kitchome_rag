from typing import List, Dict, Any, Optional

class GroundedGenerator:
    """
    Synthesizes grounded RAG answers with parent-context injection and strict provenance citations.
    """
    def __init__(self, model_name: str = "kitchome-grounded-synthesizer"):
        self.model_name = model_name

    def format_parent_context_prompt(
        self, 
        query: str, 
        retrieved_chunks: List[Dict[str, Any]],
        document_summaries: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Assembles prompt with Parent-Context Injection:
        Groups chunks by parent document, prepends document summary header, and lists excerpts.
        Resolves summary from document_summaries map or chunk metadata.
        """
        # Deduplicate parent documents and group excerpts
        doc_groups: Dict[str, Dict[str, Any]] = {}
        for c in retrieved_chunks:
            doc_id = c.get("document_id", "doc_unknown")
            meta = c.get("metadata", {})
            title = meta.get("document_title", meta.get("filename", "Document"))
            doc_summary = meta.get("doc_summary", "")
            if not doc_summary and document_summaries and doc_id in document_summaries:
                doc_summary = document_summaries[doc_id]

            breadcrumb = meta.get("breadcrumb") or title
            section_heading = meta.get("section_heading") or title

            if doc_id not in doc_groups:
                doc_groups[doc_id] = {
                    "title": title,
                    "summary": doc_summary,
                    "excerpts": []
                }
            doc_groups[doc_id]["excerpts"].append({
                "chunk_id": c.get("chunk_id", ""),
                "text": c.get("text", ""),
                "similarity_score": c.get("similarity_score", 0.0),
                "breadcrumb": breadcrumb,
                "section_heading": section_heading
            })

        # Build formatted prompt text
        prompt_lines = [
            f"User Query: {query}",
            "",
            "=== AUTHORIZED GROUNDED KNOWLEDGE BASE ==="
        ]

        citations = []
        for doc_id, doc_data in doc_groups.items():
            prompt_lines.append(f"\n--- DOCUMENT: {doc_data['title']} (ID: {doc_id}) ---")
            if doc_data["summary"]:
                prompt_lines.append(f"[Document Overview]: {doc_data['summary']}")
            
            for ex in doc_data["excerpts"]:
                section_tag = f" (Section: {ex['breadcrumb']})" if ex.get("breadcrumb") else ""
                prompt_lines.append(f"  [Chunk {ex['chunk_id']}{section_tag} (Score: {ex['similarity_score']})]: {ex['text']}")
                citations.append({
                    "chunk_id": ex["chunk_id"],
                    "document_id": doc_id,
                    "document_title": doc_data["title"],
                    "similarity_score": ex["similarity_score"],
                    "breadcrumb": ex["breadcrumb"],
                    "section_heading": ex["section_heading"]
                })

        formatted_prompt = "\n".join(prompt_lines)
        return {
            "formatted_prompt": formatted_prompt,
            "citations": citations,
            "doc_groups": doc_groups
        }

    def synthesize(
        self, 
        query: str, 
        retrieved_chunks: List[Dict[str, Any]],
        document_summaries: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Generates a grounded answer with parent-context citations.
        """
        if not retrieved_chunks:
            return {
                "answer": "No relevant documents found in authorized namespaces to answer this query.",
                "citations": [],
                "grounded": False
            }

        prompt_meta = self.format_parent_context_prompt(
            query, retrieved_chunks, document_summaries=document_summaries
        )
        
        # Build synthesis text with citation tags
        lead_chunk = retrieved_chunks[0]
        lead_meta = lead_chunk.get("metadata", {})
        lead_title = lead_meta.get("document_title", "Knowledge Base")

        # Synthesize clear answer with citations
        answer_parts = []
        for c in retrieved_chunks:
            t = c.get("text", "").strip()
            cid = c.get("chunk_id", "")
            meta = c.get("metadata", {})
            ref_label = meta.get("breadcrumb") or lead_title
            if t:
                # Add formatted statement with inline citation referencing breadcrumbs
                answer_parts.append(f"{t} [Ref: {ref_label} | Chunk: {cid}]")

        answer_text = " ".join(answer_parts)

        return {
            "answer": answer_text,
            "citations": prompt_meta["citations"],
            "grounded": True,
            "formatted_prompt": prompt_meta["formatted_prompt"]
        }
