import pytest
from src.ingestion.loader import RawDocument
from src.ingestion.chunker import TextChunker
from src.rag.generator import GroundedGenerator

def test_sentence_aware_sliding_overlap_continuity():
    """
    Verify that TextChunker carries forward trailing complete sentences without
    chopping words or creating mid-sentence fragments.
    """
    # Create paragraphs where each paragraph consists of clear sentences
    para1 = "This is sentence one. This is sentence two. This is sentence three."
    para2 = "This is sentence four. This is sentence five. This is sentence six."
    para3 = "This is sentence seven. This is sentence eight. This is sentence nine."
    content = f"{para1}\n\n{para2}\n\n{para3}"

    doc = RawDocument(
        document_id="doc_overlap_test",
        title="Continuity Guide",
        source_path="/test/continuity.md",
        content=content
    )

    # Configure chunk_size small enough so each chunk can hold roughly 1.5 paragraphs,
    # with chunk_overlap of 45 characters (enough for one full sentence like 'This is sentence three.')
    chunker = TextChunker(chunk_size=120, chunk_overlap=45)
    chunks = chunker.chunk_document(doc, namespace="appliances")

    assert len(chunks) >= 2

    # Chunk 0 should have has_overlap = False
    assert chunks[0].metadata["has_overlap"] is False
    assert chunks[0].metadata["overlap_char_count"] == 0

    # Chunk 1 should have has_overlap = True
    assert chunks[1].metadata["has_overlap"] is True
    assert chunks[1].metadata["overlap_char_count"] > 0

    # The start of Chunk 1 must be a complete sentence that was at the end of Chunk 0
    chunk0_text = chunks[0].text
    chunk1_text = chunks[1].text

    # Extract the overlap text (up to the double newline)
    first_part_chunk1 = chunk1_text.split("\n\n")[0]
    assert first_part_chunk1 in chunk0_text
    # Ensure it's not a chopped fragment: should start with a capitalized word and end with a period
    assert first_part_chunk1[0].isupper()
    assert first_part_chunk1.endswith(".")


def test_zero_chunk_overlap_disables_overlap():
    """Verify that setting chunk_overlap=0 results in no overlap between chunks."""
    content = "Paragraph alpha has some text.\n\nParagraph beta has more text.\n\nParagraph gamma finishes it."
    doc = RawDocument(
        document_id="doc_zero_overlap",
        title="Zero Overlap Guide",
        source_path="/test/zero.md",
        content=content
    )

    chunker = TextChunker(chunk_size=40, chunk_overlap=0)
    chunks = chunker.chunk_document(doc, namespace="appliances")

    assert len(chunks) >= 2
    for c in chunks:
        assert c.metadata["has_overlap"] is False
        assert c.metadata["overlap_char_count"] == 0


def test_hierarchical_markdown_breadcrumbs_tracking():
    """
    Verify that nested markdown headings (#, ##, ###) are dynamically tracked,
    scoped correctly to parent levels, and recorded into metadata.
    """
    content = (
        "# Dishwasher Manual\n\n"
        "Introduction text for the dishwasher.\n\n"
        "## Safety Precautions\n\n"
        "General safety warning information.\n\n"
        "### Electrical Requirements\n\n"
        "Ensure power cord is grounded properly.\n\n"
        "### Water Connection\n\n"
        "Verify hot water supply pressure is 20-120 psi.\n\n"
        "## Troubleshooting\n\n"
        "Common issues and solutions."
    )

    doc = RawDocument(
        document_id="doc_breadcrumbs_test",
        title="Dishwasher Manual",
        source_path="/test/dishwasher.md",
        content=content
    )

    # Chunk size large enough to capture sections
    chunker = TextChunker(chunk_size=90, chunk_overlap=20)
    chunks = chunker.chunk_document(doc, namespace="appliances")

    # Find chunk discussing electrical requirements
    elec_chunks = [c for c in chunks if "power cord" in c.text]
    assert len(elec_chunks) == 1
    elec_meta = elec_chunks[0].metadata

    assert "Dishwasher Manual" in elec_meta["breadcrumb"]
    assert "Safety Precautions" in elec_meta["breadcrumb"]
    assert "Electrical Requirements" in elec_meta["breadcrumb"]
    assert elec_meta["section_heading"] == "Electrical Requirements"
    assert "Electrical Requirements" in elec_meta["heading_hierarchy"]

    # Find chunk discussing water connection (sibling subsection)
    water_chunks = [c for c in chunks if "hot water supply" in c.text]
    assert len(water_chunks) == 1
    water_meta = water_chunks[0].metadata

    assert "Water Connection" in water_meta["breadcrumb"]
    assert "Electrical Requirements" not in water_meta["breadcrumb"]
    assert water_meta["section_heading"] == "Water Connection"

    # Find chunk discussing troubleshooting (sibling of Safety Precautions)
    trouble_chunks = [c for c in chunks if "Common issues" in c.text]
    assert len(trouble_chunks) == 1
    trouble_meta = trouble_chunks[0].metadata

    assert "Troubleshooting" in trouble_meta["breadcrumb"]
    assert "Safety Precautions" not in trouble_meta["breadcrumb"]
    assert trouble_meta["section_heading"] == "Troubleshooting"


def test_rag_generator_incorporates_breadcrumbs():
    """Verify that GroundedGenerator formats citations and inline references using breadcrumbs."""
    generator = GroundedGenerator()

    retrieved_chunks = [
        {
            "chunk_id": "chk_001",
            "document_id": "doc_101",
            "text": "Disconnect power before servicing heating element.",
            "similarity_score": 0.88,
            "metadata": {
                "document_title": "Oven Guide",
                "breadcrumb": "Oven Guide > Safety Precautions > Electrical Isolation",
                "section_heading": "Electrical Isolation"
            }
        }
    ]

    res = generator.synthesize("how to safely service heating element", retrieved_chunks)

    assert res["grounded"] is True
    assert len(res["citations"]) == 1
    assert res["citations"][0]["breadcrumb"] == "Oven Guide > Safety Precautions > Electrical Isolation"
    assert res["citations"][0]["section_heading"] == "Electrical Isolation"

    # Verify that the inline reference in the synthesized answer includes the breadcrumb
    assert "[Ref: Oven Guide > Safety Precautions > Electrical Isolation | Chunk: chk_001]" in res["answer"]
    assert "Section: Oven Guide > Safety Precautions > Electrical Isolation" in res["formatted_prompt"]
