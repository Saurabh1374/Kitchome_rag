import os
import tempfile
import pytest

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from src.ingestion.parser import LocalDocumentParser, HTMLToMarkdownConverter
from src.ingestion.cursor import IngestionCursorManager, IngestionCursorRecord
from src.ingestion.queue import IngestionQueueManager
from src.ingestion.controller import IngestionController
from src.ingestion.worker import IngestionWorker, ChunkerWorker, EmbedderWorker
from src.vector_store.base import NamespaceVectorStore
from src.skills_ms.router import SkillsRouter


def test_html_table_and_content_parsing():
    """Verify that LocalDocumentParser extracts HTML headings, lists, and tables into clean GFM Markdown."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head><title>Blender Troubleshooting Guide</title></head>
    <body>
        <nav><a href="/home">Home</a></nav>
        <h1>Blender Troubleshooting</h1>
        <p>Follow these steps if your blender does not start.</p>
        <h2>Error Codes and Solutions</h2>
        <table>
            <tr>
                <th>Error Code</th>
                <th>Symptom</th>
                <th>Resolution</th>
            </tr>
            <tr>
                <td>E01</td>
                <td>Motor Overheated</td>
                <td>Wait 30 minutes for thermal fuse to cool.</td>
            </tr>
            <tr>
                <td>E02</td>
                <td>Jar Not Detected</td>
                <td>Align jar locking tabs with base sensor.</td>
            </tr>
        </table>
        <h2>Maintenance Checklist</h2>
        <ul>
            <li>Inspect drive coupling for wear.</li>
            <li>Lubricate blade bearing every 6 months.</li>
        </ul>
        <footer>Copyright 2026 Blender Corp</footer>
    </body>
    </html>
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "blender_guide.html")
        with open(input_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        parser = LocalDocumentParser()
        doc = parser.parse_and_save(input_file, declared_domain="appliances")

        # 1. Verify destination path is under processed/ directory
        expected_processed_path = os.path.join(tmpdir, "processed", "blender_guide.md")
        assert doc.source_path == expected_processed_path
        assert os.path.exists(expected_processed_path)

        # 2. Verify title extracted from <title> or <h1>
        assert "Blender Troubleshooting" in doc.title

        # 3. Read generated markdown and verify content
        with open(expected_processed_path, "r", encoding="utf-8") as f:
            md_text = f.read()

        # Headings converted to Markdown
        assert "# Blender Troubleshooting" in md_text
        assert "## Error Codes and Solutions" in md_text
        assert "## Maintenance Checklist" in md_text

        # Tables converted to GFM markdown
        assert "| Error Code | Symptom | Resolution |" in md_text
        assert "| :--- | :--- | :--- |" in md_text
        assert "| E01 | Motor Overheated | Wait 30 minutes for thermal fuse to cool. |" in md_text
        assert "| E02 | Jar Not Detected | Align jar locking tabs with base sensor. |" in md_text

        # Noise stripped
        assert "Home" not in md_text
        assert "Copyright 2026" not in md_text

        # Lists converted
        assert "- Inspect drive coupling for wear." in md_text


def test_pdf_parsing_and_processed_storage():
    """Generate a synthetic PDF with headings and tabular text, and verify parsing to .md under processed/."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "airfryer_manual.pdf")

        # Create synthetic PDF using reportlab
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.setTitle("Air Fryer Operating Manual")
        c.drawString(100, 750, "1.0 Safety Instructions")
        c.drawString(100, 720, "Always place the appliance on a horizontal, heat-resistant surface.")
        c.drawString(100, 680, "2.0 Technical Specifications")
        c.drawString(100, 650, "Parameter\tValue")
        c.drawString(100, 630, "Rated Voltage\t120V 60Hz")
        c.drawString(100, 610, "Rated Power\t1500W")
        c.drawString(100, 50, "Page 1 of 1")  # Running footer
        c.save()

        parser = LocalDocumentParser()
        doc = parser.parse_and_save(pdf_path, declared_domain="appliances")

        expected_processed_path = os.path.join(tmpdir, "processed", "airfryer_manual.md")
        assert doc.source_path == expected_processed_path
        assert os.path.exists(expected_processed_path)

        with open(expected_processed_path, "r", encoding="utf-8") as f:
            md_text = f.read()

        # Check section detection
        assert "1.0 Safety Instructions" in md_text
        assert "Always place the appliance on a horizontal" in md_text
        assert "Rated Voltage" in md_text
        assert "1500W" in md_text

        # Running page footer should be stripped
        assert "Page 1 of 1" not in md_text


def test_json_specification_sheet_to_markdown():
    """Verify that JSON specification sheets and recipes are formatted into clean Markdown."""
    data = {
        "title": "KitchenAid Artisan Stand Mixer",
        "description": "Commercial grade stand mixer for kneading and whipping.",
        "specifications": {
            "capacity": "5 Quart",
            "speeds": "10 Speeds",
            "wattage": "325 Watts"
        }
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = os.path.join(tmpdir, "mixer.json")
        with open(json_file, "w", encoding="utf-8") as f:
            import json
            json.dump(data, f)

        parser = LocalDocumentParser()
        doc = parser.parse_and_save(json_file, declared_domain="appliances")

        with open(doc.source_path, "r", encoding="utf-8") as f:
            md_text = f.read()

        assert "# KitchenAid Artisan Stand Mixer" in md_text
        assert "## Specifications" in md_text
        assert "| Parameter | Value |" in md_text
        assert "| Capacity | 5 Quart |" in md_text
        assert "| Speeds | 10 Speeds |" in md_text


def test_cursor_lifecycle_parsed_to_active():
    """
    End-to-end verification of cursor lifecycle:
    1. ChunkerWorker runs Stage 1 -> cursor has status='PARSED' and file_path=processed/*.md
    2. EmbedderWorker finishes Stage 5 -> cursor status flips to 'ACTIVE'
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cursor_db = os.path.join(tmpdir, "cursor.db")
        queue_db = os.path.join(tmpdir, "queue.db")
        cursor_mgr = IngestionCursorManager(db_path=cursor_db)
        queue_mgr = IngestionQueueManager(db_path=queue_db)
        vector_store = NamespaceVectorStore()
        router = SkillsRouter()

        html_content = "<h1>Coffee Maker Manual</h1><p>Fill reservoir with filtered water up to the 12 cup line.</p>"
        raw_file = os.path.join(tmpdir, "coffee_maker.html")
        with open(raw_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        controller = IngestionController(
            cursor_manager=cursor_mgr,
            queue_manager=queue_mgr,
            vector_store=vector_store,
            router=router,
            data_root=tmpdir
        )

        res = controller.intake_document(
            file_path=raw_file,
            title="Coffee Maker Manual",
            declared_domain="appliances"
        )
        job_id = res["job_id"]
        document_id = res["document_id"]

        # Step 1: ChunkerWorker runs Stage 1-3
        chunker = ChunkerWorker(
            worker_id="chunker_1",
            queue_manager=queue_mgr,
            cursor_manager=cursor_mgr,
            router=router
        )
        chunk_res = chunker.process_next_job()
        assert chunk_res["status"] == "READY_FOR_EMBED"

        # Verify cursor record is in status 'PARSED' and points to the processed .md file
        parsed_cursor = cursor_mgr.get_by_document_id(document_id)
        assert parsed_cursor is not None
        assert parsed_cursor.status == "PARSED"
        assert parsed_cursor.file_path.endswith(".md")
        assert "processed" in parsed_cursor.file_path
        assert os.path.exists(parsed_cursor.file_path)

        # Step 2: EmbedderWorker processes batch and finishes Stage 5
        embedder = EmbedderWorker(
            worker_id="embedder_1",
            cursor_manager=cursor_mgr,
            queue_manager=queue_mgr,
            vector_store=vector_store
        )
        batch_res = embedder.process_next_batch()
        assert batch_res["status"] == "COMPLETED"
        assert batch_res["barrier_resolved"] is True

        # Verify cursor record has officially transitioned to 'ACTIVE'
        active_cursor = cursor_mgr.get_by_document_id(document_id)
        assert active_cursor is not None
        assert active_cursor.status == "ACTIVE"
        assert active_cursor.is_latest is True


def test_pdf_xy_matrix_table_extraction():
    """Verify that pdfplumber extracts 2D X-Y matrix tables with multi-line cell normalization and cross-reference linearization."""
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "blender_matrix_guide.pdf")

        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = [
            Paragraph("Commercial Blender Manual", styles["Heading1"]),
            Spacer(1, 10),
            Paragraph("1.0 Operating Programs Matrix", styles["Heading2"]),
            Paragraph("Refer to the chart below to configure jar size and blend cycle.", styles["Normal"]),
            Spacer(1, 10),
            Table([
                ["Preset Program", "1.0L Jar", "1.5L Jar", "2.0L Jar"],
                ["Smoothies", "Speed 10\n(45s)", "Speed 10\n(60s)", "Speed 10\n(75s)"],
                ["Hot Soups", "Speed 8\n(5m)", "Speed 9\n(6m)", "Speed 10\n(6m30s)"],
                ["Frozen Dessert", "Speed 10\n(30s)", "Speed 10\n(45s)", "Speed 10\n(55s)"],
            ], style=[
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ]),
            Spacer(1, 12),
            Paragraph("2.0 Cleaning & Sanitization", styles["Heading2"]),
            Paragraph("Flush container with warm water immediately after use.", styles["Normal"]),
        ]
        doc.build(elements)

        parser = LocalDocumentParser()
        parsed_doc = parser.parse_and_save(pdf_path, declared_domain="appliances")

        expected_processed_path = os.path.join(tmpdir, "processed", "blender_matrix_guide.md")
        assert parsed_doc.source_path == expected_processed_path
        assert os.path.exists(expected_processed_path)

        with open(expected_processed_path, "r", encoding="utf-8") as f:
            md_text = f.read()

        # 1. Check Section Headings
        assert "## 1.0 Operating Programs Matrix" in md_text
        assert "## 2.0 Cleaning & Sanitization" in md_text

        # 2. Check GFM Table Header & Rows
        assert "| Preset Program | 1.0L Jar | 1.5L Jar | 2.0L Jar |" in md_text
        assert "| :--- | :--- | :--- | :--- |" in md_text

        # Multi-line cell text should be collapsed to single space-separated strings
        assert "| Smoothies | Speed 10 (45s) | Speed 10 (60s) | Speed 10 (75s) |" in md_text
        assert "| Hot Soups | Speed 8 (5m) | Speed 9 (6m) | Speed 10 (6m30s) |" in md_text
        assert "| Frozen Dessert | Speed 10 (30s) | Speed 10 (45s) | Speed 10 (55s) |" in md_text

        # 3. Check X-Y Cross-Reference Matrix Linearization
        assert "**Cross-Reference Matrix:**" in md_text
        assert "- **Smoothies** -> [1.0L Jar]: Speed 10 (45s) | [1.5L Jar]: Speed 10 (60s) | [2.0L Jar]: Speed 10 (75s)" in md_text
        assert "- **Hot Soups** -> [1.0L Jar]: Speed 8 (5m) | [1.5L Jar]: Speed 9 (6m) | [2.0L Jar]: Speed 10 (6m30s)" in md_text
        assert "- **Frozen Dessert** -> [1.0L Jar]: Speed 10 (30s) | [1.5L Jar]: Speed 10 (45s) | [2.0L Jar]: Speed 10 (55s)" in md_text

        # 4. Check Reading Order (Narrative text before table, table before outro text)
        idx_intro = md_text.find("Refer to the chart below")
        idx_table = md_text.find("| Preset Program |")
        idx_matrix = md_text.find("**Cross-Reference Matrix:**")
        idx_outro = md_text.find("## 2.0 Cleaning & Sanitization")

        assert idx_intro != -1
        assert idx_table != -1
        assert idx_matrix != -1
        assert idx_outro != -1
        assert idx_intro < idx_table < idx_matrix < idx_outro


def test_format_table_to_gfm_and_matrix_edge_cases():
    """Verify table formatting edge cases: 2-column tables, empty cells, and pipe characters."""
    parser = LocalDocumentParser()

    # Case 1: 2-column key-value table (should format clean GFM without cross-reference matrix)
    raw_2col = [
        ["Specification", "Rating"],
        ["Voltage", "120V AC"],
        ["Wattage", "1200W"]
    ]
    md_2col = parser._format_table_to_gfm_and_matrix(raw_2col)
    assert "| Specification | Rating |" in md_2col
    assert "| Voltage | 120V AC |" in md_2col
    assert "**Cross-Reference Matrix:**" not in md_2col

    # Case 2: Multi-line cells and pipes in cell text
    raw_with_pipes = [
        ["Option", "Sub-option | Mode", "Notes"],
        ["Pulse", "Low | High", "Use for\nchopping"],
    ]
    md_pipes = parser._format_table_to_gfm_and_matrix(raw_with_pipes)
    # Pipe should be escaped
    assert "Sub-option \\| Mode" in md_pipes
    assert "Low \\| High" in md_pipes
    assert "Use for chopping" in md_pipes

    # Case 3: Empty / malformed table
    assert parser._format_table_to_gfm_and_matrix([]) == ""
    assert parser._format_table_to_gfm_and_matrix([["Only One Row"]]) == ""

