import os
import re
import json
import uuid
import hashlib
from html.parser import HTMLParser
from typing import Optional, List, Dict, Any, Tuple

from .loader import RawDocument

try:
    import pdfplumber
except ImportError:
    pdfplumber = None  # type: ignore

try:
    import pypdf
except ImportError:
    pypdf = None  # type: ignore


class HTMLToMarkdownConverter(HTMLParser):
    """
    Fast, standard-library HTML-to-Markdown converter.
    Strips noise, preserves semantic headings (#, ##), lists, and converts
    HTML <table> elements into GitHub-Flavored Markdown (GFM) tables.
    """
    def __init__(self):
        super().__init__()
        self.output: List[str] = []
        self.ignore_tags = {"script", "style", "nav", "footer", "header", "aside", "noscript"}
        self.ignore_depth = 0
        self.extracted_title: Optional[str] = None
        self._in_title = False

        # Table parsing state
        self.in_table = False
        self.table_rows: List[Tuple[List[str], bool]] = []  # List of (cells, is_header)
        self.current_row: List[str] = []
        self.current_cell: List[str] = []
        self.is_header_row = False

        # List parsing state
        self.list_stack: List[str] = []  # 'ul' or 'ol'
        self.ol_index = 0

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]):
        tag = tag.lower()
        if tag in self.ignore_tags:
            self.ignore_depth += 1
            return

        if self.ignore_depth > 0:
            return

        if tag == "title":
            self._in_title = True
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag[1])
            self.output.append(f"\n\n{'#' * level} ")
        elif tag == "p":
            self.output.append("\n\n")
        elif tag == "br":
            self.output.append("\n")
        elif tag == "ul":
            self.list_stack.append("ul")
            self.output.append("\n")
        elif tag == "ol":
            self.list_stack.append("ol")
            self.ol_index = 1
            self.output.append("\n")
        elif tag == "li":
            if self.list_stack and self.list_stack[-1] == "ol":
                self.output.append(f"\n{self.ol_index}. ")
                self.ol_index += 1
            else:
                self.output.append("\n- ")
        elif tag == "table":
            self.in_table = True
            self.table_rows = []
            self.output.append("\n\n")
        elif tag == "tr":
            self.current_row = []
            self.is_header_row = False
        elif tag in ("th", "td"):
            self.current_cell = []
            if tag == "th":
                self.is_header_row = True

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag in self.ignore_tags:
            if self.ignore_depth > 0:
                self.ignore_depth -= 1
            return

        if self.ignore_depth > 0:
            return

        if tag == "title":
            self._in_title = False
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6", "p"):
            self.output.append("\n")
        elif tag in ("ul", "ol"):
            if self.list_stack:
                self.list_stack.pop()
            self.output.append("\n")
        elif tag in ("th", "td"):
            cell_text = " ".join("".join(self.current_cell).split()).strip()
            # Replace inner pipes to prevent breaking markdown table formatting
            cell_text = cell_text.replace("|", "&#124;")
            self.current_row.append(cell_text)
            self.current_cell = []
        elif tag == "tr":
            if self.current_row:
                self.table_rows.append((self.current_row, self.is_header_row))
            self.current_row = []
        elif tag == "table":
            self._render_gfm_table()
            self.in_table = False

    def handle_data(self, data: str):
        if self.ignore_depth > 0:
            return

        if self._in_title and not self.extracted_title:
            self.extracted_title = data.strip()
            return

        if self.in_table:
            self.current_cell.append(data)
        else:
            self.output.append(data)

    def _render_gfm_table(self):
        """Renders accumulated table rows as GitHub Flavored Markdown."""
        if not self.table_rows:
            return

        max_cols = max(len(row[0]) for row in self.table_rows)
        if max_cols == 0:
            return

        normalized_rows: List[List[str]] = []
        for cells, _ in self.table_rows:
            # Pad row if uneven
            padded = cells + [""] * (max_cols - len(cells))
            normalized_rows.append(padded)

        first_cells, first_is_header = self.table_rows[0]

        if first_is_header:
            header = normalized_rows[0]
            data_rows = normalized_rows[1:]
        else:
            header = [f"Column {i + 1}" for i in range(max_cols)]
            data_rows = normalized_rows

        table_lines = []
        table_lines.append("| " + " | ".join(header) + " |")
        table_lines.append("| " + " | ".join([":---"] * max_cols) + " |")
        for row in data_rows:
            table_lines.append("| " + " | ".join(row) + " |")

        self.output.append("\n".join(table_lines) + "\n\n")

    def get_markdown(self) -> str:
        text = "".join(self.output)
        # Normalize redundant newlines
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        return text


class LocalDocumentParser:
    """
    Local multi-format document parser.
    Converts PDF, HTML, JSON, and Text documents into standardized Markdown (.md),
    persists them under a sibling 'processed/' directory, and returns RawDocument objects.
    """
    def __init__(self):
        pass

    def parse_file(self, file_path: str, declared_domain: Optional[str] = None) -> Tuple[str, str, Dict[str, Any]]:
        """
        Parses raw input file and returns (markdown_content, title, metadata).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document file not found: {file_path}")

        filename = os.path.basename(file_path)
        ext = os.path.splitext(filename)[1].lower()
        title = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").title()
        metadata: Dict[str, Any] = {
            "source_format": ext.lstrip("."),
            "raw_filename": filename
        }

        if ext in (".html", ".htm"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                html_text = f.read()
            converter = HTMLToMarkdownConverter()
            converter.feed(html_text)
            md_content = converter.get_markdown()
            if converter.extracted_title:
                title = converter.extracted_title

        elif ext == ".pdf":
            md_content, pdf_title = self._parse_pdf(file_path)
            if pdf_title:
                title = pdf_title

        elif ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            md_content, json_title = self._format_json_to_markdown(data, title)
            if json_title:
                title = json_title

        else:
            # Plain text / Markdown pass-through
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                md_content = f.read().strip()

        return md_content, title, metadata

    def _parse_pdf(self, file_path: str) -> Tuple[str, Optional[str]]:
        """
        Extracts text, structured tables, and X-Y matrix cross-references from a PDF file.
        Uses pdfplumber as the primary extractor for precise layout and table extraction,
        with fallback to pypdf if pdfplumber is unavailable or encounters an error.
        """
        if pdfplumber is not None:
            try:
                return self._parse_pdf_with_pdfplumber(file_path)
            except Exception:
                pass
        return self._parse_pdf_fallback_pypdf(file_path)

    def _parse_pdf_with_pdfplumber(self, file_path: str) -> Tuple[str, Optional[str]]:
        """Extracts text and tables in visual reading order using pdfplumber."""
        pdf_title: Optional[str] = None
        pages_md: List[str] = []

        with pdfplumber.open(file_path) as pdf:
            if pdf.metadata:
                raw_title = pdf.metadata.get("Title") or pdf.metadata.get("title")
                if raw_title:
                    pdf_title = str(raw_title).strip()

            for page in pdf.pages:
                tables = page.find_tables()
                if tables:
                    sorted_tables = sorted(tables, key=lambda t: t.bbox[1])
                    current_top = 0.0
                    page_blocks: List[str] = []

                    for t in sorted_tables:
                        # Non-table text above this table
                        if t.bbox[1] > current_top:
                            crop_box = (0, current_top, page.width, t.bbox[1])
                            band_text = page.crop(crop_box).extract_text() or ""
                            cleaned_band = self._clean_text_band(band_text)
                            if cleaned_band:
                                page_blocks.append(cleaned_band)

                        # Extract and format table
                        raw_table = t.extract()
                        table_md = self._format_table_to_gfm_and_matrix(raw_table)
                        if table_md:
                            page_blocks.append(table_md)

                        current_top = max(current_top, t.bbox[3])

                    # Non-table text below the last table
                    if current_top < page.height:
                        crop_box = (0, current_top, page.width, page.height)
                        band_text = page.crop(crop_box).extract_text() or ""
                        cleaned_band = self._clean_text_band(band_text)
                        if cleaned_band:
                            page_blocks.append(cleaned_band)

                    page_text = "\n\n".join(page_blocks).strip()
                    if page_text:
                        pages_md.append(page_text)
                else:
                    text = page.extract_text() or ""
                    cleaned_text = self._clean_text_band(text)
                    if cleaned_text:
                        pages_md.append(cleaned_text)

        full_content = "\n\n".join(pages_md)
        full_content = re.sub(r'\n{3,}', '\n\n', full_content).strip()
        return full_content, pdf_title

    def _parse_pdf_fallback_pypdf(self, file_path: str) -> Tuple[str, Optional[str]]:
        """Fallback PDF extraction using pypdf."""
        if pypdf is None:
            raise RuntimeError("Neither pdfplumber nor pypdf is installed to parse PDF files.")

        reader = pypdf.PdfReader(file_path)
        pdf_title = None
        if reader.metadata and reader.metadata.title:
            pdf_title = str(reader.metadata.title).strip()

        pages_text: List[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            cleaned = self._clean_text_band(text)
            if cleaned:
                pages_text.append(cleaned)

        full_content = "\n\n".join(pages_text)
        full_content = re.sub(r'\n{3,}', '\n\n', full_content).strip()
        return full_content, pdf_title

    def _clean_text_band(self, text: str) -> str:
        """Strips running headers/footers, marks section headings, and detects tabular lines."""
        if not text:
            return ""
        lines = [l.strip() for l in text.split("\n")]
        cleaned_lines: List[str] = []

        for line in lines:
            if not line:
                continue
            # Strip running page footers (e.g. "Page 1 of 12" or "1 / 12")
            if re.match(r'^(page\s+\d+(\s+of\s+\d+)?|\d+\s*/\s*\d+)$', line, re.IGNORECASE):
                continue

            # Detect numbered section headers: "1.0 Safety" -> "## 1.0 Safety"
            if re.match(r'^\d+(\.\d+)*\s+[A-Z]', line) and len(line) < 80:
                cleaned_lines.append(f"\n## {line}\n")
            # Detect tabular lines with tabs or multi-spaces
            elif "\t" in line or re.search(r'\s{3,}', line):
                cols = [c.strip() for c in re.split(r'\t|\s{3,}', line) if c.strip()]
                if len(cols) >= 2:
                    cleaned_lines.append("| " + " | ".join(cols) + " |")
                else:
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _format_table_to_gfm_and_matrix(self, raw_table: List[List[Optional[str]]]) -> str:
        """
        Formats a 2D raw table from pdfplumber into GitHub-Flavored Markdown (GFM).
        Handles multi-line wrapped cells by collapsing whitespace and internal newlines.
        For multi-column tables (>= 3 columns), generates a semantic Cross-Reference Matrix
        linearization to maximize RAG vector retrieval and LLM context fidelity.
        """
        if not raw_table or len(raw_table) < 2:
            return ""

        cleaned_rows: List[List[str]] = []
        for row in raw_table:
            if not row:
                continue
            cleaned_cells: List[str] = []
            for cell in row:
                if cell is None:
                    cell_text = ""
                else:
                    # Collapse wrapped lines and whitespace inside cell into a single clean line
                    cell_text = " ".join(str(cell).split()).strip()
                    # Escape pipes to avoid breaking markdown table formatting
                    cell_text = cell_text.replace("|", "\\|")
                cleaned_cells.append(cell_text)

            if any(c for c in cleaned_cells):
                cleaned_rows.append(cleaned_cells)

        if len(cleaned_rows) < 2:
            return ""

        headers = cleaned_rows[0]
        num_cols = len(headers)
        if num_cols < 2:
            return ""

        lines: List[str] = []
        # GFM Table Header
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join([":---"] * num_cols) + " |")

        for row in cleaned_rows[1:]:
            padded_row = (row + [""] * num_cols)[:num_cols]
            lines.append("| " + " | ".join(padded_row) + " |")

        # X-Y Matrix Cross-Reference Linearization:
        # For tables with >= 3 columns (e.g. Row Program vs. Container Capacities),
        # generate coordinate tuples: - **Row** -> [Col1]: Val1 | [Col2]: Val2
        if num_cols >= 3 and len(cleaned_rows) > 1:
            col_labels = headers[1:]
            matrix_lines: List[str] = []
            for row in cleaned_rows[1:]:
                row_header = row[0]
                if not row_header:
                    continue
                row_entries = []
                for col_idx, col_label in enumerate(col_labels, start=1):
                    val = row[col_idx] if col_idx < len(row) else ""
                    if val:
                        col_name = col_label if col_label else f"Col {col_idx}"
                        row_entries.append(f"[{col_name}]: {val}")
                if row_entries:
                    matrix_lines.append(f"- **{row_header}** -> " + " | ".join(row_entries))

            if matrix_lines:
                lines.append("\n**Cross-Reference Matrix:**")
                lines.extend(matrix_lines)

        return "\n".join(lines)

    def _format_json_to_markdown(self, data: Any, default_title: str) -> Tuple[str, str]:
        """Converts structured JSON dictionaries (e.g. recipes or appliance specs) to Markdown."""
        if not isinstance(data, dict):
            return f"```json\n{json.dumps(data, indent=2)}\n```", default_title

        title = data.get("title") or default_title
        lines = [f"# {title}\n"]

        if "description" in data:
            lines.append(f"{data['description']}\n")

        # Appliance specifications or metadata table
        specs = data.get("specifications") or data.get("specs")
        if isinstance(specs, dict):
            lines.append("## Specifications\n")
            lines.append("| Parameter | Value |")
            lines.append("| :--- | :--- |")
            for k, v in specs.items():
                k_clean = str(k).replace("_", " ").title()
                lines.append(f"| {k_clean} | {v} |")
            lines.append("")

        # Ingredients list (for recipes)
        ingredients = data.get("ingredients")
        if isinstance(ingredients, list):
            lines.append("## Ingredients\n")
            for item in ingredients:
                lines.append(f"- {item}")
            lines.append("")

        # Instructions / Steps
        steps = data.get("instructions") or data.get("steps") or data.get("directions")
        if isinstance(steps, list):
            lines.append("## Instructions\n")
            for idx, step in enumerate(steps, 1):
                lines.append(f"{idx}. {step}")
            lines.append("")

        # Any general content field
        if "content" in data and isinstance(data["content"], str):
            lines.append(data["content"])

        md_text = "\n".join(lines).strip()
        return md_text, title

    def parse_and_save(
        self, 
        file_path: str, 
        output_dir: Optional[str] = None, 
        declared_domain: Optional[str] = None
    ) -> RawDocument:
        """
        Parses raw input file, writes clean Markdown output into a sibling 'processed/' directory,
        and returns a RawDocument pointing to the processed .md file.
        """
        md_content, title, metadata = self.parse_file(file_path, declared_domain=declared_domain)

        abs_file_path = os.path.abspath(file_path)
        parent_dir = os.path.dirname(abs_file_path)

        if output_dir:
            processed_dir = os.path.abspath(output_dir)
        elif os.path.basename(parent_dir) == "processed":
            processed_dir = parent_dir
        else:
            processed_dir = os.path.join(parent_dir, "processed")

        os.makedirs(processed_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        processed_path = os.path.join(processed_dir, f"{base_name}.md")

        # Write clean markdown
        with open(processed_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, abs_file_path))

        if not declared_domain:
            # Infer domain from parent directory if not specified
            dir_for_domain = parent_dir if os.path.basename(parent_dir) != "processed" else os.path.dirname(parent_dir)
            domain_candidate = os.path.basename(dir_for_domain)
            if domain_candidate and domain_candidate not in ("data", ".", ""):
                declared_domain = domain_candidate

        return RawDocument(
            document_id=doc_id,
            title=title,
            source_path=processed_path,
            content=md_content,
            declared_domain=declared_domain,
            metadata={
                **metadata,
                "raw_file_path": abs_file_path,
                "processed_file_path": processed_path,
                "content_hash": hashlib.sha256(md_content.encode("utf-8")).hexdigest()
            }
        )
