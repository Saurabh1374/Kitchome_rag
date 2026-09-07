import os
import json
import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class RawDocument(BaseModel):
    document_id: str
    title: str
    source_path: str
    content: str
    declared_domain: Optional[str] = None
    access_tier: str = "free"
    tenant_id: str = "global"
    clearance_level: int = 1
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentLoader:
    """
    Multi-format document loader for reading domain content files.
    """
    @staticmethod
    def load_file(file_path: str, declared_domain: Optional[str] = None) -> RawDocument:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document file not found: {file_path}")

        filename = os.path.basename(file_path)
        ext = os.path.splitext(filename)[1].lower()
        title = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").title()

        if ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    content = data.get("content") or data.get("text") or json.dumps(data)
                    title = data.get("title", title)
                    declared_domain = declared_domain or data.get("domain") or data.get("category")
                else:
                    content = json.dumps(data)
        elif ext in [".txt", ".md", ".markdown"]:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            # Fallback for plain text parsing
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

        doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, file_path))

        if not declared_domain:
            parent_dir = os.path.basename(os.path.dirname(file_path))
            if parent_dir and parent_dir not in ("data", ".", ""):
                declared_domain = parent_dir

        return RawDocument(
            document_id=doc_id,
            title=title,
            source_path=file_path,
            content=content,
            declared_domain=declared_domain,
            metadata={"filename": filename, "extension": ext}
        )

    @classmethod
    def load_directory(cls, dir_path: str, declared_domain: Optional[str] = None) -> List[RawDocument]:
        documents = []
        if not os.path.exists(dir_path):
            return documents

        valid_extensions = {".json", ".txt", ".md", ".markdown"}

        for root, _, files in os.walk(dir_path):
            for file in sorted(files):
                if file.startswith("."):
                    continue
                ext = os.path.splitext(file)[1].lower()
                if ext not in valid_extensions:
                    continue
                file_path = os.path.join(root, file)
                try:
                    doc = cls.load_file(file_path, declared_domain=declared_domain)
                    documents.append(doc)
                except Exception as e:
                    print(f"Skipping {file_path} due to load error: {e}")
        return documents
