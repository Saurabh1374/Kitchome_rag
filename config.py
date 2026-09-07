import os
from typing import Optional
from pydantic import BaseModel, Field

def mask_secret(val: Optional[str], visible_chars: int = 4, full_mask: bool = False) -> str:
    """Masks secret strings for safe logging and string representation."""
    if not val:
        return "None" if val is None else ""
    if full_mask or len(val) <= visible_chars:
        return "********"
    return f"{val[:visible_chars]}...{'*' * 6}"

class IngestionConfig(BaseModel):
    chunk_size: int = Field(default=300, description="Target chunk character length")
    chunk_overlap: int = Field(default=50, description="Character overlap between consecutive chunks")
    embedding_dimension: int = Field(default=384, description="Vector embedding dimension size")

    def get_masked_dict(self) -> dict:
        return self.model_dump()

class EmbedderConfig(BaseModel):
    provider: str = Field(
        default_factory=lambda: os.getenv("EMBEDDER_PROVIDER", "hash"),
        description="Embedding provider: 'hash', 'huggingface', 'huggingface_api', 'ollama', 'openai', 'fastembed'"
    )
    model_name: str = Field(
        default_factory=lambda: os.getenv("EMBEDDER_MODEL_NAME", "BAAI/bge-small-en-v1.5"),
        description="Model identifier or repository name"
    )
    dimension: int = Field(
        default_factory=lambda: int(os.getenv("EMBEDDER_DIMENSION", "384")),
        description="Vector embedding dimension size"
    )
    batch_size: int = Field(
        default_factory=lambda: int(os.getenv("EMBEDDER_BATCH_SIZE", "32")),
        description="Batch size for embedding calls"
    )
    api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("OPENAI_API_KEY"),
        description="API key or token for cloud / inference API providers"
    )
    base_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("EMBEDDER_BASE_URL", "http://localhost:11434"),
        description="Base URL for local or custom endpoints (e.g. Ollama)"
    )
    device: str = Field(
        default_factory=lambda: os.getenv("EMBEDDER_DEVICE", "cpu"),
        description="Compute device: 'cpu', 'mps', 'cuda'"
    )
    normalize: bool = Field(
        default=True,
        description="Whether to L2-normalize vectors"
    )
    timeout_seconds: float = Field(
        default=30.0,
        description="Network / compute timeout in seconds"
    )
    max_batch_retries: int = Field(
        default=3,
        description="Maximum attempts per batch before quarantining to DLQ"
    )

    def get_masked_dict(self) -> dict:
        d = self.model_dump()
        if d.get("api_key"):
            d["api_key"] = mask_secret(d["api_key"])
        return d

    def __repr__(self) -> str:
        return f"EmbedderConfig({self.get_masked_dict()})"

    __str__ = __repr__

class SkillsMSConfig(BaseModel):
    registry_file: str = Field(default="skills_registry.json")
    default_namespace: str = Field(default="general_home")

    def get_masked_dict(self) -> dict:
        return self.model_dump()

class PGVectorConfig(BaseModel):
    host: str = Field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    port: int = Field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    db_name: str = Field(default_factory=lambda: os.getenv("POSTGRES_DB", "kitchome_rag"))
    user: str = Field(default_factory=lambda: os.getenv("POSTGRES_USER", "postgres"))
    password: str = Field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", ""), description="PostgreSQL password")
    vector_table: str = Field(default="kitchome_vector_chunks")

    def get_masked_dict(self) -> dict:
        d = self.model_dump()
        if d.get("password"):
            d["password"] = mask_secret(d["password"])
        return d

    def __repr__(self) -> str:
        return f"PGVectorConfig({self.get_masked_dict()})"

    __str__ = __repr__

class WorkerPoolConfig(BaseModel):
    role: str = Field(default_factory=lambda: os.getenv("WORKER_ROLE", "all"), description="'chunker', 'embedder', or 'all'")
    mode: str = Field(default_factory=lambda: os.getenv("WORKER_MODE", "process"), description="'process' for multi-core or 'thread' for low-RAM")
    chunker_concurrency: int = Field(default_factory=lambda: int(os.getenv("CHUNKER_CONCURRENCY", "1")), description="Number of Chunker workers")
    embedder_concurrency: int = Field(default_factory=lambda: int(os.getenv("EMBEDDER_CONCURRENCY", "2")), description="Number of Embedder workers")
    embedder_batch_size: int = Field(default_factory=lambda: int(os.getenv("EMBEDDER_BATCH_SIZE", "32")), description="Chunks per batch claim")
    lease_duration_seconds: float = Field(default_factory=lambda: float(os.getenv("WORKER_LEASE_SECONDS", "120.0")), description="Chunker lease timeout")
    embedder_lease_seconds: float = Field(default_factory=lambda: float(os.getenv("EMBEDDER_LEASE_SECONDS", "60.0")), description="Embedder batch lease timeout")
    poll_interval_seconds: float = Field(default_factory=lambda: float(os.getenv("WORKER_POLL_INTERVAL", "1.0")), description="Idle sleep duration in seconds")

    def get_masked_dict(self) -> dict:
        return self.model_dump()

class AppConfig(BaseModel):
    data_dir: str = os.path.join(os.path.dirname(__file__), "data")
    vector_db_path: str = os.path.join(os.path.dirname(__file__), "vector_store_data.json")
    ingestion: IngestionConfig = IngestionConfig()
    embedder: EmbedderConfig = EmbedderConfig()
    skills_ms: SkillsMSConfig = SkillsMSConfig()
    pgvector: PGVectorConfig = PGVectorConfig()
    worker: WorkerPoolConfig = WorkerPoolConfig()

    def get_masked_dict(self) -> dict:
        """Returns configuration dictionary with all secrets and credentials masked."""
        data = self.model_dump()
        data["embedder"] = self.embedder.get_masked_dict()
        data["pgvector"] = self.pgvector.get_masked_dict()
        return data

    def __repr__(self) -> str:
        return f"AppConfig({self.get_masked_dict()})"

    __str__ = __repr__

config = AppConfig()

