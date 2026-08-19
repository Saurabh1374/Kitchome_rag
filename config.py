import os
from pydantic import BaseModel, Field

class IngestionConfig(BaseModel):
    chunk_size: int = Field(default=300, description="Target chunk character length")
    chunk_overlap: int = Field(default=50, description="Character overlap between consecutive chunks")
    embedding_dimension: int = Field(default=384, description="Vector embedding dimension size")

class SkillsMSConfig(BaseModel):
    registry_file: str = Field(default="skills_registry.json")
    default_namespace: str = Field(default="general_home")

class PGVectorConfig(BaseModel):
    host: str = Field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    port: int = Field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    db_name: str = Field(default_factory=lambda: os.getenv("POSTGRES_DB", "kitchome_rag"))
    user: str = Field(default_factory=lambda: os.getenv("POSTGRES_USER", "postgres"))
    password: str = Field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "postgres"))
    vector_table: str = Field(default="kitchome_vector_chunks")

class AppConfig(BaseModel):
    data_dir: str = os.path.join(os.path.dirname(__file__), "data")
    vector_db_path: str = os.path.join(os.path.dirname(__file__), "vector_store_data.json")
    ingestion: IngestionConfig = IngestionConfig()
    skills_ms: SkillsMSConfig = SkillsMSConfig()
    pgvector: PGVectorConfig = PGVectorConfig()

config = AppConfig()
