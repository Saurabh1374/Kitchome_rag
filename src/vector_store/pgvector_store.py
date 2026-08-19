import json
import logging
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from .base import VectorChunk, NamespaceVectorStore

logger = logging.getLogger(__name__)

class PGVectorStore(NamespaceVectorStore):
    """
    PostgreSQL + pgvector implementation of the Partitioned Vector Store.
    Executes vector similarity search strictly within skills.ms index namespaces using HNSW/Cosine operations.
    Falls back gracefully to local storage if PostgreSQL connection is unavailable.
    """
    def __init__(
        self, 
        host: str = "localhost", 
        port: int = 5432, 
        db_name: str = "kitchome_rag", 
        user: str = "postgres", 
        password: str = "postgres",
        table_name: str = "kitchome_vector_chunks",
        embedding_dimension: int = 384,
        storage_path: Optional[str] = None
    ):
        super().__init__(storage_path=storage_path)
        self.host = host
        self.port = port
        self.db_name = db_name
        self.user = user
        self.password = password
        self.table_name = table_name
        self.embedding_dimension = embedding_dimension
        self.is_connected = False

        self._init_db()

    def _init_db(self):
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                dbname=self.db_name,
                user=self.user,
                password=self.password,
                connect_timeout=3
            )
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                register_vector(conn)
                
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        chunk_id VARCHAR(64) PRIMARY KEY,
                        document_id VARCHAR(64) NOT NULL,
                        namespace VARCHAR(64) NOT NULL,
                        text TEXT NOT NULL,
                        metadata JSONB NOT NULL,
                        embedding vector({self.embedding_dimension}) NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_namespace ON {self.table_name} (namespace);
                """)
            conn.close()
            self.is_connected = True
            logger.info("Successfully connected to PostgreSQL + pgvector")
        except Exception as e:
            self.is_connected = False
            logger.warning(f"PostgreSQL + pgvector connection unavailable ({e}). Operating in Local Fallback Mode.")

    def _get_connection(self):
        conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.db_name,
            user=self.user,
            password=self.password
        )
        register_vector(conn)
        return conn

    def upsert_chunks(self, chunks: List[VectorChunk]) -> int:
        if not self.is_connected:
            return super().upsert_chunks(chunks)

        count = 0
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                for chunk in chunks:
                    metadata_json = json.dumps(chunk.metadata)
                    embedding_str = "[" + ",".join(map(str, chunk.embedding)) + "]"
                    
                    cur.execute(f"""
                        INSERT INTO {self.table_name} (chunk_id, document_id, namespace, text, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s::vector)
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            document_id = EXCLUDED.document_id,
                            namespace = EXCLUDED.namespace,
                            text = EXCLUDED.text,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding;
                    """, (chunk.chunk_id, chunk.document_id, chunk.namespace, chunk.text, metadata_json, embedding_str))
                    count += 1
            conn.commit()
        finally:
            conn.close()
        
        # Keep in-memory sync as well
        super().upsert_chunks(chunks)
        return count

    def search(
        self, 
        query_vector: List[float], 
        namespace: str, 
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        if not self.is_connected:
            return super().search(query_vector, namespace, top_k)

        conn = self._get_connection()
        results = []
        try:
            embedding_str = "[" + ",".join(map(str, query_vector)) + "]"
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT chunk_id, document_id, namespace, text, metadata,
                           1 - (embedding <=> %s::vector) AS similarity_score
                    FROM {self.table_name}
                    WHERE namespace = %s
                    ORDER BY embedding <=> %s::vector ASC
                    LIMIT %s;
                """, (embedding_str, namespace, embedding_str, top_k))
                
                rows = cur.fetchall()
                for row in rows:
                    results.append({
                        "chunk_id": row["chunk_id"],
                        "document_id": row["document_id"],
                        "namespace": row["namespace"],
                        "text": row["text"],
                        "metadata": row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"]),
                        "similarity_score": round(float(row["similarity_score"]), 4)
                    })
        finally:
            conn.close()

        return results

    def get_namespace_stats(self) -> Dict[str, int]:
        if not self.is_connected:
            return super().get_namespace_stats()

        conn = self._get_connection()
        stats = {}
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT namespace, COUNT(*) FROM {self.table_name} GROUP BY namespace;")
                rows = cur.fetchall()
                for ns, count in rows:
                    stats[ns] = count
        finally:
            conn.close()
        return stats
