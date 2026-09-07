import json
import logging
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from .base import VectorChunk, NamespaceVectorStore

logger = logging.getLogger(__name__)

class PooledConnectionWrapper:
    """Wrapper that returns connection to ThreadedConnectionPool when close() is called."""
    def __init__(self, pool: ThreadedConnectionPool, conn):
        self._pool = pool
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):
        if self._pool and self._conn:
            try:
                self._pool.putconn(self._conn)
            except Exception:
                pass
            self._conn = None

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
        self._pool: Optional[ThreadedConnectionPool] = None

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
            self._pool = ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                host=self.host,
                port=self.port,
                dbname=self.db_name,
                user=self.user,
                password=self.password,
                connect_timeout=3
            )
            self.is_connected = True
            logger.info("Successfully connected to PostgreSQL + pgvector with ThreadedConnectionPool")
        except Exception as e:
            self.is_connected = False
            self._pool = None
            logger.warning(f"PostgreSQL + pgvector connection unavailable ({e}). Operating in Local Fallback Mode.")

    def _get_connection(self):
        if self._pool:
            conn = self._pool.getconn()
            register_vector(conn)
            return PooledConnectionWrapper(self._pool, conn)
        conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.db_name,
            user=self.user,
            password=self.password
        )
        register_vector(conn)
        return conn

    def close(self):
        """Closes all connections in the ThreadedConnectionPool."""
        if self._pool:
            try:
                self._pool.closeall()
            except Exception:
                pass
            self._pool = None

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
        
        return count

    def search(
        self, 
        query_vector: List[float], 
        namespace: Any, 
        top_k: int = 3,
        abac_filter: Optional[Dict[str, Any]] = None,
        is_summary: Optional[bool] = None,
        is_latest: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        if not self.is_connected:
            return super().search(
                query_vector, 
                namespace, 
                top_k, 
                abac_filter=abac_filter, 
                is_summary=is_summary,
                is_latest=is_latest
            )

        if isinstance(namespace, str):
            target_namespaces = [namespace]
        elif isinstance(namespace, (list, tuple, set)):
            target_namespaces = list(namespace)
        else:
            target_namespaces = [str(namespace)]

        conn = self._get_connection()
        results = []
        try:
            embedding_str = "[" + ",".join(map(str, query_vector)) + "]"
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                where_clauses = ["namespace = ANY(%s)", "embedding IS NOT NULL"]
                params: List[Any] = [target_namespaces]

                if is_summary is not None:
                    where_clauses.append("(metadata->>'is_summary')::boolean = %s")
                    params.append(is_summary)

                if is_latest is not None:
                    where_clauses.append("COALESCE((metadata->>'is_latest')::boolean, true) = %s")
                    params.append(is_latest)

                if abac_filter:
                    tenant_id = abac_filter.get("tenant_id", "global")
                    where_clauses.append("(metadata->>'tenant_id' = %s OR metadata->>'tenant_id' = 'global')")
                    params.append(tenant_id)

                    permitted_tiers = abac_filter.get("permitted_tiers")
                    if permitted_tiers:
                        where_clauses.append("metadata->>'access_tier' = ANY(%s)")
                        params.append(permitted_tiers)

                    max_clearance = abac_filter.get("max_clearance")
                    if max_clearance is not None:
                        where_clauses.append("COALESCE((metadata->>'clearance_level')::int, 1) <= %s")
                        params.append(max_clearance)

                where_sql = " AND ".join(where_clauses)
                query_sql = f"""
                    SELECT chunk_id, document_id, namespace, text, metadata,
                           1 - (embedding <=> %s::vector) AS similarity_score
                    FROM {self.table_name}
                    WHERE {where_sql}
                    ORDER BY embedding <=> %s::vector ASC
                    LIMIT %s;
                """
                full_params = [embedding_str] + params + [embedding_str, top_k]
                cur.execute(query_sql, full_params)
                
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

    def delete_chunks_by_document_id(self, document_id: str) -> int:
        if not self.is_connected:
            return super().delete_chunks_by_document_id(document_id)

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self.table_name} WHERE document_id = %s;", (document_id,))
                count = cur.rowcount
            conn.commit()
            return count
        finally:
            conn.close()

    def delete_chunks_by_job_id(self, job_id: str) -> int:
        if not self.is_connected:
            return super().delete_chunks_by_job_id(job_id)

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self.table_name} WHERE metadata->>'job_id' = %s AND COALESCE((metadata->>'is_latest')::boolean, false) = false;", (job_id,))
                count = cur.rowcount
            conn.commit()
            return count
        finally:
            conn.close()

    def flip_is_latest(self, doc_family: str, new_active_job_id: Optional[str] = None, new_document_id: Optional[str] = None) -> int:
        if not self.is_connected:
            return super().flip_is_latest(doc_family, new_active_job_id, new_document_id)

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # 1. Flip previous versions of this doc_family to false
                cur.execute(f"""
                    UPDATE {self.table_name}
                    SET metadata = jsonb_set(metadata, '{{is_latest}}', 'false'::jsonb)
                    WHERE metadata->>'doc_family' = %s;
                """, (doc_family,))
                
                # 2. Flip active version chunks to true
                cur.execute(f"""
                    UPDATE {self.table_name}
                    SET metadata = jsonb_set(metadata, '{{is_latest}}', 'true'::jsonb)
                    WHERE metadata->>'doc_family' = %s AND (metadata->>'job_id' = %s OR document_id = %s);
                """, (doc_family, new_active_job_id, new_document_id))
                count = cur.rowcount
            conn.commit()
            return count
        finally:
            conn.close()

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
