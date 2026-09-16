import json
import logging
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
import psycopg2
import psycopg2.extensions
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from .base import VectorChunk, DocumentSummaryRecord, NamespaceVectorStore

logger = logging.getLogger(__name__)

class PooledConnectionWrapper:
    """
    Wrapper that manages connection lifecycle with ThreadedConnectionPool.
    Supports Python context management (__enter__, __exit__) for transaction boundaries.
    Guarantees that tenant context (SET LOCAL or SET) NEVER leaks across pool checkouts
    by sanitizing transactions and issuing RESET ALL upon close().
    """
    def __init__(self, pool: ThreadedConnectionPool, conn):
        self._pool = pool
        self._conn = conn

    def __enter__(self):
        return self._conn.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._conn.__exit__(exc_type, exc_val, exc_tb)

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):
        """
        Safely returns connection to the pool after sanitizing all transaction and session state.
        Ensures tenant context (SET LOCAL or SET) NEVER leaks across pool checkouts.
        """
        if self._pool and self._conn:
            try:
                # If connection is broken or closed, evict it from pool
                if getattr(self._conn, "closed", 0) != 0:
                    self._pool.putconn(self._conn, close=True)
                else:
                    # Roll back any dangling transaction so SET LOCAL variables are destroyed
                    needs_rollback = False
                    if hasattr(self._conn, "get_transaction_status"):
                        if self._conn.get_transaction_status() != psycopg2.extensions.TRANSACTION_STATUS_IDLE:
                            needs_rollback = True
                    elif hasattr(self._conn, "status") and self._conn.status != psycopg2.extensions.STATUS_READY:
                        needs_rollback = True

                    if needs_rollback:
                        self._conn.rollback()
                    
                    # Defensively wipe any lingering session settings (SET)
                    try:
                        with self._conn.cursor() as cur:
                            cur.execute("RESET ALL;")
                        self._conn.commit()
                    except Exception:
                        # If sanitization fails, discard this connection from pool rather than recycling dirty state
                        self._pool.putconn(self._conn, close=True)
                        self._conn = None
                        return

                    self._pool.putconn(self._conn, close=False)
            except Exception:
                try:
                    self._pool.putconn(self._conn, close=True)
                except Exception:
                    pass
            self._conn = None

class PGVectorStore(NamespaceVectorStore):
    """
    PostgreSQL + pgvector implementation of the Partitioned Vector Store.
    Executes vector similarity search strictly within skills.ms index namespaces using HNSW/Cosine operations.
    Maintains dedicated document_summaries table and supports PostgreSQL Row-Level Security (RLS).
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
        summaries_table_name: str = "document_summaries",
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
        self.summaries_table_name = summaries_table_name
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
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_tenant ON {self.table_name} ((metadata->>'tenant_id'));

                    CREATE TABLE IF NOT EXISTS {self.summaries_table_name} (
                        document_id VARCHAR(64) PRIMARY KEY,
                        namespace VARCHAR(64) NOT NULL,
                        document_title VARCHAR(255) NOT NULL,
                        summary_text TEXT NOT NULL,
                        token_count INT NOT NULL DEFAULT 0,
                        embedding vector({self.embedding_dimension}),
                        access_tier VARCHAR(32) NOT NULL DEFAULT 'free',
                        tenant_id VARCHAR(64) NOT NULL DEFAULT 'global',
                        clearance_level INT NOT NULL DEFAULT 1,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_{self.summaries_table_name}_namespace ON {self.summaries_table_name} (namespace);
                    CREATE INDEX IF NOT EXISTS idx_{self.summaries_table_name}_tenant ON {self.summaries_table_name} (tenant_id);
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
            self.init_rls_policies()
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
            with conn:
                # Set PostgreSQL transaction-scoped RLS session variables if ABAC filter is provided
                if abac_filter:
                    tenant_id = abac_filter.get("tenant_id", "global")
                    permitted_tiers = abac_filter.get("permitted_tiers", [])
                    max_clearance = abac_filter.get("max_clearance", 1)
                    is_internal_worker = abac_filter.get("is_internal_worker", False)
                    with conn.cursor() as rls_cur:
                        rls_cur.execute("SET LOCAL app.current_tenant = %s;", (tenant_id,))
                        rls_cur.execute("SET LOCAL app.permitted_tiers = %s;", (",".join(permitted_tiers) if permitted_tiers else "",))
                        rls_cur.execute("SET LOCAL app.clearance_level = %s;", (str(max_clearance),))
                        rls_cur.execute("SET LOCAL app.is_internal_worker = %s;", ("true" if is_internal_worker else "false",))

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

    def upsert_summary(self, summary: DocumentSummaryRecord) -> str:
        """Stores or updates a dedicated document summary record in PostgreSQL."""
        # Always mirror in memory
        super().upsert_summary(summary)

        if not self.is_connected:
            return summary.document_id

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                embedding_str = f"[{','.join(map(str, summary.embedding))}]" if summary.embedding else None
                metadata_json = json.dumps(summary.metadata)
                cur.execute(f"""
                    INSERT INTO {self.summaries_table_name} (
                        document_id, namespace, document_title, summary_text,
                        token_count, embedding, access_tier, tenant_id,
                        clearance_level, metadata, updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        {"%s::vector" if embedding_str else "NULL"},
                        %s, %s, %s, %s, CURRENT_TIMESTAMP
                    )
                    ON CONFLICT (document_id) DO UPDATE SET
                        namespace = EXCLUDED.namespace,
                        document_title = EXCLUDED.document_title,
                        summary_text = EXCLUDED.summary_text,
                        token_count = EXCLUDED.token_count,
                        embedding = EXCLUDED.embedding,
                        access_tier = EXCLUDED.access_tier,
                        tenant_id = EXCLUDED.tenant_id,
                        clearance_level = EXCLUDED.clearance_level,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP;
                """, (
                    summary.document_id,
                    summary.namespace,
                    summary.document_title,
                    summary.summary_text,
                    summary.token_count,
                    *([embedding_str] if embedding_str else []),
                    summary.access_tier,
                    summary.tenant_id,
                    summary.clearance_level,
                    metadata_json
                ))
            conn.commit()
            return summary.document_id
        finally:
            conn.close()

    def get_summary(self, document_id: str, abac_filter: Optional[Dict[str, Any]] = None) -> Optional[DocumentSummaryRecord]:
        """Retrieves a single parent document summary by document_id within a transaction-scoped RLS context."""
        if not self.is_connected:
            return super().get_summary(document_id, abac_filter=abac_filter)

        conn = self._get_connection()
        try:
            with conn:
                if abac_filter:
                    tenant_id = abac_filter.get("tenant_id", "global")
                    permitted_tiers = abac_filter.get("permitted_tiers", [])
                    max_clearance = abac_filter.get("max_clearance", 1)
                    is_internal_worker = abac_filter.get("is_internal_worker", False)
                    with conn.cursor() as rls_cur:
                        rls_cur.execute("SET LOCAL app.current_tenant = %s;", (tenant_id,))
                        rls_cur.execute("SET LOCAL app.permitted_tiers = %s;", (",".join(permitted_tiers) if permitted_tiers else "",))
                        rls_cur.execute("SET LOCAL app.clearance_level = %s;", (str(max_clearance),))
                        rls_cur.execute("SET LOCAL app.is_internal_worker = %s;", ("true" if is_internal_worker else "false",))

                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(f"""
                        SELECT document_id, namespace, document_title, summary_text,
                               token_count, access_tier, tenant_id, clearance_level, metadata
                        FROM {self.summaries_table_name}
                        WHERE document_id = %s;
                    """, (document_id,))
                    row = cur.fetchone()
                    if not row:
                        return super().get_summary(document_id, abac_filter=abac_filter)
                    return DocumentSummaryRecord(
                        document_id=row["document_id"],
                        namespace=row["namespace"],
                        document_title=row["document_title"],
                        summary_text=row["summary_text"],
                        token_count=row["token_count"],
                        access_tier=row["access_tier"],
                        tenant_id=row["tenant_id"],
                        clearance_level=row["clearance_level"],
                        metadata=row["metadata"] if isinstance(row["metadata"], dict) else {}
                    )
        finally:
            conn.close()

    def get_summaries(self, document_ids: List[str], abac_filter: Optional[Dict[str, Any]] = None) -> Dict[str, DocumentSummaryRecord]:
        """Batch retrieves parent document summaries for a list of document IDs within a transaction-scoped RLS context."""
        if not document_ids:
            return {}
        if not self.is_connected:
            return super().get_summaries(document_ids, abac_filter=abac_filter)

        conn = self._get_connection()
        try:
            with conn:
                if abac_filter:
                    tenant_id = abac_filter.get("tenant_id", "global")
                    permitted_tiers = abac_filter.get("permitted_tiers", [])
                    max_clearance = abac_filter.get("max_clearance", 1)
                    is_internal_worker = abac_filter.get("is_internal_worker", False)
                    with conn.cursor() as rls_cur:
                        rls_cur.execute("SET LOCAL app.current_tenant = %s;", (tenant_id,))
                        rls_cur.execute("SET LOCAL app.permitted_tiers = %s;", (",".join(permitted_tiers) if permitted_tiers else "",))
                        rls_cur.execute("SET LOCAL app.clearance_level = %s;", (str(max_clearance),))
                        rls_cur.execute("SET LOCAL app.is_internal_worker = %s;", ("true" if is_internal_worker else "false",))

                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(f"""
                        SELECT document_id, namespace, document_title, summary_text,
                               token_count, access_tier, tenant_id, clearance_level, metadata
                        FROM {self.summaries_table_name}
                        WHERE document_id = ANY(%s);
                    """, (list(set(document_ids)),))
                    rows = cur.fetchall()
                    results = {}
                    for row in rows:
                        results[row["document_id"]] = DocumentSummaryRecord(
                            document_id=row["document_id"],
                            namespace=row["namespace"],
                            document_title=row["document_title"],
                            summary_text=row["summary_text"],
                            token_count=row["token_count"],
                            access_tier=row["access_tier"],
                            tenant_id=row["tenant_id"],
                            clearance_level=row["clearance_level"],
                            metadata=row["metadata"] if isinstance(row["metadata"], dict) else {}
                        )
                    # Fallback to in-memory for any missing IDs
                    for doc_id in document_ids:
                        if doc_id not in results:
                            mem_sum = super().get_summary(doc_id, abac_filter=abac_filter)
                            if mem_sum:
                                results[doc_id] = mem_sum
                    return results
        finally:
            conn.close()

    def search_summaries(
        self,
        query_vector: List[float],
        namespace: Any,
        top_k: int = 3,
        abac_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Searches dedicated document summaries using PostgreSQL pgvector cosine similarity and ABAC filters."""
        if not self.is_connected:
            return super().search_summaries(query_vector, namespace, top_k, abac_filter=abac_filter)

        if isinstance(namespace, str):
            target_namespaces = [namespace]
        elif isinstance(namespace, (list, tuple, set)):
            target_namespaces = list(namespace)
        else:
            target_namespaces = [str(namespace)]

        conn = self._get_connection()
        results = []
        try:
            with conn:
                if abac_filter:
                    tenant_id = abac_filter.get("tenant_id", "global")
                    permitted_tiers = abac_filter.get("permitted_tiers", [])
                    max_clearance = abac_filter.get("max_clearance", 1)
                    is_internal_worker = abac_filter.get("is_internal_worker", False)
                    with conn.cursor() as rls_cur:
                        rls_cur.execute("SET LOCAL app.current_tenant = %s;", (tenant_id,))
                        rls_cur.execute("SET LOCAL app.permitted_tiers = %s;", (",".join(permitted_tiers) if permitted_tiers else "",))
                        rls_cur.execute("SET LOCAL app.clearance_level = %s;", (str(max_clearance),))
                        rls_cur.execute("SET LOCAL app.is_internal_worker = %s;", ("true" if is_internal_worker else "false",))

                embedding_str = "[" + ",".join(map(str, query_vector)) + "]"
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    where_clauses = ["namespace = ANY(%s)", "embedding IS NOT NULL"]
                    params: List[Any] = [target_namespaces]

                    if abac_filter:
                        tenant_id = abac_filter.get("tenant_id", "global")
                        where_clauses.append("(tenant_id = %s OR tenant_id = 'global')")
                        params.append(tenant_id)

                        permitted_tiers = abac_filter.get("permitted_tiers")
                        if permitted_tiers:
                            where_clauses.append("access_tier = ANY(%s)")
                            params.append(permitted_tiers)

                        max_clearance = abac_filter.get("max_clearance")
                        if max_clearance is not None:
                            where_clauses.append("clearance_level <= %s")
                            params.append(max_clearance)

                    where_sql = " AND ".join(where_clauses)
                    query_sql = f"""
                        SELECT document_id, namespace, document_title, summary_text,
                               token_count, access_tier, tenant_id, clearance_level, metadata,
                               1 - (embedding <=> %s::vector) AS similarity_score
                        FROM {self.summaries_table_name}
                        WHERE {where_sql}
                        ORDER BY embedding <=> %s::vector ASC
                        LIMIT %s;
                    """
                    full_params = [embedding_str] + params + [embedding_str, top_k]
                    cur.execute(query_sql, full_params)

                    rows = cur.fetchall()
                    for row in rows:
                        results.append({
                            "document_id": row["document_id"],
                            "namespace": row["namespace"],
                            "document_title": row["document_title"],
                            "summary_text": row["summary_text"],
                            "token_count": row["token_count"],
                            "access_tier": row["access_tier"],
                            "tenant_id": row["tenant_id"],
                            "clearance_level": row["clearance_level"],
                            "metadata": row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"]),
                            "similarity_score": round(float(row["similarity_score"]), 4)
                        })
        finally:
            conn.close()

        return results

    @contextmanager
    def tenant_transaction(
        self, 
        user: Optional[Any] = None, 
        abac_filter: Optional[Dict[str, Any]] = None,
        is_internal_worker: bool = False,
        conn: Optional[Any] = None
    ):
        """
        Context manager ensuring that tenant context is strictly bound to the active transaction.
        Applies SET LOCAL session variables, executes within `with conn:`, and guarantees
        all tenant context is completely wiped upon transaction completion or exit.
        """
        local_conn = False
        if conn is None:
            conn = self._get_connection()
            local_conn = True

        try:
            with conn:
                if user is not None:
                    from ..auth.abac import ABACPolicyEngine
                    rls_vars = ABACPolicyEngine.get_rls_session_vars(user)
                    with conn.cursor() as cur:
                        for k, v in rls_vars.items():
                            cur.execute(f"SET LOCAL {k} = %s;", (v,))
                        cur.execute("SET LOCAL app.is_internal_worker = %s;", ("true" if is_internal_worker else "false",))
                elif abac_filter is not None:
                    tenant_id = abac_filter.get("tenant_id", "global")
                    permitted_tiers = abac_filter.get("permitted_tiers", [])
                    max_clearance = abac_filter.get("max_clearance", 1)
                    worker_flag = abac_filter.get("is_internal_worker", is_internal_worker)
                    with conn.cursor() as cur:
                        cur.execute("SET LOCAL app.current_tenant = %s;", (tenant_id,))
                        cur.execute("SET LOCAL app.permitted_tiers = %s;", (",".join(permitted_tiers) if permitted_tiers else "",))
                        cur.execute("SET LOCAL app.clearance_level = %s;", (str(max_clearance),))
                        cur.execute("SET LOCAL app.is_internal_worker = %s;", ("true" if worker_flag else "false",))
                elif is_internal_worker:
                    with conn.cursor() as cur:
                        cur.execute("SET LOCAL app.is_internal_worker = 'true';")

                yield conn
        finally:
            if local_conn:
                conn.close()

    def apply_rls_session(self, conn, user: Any, is_internal_worker: bool = False) -> None:
        """Sets PostgreSQL transaction-scoped session variables for Row-Level Security."""
        from ..auth.abac import ABACPolicyEngine
        rls_vars = ABACPolicyEngine.get_rls_session_vars(user)
        with conn.cursor() as cur:
            for k, v in rls_vars.items():
                cur.execute(f"SET LOCAL {k} = %s;", (v,))
            cur.execute("SET LOCAL app.is_internal_worker = %s;", ("true" if is_internal_worker else "false",))

    def init_rls_policies(self) -> None:
        """Configures native PostgreSQL Row-Level Security policies on chunks and summaries."""
        if not self.is_connected:
            return
        conn = self._get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    # Enable RLS
                    cur.execute(f"ALTER TABLE {self.table_name} ENABLE ROW LEVEL SECURITY;")
                    cur.execute(f"ALTER TABLE {self.summaries_table_name} ENABLE ROW LEVEL SECURITY;")

                    # Policy for vector chunks (Fail-Closed Default)
                    cur.execute(f"""
                        DROP POLICY IF EXISTS rls_chunks_tenant_clearance ON {self.table_name};
                        CREATE POLICY rls_chunks_tenant_clearance ON {self.table_name}
                        FOR SELECT
                        USING (
                            (
                                current_setting('app.is_internal_worker', true) = 'true'
                                OR metadata->>'tenant_id' = 'global'
                                OR (
                                    NULLIF(current_setting('app.current_tenant', true), '') IS NOT NULL
                                    AND metadata->>'tenant_id' = current_setting('app.current_tenant', true)
                                )
                            )
                            AND (
                                current_setting('app.is_internal_worker', true) = 'true'
                                OR (
                                    NULLIF(current_setting('app.permitted_tiers', true), '') IS NOT NULL
                                    AND metadata->>'access_tier' = ANY(string_to_array(current_setting('app.permitted_tiers', true), ','))
                                )
                                OR (
                                    NULLIF(current_setting('app.permitted_tiers', true), '') IS NULL
                                    AND metadata->>'access_tier' = 'free'
                                )
                            )
                            AND (
                                current_setting('app.is_internal_worker', true) = 'true'
                                OR COALESCE((metadata->>'clearance_level')::int, 1) <= COALESCE(NULLIF(current_setting('app.clearance_level', true), '')::int, 1)
                            )
                        );
                    """)

                    # Policy for summaries (Fail-Closed Default)
                    cur.execute(f"""
                        DROP POLICY IF EXISTS rls_summaries_tenant_clearance ON {self.summaries_table_name};
                        CREATE POLICY rls_summaries_tenant_clearance ON {self.summaries_table_name}
                        FOR SELECT
                        USING (
                            (
                                current_setting('app.is_internal_worker', true) = 'true'
                                OR tenant_id = 'global'
                                OR (
                                    NULLIF(current_setting('app.current_tenant', true), '') IS NOT NULL
                                    AND tenant_id = current_setting('app.current_tenant', true)
                                )
                            )
                            AND (
                                current_setting('app.is_internal_worker', true) = 'true'
                                OR (
                                    NULLIF(current_setting('app.permitted_tiers', true), '') IS NOT NULL
                                    AND access_tier = ANY(string_to_array(current_setting('app.permitted_tiers', true), ','))
                                )
                                OR (
                                    NULLIF(current_setting('app.permitted_tiers', true), '') IS NULL
                                    AND access_tier = 'free'
                                )
                            )
                            AND (
                                current_setting('app.is_internal_worker', true) = 'true'
                                OR clearance_level <= COALESCE(NULLIF(current_setting('app.clearance_level', true), '')::int, 1)
                            )
                        );
                    """)

                    # Policy for vector chunks modifications (Fail-Closed Default)
                    cur.execute(f"""
                        DROP POLICY IF EXISTS rls_chunks_modifications ON {self.table_name};
                        CREATE POLICY rls_chunks_modifications ON {self.table_name}
                        FOR ALL
                        USING (
                            current_setting('app.is_internal_worker', true) = 'true'
                            OR (
                                NULLIF(current_setting('app.current_tenant', true), '') IS NOT NULL
                                AND metadata->>'tenant_id' = current_setting('app.current_tenant', true)
                            )
                        )
                        WITH CHECK (
                            current_setting('app.is_internal_worker', true) = 'true'
                            OR (
                                NULLIF(current_setting('app.current_tenant', true), '') IS NOT NULL
                                AND metadata->>'tenant_id' = current_setting('app.current_tenant', true)
                            )
                        );
                    """)

                    # Policy for summaries modifications (Fail-Closed Default)
                    cur.execute(f"""
                        DROP POLICY IF EXISTS rls_summaries_modifications ON {self.summaries_table_name};
                        CREATE POLICY rls_summaries_modifications ON {self.summaries_table_name}
                        FOR ALL
                        USING (
                            current_setting('app.is_internal_worker', true) = 'true'
                            OR (
                                NULLIF(current_setting('app.current_tenant', true), '') IS NOT NULL
                                AND tenant_id = current_setting('app.current_tenant', true)
                            )
                        )
                        WITH CHECK (
                            current_setting('app.is_internal_worker', true) = 'true'
                            OR (
                                NULLIF(current_setting('app.current_tenant', true), '') IS NOT NULL
                                AND tenant_id = current_setting('app.current_tenant', true)
                            )
                        );
                    """)
            logger.info("Successfully initialized PostgreSQL Row-Level Security policies.")
        finally:
            conn.close()
