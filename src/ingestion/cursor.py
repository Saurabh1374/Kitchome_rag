import os
import sqlite3
import time
import threading
import logging
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

logger = logging.getLogger("kitchome.ingestion.cursor")

class IngestionCursorRecord(BaseModel):
    document_id: str
    file_path: str
    content_hash: str
    doc_family: str
    version: int = 1
    is_latest: bool = True
    status: str = "ACTIVE"  # PARSED | ACTIVE | ARCHIVED | QUASHED
    declared_domain: Optional[str] = None
    resolved_namespace: Optional[str] = None
    access_tier: str = "free"
    tenant_id: str = "global"
    clearance_level: int = 1
    chunks_count: int = 0
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

class IngestionCursorManager:
    """
    Permanent Document Catalog and Idempotency Gatekeeper.
    Tracks content hashes, document families, version histories, and quash tombstones.
    """
    def __init__(self, db_path: str = "data/ingestion_cursor.db"):
        self.db_path = db_path
        self._local = threading.local()
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()
        logger.debug("IngestionCursorManager initialized at db_path='%s'", os.path.abspath(db_path))

    @contextmanager
    def _get_connection(self):
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            self._local.conn = conn
        with self._local.conn:
            yield self._local.conn

    def close(self):
        """Cleanly closes the thread-local SQLite connection."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                self._local.conn.close()
                logger.debug("Closed thread-local SQLite connection for cursor manager.")
            except Exception as e:
                logger.debug("Error closing thread-local SQLite connection: %s", e)
            self._local.conn = None

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_cursor (
                    document_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    doc_family TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    is_latest INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    declared_domain TEXT,
                    resolved_namespace TEXT,
                    access_tier TEXT DEFAULT 'free',
                    tenant_id TEXT DEFAULT 'global',
                    clearance_level INTEGER DEFAULT 1,
                    chunks_count INTEGER DEFAULT 0,
                    created_at REAL,
                    updated_at REAL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cursor_hash ON ingestion_cursor (content_hash);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cursor_family ON ingestion_cursor (doc_family, version);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cursor_path ON ingestion_cursor (file_path);")
            conn.commit()

    def _row_to_record(self, row: sqlite3.Row) -> IngestionCursorRecord:
        return IngestionCursorRecord(
            document_id=row["document_id"],
            file_path=row["file_path"],
            content_hash=row["content_hash"],
            doc_family=row["doc_family"],
            version=row["version"],
            is_latest=bool(row["is_latest"]),
            status=row["status"],
            declared_domain=row["declared_domain"],
            resolved_namespace=row["resolved_namespace"],
            access_tier=row["access_tier"],
            tenant_id=row["tenant_id"],
            clearance_level=row["clearance_level"],
            chunks_count=row["chunks_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )

    def get_by_hash(self, content_hash: str) -> Optional[IngestionCursorRecord]:
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM ingestion_cursor WHERE content_hash = ? AND status IN ('ACTIVE', 'ACTIVE_DEGRADED') LIMIT 1;", 
                (content_hash,)
            )
            row = cur.fetchone()
            record = self._row_to_record(row) if row else None
            logger.debug(
                "Cursor lookup by hash '%.12s...': %s",
                content_hash,
                f"HIT (doc_id='{record.document_id}', family='{record.doc_family}', v{record.version})" if record else "MISS"
            )
            return record

    def get_by_path(self, file_path: str) -> Optional[IngestionCursorRecord]:
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM ingestion_cursor WHERE file_path = ? ORDER BY version DESC LIMIT 1;", 
                (file_path,)
            )
            row = cur.fetchone()
            record = self._row_to_record(row) if row else None
            logger.debug(
                "Cursor lookup by path '%s': %s",
                file_path,
                f"HIT (doc_id='{record.document_id}', v{record.version})" if record else "MISS"
            )
            return record

    def get_active_by_family(self, doc_family: str) -> Optional[IngestionCursorRecord]:
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM ingestion_cursor WHERE doc_family = ? AND is_latest = 1 AND status IN ('ACTIVE', 'ACTIVE_DEGRADED') LIMIT 1;", 
                (doc_family,)
            )
            row = cur.fetchone()
            record = self._row_to_record(row) if row else None
            logger.debug(
                "Cursor lookup for active family '%s': %s",
                doc_family,
                f"HIT (v{record.version}, status={record.status})" if record else "MISS"
            )
            return record

    def get_next_version(self, doc_family: str) -> int:
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT MAX(version) as max_v FROM ingestion_cursor WHERE doc_family = ?;", 
                (doc_family,)
            )
            row = cur.fetchone()
            next_v = int(row["max_v"]) + 1 if (row and row["max_v"] is not None) else 1
            logger.debug("Computed next version for family '%s': v%d (current_max=%s)", doc_family, next_v, row["max_v"] if row else None)
            return next_v

    def get_by_document_id(self, document_id: str) -> Optional[IngestionCursorRecord]:
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM ingestion_cursor WHERE document_id = ? LIMIT 1;", 
                (document_id,)
            )
            row = cur.fetchone()
            record = self._row_to_record(row) if row else None
            logger.debug(
                "Cursor lookup by document_id='%s': %s",
                document_id,
                f"HIT (family='{record.doc_family}', v{record.version}, status={record.status})" if record else "MISS"
            )
            return record

    def record_parsed(self, record: IngestionCursorRecord) -> IngestionCursorRecord:
        """Records an intermediate parsed document record with status = 'PARSED'."""
        now = time.time()
        logger.info(
            "Cursor recording PARSED stage: doc_id='%s', family='%s', v%d, domain='%s', tenant='%s'",
            record.document_id, record.doc_family, record.version, record.declared_domain, record.tenant_id
        )
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO ingestion_cursor (
                    document_id, file_path, content_hash, doc_family, version,
                    is_latest, status, declared_domain, resolved_namespace,
                    access_tier, tenant_id, clearance_level, chunks_count,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    record.document_id, record.file_path, record.content_hash,
                    record.doc_family, record.version, 0,
                    "PARSED", record.declared_domain, record.resolved_namespace,
                    record.access_tier, record.tenant_id, record.clearance_level,
                    record.chunks_count, record.created_at or now, now
                )
            )
            conn.commit()
        logger.debug(
            "Cursor record parsed saved: hash='%.12s...', file='%s', chunks=%d",
            record.content_hash, record.file_path, record.chunks_count
        )
        return record

    def commit_version(self, record: IngestionCursorRecord) -> None:
        now = time.time()
        logger.info(
            "Cursor COMMITTING active version: family='%s', v%d, doc_id='%s', chunks=%d, status='%s'",
            record.doc_family, record.version, record.document_id, record.chunks_count, record.status
        )
        with self._get_connection() as conn:
            # 1. Archive any prior active versions of this doc_family
            conn.execute(
                """
                UPDATE ingestion_cursor 
                SET is_latest = 0, status = 'ARCHIVED', updated_at = ? 
                WHERE doc_family = ? AND is_latest = 1;
                """,
                (now, record.doc_family)
            )
            # 2. Insert new active record
            conn.execute(
                """
                INSERT OR REPLACE INTO ingestion_cursor (
                    document_id, file_path, content_hash, doc_family, version,
                    is_latest, status, declared_domain, resolved_namespace,
                    access_tier, tenant_id, clearance_level, chunks_count,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    record.document_id, record.file_path, record.content_hash,
                    record.doc_family, record.version, 1 if record.is_latest else 0,
                    record.status, record.declared_domain, record.resolved_namespace,
                    record.access_tier, record.tenant_id, record.clearance_level,
                    record.chunks_count, record.created_at, now
                )
            )
            conn.commit()
        logger.debug(
            "Cursor committed successfully: archived previous active versions for family '%s' and marked doc_id='%s' active (is_latest=1)",
            record.doc_family, record.document_id
        )

    def quash_version(self, document_id: str) -> bool:
        now = time.time()
        logger.info("Cursor QUASHING document_id='%s'", document_id)
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                UPDATE ingestion_cursor 
                SET status = 'QUASHED', is_latest = 0, updated_at = ? 
                WHERE document_id = ?;
                """,
                (now, document_id)
            )
            conn.commit()
            success = cur.rowcount > 0
            if success:
                logger.info("Cursor successfully QUASHED document_id='%s' (rows_updated=%d)", document_id, cur.rowcount)
            else:
                logger.warning("Cursor quash failed: document_id='%s' not found", document_id)
            return success

    def list_records(self, doc_family: Optional[str] = None) -> List[IngestionCursorRecord]:
        with self._get_connection() as conn:
            if doc_family:
                cur = conn.execute("SELECT * FROM ingestion_cursor WHERE doc_family = ? ORDER BY version ASC;", (doc_family,))
            else:
                cur = conn.execute("SELECT * FROM ingestion_cursor ORDER BY created_at DESC;")
            records = [self._row_to_record(r) for r in cur.fetchall()]
            logger.debug("Listed cursor records: doc_family_filter='%s', count=%d", doc_family or 'ALL', len(records))
            return records
