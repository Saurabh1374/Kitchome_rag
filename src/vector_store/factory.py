import os
import threading
import logging
from typing import Optional

from config import config
from .pgvector_store import PGVectorStore

logger = logging.getLogger(__name__)

_singleton_store: Optional[PGVectorStore] = None
_lock = threading.Lock()

def get_vector_store(force_new: bool = False) -> PGVectorStore:
    """
    Returns the centralized PGVectorStore instance.
    Uses PostgreSQL + pgvector when available; otherwise falls back gracefully
    to NamespaceVectorStore with local file persistence (config.vector_db_path).
    """
    global _singleton_store

    if force_new:
        with _lock:
            _singleton_store = _create_store()
            return _singleton_store

    if _singleton_store is None:
        with _lock:
            if _singleton_store is None:
                _singleton_store = _create_store()

    return _singleton_store

def _create_store() -> PGVectorStore:
    logger.debug(
        "Initializing PGVectorStore: host=%s, port=%s, db=%s, user=%s, table=%s, fallback_path=%s",
        config.pgvector.host,
        config.pgvector.port,
        config.pgvector.db_name,
        config.pgvector.user,
        config.pgvector.vector_table,
        config.vector_db_path
    )
    return PGVectorStore(
        host=config.pgvector.host,
        port=config.pgvector.port,
        db_name=config.pgvector.db_name,
        user=config.pgvector.user,
        password=config.pgvector.password,
        table_name=config.pgvector.vector_table,
        embedding_dimension=config.ingestion.embedding_dimension,
        storage_path=config.vector_db_path
    )
