import os
import json
import time
import hashlib
import logging
import threading
import concurrent.futures
from typing import Optional, Dict, Any, List

logger = logging.getLogger("kitchome.redis")

class RedisManager:
    """
    Central Redis Client Manager for Distributed Cache, Token Revocation, 
    Session Persistence, and Telemetry Ring Buffering.
    
    Defaults to 192.168.0.117:6379 with sub-second timeouts (0.2s), 
    asynchronous fire-and-forget background writes for telemetry and sessions,
    cached availability health checks, and seamless in-memory fallback.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: Optional[int] = None,
        password: Optional[str] = None,
        enabled: Optional[bool] = None,
        socket_timeout: float = 0.2
    ):
        self.host = host or os.getenv("REDIS_HOST", "192.168.0.117")
        self.port = int(port or os.getenv("REDIS_PORT", "6379"))
        self.db = int(db or os.getenv("REDIS_DB", "0"))
        self.password = password or os.getenv("REDIS_PASSWORD") or None
        
        env_enabled = os.getenv("REDIS_ENABLED", "true").lower() in ("true", "1", "yes")
        self.enabled = env_enabled if enabled is None else enabled
        self.socket_timeout = socket_timeout

        self._client: Optional[Any] = None
        self._is_connected: Optional[bool] = None

        # Cached availability state (15-second TTL)
        self._cached_available: Optional[bool] = None
        self._cached_available_expires_at: float = 0.0

        # Dedicated background executor for fire-and-forget async operations
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="RedisAsyncWorker"
        )

        # In-memory fallbacks when Redis is offline or disabled
        self._fallback_revoked: Dict[str, float] = {}  # token_hash -> expiry_time
        self._fallback_sessions: Dict[str, Dict[str, Any]] = {}
        self._fallback_jwks: Optional[Dict[str, Any]] = None
        self._fallback_telemetry: List[Dict[str, Any]] = []

    def _get_client(self):
        if not self.enabled:
            return None

        if self._client is None:
            try:
                import redis
                self._client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    socket_timeout=self.socket_timeout,
                    socket_connect_timeout=self.socket_timeout,
                    decode_responses=True
                )
                self._client.ping()
                self._is_connected = True
                logger.info(f"Connected to Redis at {self.host}:{self.port} (db={self.db})")
            except Exception as e:
                self._is_connected = False
                logger.warning(f"Redis connection to {self.host}:{self.port} failed ({e}). Operating in in-memory fallback mode.")
                self._client = None

        return self._client

    def is_available(self, force_refresh: bool = False) -> bool:
        """
        Fast cached availability check (15s TTL).
        Prevents synchronous network latency from stalling Streamlit renders or request loops.
        """
        if not self.enabled:
            return False

        now = time.time()
        if not force_refresh and self._cached_available is not None and now < self._cached_available_expires_at:
            return self._cached_available

        client = self._get_client()
        if not client:
            self._cached_available = False
            self._cached_available_expires_at = now + 15.0
            return False

        try:
            val = bool(client.ping())
            self._cached_available = val
            self._cached_available_expires_at = now + 15.0
            return val
        except Exception:
            self._cached_available = False
            self._cached_available_expires_at = now + 15.0
            return False

    # -------------------------------------------------------------------------
    # 1. Instant Token Revocation (Single Sign-Out / SLO)
    # -------------------------------------------------------------------------
    def _hash_token(self, token: str) -> str:
        clean = token.strip()
        if clean.lower().startswith("bearer "):
            clean = clean[7:].strip()
        return hashlib.sha256(clean.encode("utf-8")).hexdigest()

    def revoke_token(self, token: str, ttl_seconds: int = 3600) -> bool:
        """Blacklists a token by its SHA256 hash across the ecosystem."""
        token_hash = self._hash_token(token)
        # Immediate in-memory invalidation
        self._fallback_revoked[token_hash] = time.time() + ttl_seconds

        client = self._get_client()
        if client:
            try:
                client.set(f"kitchome:revoked:{token_hash}", "1", ex=max(int(ttl_seconds), 60))
                logger.info(f"Token revoked in Redis (hash: {token_hash[:12]}..., ttl: {ttl_seconds}s)")
                return True
            except Exception as e:
                logger.warning(f"Failed to revoke token in Redis ({e}). Using in-memory fallback.")

        return True

    def is_token_revoked(self, token: str) -> bool:
        """Checks if a token has been explicitly revoked via Single Sign-Out."""
        token_hash = self._hash_token(token)
        # 1. Fast local cache lookup
        exp = self._fallback_revoked.get(token_hash)
        if exp is not None:
            if time.time() < exp:
                return True
            else:
                del self._fallback_revoked[token_hash]

        # 2. Redis lookup with strict 0.2s timeout
        client = self._get_client()
        if client:
            try:
                val = client.get(f"kitchome:revoked:{token_hash}")
                return val is not None
            except Exception as e:
                logger.warning(f"Redis query failed ({e}). Checking in-memory fallback.")

        return False

    # -------------------------------------------------------------------------
    # 2. Session State Persistence (Streamlit & Cross-Container Rehydration)
    # -------------------------------------------------------------------------
    def save_user_session(self, user_id: str, state_dict: Dict[str, Any], ttl_seconds: int = 86400, async_write: bool = True) -> bool:
        """
        Persists user session state.
        Instantly updates local memory, and offloads remote Redis write to a background thread.
        """
        self._fallback_sessions[user_id] = state_dict

        client = self._get_client()
        if not client:
            return True

        def _do_write():
            try:
                client.set(f"kitchome:session:{user_id}", json.dumps(state_dict), ex=ttl_seconds)
                return True
            except Exception as e:
                logger.warning(f"Failed to save user session in Redis ({e})")
                return False

        if async_write:
            self._executor.submit(_do_write)
            return True
        else:
            return _do_write()

    def get_user_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves stored user session state with fast fallback resolution."""
        if user_id in self._fallback_sessions:
            return self._fallback_sessions[user_id]

        client = self._get_client()
        if client:
            try:
                raw = client.get(f"kitchome:session:{user_id}")
                if raw:
                    return json.loads(raw)
            except Exception as e:
                logger.warning(f"Failed to get user session from Redis ({e})")

        return self._fallback_sessions.get(user_id)

    def delete_user_session(self, user_id: str) -> bool:
        """Clears user session upon explicit logout."""
        self._fallback_sessions.pop(user_id, None)

        client = self._get_client()
        if client:
            try:
                client.delete(f"kitchome:session:{user_id}")
            except Exception as e:
                logger.warning(f"Failed to delete session from Redis ({e})")

        return True

    # -------------------------------------------------------------------------
    # 3. Cluster-Wide JWKS Cache
    # -------------------------------------------------------------------------
    def cache_jwks(self, jwks_data: Dict[str, Any], ttl_seconds: int = 3600, async_write: bool = True) -> bool:
        """Caches the public JWKS key set in Redis asynchronously."""
        self._fallback_jwks = jwks_data

        client = self._get_client()
        if not client:
            return True

        def _do_cache():
            try:
                client.set("kitchome:auth:jwks", json.dumps(jwks_data), ex=ttl_seconds)
                return True
            except Exception as e:
                logger.warning(f"Failed to cache JWKS in Redis ({e})")
                return False

        if async_write:
            self._executor.submit(_do_cache)
            return True
        else:
            return _do_cache()

    def get_cached_jwks(self) -> Optional[Dict[str, Any]]:
        """Retrieves the cached JWKS key set from Redis."""
        if self._fallback_jwks:
            return self._fallback_jwks

        client = self._get_client()
        if client:
            try:
                raw = client.get("kitchome:auth:jwks")
                if raw:
                    return json.loads(raw)
            except Exception as e:
                logger.warning(f"Failed to get cached JWKS from Redis ({e})")

        return self._fallback_jwks

    # -------------------------------------------------------------------------
    # 4. Durable Telemetry & Audit Ring Buffer
    # -------------------------------------------------------------------------
    def push_telemetry_event(self, event_dict: Dict[str, Any], max_events: int = 1000, async_write: bool = True) -> bool:
        """
        Appends an authentication trace event to the telemetry ring buffer.
        Dispatches asynchronously to background worker to prevent request blocking.
        """
        self._fallback_telemetry.append(event_dict)
        if len(self._fallback_telemetry) > max_events:
            self._fallback_telemetry.pop(0)

        client = self._get_client()
        if not client:
            return True

        def _do_push():
            try:
                raw = json.dumps(event_dict)
                pipe = client.pipeline()
                pipe.lpush("kitchome:auth:telemetry:events", raw)
                pipe.ltrim("kitchome:auth:telemetry:events", 0, max_events - 1)
                pipe.execute()
                return True
            except Exception as e:
                logger.warning(f"Failed to push telemetry event to Redis ({e})")
                return False

        if async_write:
            self._executor.submit(_do_push)
            return True
        else:
            return _do_push()

    def get_telemetry_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetches the most recent authentication telemetry events with local fallback deduplication."""
        merged: List[Dict[str, Any]] = []
        seen_trace_ids = set()

        client = self._get_client()
        if client:
            try:
                raw_list = client.lrange("kitchome:auth:telemetry:events", 0, limit - 1)
                for x in raw_list:
                    item = json.loads(x)
                    merged.append(item)
                    if "trace_id" in item:
                        seen_trace_ids.add(item["trace_id"])
            except Exception as e:
                logger.warning(f"Failed to fetch telemetry events from Redis ({e})")

        # Supplement with any pending local fallback events
        for item in reversed(self._fallback_telemetry):
            t_id = item.get("trace_id")
            if not t_id or t_id not in seen_trace_ids:
                merged.append(item)
                if t_id:
                    seen_trace_ids.add(t_id)
            if len(merged) >= limit:
                break

        return merged[:limit]

# Global Singleton Manager
redis_manager = RedisManager()
