import pytest
import time
import uuid
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from unittest.mock import MagicMock
from src.common.redis_client import RedisManager
from src.auth.service import JwksTokenVerifier

@pytest.fixture
def redis_mgr():
    """Fixture pointing to live Redis on 192.168.0.117."""
    return RedisManager(host="192.168.0.117", port=6379, db=0, socket_timeout=2.0)

def test_live_redis_connectivity(redis_mgr):
    """Tests ping against 192.168.0.117:6379."""
    assert redis_mgr.is_available() is True

def test_redis_token_revocation_lifecycle(redis_mgr):
    """Tests token revocation lifecycle in Redis."""
    test_token = f"jwt_mock_token_{uuid.uuid4().hex}"
    
    # 1. Not revoked initially
    assert redis_mgr.is_token_revoked(test_token) is False

    # 2. Revoke token with short TTL
    assert redis_mgr.revoke_token(test_token, ttl_seconds=60) is True

    # 3. Verified as revoked
    assert redis_mgr.is_token_revoked(test_token) is True

def test_jwks_verifier_rejects_revoked_token():
    """Verifies that JwksTokenVerifier rejects revoked tokens even if cryptographically valid."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    
    now = int(time.time())
    token = jwt.encode(
        {"sub": "revoked_alice", "exp": now + 3600, "iat": now},
        private_key,
        algorithm="RS256",
        headers={"kid": "key-revocation-test"}
    )

    verifier = JwksTokenVerifier(jwks_url="http://mock-auth/.well-known/jwks.json")
    mock_signing_key = MagicMock()
    mock_signing_key.key = public_key
    mock_client = MagicMock()
    mock_client.get_signing_key_from_jwt.return_value = mock_signing_key
    verifier._client = mock_client

    # Before revocation: valid
    decoded = verifier.verify_token(token)
    assert decoded["sub"] == "revoked_alice"

    # Revoke in Redis
    from src.common.redis_client import redis_manager
    redis_manager.revoke_token(token, ttl_seconds=60)

    # After revocation: immediately rejected
    with pytest.raises(PermissionError, match=r"Session revoked: Token has been invalidated via Single Sign-Out."):
        verifier.verify_token(token)

def test_redis_session_persistence(redis_mgr):
    """Tests saving, retrieving, and deleting user session state."""
    test_user = f"user_{uuid.uuid4().hex[:6]}"
    state = {
        "user_id": test_user,
        "role": "admin",
        "clearance": 3,
        "theme": "dark"
    }

    assert redis_mgr.save_user_session(test_user, state, ttl_seconds=60) is True
    loaded = redis_mgr.get_user_session(test_user)
    assert loaded == state

    assert redis_mgr.delete_user_session(test_user) is True
    assert redis_mgr.get_user_session(test_user) is None

def test_redis_telemetry_event_ring_buffer(redis_mgr):
    """Tests pushing and querying telemetry events in Redis."""
    t_id = f"trace_{uuid.uuid4().hex[:8]}"
    event = {
        "trace_id": t_id,
        "action": "TOKEN_VERIFY",
        "status": "SUCCESS",
        "user_id": "test_chef"
    }

    assert redis_mgr.push_telemetry_event(event, max_events=50) is True
    events = redis_mgr.get_telemetry_events(limit=5)
    matching = [e for e in events if e.get("trace_id") == t_id]
    assert len(matching) == 1
    assert matching[0]["action"] == "TOKEN_VERIFY"

def test_redis_fallback_when_disabled():
    """Tests that all operations gracefully fall back to in-memory mode when Redis is disabled."""
    mgr = RedisManager(enabled=False)
    assert mgr.is_available() is False

    test_token = f"fallback_tok_{uuid.uuid4().hex}"
    assert mgr.is_token_revoked(test_token) is False
    assert mgr.revoke_token(test_token, ttl_seconds=60) is True
    assert mgr.is_token_revoked(test_token) is True

    assert mgr.save_user_session("fallback_user", {"status": "ok"}) is True
    assert mgr.get_user_session("fallback_user") == {"status": "ok"}
