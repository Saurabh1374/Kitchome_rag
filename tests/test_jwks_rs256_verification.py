import time
import json
import base64
import pytest
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from unittest.mock import MagicMock, patch
from src.auth.service import JwksTokenVerifier, verify_token_hybrid
from src.auth.context import UserContext, UserTier

@pytest.fixture(scope="module")
def rsa_keypair():
    """Generates an ephemeral RSA 2048-bit key pair for testing."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    public_key = private_key.public_key()
    return private_key, public_key

def test_jwks_rs256_token_verification_success(rsa_keypair):
    """Verifies that an RS256 token signed by RSA private key is verified locally via public key."""
    private_key, public_key = rsa_keypair
    kid = "test-key-1"

    now = int(time.time())
    payload = {
        "sub": "chef_ramsay",
        "email": "ramsay@kitchen.com",
        "tenant_id": "restaurant_group",
        "tier": "enterprise",
        "roles": ["ROLE_ADMIN", "ADMIN"],
        "iat": now,
        "exp": now + 3600,
        "iss": "kitchome-auth"
    }

    token = jwt.encode(
        payload,
        private_key,
        algorithm="RS256",
        headers={"kid": kid}
    )

    verifier = JwksTokenVerifier(jwks_url="http://mock-auth/.well-known/jwks.json")
    mock_signing_key = MagicMock()
    mock_signing_key.key = public_key

    mock_client = MagicMock()
    mock_client.get_signing_key_from_jwt.return_value = mock_signing_key
    verifier._client = mock_client

    decoded = verifier.verify_token(token)
    assert decoded["sub"] == "chef_ramsay"
    assert decoded["tier"] == "enterprise"
    assert "ADMIN" in decoded["roles"]
    assert decoded["tenant_id"] == "restaurant_group"

def test_jwks_rs256_token_expired(rsa_keypair):
    """Verifies that expired RS256 tokens are rejected with PermissionError."""
    private_key, public_key = rsa_keypair
    kid = "test-key-1"

    past = int(time.time()) - 100
    payload = {
        "sub": "expired_user",
        "exp": past,
        "iat": past - 3600
    }

    token = jwt.encode(
        payload,
        private_key,
        algorithm="RS256",
        headers={"kid": kid}
    )

    verifier = JwksTokenVerifier(jwks_url="http://mock-auth/.well-known/jwks.json")
    mock_signing_key = MagicMock()
    mock_signing_key.key = public_key

    mock_client = MagicMock()
    mock_client.get_signing_key_from_jwt.return_value = mock_signing_key
    verifier._client = mock_client

    with pytest.raises(PermissionError, match=r"Token has expired"):
        verifier.verify_token(token)

def test_jwks_rs256_tampered_signature(rsa_keypair):
    """Verifies that a tampered signature triggers PermissionError."""
    private_key, public_key = rsa_keypair
    kid = "test-key-1"

    now = int(time.time())
    payload = {
        "sub": "tampered_user",
        "exp": now + 3600,
        "iat": now
    }

    token = jwt.encode(
        payload,
        private_key,
        algorithm="RS256",
        headers={"kid": kid}
    )

    # Tamper with the signature segment
    parts = token.split(".")
    tampered_sig = parts[2][:-4] + "AAAA"
    tampered_token = f"{parts[0]}.{parts[1]}.{tampered_sig}"

    verifier = JwksTokenVerifier(jwks_url="http://mock-auth/.well-known/jwks.json")
    mock_signing_key = MagicMock()
    mock_signing_key.key = public_key

    mock_client = MagicMock()
    mock_client.get_signing_key_from_jwt.return_value = mock_signing_key
    verifier._client = mock_client

    with pytest.raises(PermissionError, match=r"Invalid RS256 token signature"):
        verifier.verify_token(tampered_token)

def test_verify_token_hybrid_routes_rs256_to_jwks(rsa_keypair):
    """Verifies that verify_token_hybrid accurately detects RS256 header and routes to JWKS verifier."""
    private_key, public_key = rsa_keypair
    kid = "test-key-1"

    now = int(time.time())
    payload = {
        "sub": "hybrid_user",
        "roles": ["USER"],
        "exp": now + 3600,
        "iat": now
    }

    token = jwt.encode(
        payload,
        private_key,
        algorithm="RS256",
        headers={"kid": kid}
    )

    with patch("src.auth.service.jwks_verifier") as mock_jwks:
        mock_jwks.verify_token.return_value = payload
        result = verify_token_hybrid(f"Bearer {token}")
        assert result["sub"] == "hybrid_user"
        mock_jwks.verify_token.assert_called_once_with(token, trace_id=None)
