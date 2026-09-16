import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from src.api.main import app
from src.auth.context import UserContext, UserTier
from src.auth.scopes import ServiceScope, local_scope_manager, UserApprovalStatus
from src.auth.service import JWTTokenManager

TEST_SECRET = JWTTokenManager.DEFAULT_SECRET
client = TestClient(app)

@pytest.fixture(autouse=True)
def clean_database():
    """Ensure clean local database state for every test."""
    local_scope_manager.clear()
    yield
    local_scope_manager.clear()

def test_api_health_and_root():
    """Verify health check and root endpoints return status ok."""
    r_health = client.get("/healthz")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    r_root = client.get("/")
    assert r_root.status_code == 200
    assert "endpoints" in r_root.json()

def test_api_strict_single_doc_ingest_type_validation():
    """Verify strict type checking rejects invalid payloads with HTTP 422."""
    tenant = "tenant_validate"
    local_scope_manager.bootstrap_founding_admin(tenant_id=tenant, user_id="admin_founder")

    admin = UserContext(user_id="admin_founder", tier=UserTier.ENTERPRISE, tenant_id=tenant)
    token = JWTTokenManager.create_token(user=admin, scopes=None, secret=TEST_SECRET)
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Negative clearance level (must be 1-3)
    bad_payload_1 = {
        "document_id": "valid_id_101",
        "title": "Valid Title",
        "content": "This is valid text content longer than ten characters.",
        "clearance_level": 0,  # Invalid! Must be ge=1
        "access_tier": "free"
    }
    r1 = client.post("/api/v1/ingest", json=bad_payload_1, headers=headers)
    assert r1.status_code == 422

    # 2. Clearance level exceeding 3
    bad_payload_2 = {
        "document_id": "valid_id_102",
        "title": "Valid Title",
        "content": "This is valid text content longer than ten characters.",
        "clearance_level": 4,  # Invalid! Must be le=3
        "access_tier": "free"
    }
    r2 = client.post("/api/v1/ingest", json=bad_payload_2, headers=headers)
    assert r2.status_code == 422

    # 3. Invalid access tier (must be free, premium, scholar, or enterprise)
    bad_payload_3 = {
        "document_id": "valid_id_103",
        "title": "Valid Title",
        "content": "This is valid text content longer than ten characters.",
        "clearance_level": 1,
        "access_tier": "super_vip"  # Invalid!
    }
    r3 = client.post("/api/v1/ingest", json=bad_payload_3, headers=headers)
    assert r3.status_code == 422

    # 4. Content too short (< 10 chars)
    bad_payload_4 = {
        "document_id": "valid_id_104",
        "title": "Valid Title",
        "content": "Short",  # Invalid!
        "clearance_level": 1,
        "access_tier": "free"
    }
    r4 = client.post("/api/v1/ingest", json=bad_payload_4, headers=headers)
    assert r4.status_code == 422

def test_api_ingest_requires_scope_and_succeeds_for_authorized_caller():
    """Verify ingestion enforces 'ingestion:write' scope and caller tenant isolation."""
    tenant = "tenant_ingest_test"
    local_scope_manager.bootstrap_founding_admin(tenant_id=tenant, user_id="admin_user")

    # Regular user with only rag:read
    local_scope_manager.approve_user(
        tenant_id=tenant,
        user_id="read_only_user",
        clearance_level=1,
        role="member",
        scopes=[ServiceScope.RAG_READ]
    )

    user_read = UserContext(user_id="read_only_user", tier=UserTier.FREE, tenant_id=tenant)
    token_read = JWTTokenManager.create_token(user=user_read, scopes=None, secret=TEST_SECRET)

    valid_payload = {
        "document_id": "doc_induction_cleaning",
        "title": "Induction Stove Care",
        "content": "Always clean induction glass cooktops with a microfibre cloth and specialized cleaner.",
        "clearance_level": 1,
        "access_tier": "free",
        "declared_domain": "appliances_troubleshooting"
    }

    # 1. Unauthenticated request fails with 401
    r_no_auth = client.post("/api/v1/ingest", json=valid_payload)
    assert r_no_auth.status_code == 401

    # 2. Read-only user rejected with 403 (Lacks ingestion:write)
    r_forbidden = client.post("/api/v1/ingest", json=valid_payload, headers={"Authorization": f"Bearer {token_read}"})
    assert r_forbidden.status_code == 403

    # 3. Admin grants ingestion:write to user
    local_scope_manager.grant_scope(tenant_id=tenant, user_id="read_only_user", scope=ServiceScope.INGESTION_WRITE)

    # 4. Now ingestion succeeds immediately on the exact same token
    r_success = client.post("/api/v1/ingest", json=valid_payload, headers={"Authorization": f"Bearer {token_read}"})
    assert r_success.status_code == 200
    data = r_success.json()
    assert data["status"] == "SUCCESS"
    assert data["document_id"] == "doc_induction_cleaning"
    assert data["chunks_ingested"] >= 1

def test_api_query_flow_and_clearance_filtering():
    """Verify RAG query endpoint returns answer and enforces clearance bounds."""
    tenant = "tenant_query_test"
    local_scope_manager.bootstrap_founding_admin(tenant_id=tenant, user_id="alice_admin")

    local_scope_manager.approve_user(
        tenant_id=tenant,
        user_id="member_bob",
        clearance_level=2,
        role="member",
        scopes=[ServiceScope.RAG_READ]
    )

    user = UserContext(user_id="member_bob", tier=UserTier.PREMIUM, tenant_id=tenant)
    token = JWTTokenManager.create_token(user=user, scopes=None, secret=TEST_SECRET)
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Valid Query
    query_payload = {"query_text": "how to maintain granite countertops", "max_results": 3}
    r = client.post("/api/v1/query", json=query_payload, headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "SUCCESS"
    assert data["user_id"] == "member_bob"
    assert data["clearance_level"] == 2
    assert "answer" in data

    # 2. Exceeding clearance level rejected
    bad_clearance_payload = {"query_text": "confidential executive notes", "custom_clearance": 3}
    r_bad = client.post("/api/v1/query", json=bad_clearance_payload, headers=headers)
    assert r_bad.status_code == 403
    assert "exceeds caller's authorized clearance" in r_bad.json()["detail"]

def test_api_onboarding_lifecycle_and_admin_governance():
    """
    Verify complete onboarding lifecycle via API:
    1. Second user arrives -> 403 Forbidden with PENDING_APPROVAL and /onboarding redirect.
    2. Admin checks /api/v1/admin/pending -> sees user.
    3. Admin calls /api/v1/admin/approve -> user approved.
    4. User can now query successfully.
    """
    tenant = "tenant_lifecycle"

    # User 1: Founding Admin
    admin_ctx = UserContext(user_id="owner_amy", tier=UserTier.ENTERPRISE, tenant_id=tenant)
    token_admin = JWTTokenManager.create_token(user=admin_ctx, scopes=None, secret=TEST_SECRET)

    # Calling /auth/me boots admin
    r_admin_me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token_admin}"})
    assert r_admin_me.status_code == 200
    assert r_admin_me.json()["role"] == "admin"

    # User 2: Dave arrives -> PENDING_APPROVAL
    dave_ctx = UserContext(user_id="dave_dev", tier=UserTier.FREE, tenant_id=tenant)
    token_dave = JWTTokenManager.create_token(user=dave_ctx, scopes=None, secret=TEST_SECRET)

    r_query_dave = client.post(
        "/api/v1/query",
        json={"query_text": "clean cast iron"},
        headers={"Authorization": f"Bearer {token_dave}"}
    )
    assert r_query_dave.status_code == 403
    detail = r_query_dave.json()["detail"]
    assert detail["status"] == "PENDING_APPROVAL"
    assert detail["onboarding_url"] == "/onboarding"

    # Admin lists pending users
    r_pending = client.get("/api/v1/admin/pending", headers={"Authorization": f"Bearer {token_admin}"})
    assert r_pending.status_code == 200
    pending_list = r_pending.json()
    assert len(pending_list) == 1
    assert pending_list[0]["user_id"] == "dave_dev"

    # Non-admin trying to approve fails with 403
    r_unauthorized_approve = client.post(
        "/api/v1/admin/approve",
        json={"user_id": "dave_dev", "clearance_level": 2, "role": "member", "scopes": ["rag:read"]},
        headers={"Authorization": f"Bearer {token_dave}"}
    )
    assert r_unauthorized_approve.status_code == 403

    # Admin approves Dave with clearance=2 and rag:read
    r_approve = client.post(
        "/api/v1/admin/approve",
        json={"user_id": "dave_dev", "clearance_level": 2, "role": "member", "scopes": ["rag:read"]},
        headers={"Authorization": f"Bearer {token_admin}"}
    )
    assert r_approve.status_code == 200
    assert r_approve.json()["status"] == "SUCCESS"

    # Dave now queries successfully!
    r_dave_allowed = client.post(
        "/api/v1/query",
        json={"query_text": "clean cast iron"},
        headers={"Authorization": f"Bearer {token_dave}"}
    )
    assert r_dave_allowed.status_code == 200
    assert r_dave_allowed.json()["user_id"] == "dave_dev"
    assert r_dave_allowed.json()["clearance_level"] == 2

def test_api_token_malformed_and_tampered_returns_401():
    """Verify that malformed, corrupted, or tampered JWTs return HTTP 401 instead of crashing with HTTP 500."""
    # 1. Non-JWT string
    r1 = client.post("/api/v1/query", json={"query_text": "test"}, headers={"Authorization": "Bearer invalid_token"})
    assert r1.status_code == 401
    assert "Malformed JWT" in r1.json()["detail"]

    # 2. Invalid base64 in segments
    r2 = client.post("/api/v1/query", json={"query_text": "test"}, headers={"Authorization": "Bearer a.b.c"})
    assert r2.status_code == 401

    # 3. Signature mismatch
    r3 = client.post("/api/v1/query", json={"query_text": "test"}, headers={"Authorization": "Bearer eyJhbGciOiAiSFMyNTYifQ.eyJzdWIiOiAidGVzdCJ9.bad_signature"})
    assert r3.status_code == 401

    # 4. Ingest endpoint with malformed token
    r4 = client.post("/api/v1/ingest", json={"document_id": "doc_1", "title": "Valid Title", "content": "1234567890"}, headers={"Authorization": "Bearer malformed"})
    assert r4.status_code == 401

    # 5. Auth /me with malformed token
    r5 = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer malformed"})
    assert r5.status_code == 401

def test_api_query_custom_clearance_downscoping_and_latency():
    """Verify that custom_clearance properly downscopes clearance and latency_ms is recorded."""
    tenant = "tenant_downscope"
    local_scope_manager.bootstrap_founding_admin(tenant_id=tenant, user_id="founder")
    local_scope_manager.approve_user(
        tenant_id=tenant,
        user_id="cleared_user",
        clearance_level=3,
        role="member",
        scopes=[ServiceScope.RAG_READ]
    )

    user = UserContext(user_id="cleared_user", tier=UserTier.ENTERPRISE, tenant_id=tenant)
    token = JWTTokenManager.create_token(user=user, scopes=None, secret=TEST_SECRET)
    headers = {"Authorization": f"Bearer {token}"}

    # Query with custom_clearance=1 (downscoping from 3)
    r = client.post(
        "/api/v1/query",
        json={"query_text": "culinary knife skills", "custom_clearance": 1},
        headers=headers
    )
    assert r.status_code == 200
    data = r.json()
    assert data["clearance_level"] == 1
    assert data["latency_ms"] > 0.0

