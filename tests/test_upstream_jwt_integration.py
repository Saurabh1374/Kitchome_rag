import os
import uuid
import pytest
from fastapi.testclient import TestClient
from src.auth.service import JWTTokenManager, ServiceAuthGuard
from src.auth.context import UserContext, UserTier
from src.auth.scopes import ServiceScope, local_scope_manager
from src.api.main import app

# Shared Base64 key identical to upstream Complete-custom-auth (.env.local)
UPSTREAM_BASE64_SECRET = "JzjbCwXWdSXqGdJBIWVPiRxQUB/kIWPY3FjgGd78Hoxd04464lqv7cV4/uBRp2RVfMo3dMRvV+gsCzbx+1vALA=="

@pytest.fixture(autouse=True)
def setup_secret(monkeypatch):
    monkeypatch.setenv("AUTH_JWT_SECRET", UPSTREAM_BASE64_SECRET)
    JWTTokenManager.DEFAULT_SECRET = UPSTREAM_BASE64_SECRET

def test_upstream_token_signature_and_hydration():
    """Verifies that an upstream JWT issued with Base64 key and claims (sub, tenant_id, tier)
    is properly verified and hydrated by downstream ServiceAuthGuard."""
    test_tenant = f"corp_{uuid.uuid4().hex[:6]}"
    test_user = f"mario_{uuid.uuid4().hex[:4]}"

    user_ctx = UserContext(
        user_id=test_user,
        tenant_id=test_tenant,
        tier=UserTier.ENTERPRISE,
        clearance_level=1
    )
    token = JWTTokenManager.create_token(
        user=user_ctx,
        secret=UPSTREAM_BASE64_SECRET
    )

    # Downstream verification
    user = ServiceAuthGuard.enforce_service_auth(
        auth_header_or_token=f"Bearer {token}",
        secret=UPSTREAM_BASE64_SECRET
    )

    assert user.user_id == test_user
    assert user.tenant_id == test_tenant
    assert user.tier == UserTier.ENTERPRISE
    assert user.role == "admin" # Auto-bootstrapped as founding admin for this new tenant
    assert user.clearance_level == 3 # Founding admin receives clearance 3

def test_end_to_end_promotion_synchronization_via_api():
    """Simulates the upstream PromotionService dispatching an authenticated HTTP call
    to downstream /api/v1/admin/update-access, and verifies downstream permissions update."""
    client = TestClient(app)

    tenant_id = f"gourmet_{uuid.uuid4().hex[:6]}"
    admin_id = f"admin_{uuid.uuid4().hex[:4]}"
    sous_id = f"sous_{uuid.uuid4().hex[:4]}"

    # 1. First user in tenant bootstraps as admin
    admin_ctx = UserContext(
        user_id=admin_id,
        tenant_id=tenant_id,
        tier=UserTier.ENTERPRISE,
        clearance_level=3
    )
    admin_token = JWTTokenManager.create_token(admin_ctx, secret=UPSTREAM_BASE64_SECRET)

    # Calling /auth/me bootstraps founding admin
    res = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {admin_token}"})
    assert res.status_code == 200
    assert res.json()["role"] == "admin"

    # 2. Second user registers under tenant
    user_ctx = UserContext(
        user_id=sous_id,
        tenant_id=tenant_id,
        tier=UserTier.PREMIUM,
        clearance_level=1
    )
    user_token = JWTTokenManager.create_token(user_ctx, secret=UPSTREAM_BASE64_SECRET)

    # Calling /auth/me registers as pending approval
    res_pending = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {user_token}"})
    assert res_pending.status_code == 200
    assert res_pending.json()["status"] == "PENDING_APPROVAL"

    # Admin approves user
    approve_res = client.post(
        "/api/v1/admin/approve",
        json={"user_id": sous_id, "role": "member", "clearance_level": 1},
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert approve_res.status_code == 200

    # 3. Simulate Promotion Elevation dispatched by upstream PromotionService:
    # Upstream calls POST /api/v1/admin/update-access with target clearance: 2, scopes: [rag:read, ingestion:write]
    update_res = client.post(
        "/api/v1/admin/update-access",
        json={
            "user_id": sous_id,
            "clearance_level": 2,
            "role": "member",
            "scopes": ["rag:read", "ingestion:write"]
        },
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert update_res.status_code == 200
    resp_data = update_res.json()
    assert resp_data["status"] == "SUCCESS"
    assert resp_data["profile"]["clearance_level"] == 2

    # Verify scopes via local_scope_manager
    scopes = [s.value if hasattr(s, "value") else str(s) for s in local_scope_manager.get_user_scopes(tenant_id, sous_id)]
    assert "ingestion:write" in scopes
    assert "rag:read" in scopes

    # 4. Verify sous_chef now accesses protected endpoints with updated permissions
    profile_res = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {user_token}"})
    assert profile_res.status_code == 200
    user_profile = profile_res.json()
    assert user_profile["clearance_level"] == 2
    assert "ingestion:write" in user_profile["granted_scopes"]
