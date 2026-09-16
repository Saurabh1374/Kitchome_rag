import pytest
import os
import time
from src.auth.context import UserContext, UserTier
from src.auth.scopes import ServiceScope, ServiceScopeManager, local_scope_manager, UserApprovalStatus
from src.auth.service import JWTTokenManager, ServiceAuthGuard
from src.rag.engine import RAGQueryEngine
from src.vector_store.base import NamespaceVectorStore

TEST_SECRET = "test-secret-key-scopes-456"

@pytest.fixture(autouse=True)
def clean_scope_table():
    """Ensure clean local scope table before and after each test."""
    local_scope_manager.clear()
    yield
    local_scope_manager.clear()

def test_founding_user_auto_bootstrap_and_pending_approval():
    """
    Verify:
    1. The very first user to arrive in a tenant auto-bootstraps as the Founding Admin.
    2. Any subsequent user in that tenant is placed in PENDING_APPROVAL and blocked from access.
    3. Admin can list pending users, approve them with custom clearance and scopes.
    4. Approved user immediately gains access with their admin-configured privileges.
    """
    tenant = "tenant_alpha"

    # 1. User 1: Alice arrives first -> Becomes Founding Admin
    alice = UserContext(user_id="alice_founder", tier=UserTier.ENTERPRISE, tenant_id=tenant)
    token_alice = JWTTokenManager.create_token(user=alice, scopes=None, secret=TEST_SECRET)

    v_alice = ServiceAuthGuard.enforce_service_auth(
        auth_header_or_token=token_alice,
        required_scope=ServiceScope.ADMIN_MANAGE,
        secret=TEST_SECRET
    )
    assert v_alice.user_id == "alice_founder"
    assert v_alice.role == "admin"
    assert v_alice.clearance_level == 3
    assert ServiceScope.ALL.value in v_alice.granted_scopes

    # 2. User 2: Bob arrives second -> Registered as PENDING_APPROVAL and rejected
    bob = UserContext(user_id="bob_employee", tier=UserTier.PREMIUM, tenant_id=tenant)
    token_bob = JWTTokenManager.create_token(user=bob, scopes=None, secret=TEST_SECRET)

    with pytest.raises(PermissionError, match=r"Access Denied: Account is pending administrator approval"):
        ServiceAuthGuard.enforce_service_auth(
            auth_header_or_token=token_bob,
            required_scope=ServiceScope.RAG_READ,
            secret=TEST_SECRET
        )

    # 3. Admin lists pending users
    pending = local_scope_manager.list_pending_users(tenant_id=tenant)
    assert len(pending) == 1
    assert pending[0]["user_id"] == "bob_employee"
    assert pending[0]["status"] == UserApprovalStatus.PENDING_APPROVAL.value

    # 4. Admin approves Bob with clearance_level=2 and 'rag:read' scope
    local_scope_manager.approve_user(
        tenant_id=tenant,
        user_id="bob_employee",
        clearance_level=2,
        role="member",
        scopes=[ServiceScope.RAG_READ],
        approved_by="alice_founder"
    )

    # Bob now succeeds at RAG_READ with clearance=2
    v_bob = ServiceAuthGuard.enforce_service_auth(
        auth_header_or_token=token_bob,
        required_scope=ServiceScope.RAG_READ,
        secret=TEST_SECRET
    )
    assert v_bob.user_id == "bob_employee"
    assert v_bob.clearance_level == 2
    assert v_bob.role == "member"
    assert ServiceScope.RAG_READ.value in v_bob.granted_scopes
    assert ServiceScope.INGESTION_WRITE.value not in v_bob.granted_scopes

def test_admin_rejection_blocks_access():
    """Verify that an admin can reject a pending user, preventing any service access."""
    tenant = "tenant_beta"
    local_scope_manager.bootstrap_founding_admin(tenant_id=tenant, user_id="admin_founder")

    user = UserContext(user_id="mallory_intruder", tier=UserTier.FREE, tenant_id=tenant)
    token = JWTTokenManager.create_token(user=user, scopes=None, secret=TEST_SECRET)

    # Initially rejected as pending
    with pytest.raises(PermissionError, match=r"pending administrator approval"):
        ServiceAuthGuard.enforce_service_auth(auth_header_or_token=token, secret=TEST_SECRET)

    # Admin rejects
    local_scope_manager.reject_user(
        tenant_id=tenant,
        user_id="mallory_intruder",
        rejected_by="admin_founder",
        reason="Suspicious activity"
    )

    # Now rejected as REJECTED with reason
    with pytest.raises(PermissionError, match=r"Access Denied: Account registration has been rejected. Reason: Suspicious activity"):
        ServiceAuthGuard.enforce_service_auth(auth_header_or_token=token, secret=TEST_SECRET)

def test_token_clearance_and_role_claims_strictly_ignored():
    """
    Verify that incoming JWT claims for clearance_level or role are completely IGNORED.
    Only the local service database onboarding record governs clearance and roles.
    """
    tenant = "tenant_secure"
    local_scope_manager.bootstrap_founding_admin(tenant_id=tenant, user_id="root_admin")

    # Admin onboards user as regular member with clearance level 1
    local_scope_manager.approve_user(
        tenant_id=tenant,
        user_id="attacker_user",
        clearance_level=1,
        role="member",
        scopes=[ServiceScope.RAG_READ],
        approved_by="root_admin"
    )

    # Malicious user mints or presents a token claiming clearance_level=3 and role='admin'
    tampered_user = UserContext(
        user_id="attacker_user",
        tier=UserTier.ENTERPRISE,
        tenant_id=tenant,
        clearance_level=3,
        role="admin"
    )
    token = JWTTokenManager.create_token(user=tampered_user, scopes=None, secret=TEST_SECRET)

    # Service gatekeeper must enforce local profile (clearance=1, role='member'), rejecting ADMIN_MANAGE
    with pytest.raises(PermissionError, match=r"Access Forbidden: Token lacks required scope 'admin:manage'"):
        ServiceAuthGuard.enforce_service_auth(
            auth_header_or_token=token,
            required_scope=ServiceScope.ADMIN_MANAGE,
            secret=TEST_SECRET
        )

    # User context hydrated from service must have clearance=1 and role='member'
    verified = ServiceAuthGuard.enforce_service_auth(
        auth_header_or_token=token,
        required_scope=ServiceScope.RAG_READ,
        secret=TEST_SECRET
    )
    assert verified.clearance_level == 1
    assert verified.role == "member"

def test_default_user_scope_resolution_and_rag_read_access():
    """
    Verify that an approved standard user receives 'rag:read' scope locally,
    and lacks 'ingestion:write'.
    """
    tenant = "tenant_x"
    local_scope_manager.bootstrap_founding_admin(tenant_id=tenant, user_id="admin_x")
    local_scope_manager.approve_user(
        tenant_id=tenant,
        user_id="alice_standard",
        clearance_level=1,
        role="member",
        scopes=[ServiceScope.RAG_READ]
    )

    user = UserContext(
        user_id="alice_standard",
        tier=UserTier.PREMIUM,
        tenant_id=tenant,
        clearance_level=1
    )
    token = JWTTokenManager.create_token(user=user, scopes=None, secret=TEST_SECRET)

    # 1. RAG_READ must succeed
    verified_user = ServiceAuthGuard.enforce_service_auth(
        auth_header_or_token=f"Bearer {token}",
        required_scope=ServiceScope.RAG_READ,
        secret=TEST_SECRET
    )
    assert verified_user.user_id == "alice_standard"
    assert ServiceScope.RAG_READ.value in verified_user.granted_scopes
    assert ServiceScope.INGESTION_WRITE.value not in verified_user.granted_scopes

    # 2. INGESTION_WRITE must fail for this user
    with pytest.raises(PermissionError, match=r"Access Forbidden: Token lacks required scope 'ingestion:write'"):
        ServiceAuthGuard.enforce_service_auth(
            auth_header_or_token=token,
            required_scope=ServiceScope.INGESTION_WRITE,
            secret=TEST_SECRET
        )

def test_dynamic_grant_and_instant_revocation_of_ingestion_scope():
    """
    Verify that an approved user can be granted 'ingestion:write' by an admin,
    and revocation takes effect instantly without modifying or waiting for upstream JWT to expire.
    """
    tenant = "hospital_net"
    local_scope_manager.bootstrap_founding_admin(tenant_id=tenant, user_id="admin_sarah")
    local_scope_manager.approve_user(
        tenant_id=tenant,
        user_id="bob_ingest_requester",
        clearance_level=1,
        role="member",
        scopes=[ServiceScope.RAG_READ]
    )

    user = UserContext(
        user_id="bob_ingest_requester",
        tier=UserTier.SCHOLAR,
        tenant_id=tenant,
        clearance_level=1
    )
    token = JWTTokenManager.create_token(user=user, scopes=None, secret=TEST_SECRET, ttl_seconds=7200)

    # 1. Initially, ingestion must be rejected
    with pytest.raises(PermissionError, match=r"Access Forbidden: Token lacks required scope 'ingestion:write'"):
        ServiceAuthGuard.enforce_service_auth(
            auth_header_or_token=token,
            required_scope=ServiceScope.INGESTION_WRITE,
            secret=TEST_SECRET
        )

    # 2. Admin grants 'ingestion:write' in the local service table
    local_scope_manager.grant_scope(
        tenant_id=tenant,
        user_id="bob_ingest_requester",
        scope=ServiceScope.INGESTION_WRITE,
        granted_by="admin_sarah"
    )

    # 3. Using the EXACT SAME unchanged token, ingestion now succeeds immediately
    verified = ServiceAuthGuard.enforce_service_auth(
        auth_header_or_token=token,
        required_scope=ServiceScope.INGESTION_WRITE,
        secret=TEST_SECRET
    )
    assert verified.user_id == "bob_ingest_requester"
    assert ServiceScope.INGESTION_WRITE.value in verified.granted_scopes
    assert ServiceScope.RAG_READ.value in verified.granted_scopes

    # 4. Admin immediately revokes 'ingestion:write'
    revoked = local_scope_manager.revoke_scope(
        tenant_id=tenant,
        user_id="bob_ingest_requester",
        scope=ServiceScope.INGESTION_WRITE
    )
    assert revoked is True

    # 5. Using the EXACT SAME unchanged token, ingestion is immediately blocked
    with pytest.raises(PermissionError, match=r"Access Forbidden: Token lacks required scope 'ingestion:write'"):
        ServiceAuthGuard.enforce_service_auth(
            auth_header_or_token=token,
            required_scope=ServiceScope.INGESTION_WRITE,
            secret=TEST_SECRET
        )

    # But read access still works
    read_user = ServiceAuthGuard.enforce_service_auth(
        auth_header_or_token=token,
        required_scope=ServiceScope.RAG_READ,
        secret=TEST_SECRET
    )
    assert read_user.user_id == "bob_ingest_requester"

def test_admin_full_access_wildcard_scope():
    """
    Verify that admin users receive full access ('*').
    Admin can be established via Founding User bootstrap or explicit promotion.
    """
    # Case A: Founding User Auto-Bootstrap gives admin full access
    user_founder = UserContext(user_id="charlie_founder", tier=UserTier.ENTERPRISE, tenant_id="enterprise_hq")
    token_founder = JWTTokenManager.create_token(user=user_founder, scopes=None, secret=TEST_SECRET)

    v1 = ServiceAuthGuard.enforce_service_auth(auth_header_or_token=token_founder, required_scope=ServiceScope.RAG_READ, secret=TEST_SECRET)
    v2 = ServiceAuthGuard.enforce_service_auth(auth_header_or_token=token_founder, required_scope=ServiceScope.INGESTION_WRITE, secret=TEST_SECRET)
    v3 = ServiceAuthGuard.enforce_service_auth(auth_header_or_token=token_founder, required_scope=ServiceScope.ADMIN_MANAGE, secret=TEST_SECRET)

    assert v1.user_id == "charlie_founder"
    assert v2.user_id == "charlie_founder"
    assert v3.user_id == "charlie_founder"
    assert ServiceScope.ALL.value in v1.granted_scopes

    # Case B: Admin promotion in local table
    tenant_c = "tenant_c"
    local_scope_manager.bootstrap_founding_admin(tenant_id=tenant_c, user_id="root_c")
    local_scope_manager.approve_user(
        tenant_id=tenant_c,
        user_id="eve_promoted",
        clearance_level=1,
        role="member",
        scopes=[ServiceScope.RAG_READ]
    )

    regular_user = UserContext(user_id="eve_promoted", tier=UserTier.FREE, tenant_id=tenant_c)
    token_promoted = JWTTokenManager.create_token(user=regular_user, scopes=None, secret=TEST_SECRET)

    # Before promotion: cannot administer
    with pytest.raises(PermissionError):
        ServiceAuthGuard.enforce_service_auth(
            auth_header_or_token=token_promoted,
            required_scope=ServiceScope.ADMIN_MANAGE,
            secret=TEST_SECRET
        )

    # Promote to admin in local table
    local_scope_manager.approve_user(
        tenant_id=tenant_c,
        user_id="eve_promoted",
        clearance_level=3,
        role="admin"
    )

    v5 = ServiceAuthGuard.enforce_service_auth(
        auth_header_or_token=token_promoted,
        required_scope=ServiceScope.ADMIN_MANAGE,
        secret=TEST_SECRET
    )
    assert v5.user_id == "eve_promoted"
    assert ServiceScope.ALL.value in v5.granted_scopes

def test_tenant_isolation_in_local_scope_grants():
    """Verify that granting a scope in Tenant A does not leak or grant to the same user ID in Tenant B."""
    local_scope_manager.bootstrap_founding_admin(tenant_id="tenant_a", user_id="admin_a")
    local_scope_manager.bootstrap_founding_admin(tenant_id="tenant_b", user_id="admin_b")

    local_scope_manager.approve_user(tenant_id="tenant_a", user_id="user_shared_id", scopes=[ServiceScope.RAG_READ])
    local_scope_manager.approve_user(tenant_id="tenant_b", user_id="user_shared_id", scopes=[ServiceScope.RAG_READ])

    local_scope_manager.grant_scope(
        tenant_id="tenant_a",
        user_id="user_shared_id",
        scope=ServiceScope.INGESTION_WRITE
    )

    scopes_tenant_a = local_scope_manager.get_user_scopes(tenant_id="tenant_a", user_id="user_shared_id")
    scopes_tenant_b = local_scope_manager.get_user_scopes(tenant_id="tenant_b", user_id="user_shared_id")

    assert ServiceScope.INGESTION_WRITE.value in scopes_tenant_a
    assert ServiceScope.INGESTION_WRITE.value not in scopes_tenant_b
    assert scopes_tenant_b == [ServiceScope.RAG_READ.value]

def test_rag_query_engine_with_scopeless_token():
    """
    End-to-end integration: verifies RAGQueryEngine successfully verifies a scopeless upstream token,
    resolves 'rag:read' locally, and executes guarded search.
    """
    vector_store = NamespaceVectorStore()
    engine = RAGQueryEngine(vector_store=vector_store)

    user = UserContext(
        user_id="frontend_consumer",
        tier=UserTier.PREMIUM,
        tenant_id="tenant_frontend",
        clearance_level=1
    )
    token = JWTTokenManager.create_token(user=user, scopes=None)

    # First user in tenant_frontend auto-bootstraps and has access
    result = engine.query(
        query_text="how to clean stainless steel",
        token=token
    )
    assert result["status"] == "SUCCESS"
    assert result["user_id"] == "frontend_consumer"
    assert result["user_tier"] == "premium"
    assert result["answer"] is not None
