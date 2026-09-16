import os
import uuid
import pytest
from fastapi.testclient import TestClient

from src.auth.service import JWTTokenManager
from src.auth.context import UserContext, UserTier
from src.auth.scopes import ServiceScope, local_scope_manager
from src.ingestion.history import IngestionHistoryManager, local_ingestion_history_manager
from src.api.main import app

SECRET = "kitchome-default-secret-key-change-in-production"

@pytest.fixture(autouse=True)
def setup_secret(monkeypatch):
    monkeypatch.setenv("AUTH_JWT_SECRET", SECRET)
    JWTTokenManager.DEFAULT_SECRET = SECRET

def test_ingestion_history_manager_lifecycle(tmp_path):
    """Verifies low-level IngestionHistoryManager recording, querying, and retries."""
    db_file = str(tmp_path / "test_hist.db")
    mgr = IngestionHistoryManager(db_path=db_file)

    job = mgr.record_start(
        document_id="doc_test_1",
        title="Test Document 1",
        content="This is test content for ingestion history.",
        user_id="user_alpha",
        tenant_id="tenant_x",
        clearance_level=1,
        access_tier="free",
        declared_domain="general_home"
    )
    assert job.job_id.startswith("ing_")
    assert job.status == "IN_PROGRESS"
    assert job.retry_count == 0

    # Mark as failure
    failed_job = mgr.record_failure(job.job_id, "Simulated network timeout")
    assert failed_job.status == "FAILED"
    assert failed_job.error_message == "Simulated network timeout"

    # Query history
    history = mgr.get_history(tenant_id="tenant_x", user_id="user_alpha")
    assert len(history) == 1
    assert history[0].status == "FAILED"

    # Mark as success
    succ_job = mgr.record_success(job.job_id, namespace="general_home", chunks_ingested=3, summary="Short summary")
    assert succ_job.status == "COMPLETED"
    assert succ_job.chunks_ingested == 3
    assert succ_job.namespace == "general_home"

def test_api_ingest_records_history_and_filters_per_user():
    """Verifies that POST /api/v1/ingest automatically writes to the history ledger
    and GET /api/v1/ingest/history enforces strict per-user boundaries."""
    client = TestClient(app)
    tenant_id = f"tenant_{uuid.uuid4().hex[:6]}"
    user_a = f"user_a_{uuid.uuid4().hex[:4]}"
    user_b = f"user_b_{uuid.uuid4().hex[:4]}"

    # Bootstrap founding admin for tenant
    admin_id = f"admin_{uuid.uuid4().hex[:4]}"
    local_scope_manager.bootstrap_founding_admin(tenant_id, admin_id)

    # Approve User A and User B with ingestion:write scope
    local_scope_manager.approve_user(
        tenant_id=tenant_id,
        user_id=user_a,
        clearance_level=2,
        role="member",
        scopes=[ServiceScope.RAG_READ, ServiceScope.INGESTION_WRITE]
    )
    local_scope_manager.approve_user(
        tenant_id=tenant_id,
        user_id=user_b,
        clearance_level=2,
        role="member",
        scopes=[ServiceScope.RAG_READ, ServiceScope.INGESTION_WRITE]
    )

    token_a = JWTTokenManager.create_token(UserContext(user_id=user_a, tenant_id=tenant_id, tier=UserTier.PREMIUM))
    token_b = JWTTokenManager.create_token(UserContext(user_id=user_b, tenant_id=tenant_id, tier=UserTier.PREMIUM))
    token_admin = JWTTokenManager.create_token(UserContext(user_id=admin_id, tenant_id=tenant_id, tier=UserTier.ENTERPRISE))

    # User A ingests a document
    doc_a_id = f"doc_a_{uuid.uuid4().hex[:6]}"
    res_a = client.post(
        "/api/v1/ingest",
        json={
            "document_id": doc_a_id,
            "title": "Document by User A",
            "content": "Culinary guidance for slow cooking pot roast with root vegetables and red wine reduction.",
            "declared_domain": "recipes_culinary",
            "access_tier": "premium",
            "clearance_level": 1
        },
        headers={"Authorization": f"Bearer {token_a}"}
    )
    assert res_a.status_code == 200
    res_data_a = res_a.json()
    assert res_data_a["status"] == "SUCCESS"
    assert "job_id" in res_data_a

    # User B ingests a document
    doc_b_id = f"doc_b_{uuid.uuid4().hex[:6]}"
    res_b = client.post(
        "/api/v1/ingest",
        json={
            "document_id": doc_b_id,
            "title": "Document by User B",
            "content": "Induction range error codes and inverter board troubleshooting manual.",
            "declared_domain": "appliances_troubleshooting",
            "access_tier": "premium",
            "clearance_level": 1
        },
        headers={"Authorization": f"Bearer {token_b}"}
    )
    assert res_b.status_code == 200

    # User A queries history: must ONLY see doc_a
    hist_a = client.get("/api/v1/ingest/history", headers={"Authorization": f"Bearer {token_a}"})
    assert hist_a.status_code == 200
    docs_for_a = [item["document_id"] for item in hist_a.json()]
    assert doc_a_id in docs_for_a
    assert doc_b_id not in docs_for_a

    # User B queries history: must ONLY see doc_b
    hist_b = client.get("/api/v1/ingest/history", headers={"Authorization": f"Bearer {token_b}"})
    assert hist_b.status_code == 200
    docs_for_b = [item["document_id"] for item in hist_b.json()]
    assert doc_b_id in docs_for_b
    assert doc_a_id not in docs_for_b

    # Admin queries history with all_users=True: sees BOTH
    hist_admin = client.get("/api/v1/ingest/history?all_users=true", headers={"Authorization": f"Bearer {token_admin}"})
    assert hist_admin.status_code == 200
    docs_for_admin = [item["document_id"] for item in hist_admin.json()]
    assert doc_a_id in docs_for_admin
    assert doc_b_id in docs_for_admin

def test_manual_retry_api_and_ownership_enforcement():
    """Verifies manual retry API execution, retry count increment, status transition,
    and user ownership enforcement."""
    client = TestClient(app)
    tenant_id = f"tenant_{uuid.uuid4().hex[:6]}"
    user_x = f"user_x_{uuid.uuid4().hex[:4]}"
    user_y = f"user_y_{uuid.uuid4().hex[:4]}"
    admin_id = f"admin_{uuid.uuid4().hex[:4]}"

    local_scope_manager.bootstrap_founding_admin(tenant_id, admin_id)
    local_scope_manager.approve_user(tenant_id, user_x, clearance_level=2, role="member", scopes=[ServiceScope.INGESTION_WRITE])
    local_scope_manager.approve_user(tenant_id, user_y, clearance_level=2, role="member", scopes=[ServiceScope.INGESTION_WRITE])

    token_x = JWTTokenManager.create_token(UserContext(user_id=user_x, tenant_id=tenant_id, tier=UserTier.PREMIUM))
    token_y = JWTTokenManager.create_token(UserContext(user_id=user_y, tenant_id=tenant_id, tier=UserTier.PREMIUM))
    token_admin = JWTTokenManager.create_token(UserContext(user_id=admin_id, tenant_id=tenant_id, tier=UserTier.ENTERPRISE))

    # Create a FAILED ingestion job owned by user_x
    doc_id = f"failed_doc_{uuid.uuid4().hex[:6]}"
    job = local_ingestion_history_manager.record_start(
        document_id=doc_id,
        title="Failed Recipe Ingestion",
        content="French pastry dough technique: laminate with 82% European cultured butter for 5 turns.",
        user_id=user_x,
        tenant_id=tenant_id,
        clearance_level=1,
        access_tier="premium",
        declared_domain="recipes_culinary"
    )
    local_ingestion_history_manager.record_failure(job.job_id, "Embedding service connection timeout")

    # Verify initial status is FAILED
    initial_check = client.get("/api/v1/ingest/history", headers={"Authorization": f"Bearer {token_x}"})
    assert initial_check.status_code == 200
    matched_job = next(j for j in initial_check.json() if j["job_id"] == job.job_id)
    assert matched_job["status"] == "FAILED"
    assert matched_job["retry_count"] == 0

    # User Y attempts to retry User X's job -> 403 Forbidden
    unauth_retry = client.post(
        f"/api/v1/ingest/retry/{job.job_id}",
        headers={"Authorization": f"Bearer {token_y}"}
    )
    assert unauth_retry.status_code == 403
    assert "Access Denied" in unauth_retry.json()["detail"]

    # User X retries their own job -> 200 OK
    auth_retry = client.post(
        f"/api/v1/ingest/retry/{job.job_id}",
        headers={"Authorization": f"Bearer {token_x}"}
    )
    assert auth_retry.status_code == 200
    retry_data = auth_retry.json()
    assert retry_data["status"] == "SUCCESS"
    assert retry_data["retry_count"] == 1
    assert retry_data["chunks_ingested"] > 0

    # Verify history now reflects COMPLETED
    updated_check = client.get("/api/v1/ingest/history", headers={"Authorization": f"Bearer {token_x}"})
    updated_job = next(j for j in updated_check.json() if j["job_id"] == job.job_id)
    assert updated_job["status"] == "COMPLETED"
    assert updated_job["retry_count"] == 1
    assert updated_job["chunks_ingested"] > 0
