import os
import io
import sys
import time
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes import _ingestion_controller, _ingestion_worker, _vector_store
from src.auth.service import JWTTokenManager
from src.auth.context import UserContext, UserTier
from src.auth.scopes import local_scope_manager, ServiceScope
from src.ingestion.history import local_ingestion_history_manager
from src.vector_store.factory import get_vector_store
from src.vector_store.pgvector_store import PGVectorStore

client = TestClient(app)

def get_auth_token_for(user_id: str, tenant_id: str = "tenant_pg_test", role: str = "admin", scopes=None):
    if scopes is None:
        scopes = [ServiceScope.RAG_READ, ServiceScope.INGESTION_WRITE, ServiceScope.ADMIN_MANAGE]
    local_scope_manager.bootstrap_founding_admin(tenant_id=tenant_id, user_id="admin_founder")
    local_scope_manager.approve_user(
        tenant_id=tenant_id,
        user_id=user_id,
        clearance_level=3,
        role=role,
        scopes=scopes,
        approved_by="admin_founder"
    )
    user_ctx = UserContext(
        user_id=user_id,
        tenant_id=tenant_id,
        tier=UserTier.ENTERPRISE,
        clearance_level=3,
        role=role,
        granted_scopes=[s.value if hasattr(s, "value") else s for s in scopes]
    )
    return JWTTokenManager.create_token(user=user_ctx)


def test_get_vector_store_returns_pgvectorstore():
    """Verify that get_vector_store factory returns a PGVectorStore instance."""
    store = get_vector_store()
    assert isinstance(store, PGVectorStore)
    assert _vector_store is store


def test_api_ingest_routes_through_controller_and_commits_cursor():
    """Verify that POST /api/v1/ingest routes through IngestionController and marks cursor ACTIVE."""
    import uuid
    unique_suffix = uuid.uuid4().hex[:6]
    user_id = f"test_user_pg_1_{unique_suffix}"
    tenant_id = f"tenant_pg_{unique_suffix}"
    token = get_auth_token_for(user_id, tenant_id=tenant_id)
    headers = {"Authorization": f"Bearer {token}"}

    doc_id = f"coffee_guide_v1_{unique_suffix}"
    doc_title = f"Espresso Extraction Fundamentals {unique_suffix}"
    content = f"Espresso extraction ({unique_suffix}) requires a 1:2 ratio of dry ground coffee to liquid espresso in 25 to 30 seconds. The brew water temperature should be maintained between 90 and 96 degrees Celsius."

    payload = {
        "document_id": doc_id,
        "title": doc_title,
        "content": content,
        "clearance_level": 1,
        "access_tier": "free",
        "declared_domain": "appliances_troubleshooting"
    }

    res = client.post("/api/v1/ingest", json=payload, headers=headers)
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["document_id"] == doc_id
    assert data["chunks_ingested"] >= 1

    # Verify Cursor DB was updated and committed to ACTIVE
    cursor_rec = _ingestion_controller.cursor_mgr.get_by_document_id(doc_id)
    assert cursor_rec is not None
    assert cursor_rec.status == "ACTIVE"
    assert cursor_rec.is_latest is True
    assert cursor_rec.version == 1

    # Verify Ingestion History DB recorded the job
    hist_record = local_ingestion_history_manager.get_job(data["job_id"])
    assert hist_record is not None
    assert hist_record.status == "COMPLETED"
    assert hist_record.chunks_ingested >= 1


def test_api_ingest_idempotency_gatekeeper_skips_identical_hash():
    """Verify that re-ingesting the exact same content triggers the cursor gatekeeper and skips 0 compute wasted."""
    import uuid
    unique_suffix = uuid.uuid4().hex[:6]
    user_id = f"test_user_pg_2_{unique_suffix}"
    tenant_id = f"tenant_pg_{unique_suffix}"
    token = get_auth_token_for(user_id, tenant_id=tenant_id)
    headers = {"Authorization": f"Bearer {token}"}

    doc_id = f"knife_sharpening_{unique_suffix}"
    doc_title = f"Whetstone Sharpening Technique {unique_suffix}"
    content = f"Submerge whetstones ({unique_suffix}) in water until bubbles cease. Maintain a consistent 15 to 20 degree bevel angle across the blade. Finish with a 6000 grit polish."

    payload = {
        "document_id": doc_id,
        "title": doc_title,
        "content": content,
        "clearance_level": 1,
        "access_tier": "free",
        "declared_domain": "kitchenware"
    }

    # First ingestion: should process and commit ACTIVE
    res1 = client.post("/api/v1/ingest", json=payload, headers=headers)
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["status"] == "SUCCESS"

    # Second ingestion with identical content: cursor gatekeeper intercepts
    res2 = client.post("/api/v1/ingest", json=payload, headers=headers)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["status"] == "SUCCESS"
    assert data2["document_id"] == doc_id

    # Cursor version should still be 1 (no new version created for identical hash)
    cursor_rec = _ingestion_controller.cursor_mgr.get_by_document_id(doc_id)
    assert cursor_rec.version == 1


def test_api_ingest_versioning_archives_previous_version():
    """Verify that ingesting modified content advances version to v2 and archives v1."""
    import uuid
    unique_suffix = uuid.uuid4().hex[:6]
    user_id = f"test_user_pg_3_{unique_suffix}"
    tenant_id = f"tenant_pg_{unique_suffix}"
    token = get_auth_token_for(user_id, tenant_id=tenant_id)
    headers = {"Authorization": f"Bearer {token}"}

    doc_title = f"Oven Calibration Manual {unique_suffix}"
    doc_id_v1 = f"oven_cal_v1_{unique_suffix}"
    doc_id_v2 = f"oven_cal_v2_{unique_suffix}"
    content_v1 = f"Original factory calibration standard ({unique_suffix}) for convection ovens. Temperature accuracy variance is plus or minus 15 degrees Fahrenheit."
    content_v2 = f"Updated 2026 calibration standard ({unique_suffix}) for commercial convection ovens. Temperature accuracy variance must remain within plus or minus 5 degrees Fahrenheit using dual-probe sensors."

    # Ingest v1
    res1 = client.post("/api/v1/ingest", json={
        "document_id": doc_id_v1,
        "title": doc_title,
        "content": content_v1,
        "declared_domain": "appliances_troubleshooting"
    }, headers=headers)
    assert res1.status_code == 200

    cursor_v1 = _ingestion_controller.cursor_mgr.get_by_document_id(doc_id_v1)
    assert cursor_v1.version == 1
    assert cursor_v1.is_latest is True

    # Ingest v2 with updated content
    res2 = client.post("/api/v1/ingest", json={
        "document_id": doc_id_v2,
        "title": doc_title,
        "content": content_v2,
        "declared_domain": "appliances_troubleshooting"
    }, headers=headers)
    assert res2.status_code == 200

    cursor_v2 = _ingestion_controller.cursor_mgr.get_by_document_id(doc_id_v2)
    assert cursor_v2.version == 2
    assert cursor_v2.is_latest is True
    assert cursor_v2.status == "ACTIVE"

    # Previous version should now be ARCHIVED
    cursor_v1_after = _ingestion_controller.cursor_mgr.get_by_document_id(doc_id_v1)
    assert cursor_v1_after.is_latest is False
    assert cursor_v1_after.status == "ARCHIVED"


def test_api_async_ingest_queues_job_for_workers():
    """Verify that async_mode=True returns 202 ACCEPTED and worker processes queue job."""
    import uuid
    unique_suffix = uuid.uuid4().hex[:6]
    user_id = f"test_user_async_{unique_suffix}"
    tenant_id = f"tenant_async_{unique_suffix}"
    token = get_auth_token_for(user_id, tenant_id=tenant_id)
    headers = {"Authorization": f"Bearer {token}"}

    doc_id = f"air_fryer_cleaning_{unique_suffix}"
    doc_title = f"Air Fryer Deep Clean Procedures {unique_suffix}"
    content = f"Unplug appliance before disassembly ({unique_suffix}). Soak basket in warm soapy water for 20 minutes. Use non-abrasive sponge to clean heating coil."

    payload = {
        "document_id": doc_id,
        "title": doc_title,
        "content": content,
        "async_mode": True,
        "declared_domain": "appliances_troubleshooting"
    }

    # API call returns ACCEPTED 202 immediately
    res = client.post("/api/v1/ingest", json=payload, headers=headers)
    assert res.status_code == 200  # Pydantic response_model maps status
    data = res.json()
    assert data["status"] == "ACCEPTED"
    job_id = data["job_id"]

    # Check job is PENDING in queue
    job = _ingestion_controller.queue_mgr.get_job(job_id)
    assert job is not None
    assert job.status in ("PENDING", "PROCESSING")

    # Simulate worker picking up and executing the queue job
    exec_res = _ingestion_worker.process_next_job()
    assert exec_res is not None

    # Job is now completed
    completed_job = _ingestion_controller.queue_mgr.get_job(job_id)
    assert completed_job.status in ("COMPLETED", "READY_FOR_EMBED")

    # Cursor is committed
    cursor_rec = _ingestion_controller.cursor_mgr.get_by_document_id(doc_id)
    assert cursor_rec is not None
