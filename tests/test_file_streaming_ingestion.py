import os
import io
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes import get_domain_storage_path
from src.auth.service import JWTTokenManager
from src.auth.context import UserContext, UserTier
from src.auth.scopes import local_scope_manager, ServiceScope
from src.ingestion.history import local_ingestion_history_manager

client = TestClient(app)

def get_auth_token_for(user_id: str, tenant_id: str = "tenant_test", role: str = "admin", scopes=None):
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

def test_domain_storage_path_hierarchy_and_sanitization(tmp_path):
    base_dir = str(tmp_path)
    
    # 1. Global tenant
    p1 = get_domain_storage_path(
        declared_domain="recipes_culinary",
        tenant_id="global",
        document_id="doc_salmon_101",
        extension="md",
        base_dir=base_dir
    )
    expected_p1 = os.path.join(base_dir, "recipes_culinary", "doc_salmon_101.md")
    assert p1 == expected_p1
    assert os.path.exists(os.path.dirname(p1))

    # 2. Specific tenant
    p2 = get_domain_storage_path(
        declared_domain="appliances_troubleshooting",
        tenant_id="kitchen_corp",
        document_id="oven_manual_v2",
        extension="pdf",
        base_dir=base_dir
    )
    expected_p2 = os.path.join(base_dir, "appliances_troubleshooting", "kitchen_corp", "oven_manual_v2.pdf")
    assert p2 == expected_p2
    assert os.path.exists(os.path.dirname(p2))

    # 3. Traversal sanitization and default domain fallback
    p3 = get_domain_storage_path(
        declared_domain="../../etc/passwd",
        tenant_id="../tenant/hack",
        document_id="hack/../../id",
        extension="txt",
        base_dir=base_dir
    )
    # Must NOT contain directory traversal
    assert ".." not in p3
    assert base_dir in p3

def test_post_ingest_json_creates_domain_folder_and_stores_path():
    user_id = "test_ingest_user_1"
    tenant_id = "tenant_domain_test"
    token = get_auth_token_for(user_id, tenant_id=tenant_id)
    headers = {"Authorization": f"Bearer {token}"}

    doc_id = "recipe_crispy_salmon_domain"
    payload = {
        "document_id": doc_id,
        "title": "Crispy Skin Salmon Recipe",
        "content": "Score salmon skin lightly with a sharp knife. Pat completely dry with paper towels. Season with kosher salt right before placing skin-side down in a smoking hot stainless steel pan with avocado oil. Press gently for 30 seconds to prevent curling. Cook 80% through on the skin side.",
        "clearance_level": 1,
        "access_tier": "free",
        "declared_domain": "recipes_culinary"
    }

    res = client.post("/api/v1/ingest", json=payload, headers=headers)
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["document_id"] == doc_id
    assert "file_path" in data
    assert "recipes_culinary" in data["file_path"]
    assert tenant_id in data["file_path"]
    assert os.path.exists(data["file_path"])

    # Check history record
    job_id = data["job_id"]
    record = local_ingestion_history_manager.get_job(job_id)
    assert record is not None
    assert record.file_path == data["file_path"]
    assert record.content_preview is not None
    assert "Score salmon skin lightly" in record.content_preview
    assert record.content_hash is not None

def test_post_ingest_file_stream_multipart():
    user_id = "test_file_stream_user"
    tenant_id = "tenant_stream_test"
    token = get_auth_token_for(user_id, tenant_id=tenant_id)
    headers = {"Authorization": f"Bearer {token}"}

    doc_id = "appl_blender_guide"
    doc_content = b"Commercial Blender Operating Instructions.\n\nEnsure jar is firmly locked into the motor base before powering on. Never blend boiling liquids with the lid plug sealed to avoid pressure build-up. Clean with warm water and a drop of dish soap on pulse mode for 30 seconds."
    
    files = {
        "file": ("blender_guide.txt", io.BytesIO(doc_content), "text/plain")
    }
    form_data = {
        "document_id": doc_id,
        "title": "Commercial Blender User Guide",
        "declared_domain": "appliances_troubleshooting",
        "clearance_level": "2",
        "access_tier": "premium"
    }

    res = client.post("/api/v1/ingest/file", files=files, data=form_data, headers=headers)
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["document_id"] == doc_id
    assert "appliances_troubleshooting" in data["file_path"]
    assert tenant_id in data["file_path"]
    assert os.path.exists(data["file_path"])

    # Verify history ledger entry
    job_id = data["job_id"]
    record = local_ingestion_history_manager.get_job(job_id)
    assert record is not None
    assert record.file_path == data["file_path"]
    assert record.content_preview is not None
    assert "Commercial Blender Operating Instructions" in record.content_preview
    assert record.content_hash is not None

def test_manual_retry_from_stored_file_path():
    user_id = "test_retry_stream_user"
    tenant_id = "tenant_stream_retry"
    token = get_auth_token_for(user_id, tenant_id=tenant_id)
    headers = {"Authorization": f"Bearer {token}"}

    doc_id = "recipe_roast_chicken"
    # 1. Ingest via JSON
    res = client.post("/api/v1/ingest", json={
        "document_id": doc_id,
        "title": "Herb Butter Roast Chicken",
        "content": "Stuff chicken cavity with fresh rosemary, thyme, garlic cloves, and halved lemon. Rub soft herb butter generously under the skin and over the outside. Roast at 200C (400F) for 60 to 75 minutes until internal temperature hits 74C (165F). Rest for 15 minutes before carving.",
        "clearance_level": 1,
        "access_tier": "free",
        "declared_domain": "recipes_culinary"
    }, headers=headers)
    assert res.status_code == 200
    job_id = res.json()["job_id"]

    # 2. Simulate failure in history
    local_ingestion_history_manager.record_failure(job_id, "Simulated transient network timeout")
    failed_record = local_ingestion_history_manager.get_job(job_id)
    assert failed_record.status == "FAILED"

    # 3. Retry via API
    retry_res = client.post(f"/api/v1/ingest/retry/{job_id}", headers=headers)
    assert retry_res.status_code == 200, retry_res.text
    data = retry_res.json()
    assert data["status"] == "SUCCESS"
    assert data["retry_count"] >= 1

    # Verify status transition in history
    final_record = local_ingestion_history_manager.get_job(job_id)
    assert final_record.status == "COMPLETED"
    assert final_record.error_message is None
