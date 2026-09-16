import os
import re
import sys
import time
import uuid
import hashlib
import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Header, HTTPException, status, Depends, UploadFile, File, Form

# Ensure repo root is in sys.path for direct execution or script invocation
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from ..auth.scopes import ServiceScope, local_scope_manager, UserApprovalStatus
    from ..auth.service import ServiceAuthGuard, JWTTokenManager
    from ..auth.context import UserContext, UserTier
    from ..rag.engine import RAGQueryEngine
    from ..ingestion.pipeline import IngestionPipeline
    from ..ingestion.loader import RawDocument
    from ..ingestion.history import local_ingestion_history_manager
    from ..ingestion.controller import IngestionController
    from ..ingestion.worker import IngestionWorker
    from ..vector_store.base import NamespaceVectorStore
    from ..vector_store.factory import get_vector_store
    from .models import (
        StrictSingleDocumentIngestRequest,
        QueryRequest,
        ApproveUserRequest,
        RejectUserRequest,
        UpdateAccessRequest,
        IngestResponse,
        IngestionHistoryItemResponse,
        IngestRetryResponse,
        QueryResponse,
        UserProfileResponse,
        AdminActionResponse
    )
except ImportError:
    from src.auth.scopes import ServiceScope, local_scope_manager, UserApprovalStatus
    from src.auth.service import ServiceAuthGuard, JWTTokenManager
    from src.auth.context import UserContext, UserTier
    from src.rag.engine import RAGQueryEngine
    from src.ingestion.pipeline import IngestionPipeline
    from src.ingestion.loader import RawDocument
    from src.ingestion.history import local_ingestion_history_manager
    from src.ingestion.controller import IngestionController
    from src.ingestion.worker import IngestionWorker
    from src.vector_store.base import NamespaceVectorStore
    from src.vector_store.factory import get_vector_store
    from src.api.models import (
        StrictSingleDocumentIngestRequest,
        QueryRequest,
        ApproveUserRequest,
        RejectUserRequest,
        UpdateAccessRequest,
        IngestResponse,
        IngestionHistoryItemResponse,
        IngestRetryResponse,
        QueryResponse,
        UserProfileResponse,
        AdminActionResponse
    )

try:
    from ..telemetry import metrics_registry, trace_span
except ImportError:
    from src.telemetry import metrics_registry, trace_span

logger = logging.getLogger("kitchome.api.routes")
router = APIRouter(prefix="/api/v1")

# Shared singletons (can be injected or overridden in tests)
_vector_store = get_vector_store()
_rag_engine = RAGQueryEngine(vector_store=_vector_store)
_ingestion_controller = IngestionController(vector_store=_vector_store)
_ingestion_worker = IngestionWorker(vector_store=_vector_store)
_ingestion_pipeline = IngestionPipeline(vector_store=_vector_store)

def get_auth_token(authorization: Optional[str] = Header(None, alias="Authorization")) -> Optional[str]:
    """Helper dependency to extract bearer token from Authorization header."""
    if not authorization:
        return None
    token = authorization.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    return token

def enforce_service_auth_perimeter(
    token: Optional[str],
    required_scope: Optional[ServiceScope] = None
) -> UserContext:
    """
    Validates token presence, cryptographic signature, functional scopes,
    and user onboarding lifecycle status.
    Maps authentication and authorization failures to standard HTTP status codes:
    - 401: Missing, malformed, expired, or invalid cryptographic signature.
    - 403: Pending administrator approval, rejected registration, or missing required scope.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization bearer token."
        )

    try:
        user = ServiceAuthGuard.enforce_service_auth(
            auth_header_or_token=token,
            required_scope=required_scope
        )
        return user
    except Exception as e:
        error_msg = str(e)
        lower_err = error_msg.lower()

        # 1. Onboarding: Pending administrator approval -> 403 Forbidden with onboarding URL
        if "pending administrator approval" in lower_err:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "status": "PENDING_APPROVAL",
                    "message": error_msg,
                    "onboarding_url": "/onboarding"
                }
            )

        # 2. Onboarding: Rejected registration -> 403 Forbidden
        if "registration has been rejected" in lower_err:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=error_msg
            )

        # 3. RBAC/Scope mismatch: Caller lacks required functional scope -> 403 Forbidden
        if "lacks required functional scope" in lower_err:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=error_msg
            )

        # 4. Token validation errors: Malformed JWT, expired token, signature mismatch, bad base64 -> 401 Unauthorized
        if (
            isinstance(e, ValueError)
            or "malformed jwt" in lower_err
            or "token has expired" in lower_err
            or "invalid cryptographic token" in lower_err
            or "invalid base64" in lower_err
            or "json" in lower_err
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=error_msg
            )

        # 5. Default PermissionError -> 403 Forbidden
        if isinstance(e, PermissionError):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=error_msg
            )

        # 6. Fallback for unexpected exceptions during auth verification
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {error_msg}"
        )

# -----------------------------------------------------------------------------
# 1. RAG Query API
# -----------------------------------------------------------------------------

@router.post("/query", response_model=QueryResponse, summary="Execute Guarded RAG Query")
def execute_query(
    request: QueryRequest,
    token: Optional[str] = Depends(get_auth_token)
):
    """
    Executes a guarded multi-namespace RAG retrieval query.
    Requires 'rag:read' scope and enforces local clearance levels and RBAC tier permissions.
    """
    user = enforce_service_auth_perimeter(token, required_scope=ServiceScope.RAG_READ)

    # Optional downscoping of clearance
    effective_clearance = user.clearance_level
    if request.custom_clearance is not None:
        if request.custom_clearance > user.clearance_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requested clearance ({request.custom_clearance}) exceeds caller's authorized clearance ({user.clearance_level})."
            )
        effective_clearance = request.custom_clearance

    query_user = UserContext(
        user_id=user.user_id,
        tier=user.tier,
        tenant_id=user.tenant_id,
        clearance_level=effective_clearance,
        custom_allowed_namespaces=user.custom_allowed_namespaces,
        trace_id=user.trace_id,
        role=user.role,
        granted_scopes=user.granted_scopes
    )

    t0 = time.perf_counter()
    result = _rag_engine.query(
        query_text=request.query_text,
        user_context=query_user,
        token=None,  # Auth already enforced at perimeter; token=None ensures query_user's effective_clearance is respected
        top_k=request.max_results
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    status_label = result.get("status", "SUCCESS")
    metrics_registry.rag_queries_total.inc(
        status=status_label,
        tenant_id=user.tenant_id or "global",
        clearance_level=str(effective_clearance)
    )
    metrics_registry.rag_query_duration_seconds.observe(
        elapsed_ms / 1000.0,
        status=status_label,
        tier=str(user.tier.value if hasattr(user.tier, "value") else user.tier)
    )

    retrieved = result.get("retrieved_chunks", [])
    return QueryResponse(
        query=request.query_text,
        answer=result.get("answer", "No answer generated."),
        status=result.get("status", "SUCCESS"),
        user_id=user.user_id,
        tenant_id=user.tenant_id,
        clearance_level=effective_clearance,
        role=user.role or "member",
        results_count=len(retrieved),
        retrieved_chunks=retrieved,
        document_summaries=result.get("document_summaries", []),
        latency_ms=result.get("total_latency_ms", elapsed_ms),
        trace_id=user.trace_id or ""
    )

def get_domain_storage_path(
    declared_domain: Optional[str],
    tenant_id: str,
    document_id: str,
    extension: str = "md",
    base_dir: str = "data"
) -> str:
    """
    Generates structured canonical storage path partitioned by declared domain:
    data/{declared_domain}/{tenant_id}/{document_id}.{ext}
    (or data/{declared_domain}/{document_id}.{ext} for global tenant).
    Sanitizes components to completely prevent path traversal attacks.
    """
    clean_domain = re.sub(r'[^a-zA-Z0-9_\-]', '_', (declared_domain or "general_home").strip().lower())
    clean_domain = re.sub(r'_+', '_', clean_domain).strip('_')
    if not clean_domain:
        clean_domain = "general_home"

    clean_tenant = re.sub(r'[^a-zA-Z0-9_\-]', '_', (tenant_id or "global").strip().lower())
    clean_tenant = re.sub(r'_+', '_', clean_tenant).strip('_') or "global"

    clean_doc_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', document_id.strip())
    clean_doc_id = re.sub(r'_+', '_', clean_doc_id).strip('_') or "doc"

    clean_ext = re.sub(r'[^a-zA-Z0-9]', '', extension.strip().lower().lstrip(".")) or "md"

    if clean_tenant and clean_tenant != "global":
        target_dir = os.path.join(base_dir, clean_domain, clean_tenant)
    else:
        target_dir = os.path.join(base_dir, clean_domain)

    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, f"{clean_doc_id}.{clean_ext}")

# -----------------------------------------------------------------------------
# 2. Strict Single-Document Ingestion API
# -----------------------------------------------------------------------------

@router.post("/ingest", response_model=IngestResponse, summary="Ingest Single Document with Strict Validation")
def ingest_single_document(
    request: StrictSingleDocumentIngestRequest,
    token: Optional[str] = Depends(get_auth_token)
):
    """
    Ingests exactly ONE document at a time with strict type validation.
    Stores the raw file under data/{declared_domain}/{tenant_id}/{doc_id}.md,
    records the file path in history, and executes streaming chunking.
    Requires 'ingestion:write' scope.
    """
    user = enforce_service_auth_perimeter(token, required_scope=ServiceScope.INGESTION_WRITE)

    # Clearance sanity: Non-admin users cannot ingest documents with clearance higher than their own
    if user.role != "admin" and request.clearance_level > user.clearance_level:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Cannot ingest document with clearance {request.clearance_level}; caller clearance is {user.clearance_level}."
        )

    if not request.content and not request.file_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'content' or 'file_path' must be provided."
        )

    # 1. Resolve or create target file path following declared domain folder structure
    if request.file_path and os.path.exists(request.file_path):
        target_file_path = request.file_path
        with open(target_file_path, "rb") as f:
            content_bytes = f.read()
        content_hash = hashlib.sha256(content_bytes).hexdigest()
        content_preview = content_bytes[:250].decode("utf-8", errors="ignore")
    else:
        target_file_path = get_domain_storage_path(
            declared_domain=request.declared_domain,
            tenant_id=user.tenant_id,
            document_id=request.document_id,
            extension="md"
        )
        content_str = request.content or ""
        with open(target_file_path, "w", encoding="utf-8") as f:
            f.write(content_str)
        content_hash = hashlib.sha256(content_str.encode("utf-8")).hexdigest()
        content_preview = content_str[:250]

    # 2. Feed into IngestionController (Cursor Idempotency & Queue Dispatch)
    doc_family = (request.title or request.document_id).lower().strip().replace(" ", "_")
    intake_res = _ingestion_controller.intake_document(
        file_path=target_file_path,
        title=request.title,
        doc_family=doc_family,
        declared_domain=request.declared_domain,
        access_tier=request.access_tier,
        tenant_id=user.tenant_id,
        clearance_level=request.clearance_level,
        document_id=request.document_id,
        metadata=request.metadata
    )

    # 3. Handle Idempotency Gatekeeper Skip (0 compute wasted)
    if intake_res.get("status") == "SKIPPED":
        active_summary = _vector_store.get_summary(request.document_id)
        summary_text = active_summary.summary_text if active_summary else None
        target_ns = active_summary.namespace if active_summary else (request.declared_domain or "general_home")

        domain_val = request.declared_domain or "general_home"
        metrics_registry.cursor_idempotency_skips_total.inc(domain=domain_val)
        metrics_registry.ingestion_jobs_total.inc(status="SKIPPED", domain=domain_val, tenant_id=user.tenant_id or "global")

        job = local_ingestion_history_manager.record_start(
            document_id=request.document_id,
            title=request.title,
            user_id=user.user_id,
            tenant_id=user.tenant_id,
            file_path=target_file_path,
            content_preview=content_preview,
            content_hash=content_hash,
            content=request.content or "",
            clearance_level=request.clearance_level,
            access_tier=request.access_tier,
            declared_domain=request.declared_domain,
            metadata={"skipped": True, "reason": "identical_hash", **(request.metadata or {})}
        )
        local_ingestion_history_manager.record_success(
            job_id=job.job_id,
            namespace=target_ns,
            chunks_ingested=intake_res.get("chunks_count", 0),
            summary=summary_text
        )
        return IngestResponse(
            status="SUCCESS",
            document_id=request.document_id,
            job_id=job.job_id,
            namespace=target_ns,
            chunks_ingested=intake_res.get("chunks_count", 0),
            summary=summary_text,
            file_path=target_file_path,
            trace_id=user.trace_id or ""
        )

    # 4. Accepted job: record start in history with matching job_id
    queue_job_id = intake_res["job_id"]
    job = local_ingestion_history_manager.record_start(
        job_id=queue_job_id,
        document_id=request.document_id,
        title=request.title,
        user_id=user.user_id,
        tenant_id=user.tenant_id,
        file_path=target_file_path,
        content_preview=content_preview,
        content_hash=content_hash,
        content=request.content or "",
        clearance_level=request.clearance_level,
        access_tier=request.access_tier,
        declared_domain=request.declared_domain,
        metadata=request.metadata
    )

    # 5. Async mode check
    if getattr(request, "async_mode", False):
        return IngestResponse(
            status="ACCEPTED",
            document_id=request.document_id,
            job_id=job.job_id,
            namespace=request.declared_domain or "general_home",
            chunks_ingested=0,
            summary="Queued for background worker ingestion.",
            file_path=target_file_path,
            trace_id=user.trace_id or ""
        )

    # 6. Synchronous mode: execute job through IngestionWorker
    queue_job = _ingestion_controller.queue_mgr.get_job(queue_job_id)
    try:
        if queue_job:
            exec_res = _ingestion_worker.execute_job(queue_job)
            if exec_res.get("status") == "FAILED":
                raise RuntimeError(exec_res.get("error", "Worker execution failed"))
        else:
            _ingestion_pipeline.ingest_file(
                file_path=target_file_path,
                document_id=request.document_id,
                title=request.title,
                tenant_id=user.tenant_id,
                clearance_level=request.clearance_level,
                access_tier=request.access_tier,
                declared_domain=request.declared_domain,
                metadata=request.metadata
            )

        cursor_rec = _ingestion_controller.cursor_mgr.get_by_document_id(request.document_id)
        target_ns = cursor_rec.resolved_namespace if cursor_rec else (request.declared_domain or "general_home")
        chunks_count = cursor_rec.chunks_count if cursor_rec else 0
        summary_rec = _vector_store.get_summary(request.document_id)
        doc_summary = summary_rec.summary_text if summary_rec else None

        local_ingestion_history_manager.record_success(
            job_id=job.job_id,
            namespace=target_ns,
            chunks_ingested=chunks_count,
            summary=doc_summary
        )
        domain_val = request.declared_domain or "general_home"
        metrics_registry.ingestion_jobs_total.inc(status="SUCCESS", domain=domain_val, tenant_id=user.tenant_id or "global")
        if chunks_count:
            metrics_registry.ingestion_chunks_total.inc(chunks_count, domain=domain_val)

        return IngestResponse(
            status="SUCCESS",
            document_id=request.document_id,
            job_id=job.job_id,
            namespace=target_ns,
            chunks_ingested=chunks_count,
            summary=doc_summary,
            file_path=target_file_path,
            trace_id=user.trace_id or ""
        )
    except Exception as e:
        local_ingestion_history_manager.record_failure(
            job_id=job.job_id,
            error_message=str(e)
        )
        logger.error(f"Ingestion failed for doc '{request.document_id}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {str(e)}"
        )

@router.post("/ingest/file", response_model=IngestResponse, summary="Stream Ingest Raw File with Domain-Based Path")
async def ingest_file_stream(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    declared_domain: Optional[str] = Form(None),
    clearance_level: int = Form(1),
    access_tier: str = Form("free"),
    async_mode: bool = Form(False),
    token: Optional[str] = Depends(get_auth_token)
):
    """
    Streams incoming raw file directly to disk under data/{declared_domain}/{tenant_id}/{doc_id}.{ext},
    calculates SHA-256 hash on-the-fly, and triggers bounded-RAM streaming chunking through IngestionController.
    Requires 'ingestion:write' scope.
    """
    user = enforce_service_auth_perimeter(token, required_scope=ServiceScope.INGESTION_WRITE)

    if user.role != "admin" and clearance_level > user.clearance_level:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Cannot ingest document with clearance {clearance_level}; caller clearance is {user.clearance_level}."
        )

    orig_name = file.filename or "uploaded_file"
    ext = os.path.splitext(orig_name)[1].lstrip(".") or "txt"
    doc_id = document_id or f"doc_{uuid.uuid4().hex[:8]}"
    doc_title = title or os.path.splitext(orig_name)[0].replace("_", " ").title()

    target_file_path = get_domain_storage_path(
        declared_domain=declared_domain,
        tenant_id=user.tenant_id,
        document_id=doc_id,
        extension=ext
    )

    hasher = hashlib.sha256()
    preview_bytes = b""
    total_bytes = 0

    with open(target_file_path, "wb") as f:
        while chunk := await file.read(65536):
            f.write(chunk)
            hasher.update(chunk)
            if len(preview_bytes) < 500:
                preview_bytes += chunk
            total_bytes += len(chunk)

    content_hash = hasher.hexdigest()
    content_preview = preview_bytes[:250].decode("utf-8", errors="ignore")

    doc_family = doc_title.lower().strip().replace(" ", "_")
    intake_res = _ingestion_controller.intake_document(
        file_path=target_file_path,
        title=doc_title,
        doc_family=doc_family,
        declared_domain=declared_domain,
        access_tier=access_tier,
        tenant_id=user.tenant_id,
        clearance_level=clearance_level,
        document_id=doc_id,
        metadata={"filename": orig_name, "file_size_bytes": total_bytes}
    )

    # Idempotency Gatekeeper Skip (0 compute wasted)
    if intake_res.get("status") == "SKIPPED":
        active_summary = _vector_store.get_summary(doc_id)
        summary_text = active_summary.summary_text if active_summary else None
        target_ns = active_summary.namespace if active_summary else (declared_domain or "general_home")

        job = local_ingestion_history_manager.record_start(
            document_id=doc_id,
            title=doc_title,
            user_id=user.user_id,
            tenant_id=user.tenant_id,
            file_path=target_file_path,
            content_preview=content_preview,
            content_hash=content_hash,
            clearance_level=clearance_level,
            access_tier=access_tier,
            declared_domain=declared_domain,
            metadata={"filename": orig_name, "file_size_bytes": total_bytes, "skipped": True, "reason": "identical_hash"}
        )
        local_ingestion_history_manager.record_success(
            job_id=job.job_id,
            namespace=target_ns,
            chunks_ingested=intake_res.get("chunks_count", 0),
            summary=summary_text
        )
        return IngestResponse(
            status="SUCCESS",
            document_id=doc_id,
            job_id=job.job_id,
            namespace=target_ns,
            chunks_ingested=intake_res.get("chunks_count", 0),
            summary=summary_text,
            file_path=target_file_path,
            trace_id=user.trace_id or ""
        )

    queue_job_id = intake_res["job_id"]
    job = local_ingestion_history_manager.record_start(
        job_id=queue_job_id,
        document_id=doc_id,
        title=doc_title,
        user_id=user.user_id,
        tenant_id=user.tenant_id,
        file_path=target_file_path,
        content_preview=content_preview,
        content_hash=content_hash,
        clearance_level=clearance_level,
        access_tier=access_tier,
        declared_domain=declared_domain,
        metadata={"filename": orig_name, "file_size_bytes": total_bytes}
    )

    if async_mode:
        return IngestResponse(
            status="ACCEPTED",
            document_id=doc_id,
            job_id=job.job_id,
            namespace=declared_domain or "general_home",
            chunks_ingested=0,
            summary="Queued for background worker ingestion.",
            file_path=target_file_path,
            trace_id=user.trace_id or ""
        )

    queue_job = _ingestion_controller.queue_mgr.get_job(queue_job_id)
    try:
        if queue_job:
            exec_res = _ingestion_worker.execute_job(queue_job)
            if exec_res.get("status") == "FAILED":
                raise RuntimeError(exec_res.get("error", "Worker execution failed"))
        else:
            _ingestion_pipeline.ingest_file(
                file_path=target_file_path,
                document_id=doc_id,
                title=doc_title,
                tenant_id=user.tenant_id,
                clearance_level=clearance_level,
                access_tier=access_tier,
                declared_domain=declared_domain,
                metadata={"filename": orig_name, "file_size_bytes": total_bytes}
            )

        cursor_rec = _ingestion_controller.cursor_mgr.get_by_document_id(doc_id)
        target_ns = cursor_rec.resolved_namespace if cursor_rec else (declared_domain or "general_home")
        chunks_count = cursor_rec.chunks_count if cursor_rec else 0
        summary_rec = _vector_store.get_summary(doc_id)
        doc_summary = summary_rec.summary_text if summary_rec else None

        local_ingestion_history_manager.record_success(
            job_id=job.job_id,
            namespace=target_ns,
            chunks_ingested=chunks_count,
            summary=doc_summary
        )
        return IngestResponse(
            status="SUCCESS",
            document_id=doc_id,
            job_id=job.job_id,
            namespace=target_ns,
            chunks_ingested=chunks_count,
            summary=doc_summary,
            file_path=target_file_path,
            trace_id=user.trace_id or ""
        )
    except Exception as e:
        local_ingestion_history_manager.record_failure(
            job_id=job.job_id,
            error_message=str(e)
        )
        logger.error(f"Streaming file ingestion failed for '{doc_id}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest file: {str(e)}"
        )

@router.get("/ingest/history", response_model=List[IngestionHistoryItemResponse], summary="Get Per-User Ingestion History")
def get_ingestion_history(
    status_filter: Optional[str] = None,
    all_users: bool = False,
    limit: int = 50,
    token: Optional[str] = Depends(get_auth_token)
):
    """
    Returns the caller's ingestion history.
    Strictly isolated per-user: callers only see their own submissions unless they have role='admin'
    and explicitly specify all_users=true.
    """
    user = enforce_service_auth_perimeter(token)

    target_user = None if (all_users and user.role == "admin") else user.user_id
    records = local_ingestion_history_manager.get_history(
        tenant_id=user.tenant_id,
        user_id=target_user,
        status=status_filter,
        limit=limit
    )

    return [
        IngestionHistoryItemResponse(
            job_id=r.job_id,
            document_id=r.document_id,
            title=r.title,
            user_id=r.user_id,
            tenant_id=r.tenant_id,
            status=r.status,
            clearance_level=r.clearance_level,
            access_tier=r.access_tier,
            declared_domain=r.declared_domain,
            chunks_ingested=r.chunks_ingested,
            namespace=r.namespace,
            summary=r.summary,
            error_message=r.error_message,
            retry_count=r.retry_count,
            created_at=r.created_at,
            updated_at=r.updated_at
        )
        for r in records
    ]

@router.post("/ingest/retry/{job_id}", response_model=IngestRetryResponse, summary="Manual Retry Failed Ingestion")
def retry_failed_ingestion(
    job_id: str,
    token: Optional[str] = Depends(get_auth_token)
):
    """
    Manually retries a failed document ingestion job.
    Requires 'ingestion:write' scope.
    Enforces per-user boundary: non-admins can only retry jobs that they submitted.
    """
    user = enforce_service_auth_perimeter(token, required_scope=ServiceScope.INGESTION_WRITE)

    record = local_ingestion_history_manager.get_job(job_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ingestion job '{job_id}' not found."
        )

    if record.tenant_id != user.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access Denied: Job belongs to a different tenant."
        )

    is_admin = (user.role == "admin")
    if not is_admin and record.user_id != user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access Denied: Ingestion job '{job_id}' was submitted by '{record.user_id}'. You cannot retry it."
        )

    result = local_ingestion_history_manager.retry_job(
        job_id=job_id,
        caller_user_id=user.user_id,
        is_admin=is_admin,
        pipeline=_ingestion_pipeline
    )

    if result["status"] == "FAILED":
        return IngestRetryResponse(
            status="FAILED",
            job_id=job_id,
            document_id=result["document_id"],
            retry_count=result["retry_count"],
            chunks_ingested=0,
            message=result.get("message", "Retry execution failed."),
            error_message=result.get("error_message")
        )

    return IngestRetryResponse(
        status="SUCCESS",
        job_id=job_id,
        document_id=result["document_id"],
        retry_count=result["retry_count"],
        chunks_ingested=result.get("chunks_ingested", 0),
        namespace=result.get("namespace"),
        message=result.get("message", "Job retried successfully.")
    )

# -----------------------------------------------------------------------------
# 3. User Identity & Onboarding Status
# -----------------------------------------------------------------------------

@router.get("/auth/me", response_model=UserProfileResponse, summary="Get Caller Identity & Permissions")
def get_caller_profile(token: Optional[str] = Depends(get_auth_token)):
    """Returns the caller's decoded identity, approval status, local clearance, role, and granted scopes."""
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization token.")

    try:
        payload = JWTTokenManager.verify_and_decode_token(token)
    except (PermissionError, ValueError, Exception) as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))

    user_id = payload.get("sub", "anonymous")
    tenant_id = payload.get("tenant_id", "global")

    profile = local_scope_manager.get_user_profile(tenant_id, user_id)
    if not profile:
        # Check if founding admin
        if not local_scope_manager.has_admin(tenant_id):
            profile = local_scope_manager.bootstrap_founding_admin(tenant_id, user_id)
        else:
            profile = local_scope_manager.register_pending_user(tenant_id, user_id)

    scopes = local_scope_manager.get_user_scopes(tenant_id, user_id)

    return UserProfileResponse(
        user_id=user_id,
        tenant_id=tenant_id,
        status=profile.get("status", UserApprovalStatus.PENDING_APPROVAL.value),
        role=profile.get("role", "member"),
        clearance_level=int(profile.get("clearance_level", 1)),
        granted_scopes=scopes,
        created_at=profile.get("created_at"),
        approved_at=profile.get("approved_at"),
        rejection_reason=profile.get("rejection_reason")
    )

# -----------------------------------------------------------------------------
# 4. Administrator Governance Endpoints
# -----------------------------------------------------------------------------

@router.get("/admin/pending", response_model=List[Dict[str, Any]], summary="List Pending Onboarding Requests")
def list_pending_approvals(token: Optional[str] = Depends(get_auth_token)):
    """Lists all user accounts in PENDING_APPROVAL status for the caller's tenant."""
    user = enforce_service_auth_perimeter(token, required_scope=ServiceScope.ADMIN_MANAGE)
    return local_scope_manager.list_pending_users(tenant_id=user.tenant_id)

@router.post("/admin/approve", response_model=AdminActionResponse, summary="Approve Pending User")
def approve_pending_user(
    request: ApproveUserRequest,
    token: Optional[str] = Depends(get_auth_token)
):
    """
    Administrator action to approve a user.
    Sets their clearance level (1-3), service role, and granted scopes.
    """
    user = enforce_service_auth_perimeter(token, required_scope=ServiceScope.ADMIN_MANAGE)

    profile = local_scope_manager.approve_user(
        tenant_id=user.tenant_id,
        user_id=request.user_id,
        clearance_level=request.clearance_level,
        role=request.role,
        scopes=request.scopes,
        approved_by=user.user_id
    )

    return AdminActionResponse(
        status="SUCCESS",
        message=f"User '{request.user_id}' successfully approved with clearance {request.clearance_level} and role '{request.role}'.",
        user_id=request.user_id,
        profile=profile
    )

@router.post("/admin/reject", response_model=AdminActionResponse, summary="Reject User Registration")
def reject_user(
    request: RejectUserRequest,
    token: Optional[str] = Depends(get_auth_token)
):
    """Administrator action to reject a user registration with an optional reason."""
    user = enforce_service_auth_perimeter(token, required_scope=ServiceScope.ADMIN_MANAGE)

    profile = local_scope_manager.reject_user(
        tenant_id=user.tenant_id,
        user_id=request.user_id,
        rejected_by=user.user_id,
        reason=request.reason
    )

    return AdminActionResponse(
        status="SUCCESS",
        message=f"User '{request.user_id}' has been rejected.",
        user_id=request.user_id,
        profile=profile
    )

@router.post("/admin/update-access", response_model=AdminActionResponse, summary="Update User Permissions")
def update_user_access(
    request: UpdateAccessRequest,
    token: Optional[str] = Depends(get_auth_token)
):
    """Dynamically updates clearance level, role, or scopes for an active user."""
    user = enforce_service_auth_perimeter(token, required_scope=ServiceScope.ADMIN_MANAGE)

    try:
        profile = local_scope_manager.update_user_access(
            tenant_id=user.tenant_id,
            user_id=request.user_id,
            clearance_level=request.clearance_level,
            role=request.role,
            scopes=request.scopes,
            updated_by=user.user_id
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    return AdminActionResponse(
        status="SUCCESS",
        message=f"User '{request.user_id}' access permissions successfully updated.",
        user_id=request.user_id,
        profile=profile
    )

@router.get("/admin/users", response_model=List[Dict[str, Any]], summary="List All Tenant Users")
def list_tenant_users(token: Optional[str] = Depends(get_auth_token)):
    """Lists all user profiles in the caller's tenant."""
    user = enforce_service_auth_perimeter(token, required_scope=ServiceScope.ADMIN_MANAGE)

    with local_scope_manager._get_connection() as conn:
        cur = conn.execute(
            "SELECT * FROM service_user_profiles WHERE tenant_id = ? ORDER BY created_at DESC;",
            (user.tenant_id,)
        )
        return [dict(row) for row in cur.fetchall()]

