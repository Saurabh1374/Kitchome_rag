from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class StrictSingleDocumentIngestRequest(BaseModel):
    """
    Strictly validated single-document ingestion payload.
    Enforces required types, character constraints, and domain bounds.
    """
    document_id: str = Field(
        ...,
        min_length=3,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_\-\.]+$",
        description="Unique alphanumeric identifier for the document (e.g. 'doc_recipe_101')"
    )
    title: str = Field(
        ...,
        min_length=2,
        max_length=256,
        description="Human-readable title of the document"
    )
    content: Optional[str] = Field(
        default=None,
        min_length=10,
        description="Raw document text content to be stored and indexed"
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Optional path to raw document file on disk"
    )
    clearance_level: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Security classification level: 1 = Public, 2 = Internal, 3 = Confidential"
    )
    access_tier: str = Field(
        default="free",
        pattern=r"^(free|premium|scholar|enterprise)$",
        description="Subscription tier required to view this document: free, premium, scholar, or enterprise"
    )
    declared_domain: Optional[str] = Field(
        default=None,
        description="Optional target domain override (e.g. 'recipes_culinary', 'appliances_troubleshooting')"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata to attach to document chunks"
    )
    async_mode: bool = Field(
        default=False,
        description="If True, returns 202 Accepted and queues for background workers. If False (default), processes synchronously."
    )

class QueryRequest(BaseModel):
    """Payload for executing a guarded multi-namespace RAG retrieval query."""
    query_text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The natural language query string"
    )
    custom_clearance: Optional[int] = Field(
        default=None,
        ge=1,
        le=3,
        description="Optional clearance filter cap. Cannot exceed caller's assigned clearance level."
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of relevant chunks to retrieve"
    )

class ApproveUserRequest(BaseModel):
    """Payload for an administrator to approve a pending user account."""
    user_id: str = Field(..., min_length=1, description="Target user ID to approve")
    clearance_level: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Assigned security clearance level (1=Public, 2=Internal, 3=Confidential)"
    )
    role: str = Field(
        default="member",
        pattern=r"^(member|admin|editor|viewer)$",
        description="Assigned service role"
    )
    scopes: List[str] = Field(
        default=["rag:read"],
        description="List of functional scopes granted to the user (e.g. ['rag:read', 'ingestion:write'])"
    )

class RejectUserRequest(BaseModel):
    """Payload for an administrator to reject an account registration."""
    user_id: str = Field(..., min_length=1, description="Target user ID to reject")
    reason: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Human-readable reason for rejection"
    )

class UpdateAccessRequest(BaseModel):
    """Payload for dynamically modifying an approved user's clearance level, role, or scopes."""
    user_id: str = Field(..., min_length=1, description="Target user ID to update")
    clearance_level: Optional[int] = Field(default=None, ge=1, le=3)
    role: Optional[str] = Field(default=None, pattern=r"^(member|admin|editor|viewer)$")
    scopes: Optional[List[str]] = Field(default=None)

# -----------------------------------------------------------------------------
# Response Models
# -----------------------------------------------------------------------------

class IngestResponse(BaseModel):
    status: str
    document_id: str
    job_id: Optional[str] = None
    namespace: str
    chunks_ingested: int
    summary: Optional[str] = None
    file_path: Optional[str] = None
    trace_id: str

class IngestionHistoryItemResponse(BaseModel):
    job_id: str
    document_id: str
    title: str
    user_id: str
    tenant_id: str
    file_path: Optional[str] = None
    content_preview: Optional[str] = None
    content_hash: Optional[str] = None
    status: str
    clearance_level: int
    access_tier: str
    declared_domain: Optional[str] = None
    chunks_ingested: int
    namespace: Optional[str] = None
    summary: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int
    created_at: float
    updated_at: float

class IngestRetryResponse(BaseModel):
    status: str
    job_id: str
    document_id: str
    retry_count: int
    chunks_ingested: int = 0
    namespace: Optional[str] = None
    message: str
    error_message: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    status: str
    user_id: str
    tenant_id: str
    clearance_level: int
    role: str
    results_count: int
    retrieved_chunks: List[Dict[str, Any]]
    document_summaries: List[Dict[str, Any]]
    latency_ms: float
    trace_id: str

class UserProfileResponse(BaseModel):
    user_id: str
    tenant_id: str
    status: str
    role: str
    clearance_level: int
    granted_scopes: List[str]
    created_at: Optional[float] = None
    approved_at: Optional[float] = None
    rejection_reason: Optional[str] = None

class AdminActionResponse(BaseModel):
    status: str
    message: str
    user_id: str
    profile: Optional[Dict[str, Any]] = None
