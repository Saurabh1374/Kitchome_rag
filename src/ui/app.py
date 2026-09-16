import os
import sys
import time
from typing import Optional, List, Dict, Any

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import hashlib
from src.auth.context import UserContext, UserTier
from src.auth.scopes import ServiceScope, local_scope_manager, UserApprovalStatus
from src.auth.service import JWTTokenManager, ServiceAuthGuard, auth_telemetry, verify_token_hybrid
from src.common.redis_client import redis_manager
from src.rag.engine import RAGQueryEngine
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.loader import RawDocument
from src.ingestion.history import local_ingestion_history_manager
from src.vector_store.base import NamespaceVectorStore
from src.vector_store.factory import get_vector_store
from src.api.routes import get_domain_storage_path
from src.telemetry.tracing import bind_trace_context, generate_trace_id

# Page configuration with stylish cooking/kitchen favicon
st.set_page_config(
    page_title="Kitchome RAG Intelligence",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Correlated Distributed Trace Context for Streamlit Execution Cycle
# -----------------------------------------------------------------------------
if "_session_trace_id" not in st.session_state:
    st.session_state["_session_trace_id"] = generate_trace_id()

session_trace_id = st.session_state["_session_trace_id"]
active_ui_span = bind_trace_context(
    name="streamlit_ui_cycle",
    trace_id=session_trace_id,
    attributes={"component": "streamlit_ui", "app": "kitchome-rag"}
)


# Custom styling for high-aesthetic look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF6B6B, #FF8E53, #FFA07A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #888888;
        margin-bottom: 1.5rem;
    }
    .badge-approved {
        background-color: #2e7d32;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-pending {
        background-color: #ed6c02;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-rejected {
        background-color: #d32f2f;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-clearance-1 {
        background-color: #0288d1;
        color: white;
        padding: 3px 8px;
        border-radius: 8px;
        font-size: 0.78rem;
    }
    .badge-clearance-2 {
        background-color: #7b1fa2;
        color: white;
        padding: 3px 8px;
        border-radius: 8px;
        font-size: 0.78rem;
    }
    .badge-clearance-3 {
        background-color: #c2185b;
        color: white;
        padding: 3px 8px;
        border-radius: 8px;
        font-size: 0.78rem;
    }
    .scope-pill {
        background-color: #333333;
        color: #e0e0e0;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.78rem;
        margin-right: 4px;
    }
    .card-box {
        border: 1px solid #3d3d3d;
        border-radius: 10px;
        padding: 16px;
        background-color: #1e1e1e;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Service Singletons (cached in session state)
# -----------------------------------------------------------------------------
@st.cache_resource
def get_services():
    store = get_vector_store()
    engine = RAGQueryEngine(vector_store=store)
    pipeline = IngestionPipeline(vector_store=store)
    return store, engine, pipeline

vector_store, rag_engine, ingestion_pipeline = get_services()

# -----------------------------------------------------------------------------
# Central Identity Single Sign-In (SSO) Gate
# -----------------------------------------------------------------------------
# Check for incoming JWT in query params (from http://localhost:8080/login redirect)
incoming_jwt = st.query_params.get("jwt")
if incoming_jwt:
    try:
        verified_payload = verify_token_hybrid(incoming_jwt)
        st.session_state["jwt_token"] = incoming_jwt
        st.session_state["user"] = verified_payload
        # Clean URL bar immediately
        st.query_params.clear()
    except Exception as e:
        st.sidebar.error(f"Authentication Error: {e}")
        st.session_state.pop("jwt_token", None)
        st.session_state.pop("user", None)

user_claims = st.session_state.get("user") or {}
active_token = st.session_state.get("jwt_token") or ""

# If unauthenticated, display the Authentication Gate and halt execution
if not user_claims or not active_token:
    st.markdown('<div class="main-header">🍳 KitChome RAG Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Centralized Enterprise Identity & Knowledge Retrieval Gateway</div>', unsafe_allow_html=True)

    auth_login_url = os.getenv("AUTH_LOGIN_URL", "http://localhost:8080/login") + "?redirect_uri=http://localhost:8501"
    auth_register_url = os.getenv("AUTH_REGISTER_URL", "http://localhost:8080/register") + "?redirect_uri=http://localhost:8501"

    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 2.5rem; text-align: center; max-width: 650px; margin: 3rem auto; box-shadow: 0 20px 40px rgba(0,0,0,0.5);">
        <div style="font-size: 3.5rem; margin-bottom: 1rem;">🔐</div>
        <h2 style="margin-bottom: 0.5rem; color: #fff; font-weight: 700;">Authentication Required</h2>
        <p style="color: #94a3b8; font-size: 1rem; line-height: 1.6; margin-bottom: 2rem;">
            Access to KitChome's multi-tenant document indexes, intelligence studio, and single-document ingestion pipeline requires Single Sign-In via Central Auth.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.link_button("🔑 Sign In with KitChome Auth", auth_login_url, type="primary", use_container_width=True)
        st.markdown(
            f'<div style="text-align: center; margin-top: 1rem;">'
            f'<a href="{auth_register_url}" style="color: #60a5fa; text-decoration: none; font-size: 0.95rem;">'
            f'Don\'t have an account? <b>Sign up (First user becomes Admin)</b>'
            f'</a></div>',
            unsafe_allow_html=True
        )

    st.stop()

# -----------------------------------------------------------------------------
# Authenticated Session & Identity Hydration
# -----------------------------------------------------------------------------
active_user_id = user_claims.get("sub", "anonymous")
active_email = user_claims.get("email", "")
tenant_id = user_claims.get("tenant_id", "default")
active_tier_str = user_claims.get("tier", "free")
roles = user_claims.get("roles", [])
is_admin = any(r in ("ADMIN", "ROLE_ADMIN") for r in roles) or user_claims.get("role") == "admin"

try:
    active_tier = UserTier(active_tier_str.lower())
except Exception:
    active_tier = UserTier.FREE

active_ui_span.set_attribute("user_id", active_user_id)
active_ui_span.set_attribute("tenant_id", tenant_id)
active_ui_span.set_attribute("role", "admin" if is_admin else "member")

# Synchronize with local service database
profile = local_scope_manager.get_user_profile(tenant_id, active_user_id)
if is_admin:
    if not profile or profile.get("role") != "admin" or profile.get("status") != UserApprovalStatus.APPROVED.value:
        profile = local_scope_manager.approve_user(
            tenant_id=tenant_id,
            user_id=active_user_id,
            clearance_level=3,
            role="admin",
            scopes=[ServiceScope.ALL]
        )
elif not profile:
    profile = local_scope_manager.register_pending_user(tenant_id, active_user_id)

user_status = profile.get("status", UserApprovalStatus.PENDING_APPROVAL.value)
user_role = profile.get("role", "member")
user_clearance = int(profile.get("clearance_level", 1))
user_scopes = local_scope_manager.get_user_scopes(tenant_id, active_user_id)

# Persist authenticated session to Redis on 192.168.0.117 (async fire-and-forget, only on credential change)
if st.session_state.get("_last_synced_token") != active_token:
    st.session_state["_last_synced_token"] = active_token
    redis_manager.save_user_session(active_user_id, {
        "user_id": active_user_id,
        "email": active_email,
        "tenant_id": tenant_id,
        "tier": active_tier_str,
        "role": user_role,
        "clearance": user_clearance,
        "jwt_token": active_token
    })

# -----------------------------------------------------------------------------
# Sidebar: Authenticated Identity & Immediate Single Sign-Out
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🔐 Central Identity SSO")
redis_indicator = "🟢 Redis Live (192.168.0.117)" if redis_manager.is_available() else "🟡 In-Memory Mode"
st.sidebar.caption(f"Secured with Asymmetric JWKS (RS256) · {redis_indicator}")

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🪪 Caller Identity Card")
st.sidebar.markdown(f"**User:** `{active_user_id}`")
if active_email:
    st.sidebar.markdown(f"**Email:** `{active_email}`")
st.sidebar.markdown(f"**Tenant:** `{tenant_id}`")
st.sidebar.markdown(f"**Tier:** `{active_tier.value.upper()}`")

status_badge = {
    UserApprovalStatus.APPROVED.value: '<span class="badge-approved">● APPROVED</span>',
    UserApprovalStatus.PENDING_APPROVAL.value: '<span class="badge-pending">▲ PENDING APPROVAL</span>',
    UserApprovalStatus.REJECTED.value: '<span class="badge-rejected">✖ REJECTED</span>'
}.get(user_status, f"<span>{user_status}</span>")

clearance_badge = {
    1: '<span class="badge-clearance-1">🛡️ Level 1 (Public)</span>',
    2: '<span class="badge-clearance-2">🔒 Level 2 (Internal)</span>',
    3: '<span class="badge-clearance-3">🚨 Level 3 (Confidential)</span>'
}.get(user_clearance, f"<span>Level {user_clearance}</span>")

role_icon = "👑 Admin" if user_role == "admin" else "👤 Member"

st.sidebar.markdown(f"**Status:** {status_badge}", unsafe_allow_html=True)
st.sidebar.markdown(f"**Role:** `{role_icon}`")
st.sidebar.markdown(f"**Clearance:** {clearance_badge}", unsafe_allow_html=True)

st.sidebar.markdown("**Granted Scopes:**")
scopes_html = "".join([f'<span class="scope-pill">{s}</span>' for s in user_scopes])
st.sidebar.markdown(scopes_html, unsafe_allow_html=True)

st.sidebar.markdown("---")
if st.sidebar.button("🚪 Sign Out (Single Sign-Out)", type="secondary", use_container_width=True):
    if active_token:
        # Instant cluster-wide token revocation in Redis (192.168.0.117)
        redis_manager.revoke_token(active_token)
    redis_manager.delete_user_session(active_user_id)
    st.session_state.clear()
    st.query_params.clear()
    # Immediate browser redirect to http://localhost:8080/login?logout
    st.markdown(
        f'<meta http-equiv="refresh" content="0; url={os.getenv("AUTH_LOGIN_URL", "http://localhost:8080/login")}?logout">',
        unsafe_allow_html=True
    )
    st.stop()

# -----------------------------------------------------------------------------
# Main Application Layout
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🍳 Kitchome RAG Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Tenant Document Retrieval, Single-Doc Ingestion, and Local Clearance Governance</div>', unsafe_allow_html=True)

# =============================================================================
# View 1: PENDING_APPROVAL Waiting Room
# =============================================================================
if user_status == UserApprovalStatus.PENDING_APPROVAL.value:
    st.warning("⚠️ **Account Pending Administrator Approval**")
    st.markdown("""
    ### ⏳ Welcome to Kitchome RAG!
    Your identity has been authenticated via the Central Auth Service, but your account requires authorization from your organization's administrator before accessing document indexes.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"""
        **Application Details:**
        - **User ID:** `{active_user_id}`
        - **Organization / Tenant:** `{tenant_id}`
        - **Initial Assigned Clearance:** Level 1 (Default)
        - **Requested Service:** Kitchome Culinary & Appliance RAG
        """)
    with col2:
        st.markdown("#### 📬 What happens next?")
        st.write("1. The tenant administrator will review your account.")
        st.write("2. They will configure your clearance level (Public, Internal, or Confidential).")
        st.write("3. They will grant functional scopes (such as `rag:read` or `ingestion:write`).")
        if st.button("🔄 Check Approval Status"):
            st.rerun()

# =============================================================================
# View 2: REJECTED Notice
# =============================================================================
elif user_status == UserApprovalStatus.REJECTED.value:
    st.error("🚫 **Access Registration Rejected**")
    reason = profile.get("rejection_reason") or "No specific reason provided."
    st.markdown(f"""
    Your request for access to tenant **`{tenant_id}`** was declined by the administrator.
    
    **Reason:** *{reason}*
    
    Please contact your IT administrator or security team if you believe this is an error.
    """)

# =============================================================================
# View 3: APPROVED User (Role-Based Tabs)
# =============================================================================
else:
    # Construct Tab Navigation based on user role
    tabs = ["🔍 RAG Query Studio", "📄 Document Ingestion", "📜 Ingestion History", "👤 My Clearance Profile"]
    if user_role == "admin":
        tabs.append("👑 Admin Command Center")

    tab_objects = st.tabs(tabs)

    # -------------------------------------------------------------------------
    # Tab 1: RAG Query Studio
    # -------------------------------------------------------------------------
    with tab_objects[0]:
        st.markdown("### 🔍 Intelligent Knowledge Search")
        st.caption(f"Searching index for tenant `{tenant_id}` with clearance up to Level {user_clearance}.")

        col_q, col_opts = st.columns([3, 1])
        with col_q:
            query_input = st.text_input(
                "Search query",
                value="How should I clean and maintain an induction glass cooktop?",
                placeholder="Ask about recipes, cookware, appliances, troubleshooting..."
            )
        with col_opts:
            selected_clearance = st.slider("Filter Clearance", min_value=1, max_value=user_clearance, value=user_clearance)
            max_chunks = st.slider("Max Results", min_value=1, max_value=10, value=3)

        if st.button("🚀 Search Knowledge Base", type="primary"):
            with st.spinner("Retrieving relevant passages and synthesizing answer..."):
                t0 = time.perf_counter()
                try:
                    q_user = UserContext(
                        user_id=active_user_id,
                        tier=active_tier,
                        tenant_id=tenant_id,
                        clearance_level=selected_clearance,
                        role=user_role,
                        granted_scopes=user_scopes
                    )
                    res = rag_engine.query(
                        query_text=query_input,
                        user_context=q_user,
                        token=active_token,
                        top_k=max_chunks
                    )
                    elapsed = (time.perf_counter() - t0) * 1000

                    st.markdown("#### 💡 Synthesized Response")
                    st.success(res.get("answer", "No answer generated."))

                    # Metrics Bar
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Retrieved Chunks", len(res.get("retrieved_chunks", [])))
                    m2.metric("Execution Time", f"{elapsed:.1f} ms")
                    m3.metric("Effective Clearance", f"Level {selected_clearance}")
                    m4.metric("Status", res.get("status", "SUCCESS"))

                    # Source Chunks
                    chunks = res.get("retrieved_chunks", [])
                    if chunks:
                        st.markdown("#### 📚 Source Citations & Matched Chunks")
                        for idx, c in enumerate(chunks):
                            with st.expander(f"Chunk #{idx+1} — Score: {c.get('similarity_score', 0):.3f} | Namespace: {c.get('namespace')}"):
                                st.write(c.get("text"))
                                st.caption(f"Metadata: Title={c.get('metadata', {}).get('document_title')} | Tier={c.get('metadata', {}).get('access_tier')} | Clearance={c.get('metadata', {}).get('clearance_level')}")
                    else:
                        st.info("No matching chunks found within your current tier and clearance filters.")
                except PermissionError as e:
                    st.error(f"Access Denied: {e}")

    # -------------------------------------------------------------------------
    # Tab 2: Document Ingestion Studio (Role & Scope Protected)
    # -------------------------------------------------------------------------
    with tab_objects[1]:
        st.markdown("### 📄 Single-Document Ingestion Studio")
        st.caption("Strictly validated, one-document-at-a-time intake with dynamic skills.ms summarization.")

        # Check if caller has 'ingestion:write' scope
        has_ingestion_scope = (
            ServiceScope.INGESTION_WRITE.value in user_scopes or 
            ServiceScope.ALL.value in user_scopes or 
            user_role == "admin"
        )

        if not has_ingestion_scope:
            st.error("🔒 **Ingestion Scope Required (`ingestion:write`)**")
            st.markdown("""
            Your account currently has **Read-Only** access (`rag:read`).
            Ingestion of proprietary recipes or technical documentation requires the **`ingestion:write`** scope.
            """)
            if st.button("📩 Request Ingestion Scope from Administrator"):
                st.info("Ingestion access request submitted to your tenant administrator!")
        else:
            st.success("✅ **Ingestion Authorized**: You have `ingestion:write` capability.")
            intake_mode = st.radio("Intake Method", ["📝 Direct Text / Markdown", "📁 Upload File (.pdf, .md, .txt, .json)"], horizontal=True)

            with st.form("single_doc_ingest_form"):
                f_col1, f_col2 = st.columns(2)
                with f_col1:
                    doc_id = st.text_input("Document ID*", value="recipe_sous_vide_salmon", help="Alphanumeric slug, min 3 chars")
                    doc_title = st.text_input("Document Title*", value="Sous Vide Wild Salmon with Dill Butter")
                    doc_domain = st.selectbox("Target Domain", ["recipes_culinary", "appliances_troubleshooting", "home_decor_design", "general_home"])
                with f_col2:
                    # Clearance cannot exceed caller's clearance
                    max_clear = user_clearance if user_role != "admin" else 3
                    doc_clearance = st.selectbox(
                        "Document Clearance Level*",
                        options=list(range(1, max_clear + 1)),
                        format_func=lambda x: {1: "1 - Public", 2: "2 - Internal", 3: "3 - Confidential"}[x]
                    )
                    doc_tier = st.selectbox("Access Tier*", ["free", "premium", "scholar", "enterprise"])
                    doc_tags = st.text_input("Tags (comma separated)", value="fish, sous-vide, gourmet")

                uploaded_file = None
                doc_content = ""
                if intake_mode == "📁 Upload File (.pdf, .md, .txt, .json)":
                    uploaded_file = st.file_uploader("Select File to Stream & Ingest", type=["pdf", "md", "txt", "json"])
                else:
                    doc_content = st.text_area(
                        "Raw Content / Markdown*",
                        height=180,
                        value="Sous vide cooking ensures delicate fish proteins do not coagulate harshly. Set water bath to 46°C (115°F) for tender wild sockeye salmon. Vacuum seal with clarified butter, fresh dill sprigs, and lemon zest. Cook for 35 minutes, then finish with a 30-second sear on cast iron for color."
                    )

                submitted = st.form_submit_button("🚀 Ingest & Index Document", type="primary")

                if submitted:
                    if len(doc_id) < 3:
                        st.error("Validation Error: Document ID must be at least 3 characters.")
                    elif len(doc_title) < 2:
                        st.error("Validation Error: Document title must be at least 2 characters.")
                    elif intake_mode == "📁 Upload File (.pdf, .md, .txt, .json)" and uploaded_file is None:
                        st.error("Validation Error: Please select a file to upload.")
                    elif intake_mode == "📝 Direct Text / Markdown" and len(doc_content) < 10:
                        st.error("Validation Error: Content must be at least 10 characters.")
                    else:
                        with st.spinner("Streaming document to domain folder, generating summary, and indexing chunks..."):
                            if uploaded_file is not None:
                                ext = os.path.splitext(uploaded_file.name)[1].lstrip(".") or "txt"
                                target_file_path = get_domain_storage_path(doc_domain, tenant_id, doc_id, ext)
                                file_bytes = uploaded_file.getvalue()
                                with open(target_file_path, "wb") as f:
                                    f.write(file_bytes)
                                content_hash = hashlib.sha256(file_bytes).hexdigest()
                                content_preview = file_bytes[:250].decode("utf-8", errors="ignore")
                            else:
                                target_file_path = get_domain_storage_path(doc_domain, tenant_id, doc_id, "md")
                                with open(target_file_path, "w", encoding="utf-8") as f:
                                    f.write(doc_content)
                                content_hash = hashlib.sha256(doc_content.encode("utf-8")).hexdigest()
                                content_preview = doc_content[:250]

                            hist_record = local_ingestion_history_manager.record_start(
                                document_id=doc_id,
                                title=doc_title,
                                user_id=active_user_id,
                                tenant_id=tenant_id,
                                file_path=target_file_path,
                                content_preview=content_preview,
                                content_hash=content_hash,
                                clearance_level=doc_clearance,
                                access_tier=doc_tier,
                                declared_domain=doc_domain,
                                metadata={"tags": [t.strip() for t in doc_tags.split(",") if t.strip()]}
                            )

                            try:
                                res = ingestion_pipeline.ingest_file(
                                    file_path=target_file_path,
                                    document_id=doc_id,
                                    title=doc_title,
                                    tenant_id=tenant_id,
                                    clearance_level=doc_clearance,
                                    access_tier=doc_tier,
                                    declared_domain=doc_domain,
                                    metadata={"tags": [t.strip() for t in doc_tags.split(",") if t.strip()]}
                                )
                                local_ingestion_history_manager.record_success(
                                    job_id=hist_record.job_id,
                                    namespace=res.get("namespace", "general_home"),
                                    chunks_ingested=res.get("chunks_ingested", 0),
                                    summary=res.get("summary")
                                )
                                st.success(f"🎉 Successfully ingested `{doc_id}` into namespace `{res.get('namespace')}`!")
                                st.caption(f"📁 Stored at: `{target_file_path}`")
                                st.info(f"**Auto-Generated Document Summary:** {res.get('summary')}")
                                st.metric("Chunks Indexed", res.get("chunks_ingested", 0))
                            except Exception as e:
                                local_ingestion_history_manager.record_failure(
                                    job_id=hist_record.job_id,
                                    error_message=str(e)
                                )
                                st.error(f"Ingestion failed: {e}")

    # -------------------------------------------------------------------------
    # Tab 3: Unified Ingestion History & DLQ Recovery
    # -------------------------------------------------------------------------
    with tab_objects[2]:
        st.markdown("### 📜 Unified Ingestion History & DLQ Recovery")
        st.caption(f"Per-user ledger tracking document intake, indexing status, and failure recovery for `{active_user_id}` in tenant `{tenant_id}`.")

        # Top Bar: Filters & Controls
        col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
        with col_f1:
            status_filter_choice = st.selectbox(
                "Filter by Ingestion Status",
                ["All Statuses", "FAILED Only", "COMPLETED Only", "IN_PROGRESS / PENDING"]
            )
        with col_f2:
            show_all_tenant = False
            if user_role == "admin":
                show_all_tenant = st.checkbox("Show All Tenant Users", value=False)
        with col_f3:
            if st.button("🔄 Refresh History", use_container_width=True):
                st.rerun()

        filter_val = None
        if status_filter_choice == "FAILED Only":
            filter_val = "FAILED"
        elif status_filter_choice == "COMPLETED Only":
            filter_val = "COMPLETED"
        elif status_filter_choice == "IN_PROGRESS / PENDING":
            filter_val = "IN_PROGRESS"

        target_u = None if (show_all_tenant and user_role == "admin") else active_user_id
        records = local_ingestion_history_manager.get_history(
            tenant_id=tenant_id,
            user_id=target_u,
            status=filter_val,
            limit=50
        )

        # Aggregate Metrics Bar
        all_user_records = local_ingestion_history_manager.get_history(tenant_id=tenant_id, user_id=target_u, limit=200)
        total_count = len(all_user_records)
        completed_count = sum(1 for r in all_user_records if r.status == "COMPLETED")
        failed_count = sum(1 for r in all_user_records if r.status == "FAILED")
        retries_count = sum(r.retry_count for r in all_user_records)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Submissions", total_count)
        m2.metric("Completed", completed_count)
        m3.metric("Failed / DLQ", failed_count, delta=f"{failed_count} errors" if failed_count else None, delta_color="inverse")
        m4.metric("Total Retries Triggered", retries_count)

        st.markdown("---")

        if not records:
            st.info("No ingestion history records found for the selected filter.")
        else:
            for r in records:
                with st.container():
                    st.markdown('<div class="card-box">', unsafe_allow_html=True)
                    c_info, c_status, c_action = st.columns([3, 2, 2])
                    
                    with c_info:
                        st.markdown(f"**Document:** `{r.document_id}` ({r.title})")
                        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(r.created_at))
                        st.caption(f"Job: `{r.job_id}` | User: `{r.user_id}` | Submitted: {time_str}")
                        st.caption(f"Domain: `{r.declared_domain or 'general_home'}` | Tier: `{r.access_tier}` | Level {r.clearance_level}")
                        if r.file_path:
                            st.caption(f"📁 Path: `{r.file_path}`")

                    with c_status:
                        if r.status == "COMPLETED":
                            st.markdown('<span class="badge-approved">● COMPLETED</span>', unsafe_allow_html=True)
                            st.write(f"Chunks: **{r.chunks_ingested}** (ns: `{r.namespace}`)")
                        elif r.status == "FAILED":
                            st.markdown('<span class="badge-rejected">✖ FAILED</span>', unsafe_allow_html=True)
                            st.caption(f"Retry attempts: {r.retry_count}")
                            if r.error_message:
                                with st.expander("🔍 Error Reason"):
                                    st.error(r.error_message)
                        else:
                            st.markdown('<span class="badge-pending">▲ IN PROGRESS</span>', unsafe_allow_html=True)
                            st.caption(f"Status: {r.status}")

                    with c_action:
                        if r.status == "FAILED":
                            has_retry_perm = (
                                ServiceScope.INGESTION_WRITE.value in user_scopes or 
                                ServiceScope.ALL.value in user_scopes or 
                                user_role == "admin"
                            )
                            if not has_retry_perm:
                                st.caption("🔒 Ingestion scope required to retry")
                            else:
                                if st.button(f"🔄 Retry Ingestion", key=f"btn_retry_{r.job_id}", type="primary", use_container_width=True):
                                    with st.spinner(f"Retrying ingestion for '{r.document_id}'..."):
                                        res = local_ingestion_history_manager.retry_job(
                                            job_id=r.job_id,
                                            caller_user_id=active_user_id,
                                            is_admin=(user_role == "admin"),
                                            pipeline=ingestion_pipeline
                                        )
                                        if res.get("status") == "SUCCESS":
                                            st.success(f"🎉 Job '{r.job_id}' retried successfully! Ingested {res.get('chunks_ingested')} chunks.")
                                            st.rerun()
                                        else:
                                            st.error(f"Retry failed: {res.get('error_message')}")
                        elif r.status == "COMPLETED" and r.summary:
                            with st.expander("📄 Summary"):
                                st.write(r.summary)

                    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Tab 4: My Clearance Profile
    # -------------------------------------------------------------------------
    with tab_objects[3]:
        st.markdown("### 👤 User Security & Clearance Profile")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown(f"""
            **Identity Details:**
            - **User ID:** `{active_user_id}`
            - **Tenant Organization:** `{tenant_id}`
            - **Account Status:** `{user_status}`
            - **Assigned Role:** `{user_role}`
            - **Clearance Level:** Level {user_clearance}
            """)
        with col_p2:
            st.markdown("#### 🛡️ Classification Hierarchy")
            st.markdown("""
            - 🟢 **Level 1 (Public)**: General recipes, standard home guidance.
            - 🟡 **Level 2 (Internal)**: Operational manuals, proprietary equipment specs.
            - 🔴 **Level 3 (Confidential)**: Executive restaurant formulas, trade secrets.
            """)

    # -------------------------------------------------------------------------
    # Tab 5: Admin Command Center (Admin Only)
    # -------------------------------------------------------------------------
    if user_role == "admin" and len(tab_objects) > 4:
        with tab_objects[4]:
            st.markdown("### 👑 Tenant Administration & Governance")
            st.caption(f"Managing user accounts and scope permissions for tenant `{tenant_id}`.")

            adm_tab1, adm_tab2, adm_tab3 = st.tabs(["⏳ Pending Approvals", "👥 Active Users", "📊 Audit Telemetry"])

            # Subtab A: Pending Approvals
            with adm_tab1:
                pending_users = local_scope_manager.list_pending_users(tenant_id)
                if not pending_users:
                    st.info("No user accounts currently pending approval.")
                else:
                    st.markdown(f"**Found {len(pending_users)} pending registration(s):**")
                    for p in pending_users:
                        with st.container():
                            st.markdown(f'<div class="card-box">', unsafe_allow_html=True)
                            c1, c2, c3 = st.columns([2, 2, 2])
                            with c1:
                                st.write(f"**User:** `{p['user_id']}`")
                                st.caption(f"Applied: {time.strftime('%Y-%m-%d %H:%M', time.localtime(p.get('created_at', time.time())))}")
                            with c2:
                                assign_clear = st.selectbox(f"Clearance for {p['user_id']}", [1, 2, 3], key=f"clr_{p['user_id']}")
                                assign_role = st.selectbox(f"Role for {p['user_id']}", ["member", "admin"], key=f"rol_{p['user_id']}")
                            with c3:
                                grant_ingest = st.checkbox(f"Grant Ingestion Scope", key=f"ing_{p['user_id']}")
                                b_col1, b_col2 = st.columns(2)
                                with b_col1:
                                    if st.button("✅ Approve", key=f"app_{p['user_id']}", type="primary"):
                                        scopes_to_grant = [ServiceScope.RAG_READ]
                                        if grant_ingest:
                                            scopes_to_grant.append(ServiceScope.INGESTION_WRITE)
                                        local_scope_manager.approve_user(
                                            tenant_id=tenant_id,
                                            user_id=p["user_id"],
                                            clearance_level=assign_clear,
                                            role=assign_role,
                                            scopes=scopes_to_grant,
                                            approved_by=active_user_id
                                        )
                                        st.success(f"Approved {p['user_id']}!")
                                        st.rerun()
                                with b_col2:
                                    if st.button("❌ Reject", key=f"rej_{p['user_id']}"):
                                        local_scope_manager.reject_user(
                                            tenant_id=tenant_id,
                                            user_id=p["user_id"],
                                            rejected_by=active_user_id,
                                            reason="Denied by admin"
                                        )
                                        st.warning(f"Rejected {p['user_id']}.")
                                        st.rerun()
                            st.markdown('</div>', unsafe_allow_html=True)

            # Subtab B: Active Users
            with adm_tab2:
                with local_scope_manager._get_connection() as conn:
                    cur = conn.execute("SELECT * FROM service_user_profiles WHERE tenant_id = ? ORDER BY created_at DESC;", (tenant_id,))
                    all_users = [dict(r) for r in cur.fetchall()]

                if all_users:
                    st.dataframe(all_users)
                else:
                    st.write("No users registered.")

            # Subtab C: Audit Telemetry
            with adm_tab3:
                metrics = auth_telemetry.get_metrics()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Auth Events", metrics.get("total_events", 0))
                c2.metric("Success Count", metrics.get("success_count", 0))
                c3.metric("Denied Count", metrics.get("denied_count", 0))
                c4.metric("Avg Latency", f"{metrics.get('avg_duration_ms', 0):.2f} ms")
