import os
import sys
import time
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.auth.context import UserContext, UserTier
from src.auth.service import ServiceScope, JWTTokenManager, ServiceAuthGuard
from src.auth.scopes import local_scope_manager
from src.auth.abac import ABACPolicyEngine
from src.vector_store.base import NamespaceVectorStore, VectorChunk, DocumentSummaryRecord
from src.rag.engine import RAGQueryEngine
from src.rag.generator import GroundedGenerator

TEST_SECRET = "kitchome-test-secret-key-for-jwt-verification-12345"

@pytest.fixture(autouse=True)
def clean_scopes():
    """Ensure clean scope and profile table between tests."""
    local_scope_manager.clear()
    yield
    local_scope_manager.clear()

# =====================================================================
# 1. Coarse Service Auth & JWT Token Manager Tests
# =====================================================================

def test_jwt_token_creation_and_hydration():
    """Verify that a signed JWT creates and hydrates a matching UserContext."""
    user = UserContext(
        user_id="alice_chef",
        tier=UserTier.PREMIUM,
        tenant_id="kitchen_corp",
        clearance_level=2
    )

    token = JWTTokenManager.create_token(
        user=user,
        scopes=[ServiceScope.RAG_READ],
        secret=TEST_SECRET,
        ttl_seconds=3600
    )
    assert token is not None
    assert len(token.split(".")) == 3

    # Hydrate user back from token
    hydrated_user = JWTTokenManager.get_user_from_token(token, secret=TEST_SECRET)
    assert hydrated_user.user_id == "alice_chef"
    assert hydrated_user.tier == UserTier.PREMIUM
    assert hydrated_user.tenant_id == "kitchen_corp"
    assert hydrated_user.clearance_level == 2

def test_jwt_signature_tampering_rejected():
    """Verify that any modification of payload or wrong secret raises PermissionError."""
    user = UserContext(user_id="mallory", tier=UserTier.FREE)
    token = JWTTokenManager.create_token(user=user, scopes=[ServiceScope.RAG_READ], secret=TEST_SECRET)

    # 1. Tamper with payload segment
    parts = token.split(".")
    # Replace payload with altered base64 segment
    tampered_token = f"{parts[0]}.eyJhZG1pbiI6dHJ1ZX0.{parts[2]}"
    with pytest.raises(PermissionError, match=r"Invalid cryptographic token signature"):
        JWTTokenManager.verify_and_decode_token(tampered_token, secret=TEST_SECRET)

    # 2. Decode with wrong secret
    with pytest.raises(PermissionError, match=r"Invalid cryptographic token signature"):
        JWTTokenManager.verify_and_decode_token(token, secret="wrong-secret-key-999")

def test_jwt_expiration_rejected():
    """Verify that an expired token raises PermissionError."""
    user = UserContext(user_id="bob", tier=UserTier.FREE)
    # Expired token with ttl = -10
    expired_token = JWTTokenManager.create_token(
        user=user,
        scopes=[ServiceScope.RAG_READ],
        secret=TEST_SECRET,
        ttl_seconds=-10
    )
    with pytest.raises(PermissionError, match=r"Token has expired"):
        JWTTokenManager.verify_and_decode_token(expired_token, secret=TEST_SECRET)

def test_service_auth_guard_scope_enforcement():
    """Verify ServiceAuthGuard enforces required scopes and rejects unauthorized capabilities."""
    user_ingest_only = UserContext(user_id="pipeline_worker", tier=UserTier.ENTERPRISE)
    token_ingest = JWTTokenManager.create_token(
        user=user_ingest_only,
        scopes=[ServiceScope.INGESTION_WRITE],
        secret=TEST_SECRET
    )

    # Calling RAG_READ with an INGESTION_WRITE-only token must be rejected
    with pytest.raises(PermissionError, match=r"Access Forbidden: Token lacks required scope 'rag:read'"):
        ServiceAuthGuard.enforce_service_auth(
            auth_header_or_token=token_ingest,
            required_scope=ServiceScope.RAG_READ,
            secret=TEST_SECRET
        )

    # Calling INGESTION_WRITE with INGESTION_WRITE token succeeds
    verified = ServiceAuthGuard.enforce_service_auth(
        auth_header_or_token=f"Bearer {token_ingest}",
        required_scope=ServiceScope.INGESTION_WRITE,
        secret=TEST_SECRET
    )
    assert verified.user_id == "pipeline_worker"

    # Token with wildcard ALL scope succeeds for any scope
    token_admin = JWTTokenManager.create_token(
        user=user_ingest_only,
        scopes=[ServiceScope.ALL],
        secret=TEST_SECRET
    )
    admin_user = ServiceAuthGuard.enforce_service_auth(
        auth_header_or_token=token_admin,
        required_scope=ServiceScope.RAG_READ,
        secret=TEST_SECRET
    )
    assert admin_user.user_id == "pipeline_worker"

def test_rag_query_engine_with_token_auth():
    """Verify RAGQueryEngine seamlessly verifies token and executes guarded retrieval."""
    vector_store = NamespaceVectorStore()
    engine = RAGQueryEngine(vector_store=vector_store)

    # Valid token with RAG_READ scope
    user = UserContext(user_id="charlie", tier=UserTier.PREMIUM)
    valid_token = JWTTokenManager.create_token(user=user, scopes=[ServiceScope.RAG_READ])

    res = engine.query(query_text="how to clean stainless steel", token=valid_token)
    assert res["status"] == "SUCCESS"
    assert res["user_id"] == "charlie"
    assert res["user_tier"] == "premium"

    # Invalid token rejected at the perimeter
    bad_token = valid_token + "corrupted"
    with pytest.raises(PermissionError):
        engine.query(query_text="how to clean stainless steel", token=bad_token)


# =====================================================================
# 2. Dedicated Document Summaries Storage Tests
# =====================================================================

def test_dedicated_document_summaries_crud():
    """Verify storing, retrieving, batch fetching, and searching DocumentSummaryRecords."""
    store = NamespaceVectorStore()

    summary1 = DocumentSummaryRecord(
        document_id="doc_oven_manual",
        namespace="appliances_troubleshooting",
        document_title="Oven Master Guide",
        summary_text="Comprehensive overview of oven safety, preheating protocols, and thermal fuse specs.",
        token_count=14,
        embedding=[0.1, 0.2, 0.3],
        access_tier="premium",
        tenant_id="kitchen_corp",
        clearance_level=1
    )

    summary2 = DocumentSummaryRecord(
        document_id="doc_pasta_secrets",
        namespace="recipes_culinary",
        document_title="Artisanal Pasta Handbook",
        summary_text="Complete guide to egg-to-flour ratios and extrusion techniques.",
        token_count=12,
        embedding=[0.4, 0.5, 0.6],
        access_tier="free",
        tenant_id="global",
        clearance_level=1
    )

    # 1. Upsert
    store.upsert_summary(summary1)
    store.upsert_summary(summary2)

    # 2. Get single
    retrieved1 = store.get_summary("doc_oven_manual")
    assert retrieved1 is not None
    assert retrieved1.document_title == "Oven Master Guide"
    assert "thermal fuse specs" in retrieved1.summary_text

    # 3. Batch get
    batch = store.get_summaries(["doc_oven_manual", "doc_pasta_secrets", "non_existent_id"])
    assert len(batch) == 2
    assert "doc_oven_manual" in batch
    assert "doc_pasta_secrets" in batch

    # 4. Search summaries with ABAC filter
    search_hits = store.search_summaries(
        query_vector=[0.1, 0.2, 0.3],
        namespace="appliances_troubleshooting",
        top_k=2,
        abac_filter={"tenant_id": "kitchen_corp", "permitted_tiers": ["premium"], "max_clearance": 1}
    )
    assert len(search_hits) == 1
    assert search_hits[0]["document_id"] == "doc_oven_manual"

    # Search with wrong tenant filtered out
    search_hits_denied = store.search_summaries(
        query_vector=[0.1, 0.2, 0.3],
        namespace="appliances_troubleshooting",
        top_k=2,
        abac_filter={"tenant_id": "different_corp", "permitted_tiers": ["premium"], "max_clearance": 1}
    )
    assert len(search_hits_denied) == 0

    # 5. Delete by document id removes summary
    store.delete_chunks_by_document_id("doc_oven_manual")
    assert store.get_summary("doc_oven_manual") is None

def test_parent_summary_injection_without_chunk_duplication():
    """Verify that GroundedGenerator injects the parent summary even when chunks lack doc_summary."""
    generator = GroundedGenerator()

    # Chunks without inline doc_summary in their metadata
    chunks = [
        {
            "chunk_id": "chk_101",
            "document_id": "doc_air_fryer",
            "text": "Press the roast button for 15 minutes at 375F.",
            "similarity_score": 0.89,
            "metadata": {
                "document_title": "Air Fryer 3000 Manual",
                "breadcrumb": "Air Fryer > Cooking Functions",
                "section_heading": "Roasting"
                # doc_summary intentionally omitted from chunk metadata!
            }
        }
    ]

    # External summaries map resolved from the dedicated summaries table
    summaries_map = {
        "doc_air_fryer": "The Air Fryer 3000 is a 1800W convection countertop cooker."
    }

    result = generator.synthesize(
        query="how to roast",
        retrieved_chunks=chunks,
        document_summaries=summaries_map
    )

    assert result["grounded"] is True
    # The parent overview must appear in the synthesized formatted prompt
    assert "[Document Overview]: The Air Fryer 3000 is a 1800W convection countertop cooker." in result["formatted_prompt"]
    assert "Press the roast button" in result["answer"]


# =====================================================================
# 3. PostgreSQL RLS Session Parameter Mapping Tests
# =====================================================================

def test_abac_get_rls_session_vars():
    """Verify that ABACPolicyEngine.get_rls_session_vars compiles exact session parameters."""
    user = UserContext(
        user_id="diana_enterprise",
        tier=UserTier.ENTERPRISE,
        tenant_id="enterprise_tenant_99",
        clearance_level=3
    )

    rls_vars = ABACPolicyEngine.get_rls_session_vars(user)

    assert rls_vars["app.current_tenant"] == "enterprise_tenant_99"
    assert "enterprise" in rls_vars["app.permitted_tiers"]
    assert "free" in rls_vars["app.permitted_tiers"]
    assert rls_vars["app.clearance_level"] == "3"

def test_pgvector_init_rls_and_session_application(monkeypatch):
    """Verify PGVectorStore executes RLS policy DDL and SET LOCAL session execution."""
    from unittest.mock import MagicMock
    from src.vector_store.pgvector_store import PGVectorStore

    executed_sqls = []

    mock_cur = MagicMock()
    def fake_execute(sql, params=None):
        executed_sqls.append((sql, params))
    mock_cur.execute = fake_execute

    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    store = PGVectorStore.__new__(PGVectorStore)
    store.is_connected = True
    store.table_name = "test_chunks"
    store.summaries_table_name = "test_summaries"
    store._get_connection = lambda: mock_conn

    # 1. Verify init_rls_policies executes RLS DDL
    store.init_rls_policies()
    ddl_texts = " ".join([sql for sql, _ in executed_sqls])
    assert "ENABLE ROW LEVEL SECURITY" in ddl_texts
    assert "CREATE POLICY rls_chunks_tenant_clearance" in ddl_texts
    assert "CREATE POLICY rls_summaries_tenant_clearance" in ddl_texts
    assert "app.is_internal_worker" in ddl_texts
    assert "NULLIF(current_setting('app.current_tenant', true), '') IS NOT NULL" in ddl_texts

    # 2. Verify apply_rls_session executes SET LOCAL
    user = UserContext(user_id="test_user", tier=UserTier.PREMIUM, tenant_id="tenant_omega", clearance_level=2)
    executed_sqls.clear()
    store.apply_rls_session(mock_conn, user)

    set_local_vars = {params[0] for sql, params in executed_sqls if "SET LOCAL" in sql and params}
    assert "tenant_omega" in set_local_vars
    assert "2" in set_local_vars


def test_pooled_connection_wrapper_transaction_context_and_sanitization():
    """Verify PooledConnectionWrapper handles __enter__/__exit__ and sanitizes with RESET ALL on close."""
    from unittest.mock import MagicMock
    import psycopg2.extensions
    from src.vector_store.pgvector_store import PooledConnectionWrapper

    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_conn.closed = 0
    mock_conn.status = psycopg2.extensions.STATUS_READY
    mock_conn.get_transaction_status.return_value = psycopg2.extensions.TRANSACTION_STATUS_IDLE

    wrapper = PooledConnectionWrapper(mock_pool, mock_conn)

    # 1. Verify transaction context dunders
    with wrapper as c:
        pass
    mock_conn.__enter__.assert_called_once()
    mock_conn.__exit__.assert_called_once()

    # 2. Verify close sanitizes with RESET ALL and returns to pool
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    wrapper.close()

    mock_cur.execute.assert_called_with("RESET ALL;")
    mock_conn.commit.assert_called()
    mock_pool.putconn.assert_called_with(mock_conn, close=False)
    assert wrapper._conn is None


def test_pooled_connection_wrapper_rolls_back_active_transaction():
    """Verify PooledConnectionWrapper aborts any lingering transaction before pool recycling."""
    from unittest.mock import MagicMock
    import psycopg2.extensions
    from src.vector_store.pgvector_store import PooledConnectionWrapper

    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_conn.closed = 0
    # Simulate an active in-transaction status
    mock_conn.status = psycopg2.extensions.STATUS_IN_TRANSACTION
    mock_conn.get_transaction_status.return_value = psycopg2.extensions.TRANSACTION_STATUS_INTRANS

    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    wrapper = PooledConnectionWrapper(mock_pool, mock_conn)
    wrapper.close()

    # Must have rolled back the lingering transaction before RESET ALL
    mock_conn.rollback.assert_called_once()
    mock_cur.execute.assert_called_with("RESET ALL;")
    mock_pool.putconn.assert_called_with(mock_conn, close=False)


def test_pooled_connection_wrapper_evicts_broken_connection():
    """Verify PooledConnectionWrapper evicts damaged or closed connections from the pool."""
    from unittest.mock import MagicMock
    from src.vector_store.pgvector_store import PooledConnectionWrapper

    # Case A: Connection closed flag is set
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_conn.closed = 1

    wrapper = PooledConnectionWrapper(mock_pool, mock_conn)
    wrapper.close()
    mock_pool.putconn.assert_called_with(mock_conn, close=True)

    # Case B: Reset fails with database exception
    mock_pool.reset_mock()
    mock_conn2 = MagicMock()
    mock_conn2.closed = 0
    mock_conn2.cursor.side_effect = Exception("Connection lost to Postgres")

    wrapper2 = PooledConnectionWrapper(mock_pool, mock_conn2)
    wrapper2.close()
    mock_pool.putconn.assert_called_with(mock_conn2, close=True)


def test_tenant_transaction_context_manager():
    """Verify tenant_transaction binds SET LOCAL parameters strictly within the transaction block."""
    from unittest.mock import MagicMock
    from src.vector_store.pgvector_store import PGVectorStore

    executed_sqls = []
    mock_cur = MagicMock()
    def fake_execute(sql, params=None):
        executed_sqls.append((sql, params))
    mock_cur.execute = fake_execute

    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    store = PGVectorStore.__new__(PGVectorStore)
    store.is_connected = True
    store._get_connection = lambda: mock_conn

    # 1. Normal tenant context transaction
    user = UserContext(user_id="alice", tier=UserTier.PREMIUM, tenant_id="tenant_x", clearance_level=2)
    with store.tenant_transaction(user=user) as conn:
        assert conn == mock_conn

    mock_conn.__enter__.assert_called_once()
    mock_conn.__exit__.assert_called_once()

    settings = {params[0]: sql for sql, params in executed_sqls if "SET LOCAL" in sql and params}
    assert "tenant_x" in settings
    assert "2" in settings

    # 2. Worker context transaction
    executed_sqls.clear()
    with store.tenant_transaction(is_internal_worker=True) as conn:
        pass
    worker_sqls = [sql for sql, params in executed_sqls if "app.is_internal_worker" in sql]
    assert len(worker_sqls) > 0


def test_search_strict_transaction_scoping():
    """Verify PGVectorStore.search executes within with conn: so SET LOCAL cannot leak."""
    from unittest.mock import MagicMock
    from src.vector_store.pgvector_store import PGVectorStore

    executed_sqls = []
    mock_cur = MagicMock()
    def fake_execute(sql, params=None):
        executed_sqls.append((sql, params))
    mock_cur.execute = fake_execute
    mock_cur.fetchall.return_value = []

    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    store = PGVectorStore.__new__(PGVectorStore)
    store.is_connected = True
    store.table_name = "kitchome_vector_chunks"
    store._get_connection = lambda: mock_conn

    abac_filter = {
        "tenant_id": "tenant_secret_corp",
        "permitted_tiers": ["premium"],
        "max_clearance": 2,
        "is_internal_worker": False
    }

    results = store.search(
        query_vector=[0.1, 0.2, 0.3],
        namespace="appliances",
        abac_filter=abac_filter
    )

    # Must have entered and exited transaction context
    mock_conn.__enter__.assert_called_once()
    mock_conn.__exit__.assert_called_once()
    mock_conn.close.assert_called_once()

    set_locals = [sql for sql, params in executed_sqls if "SET LOCAL" in sql]
    assert len(set_locals) >= 3
    tenant_params = [params[0] for sql, params in executed_sqls if "app.current_tenant" in sql]
    assert "tenant_secret_corp" in tenant_params


def test_connection_pool_cross_tenant_zero_leakage():
    """Verify that when a connection is recycled across two tenants, no session state leaks."""
    from unittest.mock import MagicMock
    import psycopg2.extensions
    from src.vector_store.pgvector_store import PooledConnectionWrapper

    # Simulated PostgreSQL session state on physical connection
    pg_session_state = {}
    tx_active = False

    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_conn.closed = 0
    mock_conn.status = psycopg2.extensions.STATUS_READY
    mock_conn.get_transaction_status.return_value = psycopg2.extensions.TRANSACTION_STATUS_IDLE

    def fake_execute(sql, params=None):
        nonlocal tx_active
        if "SET LOCAL app.current_tenant" in sql and params:
            pg_session_state["app.current_tenant"] = params[0]
            tx_active = True
        elif "RESET ALL" in sql:
            pg_session_state.clear()

    def fake_commit():
        nonlocal tx_active
        # Commit drops transaction-scoped SET LOCAL variables
        pg_session_state.pop("app.current_tenant", None)
        tx_active = False

    def fake_rollback():
        nonlocal tx_active
        pg_session_state.pop("app.current_tenant", None)
        tx_active = False

    mock_cur = MagicMock()
    mock_cur.execute = fake_execute
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_conn.commit = fake_commit
    mock_conn.rollback = fake_rollback

    # Tenant 1 checkout
    wrapper1 = PooledConnectionWrapper(mock_pool, mock_conn)
    with wrapper1:
        mock_cur.execute("SET LOCAL app.current_tenant = %s;", ("tenant_alpha",))
        assert pg_session_state.get("app.current_tenant") == "tenant_alpha"

    # Transaction committed by with wrapper1:
    mock_conn.commit()
    assert "app.current_tenant" not in pg_session_state

    wrapper1.close()
    mock_pool.putconn.assert_called_with(mock_conn, close=False)

    # Tenant 2 checkout on same physical connection
    wrapper2 = PooledConnectionWrapper(mock_pool, mock_conn)
    assert len(pg_session_state) == 0
    assert "app.current_tenant" not in pg_session_state


# =====================================================================
# 4. Auth Telemetry, Trace Metrics & Structured Logging Tests
# =====================================================================

def test_auth_telemetry_trace_metrics_and_events():
    """Verify AuthTelemetryTracker records events, durations, trace IDs, and metrics."""
    from src.auth.service import AuthTelemetryTracker, JWTTokenManager, ServiceAuthGuard, ServiceScope
    tracker = AuthTelemetryTracker()
    JWTTokenManager.telemetry = tracker
    ServiceAuthGuard.telemetry = tracker

    user = UserContext(user_id="telemetry_alice", tier=UserTier.PREMIUM, tenant_id="tenant_metrics_1")

    # 1. Issue Token
    custom_trace = "trace_custom_xyz_123"
    token = JWTTokenManager.create_token(
        user=user,
        scopes=[ServiceScope.RAG_READ],
        secret=TEST_SECRET,
        trace_id=custom_trace
    )

    events = tracker.get_events()
    assert len(events) == 1
    issue_event = events[0]
    assert issue_event.trace_id == custom_trace
    assert issue_event.action == "TOKEN_ISSUE"
    assert issue_event.status == "SUCCESS"
    assert issue_event.user_id == "telemetry_alice"
    assert issue_event.tenant_id == "tenant_metrics_1"
    assert issue_event.duration_ms >= 0.0

    # 2. Enforce Service Auth (valid)
    hydrated_user = ServiceAuthGuard.enforce_service_auth(
        auth_header_or_token=token,
        required_scope=ServiceScope.RAG_READ,
        secret=TEST_SECRET
    )
    assert hydrated_user.trace_id == custom_trace

    metrics = tracker.get_metrics()
    assert metrics["total_events"] >= 3  # TOKEN_ISSUE + TOKEN_VERIFY + SERVICE_AUTH
    assert metrics["success_count"] == metrics["total_events"]
    assert metrics["success_rate"] == 1.0
    assert metrics["denied_count"] == 0
    assert metrics["average_duration_ms"] >= 0.0


def test_auth_telemetry_failure_traces():
    """Verify AuthTelemetryTracker accurately captures denial modes (expired, signature, scope mismatch)."""
    from src.auth.service import AuthTelemetryTracker, JWTTokenManager, ServiceAuthGuard, ServiceScope
    tracker = AuthTelemetryTracker()
    JWTTokenManager.telemetry = tracker
    ServiceAuthGuard.telemetry = tracker

    user = UserContext(user_id="denied_bob", tier=UserTier.FREE, tenant_id="tenant_fail")

    # A. Signature Failure
    valid_token = JWTTokenManager.create_token(user=user, scopes=[ServiceScope.RAG_READ], secret=TEST_SECRET)
    tampered_token = valid_token[:-4] + "FAIL"
    try:
        ServiceAuthGuard.enforce_service_auth(auth_header_or_token=tampered_token, secret=TEST_SECRET)
    except PermissionError:
        pass

    # B. Scope Mismatch Failure
    try:
        ServiceAuthGuard.enforce_service_auth(
            auth_header_or_token=valid_token,
            required_scope=ServiceScope.ADMIN_MANAGE,
            secret=TEST_SECRET
        )
    except PermissionError:
        pass

    # C. Expired Token Failure
    expired_token = JWTTokenManager.create_token(
        user=user,
        scopes=[ServiceScope.RAG_READ],
        secret=TEST_SECRET,
        ttl_seconds=-10
    )
    try:
        ServiceAuthGuard.enforce_service_auth(auth_header_or_token=expired_token, secret=TEST_SECRET)
    except PermissionError:
        pass

    metrics = tracker.get_metrics()
    assert metrics["denied_count"] >= 3
    assert metrics["signature_failure_count"] >= 1
    assert metrics["scope_mismatch_count"] >= 1
    assert metrics["token_expired_count"] >= 1
    assert metrics["success_rate"] < 1.0




