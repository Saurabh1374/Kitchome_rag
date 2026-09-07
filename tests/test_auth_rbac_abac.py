import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.auth.context import UserContext, UserTier
from src.auth.rbac import RBACPolicyEngine
from src.auth.abac import ABACPolicyEngine
from src.vector_store.base import NamespaceVectorStore, VectorChunk

def test_rbac_tier_namespace_permissions():
    free_user = UserContext(user_id="u_free", tier=UserTier.FREE)
    premium_user = UserContext(user_id="u_premium", tier=UserTier.PREMIUM)
    scholar_user = UserContext(user_id="u_scholar", tier=UserTier.SCHOLAR)
    enterprise_user = UserContext(user_id="u_ent", tier=UserTier.ENTERPRISE, tenant_id="acme")

    # Free user can only access recipes and general
    free_allowed = RBACPolicyEngine.get_allowed_namespaces(free_user)
    assert "recipes_culinary" in free_allowed
    assert "general_home" in free_allowed
    assert "appliances_troubleshooting" not in free_allowed
    assert "academia_research" not in free_allowed

    # Premium adds appliance troubleshooting and home decor
    premium_allowed = RBACPolicyEngine.get_allowed_namespaces(premium_user)
    assert "appliances_troubleshooting" in premium_allowed
    assert "home_decor_design" in premium_allowed
    assert "academia_research" not in premium_allowed

    # Scholar adds academic research
    scholar_allowed = RBACPolicyEngine.get_allowed_namespaces(scholar_user)
    assert "academia_research" in scholar_allowed

    # Enterprise adds private tenant namespace
    ent_allowed = RBACPolicyEngine.get_allowed_namespaces(enterprise_user)
    assert "tenant_acme_private" in ent_allowed

def test_abac_metadata_filtering_in_vector_store():
    store = NamespaceVectorStore()
    
    # Chunk 1: Global public recipe
    chunk_public = VectorChunk(
        chunk_id="c_pub",
        document_id="doc_1",
        namespace="recipes_culinary",
        text="Pasta carbonara with eggs and guanciale.",
        metadata={"tenant_id": "global", "access_tier": "free", "clearance_level": 1},
        embedding=[1.0, 0.0, 0.0]
    )

    # Chunk 2: Tenant-specific proprietary recipe
    chunk_tenant_b = VectorChunk(
        chunk_id="c_tenant_b",
        document_id="doc_2",
        namespace="recipes_culinary",
        text="Secret restaurant spice recipe.",
        metadata={"tenant_id": "tenant_restaurant_b", "access_tier": "enterprise", "clearance_level": 2},
        embedding=[0.9, 0.1, 0.0]
    )

    # Chunk 3: Scholar-tier confidential research
    chunk_scholar = VectorChunk(
        chunk_id="c_scholar",
        document_id="doc_3",
        namespace="academia_research",
        text="Empirical transformer convergence bounds.",
        metadata={"tenant_id": "global", "access_tier": "scholar", "clearance_level": 2},
        embedding=[0.8, 0.2, 0.0]
    )

    store.upsert_chunks([chunk_public, chunk_tenant_b, chunk_scholar])

    # User A: Free global user
    user_free = UserContext(user_id="u_free", tier=UserTier.FREE, tenant_id="global")
    abac_filter_free = ABACPolicyEngine.build_abac_filter(user_free)

    res_free = store.search(
        query_vector=[1.0, 0.0, 0.0],
        namespace="recipes_culinary",
        abac_filter=abac_filter_free
    )
    # Free user should only see global free chunk, NOT tenant_b secret chunk
    assert len(res_free) == 1
    assert res_free[0]["chunk_id"] == "c_pub"

    # User B: Tenant B Enterprise User
    user_tenant_b = UserContext(user_id="u_b", tier=UserTier.ENTERPRISE, tenant_id="tenant_restaurant_b", clearance_level=3)
    abac_filter_b = ABACPolicyEngine.build_abac_filter(user_tenant_b)

    res_b = store.search(
        query_vector=[1.0, 0.0, 0.0],
        namespace="recipes_culinary",
        abac_filter=abac_filter_b
    )
    # Tenant B user can see both global chunk and their own tenant chunk
    assert len(res_b) == 2
    assert any(c["chunk_id"] == "c_tenant_b" for c in res_b)
