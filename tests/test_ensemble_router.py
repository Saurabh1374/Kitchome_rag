import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.skills_ms.registry import SkillNamespaceRegistry
from src.skills_ms.router import SkillsRouter

def test_ensemble_router_disambiguation():
    router = SkillsRouter()

    # 1. Culinary prep with 'clean' verb should route to recipes_culinary due to domain damping
    q_culinary = "how to clean and prep salmon for pan searing"
    res_culinary = router.resolve_namespaces_with_confidence(q_culinary)
    assert res_culinary["namespaces"][0] == "recipes_culinary"
    assert res_culinary["confidence_margin"] > 0.15

    # 2. Clear appliance troubleshooting with error code
    q_appliance = "instant pot error code E3 heating element manual"
    res_appliance = router.resolve_namespaces_with_confidence(q_appliance)
    assert res_appliance["namespaces"][0] == "appliances_troubleshooting"
    assert res_appliance["confidence_margin"] > 0.20

    # 3. Academic paper query
    q_academic = "neural network transformer architecture attention mechanism"
    res_academic = router.resolve_namespaces_with_confidence(q_academic)
    assert res_academic["namespaces"][0] == "academia_research"

def test_ensemble_router_margin_arbitration_soft_partition():
    router = SkillsRouter()

    # Ambiguous multi-domain query touching both appliance maintenance and deep cleaning
    q_ambiguous = "how to clean air fryer basket and repair handle"
    decision = router.resolve_namespaces_with_confidence(q_ambiguous, top_k=2)

    # Should detect low margin and return top-2 namespaces
    assert len(decision["namespaces"]) >= 2
    assert "appliances_troubleshooting" in decision["namespaces"]
    assert "cleaning_maintenance" in decision["namespaces"]
    assert decision["confidence_margin"] < 0.18

def test_ensemble_router_document_resolution():
    router = SkillsRouter()

    # Declared domain priority
    ns = router.resolve_namespace_for_document(
        file_path="/dummy/path.txt",
        content="some generic text",
        declared_domain="recipes_culinary"
    )
    assert ns == "recipes_culinary"

    # File pattern matching
    ns_manual = router.resolve_namespace_for_document(
        file_path="/downloads/airfryer_user_manual.pdf",
        content="Read safety instructions before turning on appliance."
    )
    assert ns_manual == "appliances_troubleshooting"
