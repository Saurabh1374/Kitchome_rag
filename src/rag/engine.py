import time
from typing import List, Dict, Any, Optional
from ..skills_ms.router import SkillsRouter
from ..vector_store.base import NamespaceVectorStore
from ..ingestion.embedder import EmbeddingEngine
from ..auth.context import UserContext, UserTier
from ..auth.rbac import RBACPolicyEngine
from ..auth.abac import ABACPolicyEngine
from .telemetry import TelemetryTracker, RetrievalTelemetry
from .generator import GroundedGenerator

class RAGQueryEngine:
    """
    Guarded RAG Query Engine & Headless Tool:
    1. Validates UserContext (RBAC + ABAC Policy Decision Point)
    2. Resolves target namespaces via skills.ms Ensemble Router (Lexical + Semantic Prototype)
    3. Executes database-level ABAC vector retrieval (pgvector / in-memory)
    4. Evaluates Strategy E Fallback: Expands scope if top similarity is below threshold
    5. Assembles Parent-Context prompt & returns grounded synthesis with source citations
    """
    def __init__(
        self,
        router: Optional[SkillsRouter] = None,
        vector_store: Optional[NamespaceVectorStore] = None,
        embedder: Optional[EmbeddingEngine] = None,
        generator: Optional[GroundedGenerator] = None,
        telemetry: Optional[TelemetryTracker] = None,
        relevance_threshold: float = 0.40
    ):
        self.router = router or SkillsRouter()
        self.vector_store = vector_store or NamespaceVectorStore()
        self.embedder = embedder or EmbeddingEngine()
        self.generator = generator or GroundedGenerator()
        self.telemetry = telemetry or TelemetryTracker()
        self.relevance_threshold = relevance_threshold

    def query(
        self, 
        query_text: str, 
        user_context: Optional[UserContext] = None, 
        top_k: int = 3,
        is_summary: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Executes a guarded RAG query flow.
        """
        # 1. Identity & Auth Context Defaulting
        user = user_context or UserContext(user_id="anonymous", tier=UserTier.FREE)
        all_namespaces = self.router.registry.get_all_namespaces()

        # 2. Ensemble Namespace Resolution (skills.ms)
        router_decision = self.router.resolve_namespaces_with_confidence(query_text, top_k=2)
        predicted_namespaces = router_decision["namespaces"]
        confidence_margin = router_decision["confidence_margin"]

        # 3. RBAC Policy Enforcement: Intersect predicted with user's permitted namespaces
        user_allowed = RBACPolicyEngine.get_allowed_namespaces(user, all_namespaces)
        authorized_namespaces = [ns for ns in predicted_namespaces if ns in user_allowed]

        if not authorized_namespaces:
            # RBAC Gate: User tier does not have permission to query this domain
            denied_event = RetrievalTelemetry(
                query_text=query_text,
                user_tier=user.tier.value if isinstance(user.tier, UserTier) else str(user.tier),
                predicted_namespaces=predicted_namespaces,
                authorized_namespaces=[],
                confidence_margin=confidence_margin,
                top_similarity_score=0.0,
                fallback_triggered=False,
                final_chunk_count=0
            )
            self.telemetry.record_event(denied_event)

            return {
                "status": "ACCESS_DENIED",
                "error": "FORBIDDEN",
                "message": f"User tier '{user.tier}' does not have access to requested domain knowledge: {predicted_namespaces}",
                "predicted_namespaces": predicted_namespaces,
                "authorized_namespaces": [],
                "answer": f"Access Denied: Your tier ({user.tier}) is not permitted to access this domain. Please upgrade your subscription.",
                "citations": []
            }

        # 4. Compile ABAC Filter for Database Retrieval
        abac_filter = ABACPolicyEngine.build_abac_filter(user)
        query_vec = self.embedder.embed_text(query_text)

        # 5. Dual-Resolution Vector Retrieval
        chunks = self.vector_store.search(
            query_vector=query_vec,
            namespace=authorized_namespaces,
            top_k=top_k,
            abac_filter=abac_filter,
            is_summary=is_summary
        )

        top_similarity = chunks[0]["similarity_score"] if chunks else 0.0
        fallback_triggered = False

        # 6. Strategy E Fallback Loop: If top similarity < threshold, broaden retrieval
        if top_similarity < self.relevance_threshold:
            fallback_triggered = True
            # Find remaining allowed namespaces not yet searched
            fallback_namespaces = [ns for ns in user_allowed if ns not in authorized_namespaces]
            if fallback_namespaces:
                secondary_chunks = self.vector_store.search(
                    query_vector=query_vec,
                    namespace=fallback_namespaces,
                    top_k=top_k,
                    abac_filter=abac_filter,
                    is_summary=is_summary
                )
                if secondary_chunks:
                    # Merge and keep top_k
                    combined = chunks + secondary_chunks
                    combined.sort(key=lambda x: x["similarity_score"], reverse=True)
                    chunks = combined[:top_k]
                    top_similarity = chunks[0]["similarity_score"]

        # 7. Record Telemetry Event
        telemetry_event = RetrievalTelemetry(
            query_text=query_text,
            user_tier=user.tier.value if isinstance(user.tier, UserTier) else str(user.tier),
            predicted_namespaces=predicted_namespaces,
            authorized_namespaces=authorized_namespaces,
            confidence_margin=confidence_margin,
            top_similarity_score=top_similarity,
            fallback_triggered=fallback_triggered,
            final_chunk_count=len(chunks)
        )
        self.telemetry.record_event(telemetry_event)

        # 8. Grounded Generation with Parent-Context Prompt Injection
        generation_result = self.generator.synthesize(query=query_text, retrieved_chunks=chunks)

        return {
            "status": "SUCCESS",
            "query": query_text,
            "user_id": user.user_id,
            "user_tier": user.tier.value if isinstance(user.tier, UserTier) else str(user.tier),
            "predicted_namespaces": predicted_namespaces,
            "authorized_namespaces": authorized_namespaces,
            "confidence_margin": confidence_margin,
            "top_similarity_score": top_similarity,
            "fallback_triggered": fallback_triggered,
            "chunks_retrieved": len(chunks),
            "retrieved_chunks": chunks,
            "answer": generation_result["answer"],
            "citations": generation_result["citations"],
            "formatted_prompt": generation_result.get("formatted_prompt", "")
        }
