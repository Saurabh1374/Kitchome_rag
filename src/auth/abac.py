from typing import Dict, Any, List
from .context import UserContext, UserTier

class ABACPolicyEngine:
    """
    Attribute-Based Access Control (ABAC) Engine:
    Evaluates fine-grained attributes on User (subject), Chunk (resource), and Action.
    Builds database search predicates for pgvector and evaluates in-memory chunk access.
    """
    TIER_HIERARCHY = {
        UserTier.FREE: 1,
        UserTier.PREMIUM: 2,
        UserTier.SCHOLAR: 3,
        UserTier.ENTERPRISE: 4
    }

    @classmethod
    def evaluate_chunk_access(cls, user: UserContext, chunk_metadata: Dict[str, Any]) -> bool:
        """
        Evaluates whether a user context satisfies the ABAC policy for a retrieved chunk.
        Checks:
        1. Tenant isolation: chunk tenant must match user tenant or be 'global'.
        2. Access tier: user's tier rank must be >= chunk's required access tier rank.
        3. Clearance level: user's clearance >= chunk's clearance level.
        """
        # 1. Tenant Check
        chunk_tenant = chunk_metadata.get("tenant_id", "global")
        if chunk_tenant != "global" and chunk_tenant != user.tenant_id:
            return False

        # 2. Access Tier Check
        user_tier = user.tier if isinstance(user.tier, UserTier) else UserTier(user.tier)
        user_rank = cls.TIER_HIERARCHY.get(user_tier, 1)

        chunk_tier_str = chunk_metadata.get("access_tier", "free")
        try:
            chunk_tier = UserTier(chunk_tier_str.lower())
            chunk_rank = cls.TIER_HIERARCHY.get(chunk_tier, 1)
        except ValueError:
            chunk_rank = 1

        if user_rank < chunk_rank:
            return False

        # 3. Clearance Check
        chunk_clearance = int(chunk_metadata.get("clearance_level", 1))
        if user.clearance_level < chunk_clearance:
            return False

        return True

    @classmethod
    def build_abac_filter(cls, user: UserContext) -> Dict[str, Any]:
        """
        Generates an ABAC filter dictionary for vector store querying.
        """
        user_tier = user.tier if isinstance(user.tier, UserTier) else UserTier(user.tier)
        user_rank = cls.TIER_HIERARCHY.get(user_tier, 1)
        
        permitted_tiers = [
            tier.value for tier, rank in cls.TIER_HIERARCHY.items() 
            if rank <= user_rank
        ]

        return {
            "tenant_id": user.tenant_id,
            "permitted_tiers": permitted_tiers,
            "max_clearance": user.clearance_level
        }

    @classmethod
    def get_rls_session_vars(cls, user: UserContext) -> Dict[str, str]:
        """
        Generates PostgreSQL session settings (for SET LOCAL) to enforce RLS at the kernel level.
        """
        user_tier = user.tier if isinstance(user.tier, UserTier) else UserTier(user.tier)
        user_rank = cls.TIER_HIERARCHY.get(user_tier, 1)
        permitted_tiers = [
            tier.value for tier, rank in cls.TIER_HIERARCHY.items() 
            if rank <= user_rank
        ]
        return {
            "app.current_tenant": str(user.tenant_id),
            "app.permitted_tiers": ",".join(permitted_tiers),
            "app.clearance_level": str(user.clearance_level)
        }
