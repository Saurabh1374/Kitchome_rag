from typing import List, Optional, Set
from .context import UserContext, UserTier

class RBACPolicyEngine:
    """
    Role-Based Access Control (RBAC) Engine:
    Maps user tiers [free, premium, scholar, enterprise] to permitted vector namespaces.
    """
    TIER_NAMESPACE_MAP = {
        UserTier.FREE: ["general_home", "recipes_culinary"],
        UserTier.PREMIUM: [
            "general_home", 
            "recipes_culinary", 
            "appliances_troubleshooting", 
            "cleaning_maintenance", 
            "home_decor_design"
        ],
        UserTier.SCHOLAR: [
            "general_home", 
            "recipes_culinary", 
            "appliances_troubleshooting", 
            "cleaning_maintenance", 
            "home_decor_design",
            "academia_research"
        ],
        UserTier.ENTERPRISE: [
            "general_home", 
            "recipes_culinary", 
            "appliances_troubleshooting", 
            "cleaning_maintenance", 
            "home_decor_design",
            "academia_research"
        ]
    }

    @classmethod
    def get_allowed_namespaces(cls, user: UserContext, all_known_namespaces: Optional[List[str]] = None) -> List[str]:
        """
        Returns the set of vector namespaces accessible to the user based on their tier and custom grants.
        Enterprise users have access to all public namespaces + their specific tenant namespaces.
        """
        if user.custom_allowed_namespaces:
            return list(set(user.custom_allowed_namespaces))

        tier = user.tier if isinstance(user.tier, UserTier) else UserTier(user.tier)
        allowed: Set[str] = set(cls.TIER_NAMESPACE_MAP.get(tier, cls.TIER_NAMESPACE_MAP[UserTier.FREE]))

        if tier == UserTier.ENTERPRISE:
            if all_known_namespaces:
                # Include all known public namespaces + tenant specific namespaces
                for ns in all_known_namespaces:
                    if not ns.startswith("tenant_") or f"tenant_{user.tenant_id}" in ns:
                        allowed.add(ns)
            allowed.add(f"tenant_{user.tenant_id}_private")

        return list(allowed)

    @classmethod
    def can_access_namespace(cls, user: UserContext, namespace: str, all_known_namespaces: Optional[List[str]] = None) -> bool:
        allowed = cls.get_allowed_namespaces(user, all_known_namespaces)
        return namespace in allowed
