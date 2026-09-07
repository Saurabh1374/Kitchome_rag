from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class UserTier(str, Enum):
    FREE = "free"
    PREMIUM = "premium"
    SCHOLAR = "scholar"
    ENTERPRISE = "enterprise"

class UserContext(BaseModel):
    """
    Headless security context capturing user identity, tier, tenant, and clearance.
    Passed via bearer tokens or API key claims into the guarded RAG pipeline.
    """
    user_id: str
    tier: UserTier = UserTier.FREE
    tenant_id: str = "global"
    clearance_level: int = 1  # 1 = Public, 2 = Internal, 3 = Confidential
    custom_allowed_namespaces: Optional[List[str]] = None
