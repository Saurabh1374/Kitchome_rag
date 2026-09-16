import os
import time
import sqlite3
import threading
import logging
from enum import Enum
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field

logger = logging.getLogger("kitchome.auth.scopes")

class ServiceScope(str, Enum):
    """Functional scopes governing service-level API capabilities."""
    RAG_READ = "rag:read"
    INGESTION_WRITE = "ingestion:write"
    ADMIN_MANAGE = "admin:manage"
    ALL = "*"

class UserApprovalStatus(str, Enum):
    """Onboarding lifecycle status for tenant users."""
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"

class LocalScopeRecord(BaseModel):
    """Data model representing a locally granted service scope."""
    tenant_id: str
    user_id: str
    scope: str
    granted_by: Optional[str] = "admin"
    granted_at: float = Field(default_factory=time.time)

class UserProfileRecord(BaseModel):
    """Data model representing a user's local service onboarding profile."""
    tenant_id: str
    user_id: str
    status: UserApprovalStatus = UserApprovalStatus.PENDING_APPROVAL
    role: str = "member"
    clearance_level: int = 1
    created_at: float = Field(default_factory=time.time)
    approved_at: Optional[float] = None
    approved_by: Optional[str] = None
    rejection_reason: Optional[str] = None

class ServiceScopeManager:
    """
    Local Service Authorization and Onboarding Manager.
    Governs user approval status, local clearance levels, roles, and functional scopes.
    Completely decoupled from upstream Identity Provider JWT claims.
    """
    def __init__(self, db_path: str = "data/service_scopes.db"):
        self.db_path = db_path
        self._local = threading.local()
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()
        logger.debug("ServiceScopeManager initialized with database at '%s'", self.db_path)

    @contextmanager
    def _get_connection(self):
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            if self.db_path != ":memory:":
                conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            self._local.conn = conn
        with self._local.conn:
            yield self._local.conn

    def close(self):
        """Cleanly closes the thread-local SQLite connection."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                self._local.conn.close()
                logger.debug("Closed thread-local SQLite connection for scope manager.")
            except Exception as e:
                logger.debug("Error closing thread-local SQLite connection: %s", e)
            self._local.conn = None

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS service_user_profiles (
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING_APPROVAL',
                    role TEXT NOT NULL DEFAULT 'member',
                    clearance_level INTEGER NOT NULL DEFAULT 1,
                    created_at REAL,
                    approved_at REAL,
                    approved_by TEXT,
                    rejection_reason TEXT,
                    PRIMARY KEY (tenant_id, user_id)
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS service_user_scopes (
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    granted_by TEXT,
                    granted_at REAL,
                    PRIMARY KEY (tenant_id, user_id, scope)
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS service_user_roles (
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    assigned_at REAL,
                    PRIMARY KEY (tenant_id, user_id, role)
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_profiles ON service_user_profiles (tenant_id, user_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_scopes ON service_user_scopes (tenant_id, user_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_roles ON service_user_roles (tenant_id, user_id);")
            conn.commit()

    # -------------------------------------------------------------------------
    # Onboarding & Lifecycle Management (Founding User + Approvals)
    # -------------------------------------------------------------------------

    def has_admin(self, tenant_id: str) -> bool:
        """Checks whether the given tenant has at least one approved administrator."""
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                SELECT 1 FROM service_user_profiles 
                WHERE tenant_id = ? AND status = 'APPROVED' AND (role = 'admin' OR clearance_level >= 3)
                LIMIT 1;
                """,
                (tenant_id,)
            )
            if cur.fetchone():
                return True
            cur2 = conn.execute(
                "SELECT 1 FROM service_user_roles WHERE tenant_id = ? AND role = 'admin' LIMIT 1;",
                (tenant_id,)
            )
            return bool(cur2.fetchone())

    def bootstrap_founding_admin(self, tenant_id: str, user_id: str) -> Dict[str, Any]:
        """
        Auto-bootstraps the founding administrator for a tenant with zero existing admins.
        Grants APPROVED status, 'admin' role, clearance level 3, and wildcard '*' scope.
        """
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO service_user_profiles (
                    tenant_id, user_id, status, role, clearance_level, created_at, approved_at, approved_by
                ) VALUES (?, ?, 'APPROVED', 'admin', 3, ?, ?, 'system_founding_user');
                """,
                (tenant_id, user_id, now, now)
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO service_user_roles (
                    tenant_id, user_id, role, assigned_at
                ) VALUES (?, ?, 'admin', ?);
                """,
                (tenant_id, user_id, now)
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO service_user_scopes (
                    tenant_id, user_id, scope, granted_by, granted_at
                ) VALUES (?, ?, '*', 'system_founding_user', ?);
                """,
                (tenant_id, user_id, now)
            )
            conn.commit()
        logger.info("ServiceScopeManager BOOTSTRAPPED founding admin user='%s' for tenant='%s'", user_id, tenant_id)
        profile = self.get_user_profile(tenant_id, user_id)
        return profile or {}

    def register_pending_user(self, tenant_id: str, user_id: str) -> Dict[str, Any]:
        """
        Registers a new un-onboarded user in PENDING_APPROVAL status.
        Unauthorized to perform RAG queries or ingestion until approved by an admin.
        """
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO service_user_profiles (
                    tenant_id, user_id, status, role, clearance_level, created_at
                ) VALUES (?, ?, 'PENDING_APPROVAL', 'member', 1, ?);
                """,
                (tenant_id, user_id, now)
            )
            conn.commit()
        logger.info("ServiceScopeManager REGISTERED pending user='%s' for tenant='%s'", user_id, tenant_id)
        profile = self.get_user_profile(tenant_id, user_id)
        return profile or {}

    def approve_user(
        self, 
        tenant_id: str, 
        user_id: str, 
        clearance_level: int = 1, 
        role: str = "member", 
        scopes: Optional[List[Union[ServiceScope, str]]] = None, 
        approved_by: str = "admin"
    ) -> Dict[str, Any]:
        """
        Administrator action to approve a pending user.
        Configures their local clearance level, service role, and granted scopes.
        """
        now = time.time()
        role_clean = role.lower().strip()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO service_user_profiles (
                    tenant_id, user_id, status, role, clearance_level, created_at, approved_at, approved_by
                ) VALUES (?, ?, 'APPROVED', ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, user_id) DO UPDATE SET
                    status = 'APPROVED',
                    role = EXCLUDED.role,
                    clearance_level = EXCLUDED.clearance_level,
                    approved_at = EXCLUDED.approved_at,
                    approved_by = EXCLUDED.approved_by,
                    rejection_reason = NULL;
                """,
                (tenant_id, user_id, role_clean, clearance_level, now, now, approved_by)
            )
            conn.execute(
                "INSERT OR REPLACE INTO service_user_roles (tenant_id, user_id, role, assigned_at) VALUES (?, ?, ?, ?);",
                (tenant_id, user_id, role_clean, now)
            )
            conn.commit()

        # Scope assignment
        if scopes:
            for sc in scopes:
                self.grant_scope(tenant_id, user_id, sc, granted_by=approved_by)
        elif role_clean == "admin":
            self.grant_scope(tenant_id, user_id, "*", granted_by=approved_by)
        else:
            self.grant_scope(tenant_id, user_id, ServiceScope.RAG_READ, granted_by=approved_by)

        logger.info(
            "ServiceScopeManager APPROVED user='%s' (tenant='%s', clearance=%d, role='%s', approved_by='%s')",
            user_id, tenant_id, clearance_level, role_clean, approved_by
        )
        profile = self.get_user_profile(tenant_id, user_id)
        return profile or {}

    def reject_user(
        self, 
        tenant_id: str, 
        user_id: str, 
        rejected_by: str = "admin", 
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Administrator action to reject a user.
        Revokes all local scopes and marks profile as REJECTED.
        """
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO service_user_profiles (
                    tenant_id, user_id, status, created_at, approved_by, rejection_reason
                ) VALUES (?, ?, 'REJECTED', ?, ?, ?)
                ON CONFLICT(tenant_id, user_id) DO UPDATE SET
                    status = 'REJECTED',
                    approved_by = EXCLUDED.approved_by,
                    rejection_reason = EXCLUDED.rejection_reason;
                """,
                (tenant_id, user_id, now, rejected_by, reason)
            )
            conn.execute(
                "DELETE FROM service_user_scopes WHERE tenant_id = ? AND user_id = ?;",
                (tenant_id, user_id)
            )
            conn.commit()
        logger.info("ServiceScopeManager REJECTED user='%s' (tenant='%s', rejected_by='%s', reason='%s')",
                    user_id, tenant_id, rejected_by, reason)
        profile = self.get_user_profile(tenant_id, user_id)
        return profile or {}

    def update_user_access(
        self,
        tenant_id: str,
        user_id: str,
        clearance_level: Optional[int] = None,
        role: Optional[str] = None,
        scopes: Optional[List[Union[ServiceScope, str]]] = None,
        updated_by: str = "admin"
    ) -> Dict[str, Any]:
        """Updates clearance level, role, or scopes for an existing approved user."""
        profile = self.get_user_profile(tenant_id, user_id)
        if not profile:
            raise ValueError(f"Cannot update access: user '{user_id}' does not exist in tenant '{tenant_id}'")

        new_clearance = clearance_level if clearance_level is not None else profile["clearance_level"]
        new_role = role.lower().strip() if role is not None else profile["role"]

        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE service_user_profiles 
                SET clearance_level = ?, role = ?
                WHERE tenant_id = ? AND user_id = ?;
                """,
                (new_clearance, new_role, tenant_id, user_id)
            )
            if role is not None:
                conn.execute(
                    "INSERT OR REPLACE INTO service_user_roles (tenant_id, user_id, role, assigned_at) VALUES (?, ?, ?, ?);",
                    (tenant_id, user_id, new_role, time.time())
                )
            conn.commit()

        if scopes is not None:
            # Replace user scopes with the updated set
            with self._get_connection() as conn:
                conn.execute("DELETE FROM service_user_scopes WHERE tenant_id = ? AND user_id = ?;", (tenant_id, user_id))
                conn.commit()
            for sc in scopes:
                self.grant_scope(tenant_id, user_id, sc, granted_by=updated_by)

        logger.info("ServiceScopeManager UPDATED access for user='%s' (tenant='%s', clearance=%d, role='%s')",
                    user_id, tenant_id, new_clearance, new_role)
        return self.get_user_profile(tenant_id, user_id) or {}

    def get_user_profile(self, tenant_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Fetches the complete local onboarding profile for a user."""
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM service_user_profiles WHERE tenant_id = ? AND user_id = ?;",
                (tenant_id, user_id)
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def get_user_clearance(self, tenant_id: str, user_id: str) -> int:
        """Returns the user's local clearance level, or default 1 if not onboarded."""
        profile = self.get_user_profile(tenant_id, user_id)
        if profile and profile.get("clearance_level") is not None:
            return int(profile["clearance_level"])
        return 1

    def get_primary_role(self, tenant_id: str, user_id: str) -> str:
        """Returns the user's primary local role, or default 'member'."""
        profile = self.get_user_profile(tenant_id, user_id)
        if profile and profile.get("role"):
            return str(profile["role"])
        roles = self.get_user_roles(tenant_id, user_id)
        if "admin" in roles:
            return "admin"
        return roles[0] if roles else "member"

    def list_pending_users(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Returns all user profiles awaiting administrator approval."""
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM service_user_profiles WHERE tenant_id = ? AND status = 'PENDING_APPROVAL' ORDER BY created_at ASC;",
                (tenant_id,)
            )
            return [dict(row) for row in cur.fetchall()]

    # -------------------------------------------------------------------------
    # Granular Scope & Role Granting / Revocation
    # -------------------------------------------------------------------------

    def grant_scope(
        self, 
        tenant_id: str, 
        user_id: str, 
        scope: Union[ServiceScope, str], 
        granted_by: str = "admin"
    ) -> None:
        """Dynamically grants a specific functional scope to a user."""
        scope_str = scope.value if isinstance(scope, ServiceScope) else str(scope)
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO service_user_scopes (
                    tenant_id, user_id, scope, granted_by, granted_at
                ) VALUES (?, ?, ?, ?, ?);
                """,
                (tenant_id, user_id, scope_str, granted_by, now)
            )
            conn.commit()
        logger.info(
            "ServiceScopeManager GRANTED scope '%s' to user='%s' (tenant='%s', granted_by='%s')",
            scope_str, user_id, tenant_id, granted_by
        )

    def revoke_scope(
        self, 
        tenant_id: str, 
        user_id: str, 
        scope: Union[ServiceScope, str]
    ) -> bool:
        """Dynamically revokes a functional scope from a user."""
        scope_str = scope.value if isinstance(scope, ServiceScope) else str(scope)
        with self._get_connection() as conn:
            cur = conn.execute(
                "DELETE FROM service_user_scopes WHERE tenant_id = ? AND user_id = ? AND scope = ?;",
                (tenant_id, user_id, scope_str)
            )
            conn.commit()
            revoked = cur.rowcount > 0

        if revoked:
            logger.info("ServiceScopeManager REVOKED scope '%s' from user='%s' (tenant='%s')", scope_str, user_id, tenant_id)
        else:
            logger.debug("ServiceScopeManager revoke no-op: user='%s' did not have scope '%s'", user_id, scope_str)
        return revoked

    def assign_role(self, tenant_id: str, user_id: str, role: str) -> None:
        """Assigns a local service role to a user."""
        now = time.time()
        role_clean = role.lower().strip()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO service_user_roles (
                    tenant_id, user_id, role, assigned_at
                ) VALUES (?, ?, ?, ?);
                """,
                (tenant_id, user_id, role_clean, now)
            )
            # If profile exists, also keep profile role updated
            conn.execute(
                "UPDATE service_user_profiles SET role = ? WHERE tenant_id = ? AND user_id = ?;",
                (role_clean, tenant_id, user_id)
            )
            conn.commit()
        logger.info("ServiceScopeManager ASSIGNED role '%s' to user='%s' (tenant='%s')", role_clean, user_id, tenant_id)

    def revoke_role(self, tenant_id: str, user_id: str, role: str) -> bool:
        """Revokes a local service role from a user."""
        role_clean = role.lower().strip()
        with self._get_connection() as conn:
            cur = conn.execute(
                "DELETE FROM service_user_roles WHERE tenant_id = ? AND user_id = ? AND role = ?;",
                (tenant_id, user_id, role_clean)
            )
            conn.commit()
            return cur.rowcount > 0

    def get_user_roles(self, tenant_id: str, user_id: str) -> List[str]:
        """Returns the list of roles assigned to a user in the local table."""
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT role FROM service_user_roles WHERE tenant_id = ? AND user_id = ?;",
                (tenant_id, user_id)
            )
            return [row["role"] for row in cur.fetchall()]

    # -------------------------------------------------------------------------
    # Evaluation & Resolution
    # -------------------------------------------------------------------------

    def is_admin(self, tenant_id: str, user_id: str, user_context: Optional[Any] = None) -> bool:
        """
        Determines whether a caller has administrative access based strictly on local authority:
        1. User profile is APPROVED and has role='admin' or clearance_level >= 3, OR
        2. Explicit 'admin' role in service_user_roles table, OR
        3. Explicit '*' or 'admin:manage' grant in service_user_scopes table.
        Does NOT check token claims.
        """
        profile = self.get_user_profile(tenant_id, user_id)
        if profile:
            if profile.get("status") != UserApprovalStatus.APPROVED.value:
                return False
            if profile.get("role") == "admin" or int(profile.get("clearance_level", 1)) >= 3:
                return True

        roles = self.get_user_roles(tenant_id, user_id)
        if "admin" in roles:
            return True

        with self._get_connection() as conn:
            cur = conn.execute(
                """
                SELECT 1 FROM service_user_scopes 
                WHERE tenant_id = ? AND user_id = ? AND scope IN ('*', 'admin:manage') 
                LIMIT 1;
                """,
                (tenant_id, user_id)
            )
            if cur.fetchone():
                return True

        return False

    def get_user_scopes(
        self, 
        tenant_id: str, 
        user_id: str, 
        user_context: Optional[Any] = None
    ) -> List[str]:
        """
        Resolves the full effective set of functional scopes for an approved user:
        1. Only for admin users: Full access wildcard ('*') and all functional scopes.
        2. For approved users: Scopes explicitly granted in service_user_scopes (defaulting to ['rag:read']).
        """
        # 1. Admin Full Access Gate
        if self.is_admin(tenant_id, user_id, user_context):
            logger.debug(
                "ServiceScopeManager resolved scopes for ADMIN user='%s' (tenant='%s'): Full Access granted (*)",
                user_id, tenant_id
            )
            return [
                ServiceScope.ALL.value,
                ServiceScope.RAG_READ.value,
                ServiceScope.INGESTION_WRITE.value,
                ServiceScope.ADMIN_MANAGE.value
            ]

        # 2. Query Local Table for Explicit Grants
        scopes = set()
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT scope FROM service_user_scopes WHERE tenant_id = ? AND user_id = ?;",
                (tenant_id, user_id)
            )
            for row in cur.fetchall():
                scopes.add(row["scope"])

        # Baseline fallback for approved users with empty explicit scope records
        if not scopes:
            scopes.add(ServiceScope.RAG_READ.value)

        resolved = sorted(list(scopes))
        logger.debug(
            "ServiceScopeManager resolved scopes for user='%s' (tenant='%s'): %s (grants_count=%d)",
            user_id, tenant_id, resolved, len(resolved)
        )
        return resolved

    def list_user_grants(self, tenant_id: str, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns records of all explicit scope grants, optionally filtered by user."""
        with self._get_connection() as conn:
            if user_id:
                cur = conn.execute(
                    "SELECT * FROM service_user_scopes WHERE tenant_id = ? AND user_id = ? ORDER BY granted_at DESC;",
                    (tenant_id, user_id)
                )
            else:
                cur = conn.execute(
                    "SELECT * FROM service_user_scopes WHERE tenant_id = ? ORDER BY granted_at DESC;",
                    (tenant_id,)
                )
            return [dict(row) for row in cur.fetchall()]

    def clear(self) -> None:
        """Purges all user profiles, scopes, and roles (used for unit testing isolation)."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM service_user_profiles;")
            conn.execute("DELETE FROM service_user_scopes;")
            conn.execute("DELETE FROM service_user_roles;")
            conn.commit()
        logger.debug("ServiceScopeManager tables cleared.")

local_scope_manager = ServiceScopeManager()
