import os
import time
import json
import hmac
import hashlib
import base64
import uuid
import logging
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from .context import UserContext, UserTier
from .scopes import ServiceScope, ServiceScopeManager, local_scope_manager, UserApprovalStatus
from src.common.redis_client import redis_manager

logger = logging.getLogger("kitchome.auth.telemetry")

class AuthTelemetryEvent(BaseModel):
    """Structured telemetry event capturing authentication, token verification, and perimeter authorization trace metrics."""
    trace_id: str = Field(default_factory=lambda: f"auth_{uuid.uuid4().hex[:12]}")
    timestamp: float = Field(default_factory=time.time)
    action: str  # "TOKEN_ISSUE", "TOKEN_VERIFY", "SERVICE_AUTH"
    status: str  # "SUCCESS", "DENIED", "EXPIRED", "INVALID_SIGNATURE", "MALFORMED", "SCOPE_MISMATCH", "MISSING_CREDENTIALS"
    user_id: Optional[str] = None
    tier: Optional[str] = None
    tenant_id: Optional[str] = None
    clearance_level: Optional[int] = None
    required_scope: Optional[str] = None
    token_scopes: Optional[List[str]] = None
    duration_ms: float = 0.0
    error_message: Optional[str] = None

class AuthTelemetryTracker:
    """
    In-memory and observable telemetry tracker for authentication and perimeter authorization.
    Maintains time-series trace metrics (latencies, counts, failure modes, success rates).
    """
    def __init__(self, max_events: int = 1000):
        self.max_events = max_events
        self._events: List[AuthTelemetryEvent] = []

    def record_event(self, event: AuthTelemetryEvent) -> None:
        if len(self._events) >= self.max_events:
            self._events.pop(0)
        self._events.append(event)
        redis_manager.push_telemetry_event(event.model_dump(), max_events=self.max_events)
        logger.debug(
            f"[TRACE {event.trace_id}] Telemetry recorded: action={event.action}, status={event.status}, "
            f"user='{event.user_id}', tenant='{event.tenant_id}', duration={event.duration_ms:.2f}ms (queue_depth={len(self._events)})"
        )

    def get_events(self) -> List[AuthTelemetryEvent]:
        return list(self._events)

    def get_metrics(self) -> Dict[str, Any]:
        total = len(self._events)
        if total == 0:
            return {
                "total_events": 0,
                "success_count": 0,
                "denied_count": 0,
                "success_rate": 1.0,
                "token_expired_count": 0,
                "signature_failure_count": 0,
                "scope_mismatch_count": 0,
                "average_duration_ms": 0.0,
                "p95_duration_ms": 0.0
            }

        successes = sum(1 for e in self._events if e.status == "SUCCESS")
        denials = total - successes
        expired = sum(1 for e in self._events if e.status == "EXPIRED")
        sig_failures = sum(1 for e in self._events if e.status == "INVALID_SIGNATURE")
        scope_mismatches = sum(1 for e in self._events if e.status == "SCOPE_MISMATCH")

        durations = sorted(e.duration_ms for e in self._events)
        avg_duration = round(sum(durations) / total, 3)
        p95_index = min(int(total * 0.95), total - 1)
        p95_duration = round(durations[p95_index], 3)

        metrics = {
            "total_events": total,
            "success_count": successes,
            "denied_count": denials,
            "success_rate": round(successes / total, 4),
            "token_expired_count": expired,
            "signature_failure_count": sig_failures,
            "scope_mismatch_count": scope_mismatches,
            "average_duration_ms": avg_duration,
            "p95_duration_ms": p95_duration
        }
        logger.debug(
            f"Auth telemetry metrics retrieved: total={total}, successes={successes}, denials={denials}, "
            f"success_rate={metrics['success_rate']}, p95={p95_duration}ms"
        )
        return metrics

    def clear(self) -> None:
        self._events.clear()
        logger.debug("Auth telemetry event tracker history cleared.")

auth_telemetry = AuthTelemetryTracker()

class JWTTokenManager:
    """
    Standard RFC 7519 HMAC-SHA256 (HS256) JWT Encoder and Decoder.
    Uses pure Python standard library (hmac, hashlib, base64, json) for zero-dependency portability.
    Emits structured telemetry trace events and logs for all token operations.
    """
    DEFAULT_SECRET = os.getenv("AUTH_JWT_SECRET", "kitchome-default-secret-key-change-in-production")
    telemetry: AuthTelemetryTracker = auth_telemetry

    @classmethod
    def _b64url_encode(cls, data: bytes) -> str:
        return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")

    @classmethod
    def _b64url_decode(cls, s: str) -> bytes:
        rem = len(s) % 4
        if rem > 0:
            s += "=" * (4 - rem)
        return base64.urlsafe_b64decode(s)

    @classmethod
    def _get_secret_bytes(cls, secret: Optional[str] = None) -> bytes:
        raw = secret or cls.DEFAULT_SECRET
        try:
            decoded = base64.b64decode(raw, validate=True)
            if len(decoded) >= 32:
                return decoded
        except Exception:
            pass
        return raw.encode("utf-8")

    @classmethod
    def create_token(
        cls,
        user: UserContext,
        scopes: Optional[List[ServiceScope]] = None,
        secret: Optional[str] = None,
        ttl_seconds: int = 3600,
        trace_id: Optional[str] = None
    ) -> str:
        """
        Creates a signed HS256 JWT containing user attributes and functional scopes.
        Emits telemetry trace metrics and logs for the issuance event.
        """
        start_time = time.perf_counter()
        t_id = trace_id or user.trace_id or f"auth_{uuid.uuid4().hex[:12]}"
        secret_key = cls._get_secret_bytes(secret)
        header = {"alg": "HS256", "typ": "JWT"}
        now = int(time.time())
        scope_vals = None
        if scopes is not None:
            scope_vals = [s.value if isinstance(s, ServiceScope) else str(s) for s in scopes]
        tier_val = user.tier.value if isinstance(user.tier, UserTier) else str(user.tier)

        payload = {
            "sub": user.user_id,
            "tier": tier_val,
            "tenant_id": user.tenant_id,
            "clearance_level": user.clearance_level,
            "iat": now,
            "exp": now + ttl_seconds,
            "iss": "kitchome-auth",
            "trace_id": t_id
        }
        if scope_vals is not None:
            payload["scopes"] = scope_vals
        if user.role:
            payload["role"] = user.role
        if user.custom_allowed_namespaces:
            payload["custom_namespaces"] = user.custom_allowed_namespaces

        header_b64 = cls._b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
        payload_b64 = cls._b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")

        sig = hmac.new(secret_key, signing_input, hashlib.sha256).digest()
        sig_b64 = cls._b64url_encode(sig)

        token = f"{header_b64}.{payload_b64}.{sig_b64}"
        duration_ms = round((time.perf_counter() - start_time) * 1000, 3)

        logger.debug(
            f"[TRACE {t_id}] Minting JWT: sub='{user.user_id}', tenant='{user.tenant_id}', clearance={user.clearance_level}, "
            f"custom_namespaces={user.custom_allowed_namespaces}, scopes={scope_vals}, iat={now}, exp={now + ttl_seconds}, "
            f"secret_source={'custom' if secret else 'env_default'}"
        )

        event = AuthTelemetryEvent(
            trace_id=t_id,
            action="TOKEN_ISSUE",
            status="SUCCESS",
            user_id=user.user_id,
            tier=tier_val,
            tenant_id=user.tenant_id,
            clearance_level=user.clearance_level,
            token_scopes=scope_vals,
            duration_ms=duration_ms
        )
        cls.telemetry.record_event(event)
        logger.info(
            f"[TRACE {t_id}] Issued JWT token: sub='{user.user_id}', tenant='{user.tenant_id}', "
            f"tier='{tier_val}', scopes={scope_vals}, ttl={ttl_seconds}s (latency: {duration_ms:.2f}ms)"
        )
        logger.debug(
            f"[TRACE {t_id}] Generated JWT structure: header_bytes={len(header_b64)}, "
            f"payload_bytes={len(payload_b64)}, sig_bytes={len(sig_b64)}"
        )
        return token

    @classmethod
    def verify_and_decode_token(
        cls, 
        token: str, 
        secret: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verifies cryptographic HMAC signature and expiry.
        Raises ValueError or PermissionError on invalid signature or expired token.
        Emits telemetry trace metrics and logs for success and failure events.
        """
        start_time = time.perf_counter()
        t_id = trace_id or f"auth_{uuid.uuid4().hex[:12]}"
        secret_key = cls._get_secret_bytes(secret)

        if redis_manager.is_token_revoked(token):
            duration_ms = round((time.perf_counter() - start_time) * 1000, 3)
            cls.telemetry.record_event(AuthTelemetryEvent(
                trace_id=t_id,
                action="TOKEN_VERIFY",
                status="REVOKED",
                duration_ms=duration_ms,
                error_message="Session revoked: Token has been invalidated via Single Sign-Out."
            ))
            logger.warning(f"[TRACE {t_id}] Token verification failed: token is revoked in Redis ({duration_ms:.2f}ms)")
            raise PermissionError("Session revoked: Token has been invalidated via Single Sign-Out.")

        logger.debug(
            f"[TRACE {t_id}] Verifying token: raw_length={len(token.strip())}, "
            f"secret_source={'custom' if secret else 'env_default'}"
        )
        parts = token.strip().split(".")
        if len(parts) != 3:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 3)
            cls.telemetry.record_event(AuthTelemetryEvent(
                trace_id=t_id,
                action="TOKEN_VERIFY",
                status="MALFORMED",
                duration_ms=duration_ms,
                error_message="Malformed JWT: expected 3 dot-separated segments."
            ))
            logger.warning(f"[TRACE {t_id}] Token verification failed: malformed JWT structure (parts={len(parts)}) ({duration_ms:.2f}ms)")
            logger.debug(f"[TRACE {t_id}] Malformed token raw prefix: '{token[:20]}...'")
            raise ValueError("Malformed JWT: expected 3 dot-separated segments.")

        header_b64, payload_b64, sig_b64 = parts
        logger.debug(
            f"[TRACE {t_id}] Token segments extracted: header_b64_len={len(header_b64)}, "
            f"payload_b64_len={len(payload_b64)}, sig_b64_len={len(sig_b64)}"
        )
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        expected_sig = hmac.new(secret_key, signing_input, hashlib.sha256).digest()
        provided_sig = cls._b64url_decode(sig_b64)

        if not hmac.compare_digest(expected_sig, provided_sig):
            duration_ms = round((time.perf_counter() - start_time) * 1000, 3)
            cls.telemetry.record_event(AuthTelemetryEvent(
                trace_id=t_id,
                action="TOKEN_VERIFY",
                status="INVALID_SIGNATURE",
                duration_ms=duration_ms,
                error_message="Invalid cryptographic token signature."
            ))
            logger.warning(f"[TRACE {t_id}] Token verification failed: cryptographic signature mismatch ({duration_ms:.2f}ms)")
            logger.debug(
                f"[TRACE {t_id}] Signature mismatch details: expected_prefix={expected_sig[:4].hex()}..., "
                f"provided_prefix={provided_sig[:4].hex() if len(provided_sig) >= 4 else 'short'}..."
            )
            raise PermissionError("Invalid cryptographic token signature.")

        payload_bytes = cls._b64url_decode(payload_b64)
        payload = json.loads(payload_bytes.decode("utf-8"))
        token_trace_id = payload.get("trace_id") or t_id

        # Verify expiration
        exp = payload.get("exp")
        now = time.time()
        if exp is not None and now >= exp:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 3)
            cls.telemetry.record_event(AuthTelemetryEvent(
                trace_id=token_trace_id,
                action="TOKEN_VERIFY",
                status="EXPIRED",
                user_id=payload.get("sub"),
                tenant_id=payload.get("tenant_id"),
                tier=payload.get("tier"),
                duration_ms=duration_ms,
                error_message=f"Token has expired (exp={exp}, now={now:.0f})"
            ))
            logger.warning(
                f"[TRACE {token_trace_id}] Token verification failed: token expired "
                f"(user='{payload.get('sub')}', exp={exp}, now={now:.0f}) ({duration_ms:.2f}ms)"
            )
            logger.debug(
                f"[TRACE {token_trace_id}] Expiry check failure details: expired by {now - exp:.2f}s "
                f"(exp_timestamp={exp}, current_timestamp={now:.2f})"
            )
            raise PermissionError("Token has expired.")

        duration_ms = round((time.perf_counter() - start_time) * 1000, 3)
        cls.telemetry.record_event(AuthTelemetryEvent(
            trace_id=token_trace_id,
            action="TOKEN_VERIFY",
            status="SUCCESS",
            user_id=payload.get("sub"),
            tenant_id=payload.get("tenant_id"),
            tier=payload.get("tier"),
            clearance_level=payload.get("clearance_level"),
            token_scopes=payload.get("scopes"),
            duration_ms=duration_ms
        ))
        remaining_ttl = round(exp - now, 2) if exp else None
        logger.info(
            f"[TRACE {token_trace_id}] Token verified successfully: sub='{payload.get('sub')}', "
            f"tenant='{payload.get('tenant_id')}', tier='{payload.get('tier')}', "
            f"scopes={payload.get('scopes')}, remaining_ttl={remaining_ttl}s ({duration_ms:.2f}ms)"
        )
        logger.debug(
            f"[TRACE {token_trace_id}] Token payload claims verified: iss='{payload.get('iss')}', "
            f"clearance_level={payload.get('clearance_level')}, custom_namespaces={payload.get('custom_namespaces')}"
        )
        return payload

    @classmethod
    def get_user_from_token(
        cls, 
        token: str, 
        secret: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> UserContext:
        """
        Decodes a verified token and hydrates a tamper-proof UserContext with trace ID.
        """
        payload = cls.verify_and_decode_token(token, secret=secret, trace_id=trace_id)
        tier_str = payload.get("tier", "free")
        try:
            tier = UserTier(tier_str)
        except ValueError:
            logger.debug(f"Unrecognized tier '{tier_str}' in token, defaulting to FREE.")
            tier = UserTier.FREE

        user = UserContext(
            user_id=payload.get("sub", "anonymous"),
            tier=tier,
            tenant_id=payload.get("tenant_id", "global"),
            clearance_level=int(payload.get("clearance_level", 1)),
            custom_allowed_namespaces=payload.get("custom_namespaces"),
            trace_id=payload.get("trace_id") or trace_id,
            role=payload.get("role")
        )
        logger.debug(
            f"[TRACE {user.trace_id}] Hydrated UserContext from token: "
            f"user_id='{user.user_id}', tier='{user.tier.value}', tenant_id='{user.tenant_id}', "
            f"clearance={user.clearance_level}, role='{user.role}', custom_namespaces={user.custom_allowed_namespaces}"
        )
        return user

class JwksTokenVerifier:
    """
    Industry-Standard Asymmetric Cryptography (RS256) Token Verifier.
    Fetches and caches public keys from /.well-known/jwks.json (RFC 7517).
    Verifies cryptographic signatures locally in memory in < 0.1ms with zero shared secrets.
    """
    def __init__(self, jwks_url: Optional[str] = None):
        self.jwks_url = jwks_url or os.getenv("JWKS_URL", "http://localhost:8080/.well-known/jwks.json")
        self._client: Optional[Any] = None

    def _get_client(self) -> Optional[Any]:
        if self._client is None:
            try:
                import jwt
                self._client = jwt.PyJWKClient(self.jwks_url, cache_keys=True, max_cached_keys=16)
            except Exception as e:
                logger.debug(f"PyJWKClient initialization note ({self.jwks_url}): {e}")
        return self._client

    def verify_token(self, token: str, trace_id: Optional[str] = None) -> Dict[str, Any]:
        if redis_manager.is_token_revoked(token):
            raise PermissionError("Session revoked: Token has been invalidated via Single Sign-Out.")

        try:
            import jwt
        except ImportError:
            raise RuntimeError("pyjwt[crypto] package is required for RS256 token verification.")

        client = self._get_client()
        if client is None:
            raise PermissionError(f"JWKS service endpoint unavailable at {self.jwks_url}")

        t_id = trace_id or f"auth_{uuid.uuid4().hex[:12]}"
        try:
            signing_key = client.get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                options={"verify_exp": True}
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise PermissionError("Token has expired.")
        except Exception as e:
            raise PermissionError(f"Invalid RS256 token signature or claims: {e}")

jwks_verifier = JwksTokenVerifier()

def verify_token_hybrid(token: str, secret: Optional[str] = None, trace_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Verifies a token using the appropriate cryptographic strategy:
    - If alg == RS256: verifies against Central Auth Service JWKS public key.
    - If alg == HS256: verifies against symmetric secret.
    """
    t = token.strip()
    if t.lower().startswith("bearer "):
        t = t[7:].strip()

    alg = "HS256"
    try:
        import jwt
        header = jwt.get_unverified_header(t)
        alg = header.get("alg", "HS256")
    except Exception:
        parts = t.split(".")
        if len(parts) == 3:
            try:
                rem = len(parts[0]) % 4
                padded = parts[0] + ("=" * (4 - rem) if rem else "")
                hdr = json.loads(base64.urlsafe_b64decode(padded.encode()).decode("utf-8"))
                alg = hdr.get("alg", "HS256")
            except Exception:
                pass

    if alg == "RS256":
        return jwks_verifier.verify_token(t, trace_id=trace_id)
    else:
        return JWTTokenManager.verify_and_decode_token(t, secret=secret, trace_id=trace_id)

class ServiceAuthGuard:
    """
    Perimeter Gatekeeper executing coarse-grained service authorization.
    Validates token signatures, resolves functional scopes via local ServiceScopeManager,
    and emits structured trace metrics and logs.
    """
    telemetry: AuthTelemetryTracker = auth_telemetry
    scope_manager: ServiceScopeManager = local_scope_manager

    @classmethod
    def enforce_service_auth(
        cls,
        auth_header_or_token: Optional[str] = None,
        required_scope: Optional[ServiceScope] = None,
        fallback_user: Optional[UserContext] = None,
        secret: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> UserContext:
        """
        1. If token is provided: verifies signature, resolves scopes (from token or local table),
           and returns hydrated UserContext.
        2. If token is omitted but fallback_user is provided: permits execution (backward compatibility).
        3. If neither is provided: rejects caller with PermissionError.
        Emits structured telemetry trace logs and records AuthTelemetryEvent metrics.
        """
        start_time = time.perf_counter()
        t_id = trace_id or (fallback_user.trace_id if fallback_user else None) or f"auth_{uuid.uuid4().hex[:12]}"
        req_val = required_scope.value if isinstance(required_scope, ServiceScope) else (str(required_scope) if required_scope else None)

        logger.debug(
            f"[TRACE {t_id}] ServiceAuthGuard evaluating perimeter request: "
            f"has_token={bool(auth_header_or_token)}, has_fallback_user={bool(fallback_user)}, "
            f"required_scope='{req_val}'"
        )

        if auth_header_or_token:
            token = auth_header_or_token.strip()
            had_bearer_prefix = token.lower().startswith("bearer ")
            if had_bearer_prefix:
                token = token[7:].strip()
            logger.debug(f"[TRACE {t_id}] Header parsed: had_bearer_prefix={had_bearer_prefix}, token_len={len(token)}")

            payload = verify_token_hybrid(token, secret=secret, trace_id=t_id)
            token_trace_id = payload.get("trace_id") or t_id

            tier_str = payload.get("tier", "free")
            try:
                tier = UserTier(tier_str)
            except ValueError:
                logger.debug(f"Unrecognized tier '{tier_str}' in token, defaulting to FREE.")
                tier = UserTier.FREE

            user_id = payload.get("sub", "anonymous")
            tenant_id = payload.get("tenant_id", "global")

            # Check if token carries authoritative ADMIN role from Central Auth Service
            roles = payload.get("roles", [])
            is_upstream_admin = any(r in ("ADMIN", "ROLE_ADMIN") for r in roles)

            # 1. Onboarding & Approval Lifecycle Resolution:
            profile = cls.scope_manager.get_user_profile(tenant_id, user_id)

            if profile is None:
                # Check if this user is a Central Auth admin or if tenant has no existing administrator
                if is_upstream_admin or not cls.scope_manager.has_admin(tenant_id):
                    # Founding User Auto-Bootstrap Pattern
                    profile = cls.scope_manager.bootstrap_founding_admin(tenant_id, user_id)
                    logger.info(
                        f"[TRACE {token_trace_id}] Founding admin auto-bootstrapped: "
                        f"user_id='{user_id}', tenant='{tenant_id}' (upstream_admin={is_upstream_admin})"
                    )
                else:
                    # Tenant already has an admin; new user must be registered as PENDING_APPROVAL
                    profile = cls.scope_manager.register_pending_user(tenant_id, user_id)
                    duration_ms = round((time.perf_counter() - start_time) * 1000, 3)
                    cls.telemetry.record_event(AuthTelemetryEvent(
                        trace_id=token_trace_id,
                        action="SERVICE_AUTH",
                        status="PENDING_APPROVAL",
                        user_id=user_id,
                        tenant_id=tenant_id,
                        tier=tier.value,
                        clearance_level=1,
                        required_scope=req_val,
                        token_scopes=[],
                        duration_ms=duration_ms,
                        error_message="Access Denied: Account is pending administrator approval. Please visit /onboarding to view request status."
                    ))
                    logger.warning(
                        f"[TRACE {token_trace_id}] Service authorization DENIED: user='{user_id}' (tenant='{tenant_id}') "
                        f"is PENDING_APPROVAL ({duration_ms:.2f}ms)"
                    )
                    raise PermissionError(
                        "Access Denied: Account is pending administrator approval. "
                        "Please visit /onboarding to view request status."
                    )

            # Check Approval Status
            status = profile.get("status")
            if status == UserApprovalStatus.PENDING_APPROVAL.value:
                duration_ms = round((time.perf_counter() - start_time) * 1000, 3)
                cls.telemetry.record_event(AuthTelemetryEvent(
                    trace_id=token_trace_id,
                    action="SERVICE_AUTH",
                    status="PENDING_APPROVAL",
                    user_id=user_id,
                    tenant_id=tenant_id,
                    tier=tier.value,
                    clearance_level=int(profile.get("clearance_level", 1)),
                    required_scope=req_val,
                    token_scopes=[],
                    duration_ms=duration_ms,
                    error_message="Access Denied: Account is pending administrator approval. Please visit /onboarding to view request status."
                ))
                raise PermissionError(
                    "Access Denied: Account is pending administrator approval. "
                    "Please visit /onboarding to view request status."
                )
            elif status == UserApprovalStatus.REJECTED.value:
                duration_ms = round((time.perf_counter() - start_time) * 1000, 3)
                reason_str = f" Reason: {profile.get('rejection_reason')}" if profile.get("rejection_reason") else ""
                cls.telemetry.record_event(AuthTelemetryEvent(
                    trace_id=token_trace_id,
                    action="SERVICE_AUTH",
                    status="REGISTRATION_REJECTED",
                    user_id=user_id,
                    tenant_id=tenant_id,
                    tier=tier.value,
                    clearance_level=int(profile.get("clearance_level", 1)),
                    required_scope=req_val,
                    token_scopes=[],
                    duration_ms=duration_ms,
                    error_message=f"Access Denied: Account registration has been rejected.{reason_str}"
                ))
                raise PermissionError(f"Access Denied: Account registration has been rejected.{reason_str}")

            # 2. Approved User: Hydrate UserContext strictly from local profile
            user = UserContext(
                user_id=user_id,
                tier=tier,
                tenant_id=tenant_id,
                clearance_level=int(profile.get("clearance_level", 1)),
                custom_allowed_namespaces=payload.get("custom_namespaces"),
                trace_id=token_trace_id,
                role=profile.get("role", "member")
            )

            # 3. Resolve Scopes from Local Table
            effective_scopes = cls.scope_manager.get_user_scopes(
                tenant_id=user.tenant_id,
                user_id=user.user_id,
                user_context=user
            )

            # Capability ceiling / downscoping:
            # If the token explicitly requests restricted scopes (e.g. M2M worker token or downscoped session),
            # restrict effective_scopes to what the token asked for (unless token requested '*').
            token_scopes_claim = payload.get("scopes")
            if token_scopes_claim is not None and len(token_scopes_claim) > 0:
                claim_list = list(token_scopes_claim)
                if ServiceScope.ALL.value not in claim_list and "*" not in claim_list:
                    effective_scopes = [s for s in claim_list if s in effective_scopes or ServiceScope.ALL.value in effective_scopes or "*" in effective_scopes]
                    scope_source = "token_downscoped"
                else:
                    scope_source = "token_claim"
            else:
                scope_source = "local_service_table"

            user.granted_scopes = effective_scopes

            if required_scope:
                has_exact = req_val in effective_scopes
                has_wildcard = ServiceScope.ALL.value in effective_scopes
                logger.debug(
                    f"[TRACE {token_trace_id}] Scope evaluation: required='{req_val}', "
                    f"granted={effective_scopes}, exact_match={has_exact}, wildcard_match={has_wildcard} "
                    f"(source={scope_source})"
                )
                if not has_exact and not has_wildcard:
                    duration_ms = round((time.perf_counter() - start_time) * 1000, 3)
                    cls.telemetry.record_event(AuthTelemetryEvent(
                        trace_id=token_trace_id,
                        action="SERVICE_AUTH",
                        status="SCOPE_MISMATCH",
                        user_id=user.user_id,
                        tenant_id=user.tenant_id,
                        tier=user.tier.value,
                        clearance_level=user.clearance_level,
                        required_scope=req_val,
                        token_scopes=effective_scopes,
                        duration_ms=duration_ms,
                        error_message=f"Access Forbidden: Token lacks required scope '{req_val}'."
                    ))
                    logger.warning(
                        f"[TRACE {token_trace_id}] Service authorization DENIED: sub='{user.user_id}', "
                        f"tenant='{user.tenant_id}' lacks required scope '{req_val}' "
                        f"(granted: {effective_scopes}, source={scope_source}) ({duration_ms:.2f}ms)"
                    )
                    raise PermissionError(f"Access Forbidden: Token lacks required scope '{req_val}'.")

            duration_ms = round((time.perf_counter() - start_time) * 1000, 3)
            cls.telemetry.record_event(AuthTelemetryEvent(
                trace_id=token_trace_id,
                action="SERVICE_AUTH",
                status="SUCCESS",
                user_id=user.user_id,
                tier=user.tier.value,
                tenant_id=user.tenant_id,
                clearance_level=user.clearance_level,
                required_scope=req_val,
                token_scopes=effective_scopes,
                duration_ms=duration_ms
            ))
            logger.info(
                f"[TRACE {token_trace_id}] Service authorization SUCCESS: user='{user.user_id}', "
                f"tenant='{user.tenant_id}', tier='{user.tier.value}', scope='{req_val}' ({duration_ms:.2f}ms)"
            )
            logger.debug(
                f"[TRACE {token_trace_id}] Security context: clearance={user.clearance_level}, "
                f"namespaces={user.custom_allowed_namespaces}, scopes={effective_scopes} (source={scope_source})"
            )
            return user

        if fallback_user:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 3)
            if not fallback_user.trace_id:
                fallback_user.trace_id = t_id

            if not fallback_user.granted_scopes:
                fallback_user.granted_scopes = cls.scope_manager.get_user_scopes(
                    tenant_id=fallback_user.tenant_id,
                    user_id=fallback_user.user_id,
                    user_context=fallback_user
                )

            if required_scope:
                has_exact = req_val in fallback_user.granted_scopes
                has_wildcard = ServiceScope.ALL.value in fallback_user.granted_scopes
                if not has_exact and not has_wildcard:
                    cls.telemetry.record_event(AuthTelemetryEvent(
                        trace_id=t_id,
                        action="SERVICE_AUTH",
                        status="SCOPE_MISMATCH",
                        user_id=fallback_user.user_id,
                        tenant_id=fallback_user.tenant_id,
                        tier=fallback_user.tier.value if isinstance(fallback_user.tier, UserTier) else str(fallback_user.tier),
                        clearance_level=fallback_user.clearance_level,
                        required_scope=req_val,
                        token_scopes=fallback_user.granted_scopes,
                        duration_ms=duration_ms,
                        error_message=f"Access Forbidden: Token lacks required scope '{req_val}'."
                    ))
                    logger.warning(
                        f"[TRACE {t_id}] Service authorization DENIED (fallback user): sub='{fallback_user.user_id}' lacks required scope '{req_val}'"
                    )
                    raise PermissionError(f"Access Forbidden: Token lacks required scope '{req_val}'.")

            cls.telemetry.record_event(AuthTelemetryEvent(
                trace_id=t_id,
                action="SERVICE_AUTH",
                status="SUCCESS",
                user_id=fallback_user.user_id,
                tier=fallback_user.tier.value if isinstance(fallback_user.tier, UserTier) else str(fallback_user.tier),
                tenant_id=fallback_user.tenant_id,
                clearance_level=fallback_user.clearance_level,
                required_scope=req_val,
                duration_ms=duration_ms
            ))
            logger.info(
                f"[TRACE {t_id}] Service authorization SUCCESS (fallback user): "
                f"user='{fallback_user.user_id}', tenant='{fallback_user.tenant_id}' ({duration_ms:.2f}ms)"
            )
            logger.debug(
                f"[TRACE {t_id}] Permitted via fallback context: tier='{fallback_user.tier.value}', "
                f"clearance={fallback_user.clearance_level}, scopes={fallback_user.granted_scopes}"
            )
            return fallback_user

        duration_ms = round((time.perf_counter() - start_time) * 1000, 3)
        cls.telemetry.record_event(AuthTelemetryEvent(
            trace_id=t_id,
            action="SERVICE_AUTH",
            status="MISSING_CREDENTIALS",
            required_scope=req_val,
            duration_ms=duration_ms,
            error_message="Unauthorized: Missing authentication credentials."
        ))
        logger.warning(
            f"[TRACE {t_id}] Service authorization DENIED: missing authentication credentials "
            f"(required_scope='{req_val}') ({duration_ms:.2f}ms)"
        )
        logger.debug(f"[TRACE {t_id}] Neither bearer token nor fallback user was supplied to perimeter gatekeeper.")
        raise PermissionError("Unauthorized: Missing authentication credentials.")

