"""
Multi-tenant authentication and authorization.

Provides API key → namespace mapping for tenant isolation.
Each tenant's memories are stored in a separate namespace (separate tables).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from dataclasses import dataclass

from cortiloop.config import AuthConfig

logger = logging.getLogger("cortiloop.auth")


class AuthError(Exception):
    """Raised when authentication fails."""


@dataclass
class AuthContext:
    """Authenticated context for a request."""
    namespace: str
    is_admin: bool = False


class AuthManager:
    """
    API key based authentication with namespace isolation.

    Keys map to namespaces — each key can only access its own namespace.
    Admin key can access any namespace.
    """

    def __init__(self, config: AuthConfig):
        self.config = config
        # Hash stored keys for comparison security
        self._key_map: dict[str, str] = {}  # hash(key) → namespace
        self._admin_hash: str = ""

        if config.enabled:
            for key, ns in config.api_keys.items():
                self._key_map[self._hash(key)] = ns
            if config.admin_key:
                self._admin_hash = self._hash(config.admin_key)
            logger.info(
                "Auth enabled: %d tenant keys configured", len(config.api_keys)
            )

    @staticmethod
    def _hash(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def authenticate(self, api_key: str, requested_namespace: str = "") -> AuthContext:
        """
        Authenticate a request and return the authorized namespace.

        Args:
            api_key: The API key from the request
            requested_namespace: Optional namespace override (admin only)

        Returns:
            AuthContext with the authorized namespace

        Raises:
            AuthError: If authentication fails
        """
        if not self.config.enabled:
            return AuthContext(namespace=requested_namespace or "default")

        if not api_key:
            raise AuthError("API key required")

        key_hash = self._hash(api_key)

        # Check admin key
        if self._admin_hash and hmac.compare_digest(key_hash, self._admin_hash):
            ns = requested_namespace or "default"
            return AuthContext(namespace=ns, is_admin=True)

        # Check tenant key
        namespace = self._key_map.get(key_hash)
        if namespace is None:
            raise AuthError("Invalid API key")

        # Tenant can only access their own namespace
        if requested_namespace and requested_namespace != namespace:
            raise AuthError(
                f"Key authorized for namespace '{namespace}', "
                f"not '{requested_namespace}'"
            )

        return AuthContext(namespace=namespace)

    @staticmethod
    def generate_key() -> str:
        """Generate a secure API key."""
        return f"cl_{secrets.token_urlsafe(32)}"

    def list_namespaces(self, api_key: str) -> list[str]:
        """List namespaces accessible to the given key."""
        ctx = self.authenticate(api_key)
        if ctx.is_admin:
            return list(set(self._key_map.values())) + ["default"]
        return [ctx.namespace]
