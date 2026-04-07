"""Tests for multi-tenant authentication."""

import pytest

from cortiloop.auth import AuthManager, AuthContext, AuthError
from cortiloop.config import AuthConfig


def test_auth_disabled_passes_all():
    config = AuthConfig(enabled=False)
    mgr = AuthManager(config)
    ctx = mgr.authenticate("anything", "any_ns")
    assert ctx.namespace == "any_ns"
    assert not ctx.is_admin


def test_auth_disabled_default_namespace():
    config = AuthConfig(enabled=False)
    mgr = AuthManager(config)
    ctx = mgr.authenticate("")
    assert ctx.namespace == "default"


def test_valid_tenant_key():
    config = AuthConfig(
        enabled=True,
        api_keys={"key_a": "tenant_a", "key_b": "tenant_b"},
    )
    mgr = AuthManager(config)
    ctx = mgr.authenticate("key_a")
    assert ctx.namespace == "tenant_a"
    assert not ctx.is_admin


def test_invalid_key_raises():
    config = AuthConfig(enabled=True, api_keys={"key_a": "tenant_a"})
    mgr = AuthManager(config)
    with pytest.raises(AuthError, match="Invalid API key"):
        mgr.authenticate("wrong_key")


def test_empty_key_raises():
    config = AuthConfig(enabled=True, api_keys={"key_a": "tenant_a"})
    mgr = AuthManager(config)
    with pytest.raises(AuthError, match="API key required"):
        mgr.authenticate("")


def test_tenant_cannot_access_other_namespace():
    config = AuthConfig(
        enabled=True,
        api_keys={"key_a": "tenant_a", "key_b": "tenant_b"},
    )
    mgr = AuthManager(config)
    with pytest.raises(AuthError, match="not 'tenant_b'"):
        mgr.authenticate("key_a", requested_namespace="tenant_b")


def test_admin_key_can_access_any_namespace():
    config = AuthConfig(
        enabled=True,
        api_keys={"key_a": "tenant_a"},
        admin_key="admin_secret",
    )
    mgr = AuthManager(config)
    ctx = mgr.authenticate("admin_secret", requested_namespace="tenant_a")
    assert ctx.namespace == "tenant_a"
    assert ctx.is_admin


def test_admin_default_namespace():
    config = AuthConfig(enabled=True, admin_key="admin_secret")
    mgr = AuthManager(config)
    ctx = mgr.authenticate("admin_secret")
    assert ctx.namespace == "default"
    assert ctx.is_admin


def test_generate_key_format():
    key = AuthManager.generate_key()
    assert key.startswith("cl_")
    assert len(key) > 20


def test_list_namespaces_tenant():
    config = AuthConfig(
        enabled=True,
        api_keys={"key_a": "tenant_a", "key_b": "tenant_b"},
    )
    mgr = AuthManager(config)
    ns = mgr.list_namespaces("key_a")
    assert ns == ["tenant_a"]


def test_list_namespaces_admin():
    config = AuthConfig(
        enabled=True,
        api_keys={"key_a": "tenant_a", "key_b": "tenant_b"},
        admin_key="admin_secret",
    )
    mgr = AuthManager(config)
    ns = mgr.list_namespaces("admin_secret")
    assert "tenant_a" in ns
    assert "tenant_b" in ns
