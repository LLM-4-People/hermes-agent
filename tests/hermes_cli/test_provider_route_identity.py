"""Provider-scoped route metadata for several providers sharing one endpoint.

A router in front of many upstreams presents ONE base URL to Hermes. Per-provider metadata
(headers, TLS, session affinity, context length, capabilities, api mode) therefore belongs to
the provider identity, not to the URL: a URL-first scan handed every provider the first entry's
routing header. These are the behavior contracts for the identity-aware owner.
"""

import pytest

from hermes_cli.config import (
    _normalize_custom_provider_entry,
    apply_custom_provider_extra_headers_to_client_kwargs,
    get_custom_provider_api_mode,
    get_custom_provider_context_length,
    get_custom_provider_extra_headers,
    get_custom_provider_model_capability,
    get_custom_provider_session_affinity_header,
    get_custom_provider_tls_settings,
    resolve_provider_extra_headers,
)
from hermes_cli.runtime_provider_custom import _lift_common_custom_fields

_ROUTE = "https://router.example.com/v1"


def _entries():
    return [
        {
            "name": "Provider A",
            "provider_key": "alpha",
            "base_url": _ROUTE,
            "api_mode": "codex_responses",
            "extra_headers": {"X-Proxy-Base-URL": "https://upstream-a.example.com/v1"},
            "ssl_ca_cert": "/etc/ssl/coral.pem",
            "session_affinity_header": "X-Alpha-Session",
            "context_length": 111,
            "models": {"m": {"answer_in_reasoning": True}},
            "capabilities": {"answer_in_reasoning": True},
        },
        {
            "name": "Provider B",
            "provider_key": "beta",
            "base_url": _ROUTE,
            "api_mode": "chat_completions",
            "proxy": {"base_url": "https://upstream-b.example.com/v1"},
            "ssl_verify": False,
            "session_affinity_header": "X-Beta-Session",
            "context_length": 222,
            "models": {"m": {"answer_in_reasoning": False}},
            "capabilities": {"answer_in_reasoning": False},
        },
    ]


@pytest.mark.parametrize("identity", ["beta", "Provider B", "custom:beta"])
def test_extra_headers_are_selected_by_provider_identity(identity):
    assert get_custom_provider_extra_headers(_ROUTE, _entries(), provider=identity) == {
        "X-Proxy-Base-URL": "https://upstream-b.example.com/v1",
    }


def test_a_provider_without_headers_does_not_borrow_a_sibling():
    entries = [
        {"name": "A", "provider_key": "a", "base_url": _ROUTE, "extra_headers": {"H": "a"}},
        {"name": "B", "provider_key": "b", "base_url": _ROUTE},
    ]
    assert get_custom_provider_extra_headers(_ROUTE, entries, provider="b") == {}


def test_no_identity_keeps_the_legacy_url_scan():
    entries = _entries()
    assert get_custom_provider_extra_headers(_ROUTE, entries) == {
        "X-Proxy-Base-URL": "https://upstream-a.example.com/v1",
    }
    assert get_custom_provider_extra_headers(_ROUTE, entries, provider="custom") == {
        "X-Proxy-Base-URL": "https://upstream-a.example.com/v1",
    }


def test_a_named_identity_with_no_matching_entry_returns_no_metadata():
    """A named entry that is disabled, renamed or on another route must not inherit a sibling's
    routing header."""
    assert get_custom_provider_extra_headers(_ROUTE, _entries(), provider="custom:unconfigured") == {}
    entries = _entries() + [{"name": "elsewhere", "base_url": "https://other.example/v1"}]
    assert get_custom_provider_extra_headers(_ROUTE, entries, provider="elsewhere") == {}


def test_a_builtin_identity_keeps_a_custom_endpoint_entry():
    """A built-in provider pointed at a user-declared custom endpoint still gets its entry."""
    entries = [{"name": "my-proxy", "base_url": _ROUTE, "extra_headers": {"H": "proxy"}}]
    assert get_custom_provider_extra_headers(_ROUTE, entries, provider="openai") == {"H": "proxy"}


def test_tls_affinity_context_and_capabilities_are_provider_scoped():
    entries = _entries()
    assert get_custom_provider_tls_settings(_ROUTE, entries, provider="beta") == {"ssl_verify": False}
    assert get_custom_provider_tls_settings(_ROUTE, entries, provider="alpha") == {
        "ssl_ca_cert": "/etc/ssl/coral.pem",
    }
    assert get_custom_provider_session_affinity_header(_ROUTE, entries, provider="beta") == "X-Beta-Session"
    assert get_custom_provider_session_affinity_header(_ROUTE, entries, provider="alpha") == "X-Alpha-Session"
    assert get_custom_provider_context_length("m", _ROUTE, entries, provider="beta") == 222
    assert get_custom_provider_context_length("m", _ROUTE, entries, provider="alpha") == 111
    assert get_custom_provider_model_capability(
        "m", _ROUTE, "answer_in_reasoning", entries, provider="beta") is False
    assert get_custom_provider_model_capability(
        "m", _ROUTE, "answer_in_reasoning", entries, provider="alpha") is True
    assert get_custom_provider_api_mode(_ROUTE, entries, provider="beta") == "chat_completions"
    assert get_custom_provider_api_mode(_ROUTE, entries, provider="alpha") == "codex_responses"


def test_apply_headers_is_provider_scoped():
    kwargs = {"default_headers": {"X-Keep": "1"}}
    apply_custom_provider_extra_headers_to_client_kwargs(
        kwargs, _ROUTE, _entries(), provider="beta")
    assert kwargs["default_headers"] == {"X-Keep": "1", "X-Proxy-Base-URL": "https://upstream-b.example.com/v1"}


def test_proxy_block_expands_to_the_default_routing_header():
    entry = {"name": "beta", "base_url": _ROUTE, "proxy": {"base_url": "https://upstream-b.example.com/v1"}}
    assert _normalize_custom_provider_entry(entry, provider_key="beta")["extra_headers"] == {
        "X-Proxy-Base-URL": "https://upstream-b.example.com/v1",
    }


def test_proxy_header_name_and_headers_are_configurable():
    entry = {
        "name": "beta",
        "base_url": _ROUTE,
        "proxy": {
            "base_url": "https://upstream-b.example.com/v1",
            "header": "X-Route-Upstream",
            "headers": {"X-Route-Tenant": "acme"},
        },
    }
    assert resolve_provider_extra_headers(entry) == {
        "X-Route-Upstream": "https://upstream-b.example.com/v1",
        "X-Route-Tenant": "acme",
    }


def test_proxy_headers_may_override_the_derived_header():
    entry = {
        "name": "beta",
        "base_url": _ROUTE,
        "proxy": {
            "base_url": "https://upstream-b.example.com/v1",
            "headers": {"x-proxy-base-url": "https://override.example/v1"},
        },
    }
    assert resolve_provider_extra_headers(entry) == {
        "x-proxy-base-url": "https://override.example/v1",
    }


def test_explicit_extra_headers_override_case_insensitively():
    entry = {
        "name": "beta",
        "base_url": _ROUTE,
        "proxy": {
            "base_url": "https://upstream-b.example.com/v1",
            "headers": {"X-Proxy-Base-URL": "https://override.example/v1"},
        },
        "extra_headers": {"x-proxy-base-url": "https://extra.example/v1"},
    }
    assert resolve_provider_extra_headers(entry) == {
        "x-proxy-base-url": "https://extra.example/v1",
    }


def test_explicit_extra_headers_win_over_the_proxy_derived_header():
    entry = {
        "name": "beta",
        "base_url": _ROUTE,
        "proxy": {"base_url": "https://upstream-b.example.com/v1"},
        "extra_headers": {"X-Proxy-Base-URL": "https://override.example/v1"},
    }
    assert resolve_provider_extra_headers(entry) == {
        "X-Proxy-Base-URL": "https://override.example/v1",
    }


@pytest.mark.parametrize(
    "proxy",
    [
        "not-a-mapping",
        {},
        {"base_url": "not-a-url"},
        {"base_url": "https://upstream-b.example.com/v1", "header": "not a header"},
        {"base_url": "https://upstream-b.example.com/v1", "header": ""},
    ],
)
def test_invalid_proxy_forms_add_no_headers(proxy):
    assert resolve_provider_extra_headers({"base_url": _ROUTE, "proxy": proxy}) == {}


def test_proxy_routing_header_reaches_the_runtime_lift():
    entry = {"name": "Provider B", "base_url": _ROUTE, "proxy": {"base_url": "https://upstream-b.example.com/v1"}}
    result = {}
    _lift_common_custom_fields(
        entry, result, provider_key="beta", key_env="BETA_API_KEY", api_mode="codex_responses")
    assert result["extra_headers"] == {"X-Proxy-Base-URL": "https://upstream-b.example.com/v1"}
    assert result["provider_key"] == "beta"


def test_model_metadata_reads_the_provider_entry():
    """Context length and wire mode probes must resolve the live provider on a shared endpoint."""
    from agent.model_metadata import _config_override_context_length, _is_codex_route

    assert _config_override_context_length("m", _ROUTE, "beta", _entries()) == 222
    assert _config_override_context_length("m", _ROUTE, "alpha", _entries()) == 111
    assert _is_codex_route("beta", _ROUTE, _entries()) is False
    assert _is_codex_route("alpha", _ROUTE, _entries()) is True
