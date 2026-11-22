"""Per-provider ``extra_headers`` applied to the OpenAI client (#3526 salvage).

Custom providers (``providers`` / ``custom_providers`` in config.yaml) can
declare an ``extra_headers`` dict that must land on the OpenAI client's
``default_headers`` at construction and survive header re-application on
credential swaps / rebuilds. Values may carry credentials — the plumbing must
never log them.
"""
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent

_PROXY_URL = "https://llm.internal.example.com/v1"
_PROXY_CONFIG = {
    "custom_providers": [
        {
            "name": "my-proxy",
            "base_url": _PROXY_URL,
            "api_key": "proxy-key",
            "extra_headers": {
                "CF-Access-Client-Id": "xxxx.access",
                "X-Client-Name": "hermes-agent",
            },
        }
    ]
}


@patch("agent.process_bootstrap.OpenAI")
def test_custom_provider_extra_headers_applied_at_construction(mock_openai):
    mock_openai.return_value = MagicMock()
    with patch("hermes_cli.config.load_config", return_value=_PROXY_CONFIG):
        agent = AIAgent(
            api_key="proxy-key",
            base_url=_PROXY_URL,
            model="my-model",
            provider="custom",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    headers = agent._client_kwargs["default_headers"]
    assert headers["CF-Access-Client-Id"] == "xxxx.access"
    assert headers["X-Client-Name"] == "hermes-agent"


_DUAL_PROVIDER_CONFIG = {
    "providers": {
        "alpha": {
            "api": _PROXY_URL,
            "name": "Provider A",
            "api_key": "coral-key",
            "extra_headers": {"X-Proxy-Base-URL": "https://upstream-a.example.com/v1"},
        },
        "beta": {
            "api": _PROXY_URL,
            "name": "Provider B",
            "api_key": "beta-key",
            "proxy": {"base_url": "https://upstream-b.example.com/v1"},
        },
    },
}


@patch("agent.process_bootstrap.OpenAI")
@pytest.mark.parametrize(
    ("identity", "expected"),
    [
        ("custom:beta", "https://upstream-b.example.com/v1"),
        ("alpha", "https://upstream-a.example.com/v1"),
    ],
)
def test_two_providers_on_one_endpoint_keep_their_own_routing_header(mock_openai, identity, expected):
    """The client is rebuilt from config on every credential swap; identity, not URL, must win."""
    mock_openai.return_value = MagicMock()
    with patch("hermes_cli.config.load_config", return_value=_DUAL_PROVIDER_CONFIG):
        agent = AIAgent(
            api_key="proxy-key",
            base_url=_PROXY_URL,
            model="shared-model",
            provider="custom",
            requested_provider=identity,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert agent._client_kwargs["default_headers"]["X-Proxy-Base-URL"] == expected






