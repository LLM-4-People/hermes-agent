"""Gateway hygiene resolves context metadata for the live provider, not the bare adapter.

On a shared routing endpoint the adapter class is ``custom`` for every named entry; sizing the
pending turn from the first sibling's context window mis-times compression.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.run_turn_hygiene import GatewayTurnHygieneMixin

_ROUTE = "https://router.example.com/v1"


class _Runner(GatewayTurnHygieneMixin):
    _HygieneSettings = SimpleNamespace

    def _resolve_session_agent_runtime(self, *, source, session_key, user_config=None):
        return "shared-model", {
            "provider": "custom",
            "requested_provider": "beta",
            "base_url": _ROUTE,
            "api_key": "test-key",
        }


@pytest.mark.asyncio
async def test_hygiene_uses_the_requested_entry_context_length(monkeypatch):
    config = {
        "model": {"provider": "custom", "base_url": _ROUTE, "model": "shared-model"},
        "providers": {
            "alpha": {"base_url": _ROUTE,
                      "models": {"shared-model": {"context_length": 111}}},
            "beta": {"base_url": _ROUTE,
                     "models": {"shared-model": {"context_length": 222}}},
        },
    }
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: config)

    hs = await _Runner()._hmwa_hygiene_settings(source=object(), session_key="agent:main:test")

    assert hs.provider == "beta"
    assert hs.config_context_length == 222
