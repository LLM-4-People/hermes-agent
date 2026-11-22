"""Gateway context resolution on a shared endpoint follows the live provider identity.

Several providers may share one proxy URL. ``model.provider`` stays the bare ``custom`` adapter
while ``model.requested_provider`` (and the live runtime) names the chosen entry. A URL-first
lookup hands the session the first sibling's context window, capabilities and api mode.
"""

from __future__ import annotations

import gateway.run as gateway_run

_ROUTE = "https://router.example.com/v1"

_CONFIG = {
    "model": {
        "default": "shared-model",
        "provider": "custom",
        "requested_provider": "custom:beta",
        "base_url": _ROUTE,
    },
    "providers": {
        "alpha": {
            "name": "Provider A",
            "api": _ROUTE,
            "api_key": "coral-key",
            "context_length": 111,
        },
        "beta": {
            "name": "Provider B",
            "api": _ROUTE,
            "api_key": "beta-key",
            "context_length": 222,
        },
    },
}


def test_gateway_context_uses_the_requested_entry(monkeypatch):
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: _CONFIG)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {
        "provider": "custom",
        "requested_provider": "custom:beta",
        "base_url": _ROUTE,
        "api_key": "beta-key",
    })

    seen: dict = {}

    def fake_context_length(model, **kwargs):
        seen.update(kwargs)
        return 222

    monkeypatch.setattr("agent.model_metadata.get_model_context_length", fake_context_length)

    resolved = gateway_run._resolve_gateway_model_context(model="shared-model")

    assert seen["provider"] == "custom:beta"
    assert resolved.provider == "custom:beta"
    assert resolved.context_length == 222
