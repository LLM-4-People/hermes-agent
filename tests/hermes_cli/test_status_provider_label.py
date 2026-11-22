"""`hermes status` provider label honors config.yaml model.base_url (#3296)."""

from unittest.mock import patch

from hermes_cli.status import _effective_provider_label


def _label_with(config, env_base_url=""):
    with patch("hermes_cli.status.resolve_requested_provider", return_value="auto"), \
         patch("hermes_cli.status.resolve_provider", return_value="openrouter"), \
         patch("hermes_cli.status.load_config", return_value=config), \
         patch("hermes_cli.status.get_env_value",
               side_effect=lambda k: env_base_url if k == "OPENAI_BASE_URL" else ""):
        return _effective_provider_label()


def test_config_base_url_labels_custom():
    label = _label_with({"model": {"base_url": "http://localhost:8080/v1"}})
    assert "OpenRouter" not in label


def test_blank_base_url_stays_openrouter():
    label = _label_with({"model": {"base_url": "   "}})
    assert "OpenRouter" in label


def test_non_openrouter_provider_untouched():
    with patch("hermes_cli.status.resolve_requested_provider", return_value="anthropic"), \
         patch("hermes_cli.status.resolve_provider", return_value="anthropic"):
        label = _effective_provider_label()
    assert "OpenRouter" not in label


def test_named_custom_provider_uses_the_configured_name():
    from hermes_cli.providers import provider_display_name

    providers = {"beta": {"name": "Provider B", "api": "https://upstream-b.example.com/v1"}}
    assert provider_display_name("custom:beta", providers, None) == "Provider B"
    assert provider_display_name("beta", providers, None) == "Provider B"


def test_unknown_custom_provider_falls_back_to_its_slug():
    from hermes_cli.providers import provider_display_name

    assert provider_display_name("custom:missing") == "custom:missing"


def test_configured_key_and_display_name_both_resolve():
    from hermes_cli.providers import provider_display_name

    providers = {"acme": {"name": "Beta", "api": "https://upstream-b.example.com/v1"}}
    assert provider_display_name("beta", providers, None) == "Beta"
    assert provider_display_name("custom:beta", providers, None) == "Beta"
    assert provider_display_name("acme", providers, None) == "Beta"


def test_bare_custom_does_not_borrow_a_legacy_entry_name():
    from hermes_cli.providers import provider_display_name

    legacy = [{"name": "Legacy", "base_url": "https://legacy.example/v1"}]
    assert provider_display_name("custom", None, legacy) == "Custom endpoint"


def test_live_provider_identity_ignores_a_stale_requested_name_for_a_concrete_adapter():
    from hermes_cli.providers import live_provider_identity

    assert live_provider_identity("custom", "custom:beta") == "custom:beta"
    assert live_provider_identity("custom", "") == "custom"
    assert live_provider_identity("openrouter", "custom:beta") == "openrouter"
