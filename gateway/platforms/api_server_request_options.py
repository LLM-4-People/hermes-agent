"""Per-request ``model_options`` decoding, lifted from ``platforms/api_server.py``.

Keeps the api_server facade under its frozen line cap (root AGENTS.md god-file
guidance: move code into a topical sibling). ``api_server.py`` re-exports
``_request_service_tier`` and ``_request_reasoning_config`` so existing importers
are unaffected; the helpers those two read stay in the facade and are imported
lazily (call time) to avoid an import cycle.
"""

import asyncio
import logging
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Optional


def _request_reasoning_config(model_options: Any) -> Optional[dict[str, Any]]:
    """Translate model_options (structured ``reasoning`` or legacy ``reasoning_effort``) into
    AIAgent reasoning_config; unknown effort values are ignored, never raised."""
    if not isinstance(model_options, dict):
        return None
    reasoning = model_options.get("reasoning")
    enabled: Any = None
    effort: Any = None
    if isinstance(reasoning, dict):
        enabled = reasoning.get("enabled")
        effort = reasoning.get("effort")
    effort = model_options.get("reasoning_effort", effort)
    effort_norm = str(effort).strip().lower() if effort is not None else ""
    if enabled is False or effort_norm == "none":
        return {"enabled": False}
    from gateway.platforms.api_server import _REASONING_EFFORTS

    if effort_norm in _REASONING_EFFORTS and effort_norm != "none":
        return {"enabled": True, "effort": effort_norm}
    if enabled is True:
        return {"enabled": True}
    return None


def _request_service_tier(model_options: Any) -> Any:
    """Return a per-request service_tier override or _REQUEST_OPTION_MISSING."""
    from gateway.platforms.api_server import _REQUEST_OPTION_MISSING, _clean_request_string, _coerce_request_bool

    if not isinstance(model_options, dict):
        return _REQUEST_OPTION_MISSING
    if "service_tier" in model_options:
        raw_tier = model_options.get("service_tier")
        return _clean_request_string(raw_tier) if isinstance(raw_tier, str) else raw_tier
    if "fast" in model_options:
        return "priority" if _coerce_request_bool(model_options.get("fast"), default=False) else None
    return _REQUEST_OPTION_MISSING


RUN_TOOL_FIELDS = ("allowed_toolsets", "excluded_tools")
_PROVIDER_TOOLSETS = ("memory", "context_engine")


class InvalidRunToolSelection(ValueError):
    """A client selection that cannot be satisfied under the served profile."""


class RunToolPolicyUnavailable(RuntimeError):
    """The profile's authoritative run policy could not be read."""


@contextmanager
def _read_policy():
    from hermes_cli.config_backend import ConfigBackendUnavailable
    try:
        yield
    except InvalidRunToolSelection:
        raise
    except (Exception, ConfigBackendUnavailable) as exc:  # health: allow BLE001 -- suppress private policy and plugin text at the API boundary
        # Config backend outages can raise SystemExit. YAML or plugin errors can include secret
        # source text. Both HTTP handlers and detached workers get the same sanitized failure.
        logging.getLogger(__name__).warning("Cannot read run tool policy (%s)", type(exc).__name__)
        raise RunToolPolicyUnavailable("Run tool policy is unavailable.") from None


def _tool_names(value, field):
    if not isinstance(value, list) or any(not isinstance(name, str) or not name for name in value):
        raise InvalidRunToolSelection(f"{field} must be an array of names")
    if len(set(value)) != len(value):
        raise InvalidRunToolSelection(f"{field} must contain distinct names")
    return sorted(value)


class RunToolInventory:
    """The profile's registered and provider-supplied tool names for discovery and selection."""

    def __init__(self, config):
        from hermes_cli.tools_config import _get_platform_tools
        self.config = config
        self.enabled = _get_platform_tools(config, "api_server")
        self._names = {}

    def names(self, name):
        from toolsets import resolve_toolset
        if name not in self._names:
            tools = set(resolve_toolset(name))
            if name in _PROVIDER_TOOLSETS:
                from plugins.plugin_loader import inspect_provider_tool_names
                tools.update(inspect_provider_tool_names(self.config, name))
            self._names[name] = sorted(tools)
        return self._names[name]

    def rows(self):
        from hermes_cli.tools_config import (
            _get_effective_configurable_toolsets, _toolset_has_keys, get_nous_subscription_features)
        features = get_nous_subscription_features(self.config)
        labels = {name: (label, desc) for name, label, desc in _get_effective_configurable_toolsets()}
        for name in sorted(self.enabled - labels.keys()):
            labels[name] = (name, "")
        return [{"name": name, "label": label, "description": desc, "enabled": name in self.enabled,
                 "configured": _toolset_has_keys(name, self.config, features=features), "tools": self.names(name)}
                for name, (label, desc) in labels.items()]

    def validate(self, result):
        if not set(result.get("allowed_toolsets", ())) <= self.enabled:
            raise InvalidRunToolSelection("allowed_toolsets contains a toolset outside the profile's enabled toolsets")
        if excluded := result.get("excluded_tools"):
            if not self._known_exclusions(excluded):
                raise InvalidRunToolSelection("excluded_tools contains a tool outside the profile's enabled toolsets")
        return result

    def _known_exclusions(self, excluded):
        from toolsets import resolve_toolset
        # Most selections contain ordinary registered tools. Only inspect an external provider
        # (which may need its own short-lived host) to resolve a remaining provider-only name.
        unknown = set(excluded).difference(tool for name in self.enabled for tool in resolve_toolset(name))
        for name in _PROVIDER_TOOLSETS:
            if unknown and name in self.enabled:
                unknown.difference_update(self.names(name))
        return not unknown


def parse_run_tool_options(body):
    options = {field: _tool_names(body[field], field) for field in RUN_TOOL_FIELDS if field in body}
    if options.get("excluded_tools") == []:
        del options["excluded_tools"]
    return options


def run_tool_inventory():
    from gateway.run import _gateway_config_home
    from hermes_cli.config_effective import load_user_config_effective
    return RunToolInventory(load_user_config_effective(_gateway_config_home() / "config.yaml", fail_closed=True))


def run_tool_rows():
    with _read_policy():
        return run_tool_inventory().rows()


def run_tool_options(body):
    options = parse_run_tool_options(body)
    with _read_policy():
        return run_tool_inventory().validate(options) if options else {}


def run_tool_scope(options):
    if not options:
        return nullcontext()
    with _read_policy():
        return _build_run_tool_scope(options)


def _build_run_tool_scope(options):
    from agent.skill_utils import parse_config_string_list
    from model_tools import get_tool_definitions
    from tools.hosted_tool_policy import restrict_tools
    from tools.tool_search_catalog import BRIDGE_TOOL_NAMES
    from toolsets import resolve_toolset

    inventory = run_tool_inventory()
    parsed = inventory.validate(parse_run_tool_options(options))
    chosen = parsed.get("allowed_toolsets", sorted(inventory.enabled))
    excluded = parsed.get("excluded_tools", ())
    disabled = parse_config_string_list((inventory.config.get("agent") or {}).get("disabled_toolsets"))
    allowed = {tool for name in chosen for tool in resolve_toolset(name)}.difference(excluded)
    # Enforce availability, schema dependencies and the operator's disabled toolsets before nested dispatch.
    with restrict_tools(chosen, allowed, excluded):
        schemas = get_tool_definitions(chosen, disabled, quiet_mode=True, skip_tool_search_assembly=True) if chosen else ()
        allowed.intersection_update(schema["function"]["name"] for schema in schemas)
    if allowed:
        # Synthetic dispatchers re-enter the same filtered registry for their underlying target.
        allowed.update(BRIDGE_TOOL_NAMES)
    return restrict_tools(chosen, allowed, excluded)


async def prepare_run_tool_scope(options):
    if not options:
        return nullcontext()
    return await asyncio.to_thread(run_tool_scope, options)
