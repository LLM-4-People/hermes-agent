#!/usr/bin/env python3
"""Mechanically generate and verify fork-owned documentation and README sections.

Single source of truth for fork documentation in ``LLM-4-People/hermes-agent``:
  1. Reads canonical fork metadata from ``docs/sources/fork_facts.json``.
  2. Introspects live repository facts directly from source code using pure-stdlib AST
     and file inspection (zero third-party dependencies, ``python3 -I`` safe).
  3. Renders:
       - ``README.md`` (only between ``<!-- BEGIN:HERMES_FORK_DOCS -->`` and
         ``<!-- END:HERMES_FORK_DOCS -->``, keeping upstream content below intact)
       - ``docs/MOBILE_COMPANION.md`` (entire file)
       - ``DOWNSTREAM.md`` (entire file)

Usage:
  python3 -I scripts/docs_fork.py --write
  python3 -I scripts/docs_fork.py --check
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import difflib
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

README_BEGIN_MARKER = "<!-- BEGIN:HERMES_FORK_DOCS -->"
README_END_MARKER = "<!-- END:HERMES_FORK_DOCS -->"
UPSTREAM_README_HEADING = "## Upstream Hermes Agent"

_TOKEN_RE = re.compile(r"\{\{([a-z0-9_]+)\}\}")
_SHARD_LIST_RE = re.compile(r"^\s*shard:\s*\[([^\]]+)\]", re.MULTILINE)

_API_FEATURE_MODULES: tuple[str, ...] = (
    "gateway/platforms/api_server_memory.py",
    "gateway/platforms/api_server_skills.py",
    "gateway/platforms/api_server_workspace.py",
    "gateway/platforms/api_server_profile.py",
    "gateway/platforms/api_server_notes.py",
    "gateway/platforms/api_server_sharing.py",
    "gateway/platforms/api_server_habits.py",
)

# Toolsets in CONFIGURABLE_TOOLSETS that are excluded from api_server's default enabled set.
_API_SERVER_EXCLUDED_TOOLSETS: frozenset[str] = frozenset({
    "tts",
    "clarify",
    "computer_use",
    "context_engine",
})

# Display grouping for /v1/capabilities flags in docs/MOBILE_COMPANION.md.
_CAPABILITY_GROUPS: tuple[tuple[str, frozenset[str]], ...] = (
    (
        "Conversations and streaming",
        frozenset({
            "chat_completions",
            "chat_completions_streaming",
            "responses_api",
            "responses_streaming",
            "session_resources",
            "session_chat",
            "session_chat_streaming",
            "session_fork",
            "session_fork_at_turn",
            "session_search",
            "session_export",
            "session_uploads",
            "session_model_lock",
            "session_continuity_header",
            "session_key_header",
        }),
    ),
    (
        "Live runs, steering, and safety",
        frozenset({
            "run_submission",
            "runs_idempotency",
            "run_status",
            "run_events_sse",
            "run_stop",
            "run_steer",
            "run_approval_response",
            "run_clarify_response",
            "run_unknown_resolution",
            "run_toolset_selection",
            "run_tool_exclusions",
            "run_scoped_instructions",
            "run_progress_hooks",
            "tool_progress_events",
            "approval_events",
            "structured_approval_diffs",
            "reasoning_streaming",
        }),
    ),
    (
        "Coding, models, and workspace",
        frozenset({
            "live_coding_stream",
            "model_options",
            "model_capability_metadata",
            "run_metrics_v2",
            "workspace_browser",
            "usage_analytics",
            "cron_run_history",
        }),
    ),
    (
        "Brain, notes, sharing, and skills",
        frozenset({
            "memory_api",
            "memory_write_api",
            "skills_api",
            "skill_studio",
            "profile_defaults",
            "smart_notes",
            "note_sharing",
            "universal_sharing",
            "habit_suggestions",
        }),
    ),
)

_TOOLSET_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Mobile companion", ("mobile_ui", "mobile_deliver", "mobile_device")),
    ("Web and research", ("web", "browser")),
    ("Coding and files", ("file", "terminal", "code_execution")),
    ("Media and creativity", ("vision", "image_gen")),
    ("Memory, knowledge, and planning", ("memory", "session_search", "skills", "todo", "cronjob")),
    ("Delegation and integrations", ("delegation", "connections")),
)

_TOOLSET_TOOL_HINTS: dict[str, str] = {
    "mobile_ui": "`mobile_ui` (`mobile_ui_patch`)",
    "mobile_deliver": "`mobile_deliver` (`mobile_deliver`)",
    "mobile_device": "`mobile_device` (`request_device_action`)",
    "web": "`web` (`web_search`, `web_extract`)",
    "browser": "`browser` (`browser_*`)",
    "file": "`file` (`read_file`, `write_file`, `patch`, `search_files`)",
    "terminal": "`terminal` (`terminal`, `process_manage`)",
    "code_execution": "`code_execution` (`execute_code`)",
    "vision": "`vision` (`vision_analyze`)",
    "image_gen": "`image_gen` (`image_generate`)",
    "memory": "`memory` (`memory`, `mobile_notes`, `mobile_share`)",
    "session_search": "`session_search`",
    "skills": "`skills` (`skills_list`, `skill_view`, `skill_manage`)",
    "todo": "`todo`",
    "cronjob": "`cronjob`",
    "delegation": "`delegation` (`delegate_task`)",
    "connections": "`connections` (`manage_connections`)",
}

_DEVICE_CAPABILITY_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Capture and location", ("camera", "location_coarse", "nearby_osm")),
    ("Maps and navigation", ("maps_handoff", "map_navigation")),
    ("Time and calendar", ("alarm_timer", "calendar_draft", "calendar_read")),
    (
        "Communication drafts (never auto-sent)",
        ("dial_draft", "sms_draft", "email_draft", "contacts_pick", "contact_draft", "share_sheet"),
    ),
    ("Clipboard and media", ("clipboard_read", "clipboard_write", "media_control")),
    ("Ambient notifications (opt-in)", ("notifications_digest", "notification_reply")),
    ("Live UI surfaces", ("ui_surfaces_v2",)),
)


class DocsForkError(RuntimeError):
    """Raised when fork metadata, live source introspection, or markers are invalid."""


@dataclass(frozen=True)
class LiveRepoState:
    live_facts: dict[str, Any]
    active_capabilities: tuple[str, ...]
    inactive_capabilities: tuple[str, ...]
    mobile_feature_flags: tuple[str, ...]
    device_capabilities: tuple[str, ...]
    deep_android_actions: tuple[str, ...]
    default_toolsets: tuple[str, ...]
    optional_toolsets: tuple[str, ...]
    fork_workflows: tuple[str, ...]
    mobile_tool_modules: tuple[str, ...]


def _eval_ast_const(node: ast.AST) -> Any:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("frozenset", "set", "tuple", "list")
        and len(node.args) == 1
    ):
        inner = _eval_ast_const(node.args[0])
        return {"frozenset": frozenset, "set": set, "tuple": tuple, "list": list}[node.func.id](inner)
    return ast.literal_eval(node)


def _parse_module_constants(root: Path, rel_path: str, names: tuple[str, ...]) -> dict[str, Any]:
    path = root / rel_path
    if not path.is_file():
        raise DocsForkError(f"Required source file missing: {rel_path}")
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except SyntaxError as exc:
        raise DocsForkError(f"Syntax error while parsing {rel_path}: {exc}") from exc

    found: dict[str, Any] = {}
    for node in tree.body:
        target_name: str | None = None
        value_node: ast.AST | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name, value_node = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            target_name, value_node = node.target.id, node.value
        if target_name in names and value_node is not None:
            try:
                found[target_name] = _eval_ast_const(value_node)
            except (ValueError, TypeError) as exc:
                raise DocsForkError(f"Cannot evaluate constant {target_name} in {rel_path}: {exc}") from exc

    missing = [name for name in names if name not in found]
    if missing:
        raise DocsForkError(f"Missing expected constants {missing} in {rel_path}")
    return found


def _read_fork_workflows(root: Path) -> tuple[str, ...]:
    for candidate in ("scripts/fork-owned-workflows.txt", ".github/fork-owned-workflows.txt"):
        path = root / candidate
        if path.is_file():
            items = [
                line.split("#", 1)[0].strip()
                for line in path.read_text(encoding="utf-8-sig").splitlines()
                if line.split("#", 1)[0].strip()
            ]
            if not items:
                raise DocsForkError(f"Workflow manifest is empty: {candidate}")
            return tuple(sorted(items))
    raise DocsForkError("Missing scripts/fork-owned-workflows.txt")


def _read_downstream_ci_shards(root: Path) -> int:
    ci_path = root / ".github" / "workflows" / "downstream-ci.yml"
    if not ci_path.is_file():
        raise DocsForkError("Missing .github/workflows/downstream-ci.yml")
    match = _SHARD_LIST_RE.search(ci_path.read_text(encoding="utf-8-sig"))
    if not match:
        raise DocsForkError("Could not find verify matrix shard list in .github/workflows/downstream-ci.yml")
    entries = [part.strip() for part in match.group(1).split(",") if part.strip()]
    if not entries:
        raise DocsForkError("Empty verify matrix shard list in .github/workflows/downstream-ci.yml")
    return len(entries)


def collect_live_repo_state(root: Path = ROOT) -> LiveRepoState:
    """Introspect live fork capabilities, endpoints, toolsets, concurrency caps, and CI invariants."""
    api_base = _parse_module_constants(
        root,
        "gateway/platforms/api_server.py",
        ("_STATIC_FEATURE_FLAGS", "_CAPABILITY_ENDPOINTS", "DEFAULT_PORT"),
    )
    api_ui = _parse_module_constants(
        root,
        "gateway/platforms/api_server_ui.py",
        ("MOBILE_FEATURE_FLAGS", "MOBILE_CAPABILITY_ENDPOINTS"),
    )

    features: dict[str, Any] = {
        "chat_completions": True,
        "chat_completions_streaming": True,
        "responses_api": True,
        "responses_streaming": True,
        "run_submission": True,
        "runs_idempotency": True,
    }
    features.update(api_base["_STATIC_FEATURE_FLAGS"])
    features.update(api_ui["MOBILE_FEATURE_FLAGS"])

    endpoints: list[tuple[str, tuple[str, ...]]] = list(api_base["_CAPABILITY_ENDPOINTS"])
    endpoints.extend(api_ui["MOBILE_CAPABILITY_ENDPOINTS"])

    for mod_rel in _API_FEATURE_MODULES:
        mod_consts = _parse_module_constants(root, mod_rel, ("CAPABILITY_FEATURES", "CAPABILITY_ENDPOINTS"))
        features.update(mod_consts["CAPABILITY_FEATURES"])
        endpoints.extend(mod_consts["CAPABILITY_ENDPOINTS"])

    # Runtime-resolved flags mirrored from APIServerAdapter._handle_capabilities.
    features["cron_run_history"] = True
    features["run_unknown_resolution"] = True
    features["run_clarify_response"] = True
    features["cors"] = False

    active_bool_caps = [k for k, v in features.items() if v is True]
    active_header_caps = [k for k, v in features.items() if isinstance(v, str) and v.strip()]
    active_caps = tuple(k for k, v in features.items() if v is True or (isinstance(v, str) and v.strip()))
    inactive_caps = tuple(k for k, v in features.items() if v is False)
    active_endpoints = [ep for ep in endpoints if len(ep[1]) < 3 or bool(features.get(ep[1][2]))]

    bridge = _parse_module_constants(
        root,
        "tools/mobile_device_bridge.py",
        ("ALLOWED_DEVICE_CAPABILITIES", "DEEP_ANDROID_ACTIONS"),
    )
    allowed_device = frozenset(bridge["ALLOWED_DEVICE_CAPABILITIES"])
    deep_actions = frozenset(bridge["DEEP_ANDROID_ACTIONS"])
    if not deep_actions.issubset(allowed_device):
        raise DocsForkError("DEEP_ANDROID_ACTIONS must be a subset of ALLOWED_DEVICE_CAPABILITIES")

    guard = _parse_module_constants(
        root,
        "gateway/platforms/api_server_auth_guard.py",
        ("DEFAULT_MAX_CONCURRENT_RUNS", "DEFAULT_MAX_CONCURRENT_RUNS_PER_PROFILE"),
    )

    tools_cfg = _parse_module_constants(
        root,
        "hermes_cli/tools_config.py",
        ("CONFIGURABLE_TOOLSETS", "_DEFAULT_OFF_TOOLSETS", "_CONFIG_ONLY_TOOLSETS"),
    )
    all_cfg_toolsets = [row[0] for row in tools_cfg["CONFIGURABLE_TOOLSETS"]]
    default_off = set(tools_cfg["_DEFAULT_OFF_TOOLSETS"])
    config_only = set(tools_cfg["_CONFIG_ONLY_TOOLSETS"])
    default_toolsets = tuple(
        name
        for name in all_cfg_toolsets
        if name not in default_off and name not in config_only and name not in _API_SERVER_EXCLUDED_TOOLSETS
    )
    optional_toolsets = tuple(name for name in all_cfg_toolsets if name not in default_toolsets)

    i18n_consts = _parse_module_constants(root, "gateway/i18n.py", ("SUPPORTED_USER_LANGS", "CATALOG"))
    supported_langs = sorted(i18n_consts["SUPPORTED_USER_LANGS"])
    catalog = i18n_consts["CATALOG"]
    if set(catalog.keys()) != set(supported_langs):
        raise DocsForkError("gateway/i18n.py CATALOG languages do not match SUPPORTED_USER_LANGS")
    en_keys = set(catalog.get("en", {}).keys())
    fr_keys = set(catalog.get("fr", {}).keys())
    if not en_keys or en_keys != fr_keys:
        raise DocsForkError("gateway/i18n.py EN/FR CATALOG keys are out of parity")

    locales_dir = root / "locales"
    if not locales_dir.is_dir():
        raise DocsForkError("Missing locales/ directory")
    locale_count = sum(1 for p in locales_dir.glob("*.yaml") if not p.name.endswith(".tui.yaml"))

    tools_dir = root / "tools"
    mobile_tool_modules = tuple(sorted(p.name for p in tools_dir.glob("mobile_*.py")))
    if not mobile_tool_modules:
        raise DocsForkError("No tools/mobile_*.py modules found")

    fork_workflows = _read_fork_workflows(root)
    downstream_ci_shards = _read_downstream_ci_shards(root)

    live_facts: dict[str, Any] = {
        "active_capability_count": len(active_caps),
        "active_boolean_capability_count": len(active_bool_caps),
        "active_header_capability_count": len(active_header_caps),
        "inactive_capability_count": len(inactive_caps),
        "endpoint_count": len(active_endpoints),
        "device_capability_count": len(allowed_device),
        "core_device_capability_count": len(allowed_device - deep_actions),
        "deep_android_action_count": len(deep_actions),
        "default_toolset_count": len(default_toolsets),
        "configurable_toolset_count": len(all_cfg_toolsets),
        "optional_toolset_count": len(optional_toolsets),
        "default_api_port": int(api_base["DEFAULT_PORT"]),
        "max_concurrent_runs": int(guard["DEFAULT_MAX_CONCURRENT_RUNS"]),
        "max_concurrent_runs_per_profile": int(guard["DEFAULT_MAX_CONCURRENT_RUNS_PER_PROFILE"]),
        "fork_workflow_count": len(fork_workflows),
        "downstream_ci_shards": downstream_ci_shards,
        "mobile_tool_module_count": len(mobile_tool_modules),
        "locale_count": locale_count,
        "supported_user_langs": supported_langs,
    }

    return LiveRepoState(
        live_facts=live_facts,
        active_capabilities=active_caps,
        inactive_capabilities=inactive_caps,
        mobile_feature_flags=tuple(api_ui["MOBILE_FEATURE_FLAGS"].keys()),
        device_capabilities=tuple(sorted(allowed_device)),
        deep_android_actions=tuple(sorted(deep_actions)),
        default_toolsets=default_toolsets,
        optional_toolsets=optional_toolsets,
        fork_workflows=fork_workflows,
        mobile_tool_modules=mobile_tool_modules,
    )


def _format_backtick_list(items: tuple[str, ...] | list[str], *, conjunction: str | None = None) -> str:
    quoted = [f"`{item}`" for item in items]
    if not quoted:
        return ""
    if conjunction is None or len(quoted) == 1:
        return ", ".join(quoted)
    if len(quoted) == 2:
        return f"{quoted[0]} {conjunction} {quoted[1]}"
    return ", ".join(quoted[:-1]) + f", {conjunction} {quoted[-1]}"


def _render_capabilities_table(state: LiveRepoState) -> str:
    active_set = set(state.active_capabilities)
    assigned: set[str] = set()
    rows: list[tuple[str, list[str]]] = []

    for label, members in _CAPABILITY_GROUPS:
        present = [cap for cap in state.active_capabilities if cap in members and cap not in assigned]
        assigned.update(present)
        if present:
            rows.append((label, present))

    native_flags = [cap for cap in state.active_capabilities if cap in state.mobile_feature_flags and cap not in assigned]
    assigned.update(native_flags)
    if native_flags:
        rows.append(("Native Android companion and bilingual", native_flags))

    remaining = [cap for cap in state.active_capabilities if cap not in assigned]
    if remaining:
        rows.append(("Additional active capabilities", remaining))
        assigned.update(remaining)

    if assigned != active_set:
        raise DocsForkError("Internal error: capability table failed to cover all active capabilities")

    lines = ["| Area | Active `/v1/capabilities` flags |", "|---|---|"]
    for label, caps in rows:
        lines.append(f"| **{label}** | {_format_backtick_list(caps)} |")
    return "\n".join(lines)


def _render_default_toolsets_table(state: LiveRepoState) -> str:
    default_set = set(state.default_toolsets)
    assigned: set[str] = set()
    lines = ["| Category | Enabled by default |", "|---|---|"]

    for label, members in _TOOLSET_CATEGORIES:
        present = [name for name in members if name in default_set and name not in assigned]
        assigned.update(present)
        if present:
            rendered = ", ".join(_TOOLSET_TOOL_HINTS.get(name, f"`{name}`") for name in present)
            lines.append(f"| **{label} ({len(present)})** | {rendered} |")

    leftover = [name for name in state.default_toolsets if name not in assigned]
    if leftover:
        rendered = ", ".join(_TOOLSET_TOOL_HINTS.get(name, f"`{name}`") for name in leftover)
        lines.append(f"| **Other ({len(leftover)})** | {rendered} |")
    return "\n".join(lines)


def _render_device_capabilities_list(state: LiveRepoState) -> str:
    all_set = set(state.device_capabilities)
    deep_set = set(state.deep_android_actions)
    assigned: set[str] = set()
    lines: list[str] = []

    for label, members in _DEVICE_CAPABILITY_GROUPS:
        present = [cap for cap in members if cap in all_set and cap not in deep_set and cap not in assigned]
        assigned.update(present)
        if present:
            lines.append(f"- **{label}:** {_format_backtick_list(present)}")

    deep_present = [cap for cap in state.deep_android_actions if cap in all_set and cap not in assigned]
    assigned.update(deep_present)
    if deep_present:
        lines.append(
            f"- **Deep Android 16 app control and actions ({len(deep_present)}):** "
            f"{_format_backtick_list(deep_present)}"
        )

    leftover = [cap for cap in state.device_capabilities if cap not in assigned]
    if leftover:
        lines.append(f"- **Additional device actions ({len(leftover)}):** {_format_backtick_list(leftover)}")
    return "\n".join(lines)


def build_template_context(facts: dict[str, Any], state: LiveRepoState) -> dict[str, str]:
    """Combine canonical metadata from ``fork_facts.json`` with live-computed repository facts."""
    ctx: dict[str, str] = {}
    for key, val in facts.items():
        if key == "live_facts" or isinstance(val, (dict, list)):
            continue
        ctx[key] = str(val)
    for key, val in state.live_facts.items():
        if isinstance(val, list):
            continue
        ctx[key] = str(val)

    ctx["supported_user_langs_inline"] = "/".join(f"`{lang}`" for lang in state.live_facts["supported_user_langs"])
    ctx["inactive_capabilities_inline"] = _format_backtick_list(state.inactive_capabilities, conjunction="and")
    ctx["optional_toolsets_inline"] = _format_backtick_list(state.optional_toolsets)
    ctx["fork_workflows_inline"] = _format_backtick_list(state.fork_workflows)
    ctx["capabilities_table"] = _render_capabilities_table(state)
    ctx["default_toolsets_table"] = _render_default_toolsets_table(state)
    ctx["device_capabilities_list"] = _render_device_capabilities_list(state)
    return ctx


def render_template(template_text: str, context: dict[str, str], *, template_name: str) -> str:
    missing: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in context:
            missing.append(key)
            return match.group(0)
        return context[key]

    rendered = _TOKEN_RE.sub(_replace, template_text)
    if missing:
        raise DocsForkError(f"Unknown placeholder(s) {sorted(set(missing))} in {template_name}")
    if "{{" in rendered or "}}" in rendered:
        raise DocsForkError(f"Malformed or unreplaced '{{{{...}}}}' marker in {template_name}")
    return rendered


def render_readme_with_markers(existing_readme: str, fork_section_body: str) -> str:
    """Replace only the marked fork section in ``README.md``, preserving upstream content below."""
    header = (
        "<!-- Generated by scripts/docs_fork.py from docs/sources/README_FORK.md.in and "
        "docs/sources/fork_facts.json. Do not edit between markers directly. -->"
    )
    block = f"{README_BEGIN_MARKER}\n{header}\n\n{fork_section_body.strip()}\n{README_END_MARKER}"

    if README_BEGIN_MARKER in existing_readme or README_END_MARKER in existing_readme:
        begin_idx = existing_readme.find(README_BEGIN_MARKER)
        end_idx = existing_readme.find(README_END_MARKER)
        if begin_idx == -1 or end_idx == -1 or end_idx < begin_idx:
            raise DocsForkError("README.md has malformed BEGIN/END HERMES_FORK_DOCS markers")
        after_end = end_idx + len(README_END_MARKER)
        prefix = existing_readme[:begin_idx]
        suffix = existing_readme[after_end:]
        return f"{prefix}{block}{suffix}"

    # Initial bootstrap when markers are not yet present: preserve everything from the upstream heading onward.
    upstream_idx = existing_readme.find(UPSTREAM_README_HEADING)
    if upstream_idx == -1:
        raise DocsForkError(
            f"README.md lacks {README_BEGIN_MARKER} markers and upstream heading {UPSTREAM_README_HEADING!r}"
        )
    # Keep the horizontal rule separator right above upstream heading when present.
    before_upstream = existing_readme[:upstream_idx].rstrip()
    if before_upstream.endswith("---"):
        tail = f"\n\n---\n\n{existing_readme[upstream_idx:]}"
    else:
        tail = f"\n\n{existing_readme[upstream_idx:]}"
    return f"{block}{tail}"


def render_standalone_doc(body: str, *, source_rel: str) -> str:
    header = (
        f"<!-- Generated by scripts/docs_fork.py from {source_rel} and "
        "docs/sources/fork_facts.json. Do not edit directly. -->\n\n"
    )
    return header + body.strip() + "\n"


def _load_facts(root: Path) -> tuple[Path, dict[str, Any]]:
    facts_path = root / "docs" / "sources" / "fork_facts.json"
    if not facts_path.is_file():
        raise DocsForkError("Missing docs/sources/fork_facts.json")
    try:
        data = json.loads(facts_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise DocsForkError(f"Invalid JSON in docs/sources/fork_facts.json: {exc}") from exc
    if not isinstance(data, dict):
        raise DocsForkError("docs/sources/fork_facts.json must contain a JSON object")
    return facts_path, data


def _verify_required_assets(root: Path, facts: dict[str, Any]) -> None:
    for key in ("social_preview_asset", "companion_preview_asset"):
        rel = facts.get(key)
        if not isinstance(rel, str) or not rel.strip():
            raise DocsForkError(f"Missing or empty '{key}' in docs/sources/fork_facts.json")
        if not (root / rel).is_file():
            raise DocsForkError(f"Referenced asset not found: {rel}")


def _compute_expected_outputs(
    root: Path,
    facts: dict[str, Any],
    state: LiveRepoState,
) -> dict[Path, str]:
    context = build_template_context(facts, state)
    sources_dir = root / "docs" / "sources"

    readme_tpl_path = sources_dir / "README_FORK.md.in"
    companion_tpl_path = sources_dir / "MOBILE_COMPANION.md.in"
    downstream_tpl_path = sources_dir / "DOWNSTREAM.md.in"
    for tpl_path in (readme_tpl_path, companion_tpl_path, downstream_tpl_path):
        if not tpl_path.is_file():
            raise DocsForkError(f"Missing template: {tpl_path.relative_to(root)}")

    readme_path = root / "README.md"
    if not readme_path.is_file():
        raise DocsForkError("Missing README.md")

    readme_body = render_template(
        readme_tpl_path.read_text(encoding="utf-8-sig"),
        context,
        template_name="docs/sources/README_FORK.md.in",
    )
    companion_body = render_template(
        companion_tpl_path.read_text(encoding="utf-8-sig"),
        context,
        template_name="docs/sources/MOBILE_COMPANION.md.in",
    )
    downstream_body = render_template(
        downstream_tpl_path.read_text(encoding="utf-8-sig"),
        context,
        template_name="docs/sources/DOWNSTREAM.md.in",
    )

    existing_readme = readme_path.read_text(encoding="utf-8-sig")
    return {
        readme_path: render_readme_with_markers(existing_readme, readme_body),
        root / "docs" / "MOBILE_COMPANION.md": render_standalone_doc(
            companion_body,
            source_rel="docs/sources/MOBILE_COMPANION.md.in",
        ),
        root / "DOWNSTREAM.md": render_standalone_doc(
            downstream_body,
            source_rel="docs/sources/DOWNSTREAM.md.in",
        ),
    }


def _find_live_fact_mismatches(facts: dict[str, Any], state: LiveRepoState) -> list[str]:
    recorded = facts.get("live_facts")
    if not isinstance(recorded, dict):
        return ["docs/sources/fork_facts.json: missing 'live_facts' object"]
    mismatches: list[str] = []
    all_keys = sorted(set(recorded.keys()) | set(state.live_facts.keys()))
    for key in all_keys:
        if key not in recorded:
            mismatches.append(f"fork_facts.json missing live_facts.{key} (live={state.live_facts[key]!r})")
        elif key not in state.live_facts:
            mismatches.append(f"fork_facts.json has obsolete live_facts.{key}={recorded[key]!r}")
        elif recorded[key] != state.live_facts[key]:
            mismatches.append(
                f"fork_facts.json stale live_facts.{key}: recorded={recorded[key]!r} != live={state.live_facts[key]!r}"
            )
    return mismatches


def run_write(root: Path = ROOT) -> list[Path]:
    """Update ``fork_facts.json`` live_facts and regenerate all fork-owned markdown docs."""
    facts_path, facts = _load_facts(root)
    state = collect_live_repo_state(root)
    _verify_required_assets(root, facts)

    updated_facts = dict(facts)
    updated_facts["live_facts"] = state.live_facts
    rendered_facts = json.dumps(updated_facts, indent=2, ensure_ascii=False) + "\n"

    expected = _compute_expected_outputs(root, updated_facts, state)
    expected[facts_path] = rendered_facts

    written: list[Path] = []
    for path, content in expected.items():
        current = path.read_text(encoding="utf-8-sig") if path.is_file() else None
        if current != content:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
            written.append(path)
    return written


def run_check(root: Path = ROOT) -> list[str]:
    """Verify zero drift between live code, ``fork_facts.json``, templates, and generated docs."""
    _, facts = _load_facts(root)
    state = collect_live_repo_state(root)
    _verify_required_assets(root, facts)

    errors: list[str] = []
    errors.extend(_find_live_fact_mismatches(facts, state))

    readme_path = root / "README.md"
    if readme_path.is_file():
        raw_readme = readme_path.read_text(encoding="utf-8-sig")
        if README_BEGIN_MARKER not in raw_readme or README_END_MARKER not in raw_readme:
            errors.append(f"README.md: missing {README_BEGIN_MARKER} / {README_END_MARKER} markers")

    expected = _compute_expected_outputs(root, facts, state)
    for path, expected_text in expected.items():
        rel = path.relative_to(root)
        if not path.is_file():
            errors.append(f"{rel}: generated file missing (run `python3 -I scripts/docs_fork.py --write`)")
            continue
        actual_text = path.read_text(encoding="utf-8-sig")
        if actual_text != expected_text:
            diff_lines = list(
                difflib.unified_diff(
                    actual_text.splitlines(),
                    expected_text.splitlines(),
                    fromfile=f"a/{rel}",
                    tofile=f"b/{rel}",
                    n=1,
                    lineterm="",
                )
            )
            preview = "\n".join(diff_lines[:24])
            errors.append(f"{rel}: out of date with docs/sources/ (run `python3 -I scripts/docs_fork.py --write`)\n{preview}")
    return errors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docs_fork.py",
        description="Mechanically generate and verify LLM-4-People/hermes-agent fork docs and README section.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="Refresh live_facts and regenerate all fork docs.")
    mode.add_argument("--check", action="store_true", help="Fail closed if live_facts or generated docs are stale.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root path (default: repo root).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = args.root.resolve()
    try:
        if args.write:
            written = run_write(root)
            rel_list = [str(p.relative_to(root)) for p in written]
            if rel_list:
                sys.stdout.write(f"docs_fork: wrote {len(rel_list)} file(s): {', '.join(rel_list)}\n")
            else:
                sys.stdout.write("docs_fork: already up to date (0 files changed)\n")
            return 0

        errors = run_check(root)
        if errors:
            for item in errors:
                sys.stderr.write(f"DOCS_FORK_DRIFT: {item}\n")
            return 1
        sys.stdout.write("docs_fork: OK (0 drift, live code facts verified)\n")
        return 0
    except DocsForkError as exc:
        sys.stderr.write(f"DOCS_FORK_ERROR: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
