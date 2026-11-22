"""Shared Skill Studio catalog operations (SSoT for API server and Dashboard).

Reuses ``tools.skills_tool``, ``tools.skill_manager_tool``, ``tools.skill_usage``,
``agent.skill_utils``, and ``hermes_cli.skills_config`` without duplicating
discovery, frontmatter parsing, linting, or config persistence.
"""

from __future__ import annotations

from contextlib import suppress
import json
import logging
from pathlib import Path
from typing import Any, Optional

from agent.skill_utils import ESSENTIAL_SKILLS, parse_frontmatter
from hermes_cli.config import cfg_get, load_config
from hermes_cli.config_backend import CONFIG_MUTATION_LOCK
from hermes_cli.skills_config import _normalize_skill_names, get_disabled_skills, save_disabled_skills
from tools.skill_manager_tool import ALLOWED_SUBDIRS, MAX_NAME_LENGTH, _find_skill
from tools.skill_usage import _read_bundled_names
from tools.skills_tool import (
    _parse_tags,
    _read_skill_text,
    _skill_catalog,
    _skills_dir,
    clear_skills_cache,
    sort_skills,
)

logger = logging.getLogger(__name__)

_MAX_LOCATOR_LENGTH = 160


def invalidate_skills_caches() -> None:
    """Clear both discovery and system-prompt skill caches after any mutation."""
    clear_skills_cache()
    with suppress(ImportError, AttributeError, RuntimeError):
        from agent.prompt_builder import clear_skills_system_prompt_cache

        clear_skills_system_prompt_cache(clear_snapshot=True)


def validate_skill_locator(name: Any) -> Optional[str]:
    """Validate a skill locator path segment before any filesystem lookup."""
    if not isinstance(name, str) or not name.strip():
        return "Skill name is required."
    if len(name) > _MAX_LOCATOR_LENGTH:
        return f"Skill name exceeds {_MAX_LOCATOR_LENGTH} characters."
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        return "Skill name must be valid UTF-8."
    if any(ch in name for ch in ("\x00", "\\", "\r", "\n")):
        return "Skill name contains forbidden characters."
    if name.startswith("/") or name.endswith("/"):
        return "Skill name cannot start or end with '/'."
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return "Path traversal or empty path segments are not allowed in skill names."
    if len(parts) > 3:
        return "Skill name path depth is too deep."
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in name):
        return "Skill name contains control characters."
    return None


def _is_within(root: Optional[Path], target: Path) -> bool:
    if root is None:
        return False
    try:
        resolved_root = root.resolve(strict=False)
        resolved_target = target.resolve(strict=False)
        return resolved_target == resolved_root or resolved_root in resolved_target.parents
    except (OSError, RuntimeError):
        return False


def _create_skills_dir() -> Optional[Path]:
    try:
        from agent.skill_utils import get_skill_create_dir

        return get_skill_create_dir()
    except (ImportError, OSError, RuntimeError):
        return None


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, allow_nan=False)
        return value
    except (TypeError, ValueError, RecursionError):
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(v) for v in value]
        return str(value)


def extract_skill_tags(frontmatter: dict[str, Any]) -> list[str]:
    """Extract normalized tag list from frontmatter (top-level or metadata.tags)."""
    raw = frontmatter.get("tags")
    if not raw and isinstance(frontmatter.get("metadata"), dict):
        raw = frontmatter["metadata"].get("tags")
    tags = _parse_tags(raw)
    return list(dict.fromkeys(tags))


def list_supporting_files(skill_dir: Path) -> list[str]:
    """Return sorted POSIX relative paths of safe supporting files under a skill."""
    try:
        resolved_root = skill_dir.resolve(strict=True)
    except (OSError, RuntimeError):
        return []
    found: list[str] = []
    for subdir in sorted(ALLOWED_SUBDIRS):
        sub_path = skill_dir / subdir
        if not sub_path.is_dir() or sub_path.is_symlink():
            continue
        try:
            for item in sub_path.rglob("*"):
                if not item.is_file():
                    continue
                resolved = item.resolve(strict=True)
                if resolved_root not in resolved.parents:
                    continue
                found.append(resolved.relative_to(resolved_root).as_posix())
        except (OSError, RuntimeError):
            continue
    return sorted(found)


def classify_skill_source(
    load_name: str,
    raw_name: str,
    skill_dir: Path,
    *,
    bundled_names: set[str],
    local_root: Optional[Path],
    create_dir: Optional[Path],
) -> tuple[str, bool]:
    """Classify skill provenance (``user``/``builtin``/``external``) and writability."""
    in_writable_root = _is_within(local_root, skill_dir) or _is_within(create_dir, skill_dir)
    if skill_dir.is_symlink() and not in_writable_root:
        return "external", False
    if raw_name in bundled_names or load_name in bundled_names or skill_dir.name in bundled_names:
        return "builtin", False
    if in_writable_root:
        return "user", True
    return "external", False


def _read_skill_frontmatter(skill_md: Path) -> tuple[dict[str, Any], str, str]:
    raw = _read_skill_text(skill_md)
    frontmatter, body = parse_frontmatter(raw)
    return frontmatter if isinstance(frontmatter, dict) else {}, body, raw


def _plugin_skill_rows(disabled: set[str], include_disabled: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        from hermes_cli.plugins import get_plugin_manager

        manager = get_plugin_manager()
        for item in manager.list_plugin_skill_metadata():
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            enabled = name not in disabled
            if not include_disabled and not enabled:
                continue
            fm = item.get("frontmatter") if isinstance(item.get("frontmatter"), dict) else {}
            rows.append({
                "name": name,
                "description": str(item.get("description") or ""),
                "category": item.get("category"),
                "enabled": enabled,
                "tags": extract_skill_tags(fm),
                "source": "plugin",
                "writable": False,
                "has_supporting_files": False,
            })
    except (ImportError, AttributeError, RuntimeError, OSError):
        logger.debug("Plugin skill enumeration skipped", exc_info=True)
    return rows


def list_skills_enriched(
    *,
    include_disabled: bool = False,
    platform: str = "api_server",
    category: Optional[str] = None,
    tag: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Return sorted, enriched skill catalog entries for the active profile."""
    config = load_config() or {}
    disabled = get_disabled_skills(config, platform=platform)
    bundled_names = _read_bundled_names()
    local_root = _skills_dir()
    create_dir = _create_skills_dir()
    catalog = _skill_catalog(skip_disabled=include_disabled, platform=platform)
    rows: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for entry in catalog:
        load_name = entry.get("load_name")
        if not load_name:
            continue
        skill_md: Path = entry["path"]
        skill_dir = skill_md.parent
        raw_name = str(entry.get("name") or load_name)
        tags: list[str] = []
        with suppress(OSError, UnicodeDecodeError):
            fm, _, _ = _read_skill_frontmatter(skill_md)
            tags = extract_skill_tags(fm)
        source, writable = classify_skill_source(
            load_name,
            raw_name,
            skill_dir,
            bundled_names=bundled_names,
            local_root=local_root,
            create_dir=create_dir,
        )
        enabled = (
            load_name not in disabled
            and raw_name not in disabled
            and skill_dir.name not in disabled
        )
        rows.append({
            "name": load_name,
            "description": entry["description"],
            "category": entry["category"],
            "enabled": enabled,
            "tags": tags,
            "source": source,
            "writable": writable,
            "has_supporting_files": bool(list_supporting_files(skill_dir)),
        })
        seen_names.add(load_name)

    for plugin_row in _plugin_skill_rows(disabled, include_disabled):
        if plugin_row["name"] not in seen_names:
            rows.append(plugin_row)

    if category:
        rows = [row for row in rows if (row.get("category") or "") == category]
    if tag:
        wanted = tag.strip().lower()
        rows = [row for row in rows if any(t.lower() == wanted for t in row.get("tags", []))]
    return sort_skills(rows)


def _locate_skill_entry(name: str, platform: str) -> tuple[Optional[Path], Optional[dict[str, Any]]]:
    catalog = _skill_catalog(skip_disabled=True, platform=platform)
    for entry in catalog:
        if entry.get("load_name") == name or entry.get("name") == name:
            return entry["path"].parent, entry
    found = _find_skill(name)
    if found is not None:
        found_dir = found["path"]
        with suppress(OSError, RuntimeError):
            resolved_found = found_dir.resolve()
            for entry in catalog:
                if entry["path"].parent.resolve() == resolved_found:
                    return found_dir, entry
        return found_dir, None
    return None, None


def _collect_lint_warnings(skill_md: Path) -> list[dict[str, str]]:
    try:
        from tools.skill_linter import lint_skill

        return [
            {"severity": f.severity, "rule": f.rule, "message": f.message}
            for f in lint_skill(skill_md)
        ]
    except (ImportError, OSError, RuntimeError, ValueError):
        return []


def read_skill_detail(
    name: str,
    *,
    platform: str = "api_server",
) -> tuple[Optional[dict[str, Any]], Optional[tuple[int, str, str]]]:
    """Read full skill detail (frontmatter, body, raw, supporting files, lint)."""
    if err := validate_skill_locator(name):
        return None, (400, "invalid_skill_name", err)

    skill_dir, catalog_entry = _locate_skill_entry(name, platform)
    if skill_dir is None:
        return None, (404, "skill_not_found", f"Skill '{name}' not found.")

    skill_md = skill_dir / "SKILL.md"
    try:
        resolved_dir = skill_dir.resolve(strict=True)
        resolved_md = skill_md.resolve(strict=True)
    except OSError:
        return None, (404, "skill_not_found", f"Skill '{name}' has no readable SKILL.md.")
    if resolved_dir not in resolved_md.parents:
        return None, (403, "skill_path_escape", "Skill file symlink escapes its skill directory.")

    try:
        frontmatter, body, raw = _read_skill_frontmatter(resolved_md)
    except (OSError, UnicodeDecodeError) as exc:
        return None, (500, "skill_read_failed", f"Could not read skill '{name}': {exc}")

    config = load_config() or {}
    disabled = get_disabled_skills(config, platform=platform)
    bundled_names = _read_bundled_names()
    local_root = _skills_dir()
    create_dir = _create_skills_dir()
    load_name = str((catalog_entry or {}).get("load_name") or name)
    raw_name = str(frontmatter.get("name") or (catalog_entry or {}).get("name") or skill_dir.name)
    category = (catalog_entry or {}).get("category")
    description = str((catalog_entry or {}).get("description") or frontmatter.get("description") or "")
    source, writable = classify_skill_source(
        load_name,
        raw_name,
        skill_dir,
        bundled_names=bundled_names,
        local_root=local_root,
        create_dir=create_dir,
    )
    enabled = (
        load_name not in disabled
        and raw_name not in disabled
        and skill_dir.name not in disabled
    )
    supporting = list_supporting_files(resolved_dir)
    return {
        "name": load_name,
        "category": category,
        "tags": extract_skill_tags(frontmatter),
        "description": description,
        "frontmatter": _json_safe(frontmatter),
        "body": body,
        "raw": raw,
        "enabled": enabled,
        "source": source,
        "writable": writable,
        "supporting_files": supporting,
        "has_supporting_files": bool(supporting),
        "lint_warnings": _collect_lint_warnings(resolved_md),
    }, None


def is_skill_writable(name: str, *, platform: str = "api_server") -> tuple[bool, Optional[tuple[int, str, str]]]:
    """Check whether an existing skill is user-owned and writable."""
    detail, err = read_skill_detail(name, platform=platform)
    if err is not None:
        return False, err
    assert detail is not None
    if not detail["writable"]:
        return False, (
            403,
            "skill_not_writable",
            f"Skill '{name}' ({detail['source']}) is read-only and cannot be modified or deleted.",
        )
    return True, None


def toggle_skill_enabled(
    name: str,
    enabled: bool,
    *,
    platform: str = "global",
) -> tuple[Optional[dict[str, Any]], Optional[tuple[int, str, str]]]:
    """Toggle a skill's enabled state in profile config.yaml (atomic SSoT writer)."""
    if err := validate_skill_locator(name):
        return None, (400, "invalid_skill_name", err)
    if platform not in ("global", "api_server"):
        return None, (400, "invalid_platform", "platform must be 'global' or 'api_server'.")

    skill_dir, catalog_entry = _locate_skill_entry(name, "api_server")
    if skill_dir is None:
        return None, (404, "skill_not_found", f"Skill '{name}' not found.")
    canonical = str((catalog_entry or {}).get("load_name") or name)
    raw_name = str((catalog_entry or {}).get("name") or skill_dir.name)
    candidates = {name, canonical, raw_name, skill_dir.name}
    if not enabled and (candidates & ESSENTIAL_SKILLS):
        return None, (403, "essential_skill_protected", f"Essential skill '{canonical}' cannot be disabled.")

    with CONFIG_MUTATION_LOCK:
        config = load_config() or {}
        if platform == "global":
            disabled = get_disabled_skills(config, platform=None)
            if enabled:
                disabled -= candidates
            else:
                disabled.add(canonical)
            save_disabled_skills(config, disabled, platform=None)
            if enabled:
                plat_raw = cfg_get(config, "skills", "platform_disabled", "api_server")
                if plat_raw is not None:
                    plat_disabled = _normalize_skill_names(plat_raw) - candidates
                    save_disabled_skills(config, plat_disabled, platform="api_server")
        else:
            plat_disabled = _normalize_skill_names(cfg_get(config, "skills", "platform_disabled", "api_server"))
            if enabled:
                plat_disabled -= candidates
            else:
                plat_disabled.add(canonical)
            save_disabled_skills(config, plat_disabled, platform="api_server")
        effective_disabled = get_disabled_skills(config, platform="api_server")
        effective_enabled = not bool(candidates & effective_disabled)

    invalidate_skills_caches()
    return {
        "name": canonical,
        "enabled": effective_enabled,
        "platform": platform,
    }, None


__all__ = [
    "MAX_NAME_LENGTH",
    "classify_skill_source",
    "extract_skill_tags",
    "invalidate_skills_caches",
    "is_skill_writable",
    "list_skills_enriched",
    "list_supporting_files",
    "read_skill_detail",
    "toggle_skill_enabled",
    "validate_skill_locator",
]
