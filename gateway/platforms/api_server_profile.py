"""Profile summary, safe behavioral defaults, and SOUL.md API routes.

Strictly scoped to the active profile via ``_profile_scope``. Never exposes or
modifies ``.env`` secrets, API keys, or raw credentials. Uses the comment-
preserving ``hermes_cli.config.save_config`` writer and ``WEBAPP_ATTACHMENT_MAX_BYTES``
SSoT limit.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import functools
import logging
from pathlib import Path
import re
from typing import Any, Optional

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]

from hermes_cli.config import load_config, save_config
from hermes_cli.config_backend import CONFIG_MUTATION_LOCK
from hermes_cli.tools_config import _get_effective_configurable_toolsets, _get_platform_tools
from hermes_cli.workspace_ops import path_is_under
from hermes_constants import (
    VALID_REASONING_EFFORTS,
    WEBAPP_ATTACHMENT_MAX_BYTES,
    get_hermes_home,
    parse_reasoning_effort,
)
from tools.memory_tool import get_builtin_memory_store_flags
from toolsets import TOOLSETS
from utils import atomic_write_text, is_truthy_value

logger = logging.getLogger("gateway.platforms.api_server")

MAX_SOUL_CHARS = 65_536
_SOUL_PREVIEW_CHARS = 240
_ALLOWED_DEFAULT_KEYS = frozenset({
    "model", "provider", "enabled_toolsets", "reasoning_effort", "preferred_language",
})
_VALID_PREFERRED_LANGUAGES = frozenset({"fr", "en", "auto"})
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")

CAPABILITY_FEATURES: dict[str, Any] = {
    "profile_defaults": True,
}

CAPABILITY_ENDPOINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("profile", ("GET", "/api/profile", "profile_defaults")),
    ("profile_defaults", ("PATCH", "/api/profile/defaults", "profile_defaults")),
    ("profile_soul", ("GET", "/api/profile/soul", "profile_defaults")),
    ("profile_soul_update", ("PUT", "/api/profile/soul", "profile_defaults")),
)


def _known_toolset_names() -> set[str]:
    names = set(TOOLSETS.keys())
    with suppress(Exception):
        names.update(name for name, _, _ in _get_effective_configurable_toolsets())
    return names


def _resolve_profile_name_and_description(home: Path) -> tuple[str, str]:
    name = "default"
    with suppress(ImportError, RuntimeError, OSError):
        from agent.file_safety import _resolve_active_profile_name

        name = _resolve_active_profile_name() or "default"
    description = ""
    with suppress(ImportError, RuntimeError, OSError):
        from hermes_cli.profiles import read_profile_meta

        meta = read_profile_meta(home) or {}
        if isinstance(meta, dict):
            description = str(meta.get("description") or "").strip()
    return name, description


def _safe_soul_path(home: Path) -> tuple[Optional[Path], Optional[tuple[int, str, str]]]:
    try:
        resolved_home = home.resolve(strict=False)
    except (OSError, RuntimeError):
        return None, (500, "profile_home_invalid", "Profile home could not be resolved.")
    soul_path = home / "SOUL.md"
    if soul_path.is_symlink():
        try:
            resolved_soul = soul_path.resolve(strict=False)
        except (OSError, RuntimeError):
            return None, (403, "soul_symlink_escape", "SOUL.md symlink is invalid.")
        if not path_is_under(resolved_home, resolved_soul):
            return None, (403, "soul_symlink_escape", "SOUL.md symlink escapes the profile home.")
    return soul_path, None


def _extract_model_defaults(config: dict[str, Any]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        default_model = str(model_cfg.get("default") or "").strip() or None
        default_provider = str(model_cfg.get("provider") or "").strip() or None
        effort = str(model_cfg.get("reasoning_effort") or "").strip() or None
    elif isinstance(model_cfg, str):
        default_model = model_cfg.strip() or None
        default_provider = None
        effort = None
    else:
        default_model = default_provider = effort = None
    if effort is None and isinstance(config.get("agent"), dict):
        effort = str(config["agent"].get("reasoning_effort") or "").strip() or None
    return default_model, default_provider, effort


def _voice_enabled(config: dict[str, Any]) -> bool:
    tts_cfg = config.get("tts") if isinstance(config.get("tts"), dict) else {}
    stt_cfg = config.get("stt") if isinstance(config.get("stt"), dict) else {}
    return bool(
        is_truthy_value(tts_cfg.get("enabled"), default=False)
        or is_truthy_value(stt_cfg.get("enabled"), default=False)
        or str(tts_cfg.get("provider") or "").strip()
        or str(stt_cfg.get("provider") or "").strip()
    )


def _sync_get_profile() -> dict[str, Any]:
    home = get_hermes_home()
    config = load_config() or {}
    name, description = _resolve_profile_name_and_description(home)
    default_model, default_provider, reasoning_effort = _extract_model_defaults(config)
    enabled_toolsets = sorted(_get_platform_tools(config, "api_server"))
    memory_enabled, user_profile_enabled = get_builtin_memory_store_flags(config)
    soul_path, soul_err = _safe_soul_path(home)
    soul_preview = ""
    soul_exists = False
    if soul_err is None and soul_path is not None and soul_path.is_file():
        with suppress(OSError, UnicodeDecodeError):
            raw = soul_path.read_text(encoding="utf-8-sig")
            soul_exists = True
            soul_preview = raw.strip()[:_SOUL_PREVIEW_CHARS]
    display_cfg = config.get("display") if isinstance(config.get("display"), dict) else {}
    raw_pref = str(display_cfg.get("preferred_language") or "").strip().lower()
    if raw_pref in _VALID_PREFERRED_LANGUAGES:
        preferred_language = raw_pref
    else:
        raw_lang = str(display_cfg.get("language") or "").strip().lower()
        preferred_language = raw_lang if raw_lang == "fr" else "auto"
    return {
        "object": "hermes.profile",
        "name": name,
        "description": description,
        "default_model": default_model,
        "default_provider": default_provider,
        "reasoning_effort": reasoning_effort,
        "preferred_language": preferred_language,
        "enabled_toolsets": enabled_toolsets,
        "memory_enabled": memory_enabled,
        "user_profile_enabled": user_profile_enabled,
        "voice_enabled": _voice_enabled(config),
        "soul_exists": soul_exists,
        "soul_preview": soul_preview,
        "max_artifact_bytes": WEBAPP_ATTACHMENT_MAX_BYTES,
    }


def _validate_profile_defaults_body(body: dict[str, Any]) -> tuple[Optional[dict[str, Any]], Optional[tuple[int, str, str]]]:
    if not body:
        return None, (400, "empty_profile_update", "Provide at least one allowed field to update.")
    unknown = sorted(set(body) - _ALLOWED_DEFAULT_KEYS)
    if unknown:
        return None, (
            400,
            "unsupported_profile_field",
            f"Unsupported or disallowed profile field(s): {', '.join(unknown)}.",
        )
    cleaned: dict[str, Any] = {}
    for key, max_len in (("model", 200), ("provider", 100)):
        if key not in body:
            continue
        val = body[key]
        if not isinstance(val, str) or not val.strip():
            return None, (400, f"invalid_{key}", f"'{key}' must be a non-empty string.")
        try:
            val.encode("utf-8")
        except UnicodeEncodeError:
            return None, (400, f"invalid_{key}", f"'{key}' must be valid UTF-8.")
        stripped = val.strip()
        if len(stripped) > max_len or _CONTROL_CHAR_RE.search(stripped):
            return None, (400, f"invalid_{key}", f"'{key}' contains invalid characters or exceeds {max_len} chars.")
        cleaned[key] = stripped

    if "reasoning_effort" in body:
        raw_effort = body["reasoning_effort"]
        if raw_effort is None or (isinstance(raw_effort, str) and not raw_effort.strip()):
            cleaned["reasoning_effort"] = None
        elif isinstance(raw_effort, str):
            norm = raw_effort.strip().lower()
            if norm != "none" and parse_reasoning_effort(norm) is None:
                valid = ", ".join(("none", *VALID_REASONING_EFFORTS))
                return None, (400, "invalid_reasoning_effort", f"reasoning_effort must be one of: {valid}.")
            cleaned["reasoning_effort"] = norm
        else:
            return None, (400, "invalid_reasoning_effort", "'reasoning_effort' must be a string or null.")

    if "enabled_toolsets" in body:
        raw_ts = body["enabled_toolsets"]
        if not isinstance(raw_ts, list) or any(not isinstance(t, str) or not t.strip() for t in raw_ts):
            return None, (400, "invalid_toolsets", "'enabled_toolsets' must be an array of non-empty strings.")
        normalized_ts = [t.strip() for t in raw_ts]
        if len(set(normalized_ts)) != len(normalized_ts):
            return None, (400, "invalid_toolsets", "'enabled_toolsets' must contain distinct toolset names.")
        known = _known_toolset_names()
        unknown_ts = sorted(t for t in normalized_ts if t not in known)
        if unknown_ts:
            return None, (400, "invalid_toolsets", f"Unknown toolset(s): {', '.join(unknown_ts)}.")
        cleaned["enabled_toolsets"] = sorted(normalized_ts)

    if "preferred_language" in body:
        raw_lang = body["preferred_language"]
        if not isinstance(raw_lang, str) or raw_lang.strip().lower() not in _VALID_PREFERRED_LANGUAGES:
            return None, (
                400,
                "invalid_preferred_language",
                "preferred_language must be one of: auto, fr, en.",
            )
        cleaned["preferred_language"] = raw_lang.strip().lower()

    return cleaned, None


def _sync_patch_profile_defaults(updates: dict[str, Any]) -> dict[str, Any]:
    with CONFIG_MUTATION_LOCK:
        config = load_config() or {}
        if any(key in updates for key in ("model", "provider", "reasoning_effort")):
            raw_model = config.get("model")
            if isinstance(raw_model, dict):
                model_cfg = dict(raw_model)
            elif isinstance(raw_model, str) and raw_model.strip():
                model_cfg = {"default": raw_model.strip()}
            else:
                model_cfg = {}
            if "model" in updates:
                model_cfg["default"] = updates["model"]
            if "provider" in updates:
                model_cfg["provider"] = updates["provider"]
            if "reasoning_effort" in updates:
                if updates["reasoning_effort"] is None:
                    model_cfg.pop("reasoning_effort", None)
                else:
                    model_cfg["reasoning_effort"] = updates["reasoning_effort"]
            config["model"] = model_cfg
        if "enabled_toolsets" in updates:
            platform_toolsets = config.get("platform_toolsets")
            if not isinstance(platform_toolsets, dict):
                platform_toolsets = {}
            platform_toolsets["api_server"] = updates["enabled_toolsets"]
            config["platform_toolsets"] = platform_toolsets
        if "preferred_language" in updates:
            display_cfg = config.get("display")
            if not isinstance(display_cfg, dict):
                display_cfg = {}
            display_cfg["preferred_language"] = updates["preferred_language"]
            if updates["preferred_language"] in ("fr", "en"):
                display_cfg["language"] = updates["preferred_language"]
            config["display"] = display_cfg
        save_config(config)
    if "preferred_language" in updates:
        with suppress(ImportError, RuntimeError):
            from agent.i18n import reset_language_cache

            reset_language_cache()
    return _sync_get_profile()


def _sync_get_soul() -> tuple[Optional[dict[str, Any]], Optional[tuple[int, str, str]]]:
    home = get_hermes_home()
    soul_path, err = _safe_soul_path(home)
    if err is not None:
        return None, err
    assert soul_path is not None
    if not soul_path.is_file():
        return {
            "object": "hermes.profile.soul",
            "content": "",
            "exists": False,
            "chars": 0,
            "max_chars": MAX_SOUL_CHARS,
        }, None
    try:
        content = soul_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        logger.warning("Could not read SOUL.md", exc_info=True)
        return None, (500, "soul_read_failed", "Could not read SOUL.md.")
    return {
        "object": "hermes.profile.soul",
        "content": content,
        "exists": True,
        "chars": len(content),
        "max_chars": MAX_SOUL_CHARS,
    }, None


def _sync_put_soul(content: str) -> tuple[Optional[dict[str, Any]], Optional[tuple[int, str, str]]]:
    home = get_hermes_home()
    soul_path, err = _safe_soul_path(home)
    if err is not None:
        return None, err
    assert soul_path is not None
    try:
        from hermes_constants import mkdir_under_hermes_home

        mkdir_under_hermes_home(soul_path.parent)
        atomic_write_text(soul_path, content, preserve_mode=True, create_mode=0o644)
    except OSError:
        logger.warning("Could not write SOUL.md", exc_info=True)
        return None, (500, "soul_write_failed", "Could not write SOUL.md.")
    return {
        "object": "hermes.profile.soul",
        "ok": True,
        "exists": True,
        "chars": len(content),
        "max_chars": MAX_SOUL_CHARS,
    }, None


async def handle_get_profile(adapter, request: web.Request) -> web.Response:
    """GET /api/profile — active profile summary & SSoT artifact limit."""
    from gateway.platforms.api_server import _error_response

    if auth_err := adapter._check_auth(request):
        return auth_err
    try:
        payload = await asyncio.to_thread(_sync_get_profile)
    except Exception:
        logger.exception("[%s] GET /api/profile failed", adapter.name)
        return _error_response("Failed to load profile.", 500, code="profile_read_failed")
    return web.json_response(payload)


async def handle_patch_profile_defaults(adapter, request: web.Request) -> web.Response:
    """PATCH /api/profile/defaults — update allowlisted safe behavioral defaults."""
    from gateway.platforms.api_server import _error_response

    if auth_err := adapter._check_auth(request):
        return auth_err
    body, err = await adapter._read_json_body(request)
    if err is not None:
        return err
    updates, val_err = await asyncio.to_thread(_validate_profile_defaults_body, body)
    if val_err is not None:
        status, code, msg = val_err
        return _error_response(msg, status, code=code)
    assert updates is not None
    try:
        payload = await asyncio.to_thread(_sync_patch_profile_defaults, updates)
    except Exception:
        logger.exception("[%s] PATCH /api/profile/defaults failed", adapter.name)
        return _error_response("Failed to save profile defaults.", 500, code="profile_write_failed")
    return web.json_response(payload)


async def handle_get_profile_soul(adapter, request: web.Request) -> web.Response:
    """GET /api/profile/soul — read active profile's SOUL.md."""
    from gateway.platforms.api_server import _error_response

    if auth_err := adapter._check_auth(request):
        return auth_err
    payload, err = await asyncio.to_thread(_sync_get_soul)
    if err is not None:
        status, code, msg = err
        return _error_response(msg, status, code=code)
    assert payload is not None
    return web.json_response(payload)


async def handle_put_profile_soul(adapter, request: web.Request) -> web.Response:
    """PUT /api/profile/soul — atomically update active profile's SOUL.md."""
    from gateway.platforms.api_server import _error_response

    if auth_err := adapter._check_auth(request):
        return auth_err
    body, err = await adapter._read_json_body(request)
    if err is not None:
        return err
    content = body.get("content")
    if not isinstance(content, str):
        return _error_response("'content' must be a string.", 400, code="invalid_soul_content")
    try:
        content.encode("utf-8")
    except UnicodeEncodeError:
        return _error_response("'content' must be valid UTF-8 text.", 400, code="invalid_soul_content")
    if len(content) > MAX_SOUL_CHARS:
        return _error_response(
            f"SOUL.md content exceeds maximum length of {MAX_SOUL_CHARS} characters.",
            413,
            code="soul_too_large",
        )
    payload, write_err = await asyncio.to_thread(_sync_put_soul, content)
    if write_err is not None:
        status, code, msg = write_err
        return _error_response(msg, status, code=code)
    assert payload is not None
    return web.json_response(payload)


def _http_routes(adapter) -> list[tuple]:
    return [
        ("GET", "/api/profile", functools.partial(handle_get_profile, adapter)),
        ("PATCH", "/api/profile/defaults", functools.partial(handle_patch_profile_defaults, adapter)),
        ("GET", "/api/profile/soul", functools.partial(handle_get_profile_soul, adapter)),
        ("PUT", "/api/profile/soul", functools.partial(handle_put_profile_soul, adapter)),
    ]
