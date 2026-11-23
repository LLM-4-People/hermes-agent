"""Single source of truth for effortless multi-profile management, admin settings, and onboarding.

Security and isolation invariants:
- First profile (``default``, or the very first profile on a fresh root) is automatically
  an admin (``is_admin=True``, ``role="admin"``). Additional profiles default to ``member``
  unless explicitly created as admin by an existing admin.
- Non-admin members can ONLY inspect/update/rotate/pair their own profile; any cross-profile
  or admin-settings call fails closed with ``403``.
- Both ``PATCH`` (demotion) and ``DELETE`` refuse with ``409 Conflict`` if the operation
  would remove the last remaining admin profile.
- Every profile creation routes through ``hermes_cli.profile_provision.provision_profile``
  and every ``config.yaml`` write routes through ``hermes_cli.config.save_config``.
"""

from __future__ import annotations

from contextlib import suppress
import json
import logging
import os
from pathlib import Path
import re
import secrets
import sqlite3
import threading
from typing import Any, Callable, Optional

from gateway.i18n import normalize_user_lang, t
from gateway.mobile_pairing import (
    DEFAULT_TTL_SECONDS,
    MAX_TTL_SECONDS,
    MIN_TTL_SECONDS,
    PairingStore,
    validate_ttl_seconds,
)
from utils import atomic_write_text

logger = logging.getLogger("gateway.mobile_profiles")

_VALID_LANGUAGES = frozenset({"auto", "en", "fr"})
_BADGE_PRESETS = frozenset({"forest", "slate", "terracotta", "indigo", "plum"})
_NON_ADMIN_EXCLUDED_TOOLSETS = frozenset({"terminal", "code_execution"})
_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")
_ENV_LINE_RE = re.compile(r"^\s*(?:export\s+)?API_SERVER_KEY\s*=")

_ADMIN_STATE_LOCK = threading.RLock()
_ENV_WRITE_LOCK = threading.RLock()

_DEFAULT_ADMIN_SETTINGS: dict[str, Any] = {
    "default_language": "auto",
    "default_model": "",
    "max_concurrent_runs": 32,
    "per_profile_concurrent_limit": 12,
    "pairing_ttl_seconds": DEFAULT_TTL_SECONDS,
    "allow_self_device_pairing": True,
}


class ProfileManagementError(Exception):
    """Typed fail-closed error for profile/admin management operations."""

    def __init__(self, message: str, *, status_code: int = 400, code: str = "profile_error") -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


def default_root_home() -> Path:
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root()


def _admin_meta_path(root_home: Path) -> Path:
    return root_home / "mobile_admin.json"


def _load_admin_state(root_home: Path) -> dict[str, Any]:
    path = _admin_meta_path(root_home)
    if not path.is_file():
        return {"admins": ["default"], "badges": {}, "settings": dict(_DEFAULT_ADMIN_SETTINGS)}
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {"admins": ["default"], "badges": {}, "settings": dict(_DEFAULT_ADMIN_SETTINGS)}
    if not isinstance(raw, dict):
        return {"admins": ["default"], "badges": {}, "settings": dict(_DEFAULT_ADMIN_SETTINGS)}
    admins = [str(s).strip().lower() for s in (raw.get("admins") or ["default"]) if str(s).strip()]
    if not admins:
        admins = ["default"]
    badges = {
        str(k).strip().lower(): str(v).strip()
        for k, v in (raw.get("badges") or {}).items()
        if isinstance(k, str) and isinstance(v, str) and v.strip()
    }
    saved_settings = raw.get("settings") if isinstance(raw.get("settings"), dict) else {}
    settings = dict(_DEFAULT_ADMIN_SETTINGS)
    settings.update({k: v for k, v in saved_settings.items() if k in _DEFAULT_ADMIN_SETTINGS})
    return {"admins": sorted(set(admins)), "badges": badges, "settings": settings}


def _save_admin_state(root_home: Path, state: dict[str, Any]) -> None:
    root_home.mkdir(parents=True, exist_ok=True)
    path = _admin_meta_path(root_home)
    payload = json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    atomic_write_text(path, payload)
    with suppress(OSError):
        os.chmod(path, 0o600)


def _profile_dir_for(slug: str, root_home: Path) -> Path:
    clean = (slug or "default").strip().lower()
    if clean == "default":
        return root_home
    return root_home / "profiles" / clean


def _all_existing_slugs(root_home: Path) -> list[str]:
    slugs = ["default"]
    profiles_root = root_home / "profiles"
    if profiles_root.is_dir():
        for entry in sorted(profiles_root.iterdir(), key=lambda p: p.name.lower()):
            if entry.is_dir() and not entry.is_symlink() and not entry.name.startswith("."):
                slugs.append(entry.name.lower())
    return slugs


def _effective_admins(root_home: Path, state: dict[str, Any]) -> set[str]:
    existing = set(_all_existing_slugs(root_home))
    admins = {s for s in state.get("admins", ["default"]) if s in existing}
    if not admins:
        # First profile invariant: default profile is always admin if no other admin exists.
        admins.add("default")
    return admins


def is_admin_profile(profile_slug: str, root_home: Optional[Path] = None) -> bool:
    """Return True when *profile_slug* holds admin privileges (fail-closed)."""
    root = Path(root_home) if root_home is not None else default_root_home()
    slug = (profile_slug or "default").strip().lower()
    with _ADMIN_STATE_LOCK:
        state = _load_admin_state(root)
        return slug in _effective_admins(root, state)


def generate_api_key() -> str:
    """Generate a 256-bit profile API key passing ``has_usable_secret(min_length=16)``."""
    return f"hmk_{secrets.token_urlsafe(32)}"


def _parse_dotenv_key(env_path: Path, key_name: str = "API_SERVER_KEY") -> str:
    if not env_path.is_file():
        return ""
    try:
        text = env_path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        k, _, v = line.partition("=")
        if k.strip() != key_name:
            continue
        val = v.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in {"'", '"'}:
            val = val[1:-1]
        return val.strip()
    return ""


def read_profile_api_key(profile_slug: str, root_home: Optional[Path] = None) -> str:
    """Read the profile's ``API_SERVER_KEY`` strictly from its own ``.env`` (never cross-profile)."""
    root = Path(root_home) if root_home is not None else default_root_home()
    slug = (profile_slug or "default").strip().lower()
    env_path = _profile_dir_for(slug, root) / ".env"
    key = _parse_dotenv_key(env_path, "API_SERVER_KEY")
    if key:
        return key
    if slug == "default":
        return os.environ.get("API_SERVER_KEY", "").strip()
    return ""


def write_profile_api_key(profile_slug: str, api_key: str, root_home: Optional[Path] = None) -> None:
    """Atomically write/replace ``API_SERVER_KEY`` in the target profile's ``.env`` with ``0600``."""
    root = Path(root_home) if root_home is not None else default_root_home()
    slug = (profile_slug or "default").strip().lower()
    pdir = _profile_dir_for(slug, root)
    pdir.mkdir(parents=True, exist_ok=True)
    env_path = pdir / ".env"
    with _ENV_WRITE_LOCK:
        existing_lines: list[str] = []
        if env_path.is_file():
            with suppress(OSError):
                existing_lines = env_path.read_text(encoding="utf-8-sig").splitlines()
        out_lines: list[str] = []
        replaced = False
        for line in existing_lines:
            if _ENV_LINE_RE.match(line):
                if not replaced:
                    out_lines.append(f"API_SERVER_KEY={api_key}")
                    replaced = True
                continue
            out_lines.append(line)
        if not replaced:
            if not out_lines:
                out_lines.append("# Per-profile secrets for this Hermes profile.")
            out_lines.append(f"API_SERVER_KEY={api_key}")
        atomic_write_text(env_path, "\n".join(out_lines) + "\n")
        with suppress(OSError):
            os.chmod(env_path, 0o600)


def _validate_display_name(raw: Any) -> str:
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raise ProfileManagementError("'display_name' must be a string.", status_code=400, code="invalid_display_name")
    cleaned = raw.strip()
    if len(cleaned) > 64 or _CONTROL_CHAR_RE.search(cleaned):
        raise ProfileManagementError(
            "'display_name' must be at most 64 safe UTF-8 characters.",
            status_code=400,
            code="invalid_display_name",
        )
    return cleaned


def _validate_preferred_language(raw: Any) -> str:
    if raw is None:
        return "auto"
    if not isinstance(raw, str) or raw.strip().lower() not in _VALID_LANGUAGES:
        raise ProfileManagementError(
            "'preferred_language' must be one of: auto, en, fr.",
            status_code=400,
            code="invalid_preferred_language",
        )
    return raw.strip().lower()


def _validate_model_name(raw: Any) -> str:
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raise ProfileManagementError("'default_model' must be a string.", status_code=400, code="invalid_model")
    cleaned = raw.strip()
    if len(cleaned) > 200 or _CONTROL_CHAR_RE.search(cleaned):
        raise ProfileManagementError(
            "'default_model' must be at most 200 safe UTF-8 characters.",
            status_code=400,
            code="invalid_model",
        )
    return cleaned


def _validate_badge_color(raw: Any) -> str:
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raise ProfileManagementError("'badge_color' must be a string.", status_code=400, code="invalid_badge_color")
    cleaned = raw.strip()
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    if lowered in _BADGE_PRESETS:
        return lowered
    if _HEX_COLOR_RE.fullmatch(cleaned):
        return cleaned.upper()
    raise ProfileManagementError(
        "'badge_color' must be a preset (forest, slate, terracotta, indigo, plum) or #RRGGBB hex.",
        status_code=400,
        code="invalid_badge_color",
    )


def _read_session_metrics(profile_dir: Path) -> tuple[int, Optional[float], bool]:
    """Read thread count, last_active, and isolated DB health strictly read-only."""
    if not profile_dir.is_dir() or profile_dir.is_symlink():
        return 0, None, False
    db_path = profile_dir / "state.db"
    if not db_path.is_file():
        return 0, None, True
    try:
        from hermes_cli.sqlite_safe_read import connect_tracked

        conn = connect_tracked(f"file:{db_path}?mode=ro", tracking_path=db_path, uri=True, timeout=2.0)
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt, MAX(COALESCE(ended_at, started_at)) AS last_ts FROM sessions"
            ).fetchone()
            if row is None:
                return 0, None, True
            count = int(row[0] or 0)
            last_ts = float(row[1]) if row[1] is not None else None
            return count, last_ts, True
        finally:
            conn.close()
    except (sqlite3.Error, RuntimeError):
        return 0, None, False


def _read_profile_config(profile_dir: Path) -> dict[str, Any]:
    from agent.runtime_scope import bind_profile_scope
    from hermes_cli.config import load_config

    try:
        with bind_profile_scope(profile_dir):
            return load_config() or {}
    except (OSError, ValueError, RuntimeError, ImportError):
        return {}


def _write_profile_config_fields(
    profile_dir: Path,
    *,
    preferred_language: Optional[str] = None,
    default_model: Optional[str] = None,
    api_server_toolsets: Optional[list[str]] = None,
) -> None:
    """Update ``config.yaml`` through the comment-preserving ``hermes_cli.config.save_config`` seam."""
    from agent.runtime_scope import bind_profile_scope
    from hermes_cli.config import load_config, save_config
    from hermes_cli.config_backend import CONFIG_MUTATION_LOCK

    with CONFIG_MUTATION_LOCK, bind_profile_scope(profile_dir):
        cfg = load_config() or {}
        if preferred_language is not None:
            raw_display = cfg.get("display")
            display_cfg = dict(raw_display) if isinstance(raw_display, dict) else {}
            display_cfg["preferred_language"] = preferred_language
            if preferred_language in ("en", "fr"):
                display_cfg["language"] = preferred_language
            else:
                display_cfg.pop("language", None)
            cfg["display"] = display_cfg
        if default_model is not None:
            raw_model = cfg.get("model")
            if isinstance(raw_model, dict):
                model_cfg = dict(raw_model)
            elif isinstance(raw_model, str) and raw_model.strip():
                model_cfg = {"default": raw_model.strip()}
            else:
                model_cfg = {}
            if default_model:
                model_cfg["default"] = default_model
            else:
                model_cfg.pop("default", None)
            cfg["model"] = model_cfg
        if api_server_toolsets is not None:
            raw_pts = cfg.get("platform_toolsets")
            pts_cfg = dict(raw_pts) if isinstance(raw_pts, dict) else {}
            pts_cfg["api_server"] = sorted(set(api_server_toolsets))
            cfg["platform_toolsets"] = pts_cfg
        save_config(cfg)


def _summarize_profile(
    slug: str,
    root_home: Path,
    *,
    admins: set[str],
    badges: dict[str, str],
    active_runs_fn: Optional[Callable[[str], int]] = None,
) -> dict[str, Any]:
    from hermes_cli.profiles import read_profile_meta

    pdir = _profile_dir_for(slug, root_home)
    meta = read_profile_meta(pdir) if pdir.is_dir() else {}
    display_name = str(meta.get("display_name") or "").strip() or slug
    cfg = _read_profile_config(pdir)
    display_cfg = cfg.get("display") if isinstance(cfg.get("display"), dict) else {}
    pref_lang = str(display_cfg.get("preferred_language") or "").strip().lower()
    if pref_lang not in _VALID_LANGUAGES:
        raw_lang = str(display_cfg.get("language") or "").strip().lower()
        pref_lang = raw_lang if raw_lang in ("en", "fr") else "auto"
    model_cfg = cfg.get("model")
    if isinstance(model_cfg, dict):
        default_model = str(model_cfg.get("default") or "").strip() or None
    elif isinstance(model_cfg, str):
        default_model = model_cfg.strip() or None
    else:
        default_model = None
    thread_count, last_active, isolated_db_ok = _read_session_metrics(pdir)
    active_runs = 0
    if active_runs_fn is not None:
        with suppress(Exception):
            active_runs = max(0, int(active_runs_fn(slug)))
    is_admin = slug in admins
    return {
        "slug": slug,
        "display_name": display_name,
        "language": pref_lang,
        "preferred_language": pref_lang,
        "default_model": default_model,
        "badge_color": badges.get(slug, "forest"),
        "active_runs": active_runs,
        "thread_count": thread_count,
        "last_active": last_active,
        "is_admin": is_admin,
        "role": "admin" if is_admin else "member",
        "is_default": slug == "default",
        "isolated_db_ok": isolated_db_ok,
        "has_api_key": bool(read_profile_api_key(slug, root_home)),
    }


def list_managed_profiles(
    caller_slug: str,
    *,
    active_runs_fn: Optional[Callable[[str], int]] = None,
    root_home: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """Return all profiles for admins, or ONLY the caller's own profile for members (fail-closed)."""
    root = Path(root_home) if root_home is not None else default_root_home()
    caller = (caller_slug or "default").strip().lower()
    with _ADMIN_STATE_LOCK:
        state = _load_admin_state(root)
        admins = _effective_admins(root, state)
        badges = dict(state["badges"])
    slugs = _all_existing_slugs(root) if caller in admins else [caller]
    return [
        _summarize_profile(slug, root, admins=admins, badges=badges, active_runs_fn=active_runs_fn)
        for slug in slugs
    ]


def get_profile_summary(
    slug: str,
    *,
    active_runs_fn: Optional[Callable[[str], int]] = None,
    root_home: Optional[Path] = None,
) -> Optional[dict[str, Any]]:
    root = Path(root_home) if root_home is not None else default_root_home()
    clean = (slug or "default").strip().lower()
    if clean not in _all_existing_slugs(root):
        return None
    with _ADMIN_STATE_LOCK:
        state = _load_admin_state(root)
        admins = _effective_admins(root, state)
        badges = dict(state["badges"])
    return _summarize_profile(clean, root, admins=admins, badges=badges, active_runs_fn=active_runs_fn)


def create_managed_profile(
    *,
    caller_slug: str,
    slug: str,
    display_name: str = "",
    preferred_language: str = "auto",
    default_model: Optional[str] = None,
    badge_color: Optional[str] = None,
    is_admin: bool = False,
    create_invite: bool = False,
    host: str = "",
    ttl_seconds: Optional[int] = None,
    root_home: Optional[Path] = None,
) -> dict[str, Any]:
    """Admin-only creation of a fully isolated profile with its own API key and optional invite."""
    root = Path(root_home) if root_home is not None else default_root_home()
    caller = (caller_slug or "default").strip().lower()
    if not is_admin_profile(caller, root):
        raise ProfileManagementError(
            "Only an admin profile can create new user profiles.",
            status_code=403,
            code="admin_required",
        )
    if not isinstance(slug, str) or not slug.strip():
        raise ProfileManagementError("'slug' is required.", status_code=400, code="invalid_profile_slug")

    from hermes_cli.profile_provision import ProfileCreateRequest, provision_profile
    from hermes_cli.profiles import normalize_profile_name, validate_profile_name, write_profile_meta

    try:
        canon = normalize_profile_name(slug)
        validate_profile_name(canon)
    except ValueError as exc:
        raise ProfileManagementError("Invalid profile slug.", status_code=400, code="invalid_profile_slug") from exc
    if canon == "default":
        raise ProfileManagementError("Cannot create reserved 'default' profile.", status_code=400, code="invalid_profile_slug")

    clean_display = _validate_display_name(display_name) or canon
    clean_lang = _validate_preferred_language(preferred_language)
    clean_model = _validate_model_name(default_model)
    clean_badge = _validate_badge_color(badge_color) or "forest"
    if not isinstance(is_admin, bool):
        raise ProfileManagementError("'is_admin' must be a boolean.", status_code=400, code="invalid_is_admin")

    admin_settings = get_admin_settings(root)
    if not clean_model and admin_settings.get("default_model"):
        clean_model = str(admin_settings["default_model"])
    if preferred_language == "auto" and admin_settings.get("default_language") in ("en", "fr"):
        clean_lang = str(admin_settings["default_language"])

    prev_home = os.environ.get("HERMES_HOME")
    try:
        os.environ["HERMES_HOME"] = str(root)
        result = provision_profile(
            ProfileCreateRequest(
                name=canon,
                no_skills=True,
                alias=False,
                quiet=True,
                mirror_credentials=False,
                share_auth=True,
            )
        )
    except FileExistsError as exc:
        raise ProfileManagementError("Profile already exists.", status_code=409, code="profile_exists") from exc
    except ValueError as exc:
        raise ProfileManagementError("Invalid profile slug.", status_code=400, code="invalid_profile_slug") from exc
    finally:
        if prev_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = prev_home

    profile_dir = result.path
    api_key = generate_api_key()
    write_profile_api_key(canon, api_key, root)
    write_profile_meta(profile_dir, display_name=clean_display)
    default_toolsets: Optional[list[str]] = None
    if not is_admin:
        from hermes_cli.tools_config import _get_platform_tools

        default_toolsets = sorted(_get_platform_tools({}, "api_server") - _NON_ADMIN_EXCLUDED_TOOLSETS)
    _write_profile_config_fields(
        profile_dir,
        preferred_language=clean_lang,
        default_model=clean_model or None,
        api_server_toolsets=default_toolsets,
    )

    with _ADMIN_STATE_LOCK:
        state = _load_admin_state(root)
        admins = _effective_admins(root, state)
        if is_admin:
            admins.add(canon)
        state["admins"] = sorted(admins)
        state["badges"][canon] = clean_badge
        _save_admin_state(root, state)

    pairing_ticket = None
    if create_invite:
        effective_ttl = ttl_seconds if ttl_seconds is not None else int(admin_settings["pairing_ttl_seconds"])
        pairing_ticket = PairingStore(root).create_ticket(
            profile_slug=canon,
            display_name=clean_display,
            mode="invite",
            created_by=caller,
            ttl_seconds=effective_ttl,
            host=host,
        )

    summary = get_profile_summary(canon, root_home=root)
    return {
        "profile": summary,
        "api_key": api_key,
        "pairing_ticket": pairing_ticket,
    }


def update_managed_profile(
    *,
    caller_slug: str,
    target_slug: str,
    updates: dict[str, Any],
    root_home: Optional[Path] = None,
) -> dict[str, Any]:
    """Update profile presentation/preferences (owner or admin) and role (admin-only with last-admin guard)."""
    root = Path(root_home) if root_home is not None else default_root_home()
    caller = (caller_slug or "default").strip().lower()
    target = (target_slug or "").strip().lower()
    if not target or target not in _all_existing_slugs(root):
        raise ProfileManagementError("Profile not found.", status_code=404, code="profile_not_found")

    caller_is_admin = is_admin_profile(caller, root)
    if caller != target and not caller_is_admin:
        raise ProfileManagementError(
            "You can only edit your own profile.",
            status_code=403,
            code="profile_access_denied",
        )
    if not isinstance(updates, dict) or not updates:
        raise ProfileManagementError("Provide at least one profile field to update.", status_code=400, code="empty_update")

    allowed = {"display_name", "preferred_language", "default_model", "badge_color", "is_admin"}
    unknown = sorted(set(updates) - allowed)
    if unknown:
        raise ProfileManagementError(
            "Unsupported profile field.",
            status_code=400,
            code="unsupported_field",
        )

    if "is_admin" in updates:
        if not caller_is_admin:
            raise ProfileManagementError(
                "Only an admin can change profile admin roles.",
                status_code=403,
                code="admin_required",
            )
        if not isinstance(updates["is_admin"], bool):
            raise ProfileManagementError("'is_admin' must be a boolean.", status_code=400, code="invalid_is_admin")

    display_name = _validate_display_name(updates["display_name"]) if "display_name" in updates else None
    preferred_language = _validate_preferred_language(updates["preferred_language"]) if "preferred_language" in updates else None
    default_model = _validate_model_name(updates["default_model"]) if "default_model" in updates else None
    badge_color = _validate_badge_color(updates["badge_color"]) if "badge_color" in updates else None

    if "is_admin" in updates:
        with _ADMIN_STATE_LOCK:
            state = _load_admin_state(root)
            admins = _effective_admins(root, state)
            if updates["is_admin"]:
                admins.add(target)
            else:
                if target in admins and len(admins) <= 1:
                    raise ProfileManagementError(
                        "Cannot demote the last remaining admin profile.",
                        status_code=409,
                        code="last_admin_protected",
                    )
                admins.discard(target)
            state["admins"] = sorted(admins)
            _save_admin_state(root, state)

    pdir = _profile_dir_for(target, root)
    if display_name is not None:
        from hermes_cli.profiles import write_profile_meta

        write_profile_meta(pdir, display_name=display_name)
    if preferred_language is not None or default_model is not None:
        _write_profile_config_fields(
            pdir,
            preferred_language=preferred_language,
            default_model=default_model,
        )
    if badge_color is not None:
        with _ADMIN_STATE_LOCK:
            state = _load_admin_state(root)
            if badge_color:
                state["badges"][target] = badge_color
            else:
                state["badges"].pop(target, None)
            _save_admin_state(root, state)

    summary = get_profile_summary(target, root_home=root)
    assert summary is not None
    return summary


def rotate_managed_profile_key(
    *,
    caller_slug: str,
    target_slug: str,
    create_invite: bool = False,
    host: str = "",
    ttl_seconds: Optional[int] = None,
    on_default_key_rotated: Optional[Callable[[str], None]] = None,
    root_home: Optional[Path] = None,
) -> dict[str, Any]:
    """Rotate a profile's API key atomically, revoking stale pairing tickets and invalidating the old key."""
    root = Path(root_home) if root_home is not None else default_root_home()
    caller = (caller_slug or "default").strip().lower()
    target = (target_slug or "").strip().lower()
    if not target or target not in _all_existing_slugs(root):
        raise ProfileManagementError("Profile not found.", status_code=404, code="profile_not_found")
    if caller != target and not is_admin_profile(caller, root):
        raise ProfileManagementError(
            "You can only rotate your own profile key.",
            status_code=403,
            code="profile_access_denied",
        )

    store = PairingStore(root)
    revoked_tickets = store.revoke_all_for_profile(target)
    new_key = generate_api_key()
    write_profile_api_key(target, new_key, root)
    if target == "default" and on_default_key_rotated is not None:
        on_default_key_rotated(new_key)

    pairing_ticket = None
    if create_invite:
        summary = get_profile_summary(target, root_home=root)
        display_name = (summary or {}).get("display_name") or target
        settings = get_admin_settings(root)
        effective_ttl = ttl_seconds if ttl_seconds is not None else int(settings["pairing_ttl_seconds"])
        pairing_ticket = store.create_ticket(
            profile_slug=target,
            display_name=display_name,
            mode="device" if caller == target else "invite",
            created_by=caller,
            ttl_seconds=effective_ttl,
            host=host,
        )

    logger.info("Rotated API_SERVER_KEY for profile=%r revoked_tickets=%d", target, revoked_tickets)
    return {
        "profile_slug": target,
        "api_key": new_key,
        "revoked_tickets": revoked_tickets,
        "pairing_ticket": pairing_ticket,
    }


def delete_managed_profile(
    *,
    caller_slug: str,
    target_slug: str,
    confirm_slug: str,
    root_home: Optional[Path] = None,
) -> dict[str, Any]:
    """Admin-only safe offboarding: refuses default/last-admin and requires explicit confirm_slug."""
    root = Path(root_home) if root_home is not None else default_root_home()
    caller = (caller_slug or "default").strip().lower()
    target = (target_slug or "").strip().lower()
    if not is_admin_profile(caller, root):
        raise ProfileManagementError(
            "Only an admin profile can delete profiles.",
            status_code=403,
            code="admin_required",
        )
    if target == "default":
        raise ProfileManagementError(
            "Cannot delete the built-in default profile.",
            status_code=409,
            code="default_profile_protected",
        )
    if not target or target not in _all_existing_slugs(root):
        raise ProfileManagementError("Profile not found.", status_code=404, code="profile_not_found")
    if (confirm_slug or "").strip().lower() != target:
        raise ProfileManagementError(
            "Explicit confirm_slug matching the target profile is required.",
            status_code=400,
            code="confirmation_mismatch",
        )
    with _ADMIN_STATE_LOCK:
        state = _load_admin_state(root)
        admins = _effective_admins(root, state)
        if target in admins and len(admins) <= 1:
            raise ProfileManagementError(
                "Cannot delete the last remaining admin profile.",
                status_code=409,
                code="last_admin_protected",
            )

    revoked_tickets = PairingStore(root).revoke_all_for_profile(target)
    from hermes_cli.profiles import delete_profile

    prev_home = os.environ.get("HERMES_HOME")
    try:
        os.environ["HERMES_HOME"] = str(root)
        delete_profile(target, yes=True)
    finally:
        if prev_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = prev_home

    with _ADMIN_STATE_LOCK:
        state = _load_admin_state(root)
        state["admins"] = sorted(s for s in state["admins"] if s != target) or ["default"]
        state["badges"].pop(target, None)
        _save_admin_state(root, state)

    logger.info("Deleted profile=%r by admin=%r revoked_tickets=%d", target, caller, revoked_tickets)
    return {"profile_slug": target, "deleted": True, "revoked_tickets": revoked_tickets}


def get_admin_settings(root_home: Optional[Path] = None) -> dict[str, Any]:
    """Return persisted server admin settings (zero secrets)."""
    root = Path(root_home) if root_home is not None else default_root_home()
    with _ADMIN_STATE_LOCK:
        state = _load_admin_state(root)
        return dict(state["settings"])


def update_admin_settings(
    *,
    caller_slug: str,
    updates: dict[str, Any],
    root_home: Optional[Path] = None,
) -> dict[str, Any]:
    """Admin-only atomic update of server-wide mobile companion admin settings."""
    root = Path(root_home) if root_home is not None else default_root_home()
    caller = (caller_slug or "default").strip().lower()
    if not is_admin_profile(caller, root):
        raise ProfileManagementError(
            "Only an admin profile can update server settings.",
            status_code=403,
            code="admin_required",
        )
    if not isinstance(updates, dict) or not updates:
        raise ProfileManagementError("Provide at least one admin setting to update.", status_code=400, code="empty_update")

    unknown = sorted(set(updates) - set(_DEFAULT_ADMIN_SETTINGS))
    if unknown:
        raise ProfileManagementError(
            "Unsupported admin setting.",
            status_code=400,
            code="unsupported_setting",
        )

    validated: dict[str, Any] = {}
    if "default_language" in updates:
        validated["default_language"] = _validate_preferred_language(updates["default_language"])
    if "default_model" in updates:
        validated["default_model"] = _validate_model_name(updates["default_model"])
    if "max_concurrent_runs" in updates:
        val = updates["max_concurrent_runs"]
        if isinstance(val, bool) or not isinstance(val, int) or not (1 <= val <= 128):
            raise ProfileManagementError(
                "'max_concurrent_runs' must be an integer between 1 and 128.",
                status_code=400,
                code="invalid_concurrency_limit",
            )
        validated["max_concurrent_runs"] = val
    if "per_profile_concurrent_limit" in updates:
        val = updates["per_profile_concurrent_limit"]
        if isinstance(val, bool) or not isinstance(val, int) or not (1 <= val <= 64):
            raise ProfileManagementError(
                "'per_profile_concurrent_limit' must be an integer between 1 and 64.",
                status_code=400,
                code="invalid_concurrency_limit",
            )
        validated["per_profile_concurrent_limit"] = val
    if "pairing_ttl_seconds" in updates:
        validated["pairing_ttl_seconds"] = validate_ttl_seconds(updates["pairing_ttl_seconds"])
    if "allow_self_device_pairing" in updates:
        val = updates["allow_self_device_pairing"]
        if not isinstance(val, bool):
            raise ProfileManagementError(
                "'allow_self_device_pairing' must be a boolean.",
                status_code=400,
                code="invalid_pairing_policy",
            )
        validated["allow_self_device_pairing"] = val

    with _ADMIN_STATE_LOCK:
        state = _load_admin_state(root)
        merged = dict(state["settings"])
        merged.update(validated)
        if merged["per_profile_concurrent_limit"] > merged["max_concurrent_runs"]:
            raise ProfileManagementError(
                "'per_profile_concurrent_limit' cannot exceed 'max_concurrent_runs'.",
                status_code=400,
                code="invalid_concurrency_limit",
            )
        state["settings"] = merged
        _save_admin_state(root, state)
        return dict(merged)


def build_localized_starters(enabled_toolsets: list[str], *, lang: str) -> list[dict[str, str]]:
    """Build real starter prompts strictly grounded in the profile's enabled toolsets."""
    norm_lang = normalize_user_lang(lang) or "en"
    ts = set(enabled_toolsets or [])
    starters: list[dict[str, str]] = [
        {
            "id": "plan_day",
            "title": t("onboarding.starter.plan_day.title", norm_lang),
            "prompt": t("onboarding.starter.plan_day.prompt", norm_lang),
            "category": "daily",
        },
    ]
    if ts & {"web", "browser"}:
        starters.append({
            "id": "web_brief",
            "title": t("onboarding.starter.web_brief.title", norm_lang),
            "prompt": t("onboarding.starter.web_brief.prompt", norm_lang),
            "category": "research",
        })
    if ts & {"file", "terminal", "code_execution"}:
        starters.append({
            "id": "workspace_review",
            "title": t("onboarding.starter.workspace_review.title", norm_lang),
            "prompt": t("onboarding.starter.workspace_review.prompt", norm_lang),
            "category": "workspace",
        })
    if "mobile_ui" in ts or "mobile_deliver" in ts:
        starters.append({
            "id": "customize_home",
            "title": t("onboarding.starter.customize_home.title", norm_lang),
            "prompt": t("onboarding.starter.customize_home.prompt", norm_lang),
            "category": "mobile",
        })
    if "cronjob" in ts or "cron" in ts:
        starters.append({
            "id": "morning_reminder",
            "title": t("onboarding.starter.morning_reminder.title", norm_lang),
            "prompt": t("onboarding.starter.morning_reminder.prompt", norm_lang),
            "category": "automation",
        })
    if len(starters) < 3:
        starters.append({
            "id": "capture_note",
            "title": t("onboarding.starter.capture_note.title", norm_lang),
            "prompt": t("onboarding.starter.capture_note.prompt", norm_lang),
            "category": "notes",
        })
    return starters


def build_onboarding_payload(
    *,
    profile_slug: str,
    lang: str = "en",
    active_model: Optional[str] = None,
    active_capabilities: int = 0,
    endpoint_count: int = 0,
    root_home: Optional[Path] = None,
) -> dict[str, Any]:
    """Assemble live first-run readiness, role, and localized starters for Hermes Mobile."""
    from agent.runtime_scope import bind_profile_scope
    from hermes_cli.tools_config import _get_platform_tools

    root = Path(root_home) if root_home is not None else default_root_home()
    slug = (profile_slug or "default").strip().lower()
    pdir = _profile_dir_for(slug, root)
    cfg = _read_profile_config(pdir)
    enabled_toolsets: list[str] = []
    with suppress(Exception), bind_profile_scope(pdir):
        enabled_toolsets = sorted(_get_platform_tools(cfg, "api_server"))
    summary = get_profile_summary(slug, root_home=root) or {
        "slug": slug,
        "display_name": slug,
        "language": "auto",
        "preferred_language": "auto",
        "default_model": active_model,
        "badge_color": "forest",
        "is_admin": slug == "default",
        "role": "admin" if slug == "default" else "member",
    }
    effective_model = active_model or summary.get("default_model") or ""
    norm_lang = normalize_user_lang(lang) or normalize_user_lang(summary.get("preferred_language")) or "en"
    is_admin = bool(summary.get("is_admin", False))
    admin_settings = get_admin_settings(root)
    return {
        "ready": bool(pdir.is_dir()),
        "active_model": effective_model,
        "active_capabilities": int(active_capabilities),
        "endpoint_count": int(endpoint_count),
        "language": norm_lang,
        "is_admin": is_admin,
        "role": "admin" if is_admin else "member",
        "profiles_manageable": is_admin,
        "allow_self_device_pairing": bool(admin_settings.get("allow_self_device_pairing", True)),
        "enabled_toolsets": enabled_toolsets,
        "profile": summary,
        "starters": build_localized_starters(enabled_toolsets, lang=norm_lang),
        "greeting": t("onboarding.greeting", norm_lang, name=summary.get("display_name") or slug),
    }
