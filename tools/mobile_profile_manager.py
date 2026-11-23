"""LLM-operable profile and pairing manager tool (``manage_profiles``) for Hermes Mobile.

Enables natural requests such as "Pair my tablet" or "Create a French profile for Alex
and give me an invite link" with strict profile isolation, fail-closed admin enforcement,
explicit approval gates on privileged mutations, and ready-to-render ``pairing_invite``
interactive cards.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from gateway.i18n import current_user_lang, normalize_user_lang, t
from gateway.mobile_pairing import PairingError, PairingStore, normalize_public_host
from gateway.mobile_profiles import (
    ProfileManagementError,
    build_onboarding_payload,
    create_managed_profile,
    default_root_home,
    get_admin_settings,
    get_profile_summary,
    is_admin_profile,
    list_managed_profiles,
    rotate_managed_profile_key,
    update_managed_profile,
)
from tools.mobile_sharing import active_profile_name
from tools.registry import registry

logger = logging.getLogger("hermes.mobile_profile_manager")

ALLOWED_ACTIONS = frozenset({
    "list_profiles",
    "get_onboarding_status",
    "update_my_profile",
    "create_device_pairing_link",
    "create_user_profile_and_invite",
    "rotate_profile_key",
})

MANAGE_PROFILES_SCHEMA: dict[str, Any] = {
    "name": "manage_profiles",
    "description": (
        "Inspect or manage Hermes Mobile profiles and ephemeral pairing links for the active user. "
        "Members can check onboarding status, update their own profile, pair another device, or "
        "rotate their own key (with explicit approval). Admins can also list all profiles and "
        "create new isolated user profiles with single-use invite links (with explicit approval)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": sorted(ALLOWED_ACTIONS),
                "description": (
                    "Action to perform: list_profiles, get_onboarding_status, update_my_profile, "
                    "create_device_pairing_link, create_user_profile_and_invite, or rotate_profile_key."
                ),
            },
            "slug": {
                "type": "string",
                "description": "Target profile slug (lowercase letters, digits, '-' or '_').",
            },
            "display_name": {
                "type": "string",
                "description": "Human-friendly profile name (max 64 characters).",
            },
            "preferred_language": {
                "type": "string",
                "enum": ["auto", "en", "fr"],
                "description": "Preferred profile language (auto, en, or fr).",
            },
            "default_model": {
                "type": "string",
                "description": "Optional default model identifier for the profile.",
            },
            "badge_color": {
                "type": "string",
                "description": "Preset (forest, slate, terracotta, indigo, plum) or #RRGGBB hex.",
            },
            "is_admin": {
                "type": "boolean",
                "description": "Whether a newly created profile should have admin privileges (default false).",
            },
            "ttl_seconds": {
                "type": "integer",
                "description": "Pairing link expiry in seconds (30 to 900, default 300).",
            },
            "host": {
                "type": "string",
                "description": "Optional public HTTPS server URL for the pairing deep link.",
            },
            "confirmed": {
                "type": "boolean",
                "description": (
                    "Must be explicitly set to true after user approval for privileged actions "
                    "(create_user_profile_and_invite and rotate_profile_key)."
                ),
            },
        },
        "required": ["action"],
    },
}


def _resolve_host_hint(raw_host: Optional[str], root: Path) -> str:
    candidate = (raw_host or "").strip()
    if not candidate:
        from gateway.mobile_profiles import _read_profile_config

        cfg = _read_profile_config(root)
        gw = cfg.get("gateway") if isinstance(cfg.get("gateway"), dict) else {}
        api_cfg = gw.get("api_server") if isinstance(gw.get("api_server"), dict) else {}
        candidate = str(api_cfg.get("public_url") or "").strip()
    if not candidate:
        return ""
    try:
        return normalize_public_host(candidate)
    except PairingError:
        return ""


def _build_pairing_card(ticket: dict[str, Any], *, lang: str) -> dict[str, Any]:
    """Build a ready-to-render ``pairing_invite`` interactive card for Hermes Mobile."""
    mode = ticket.get("mode", "device")
    title_key = "pairing.card.device.title" if mode == "device" else "pairing.card.invite.title"
    sub_key = "pairing.card.device.subtitle" if mode == "device" else "pairing.card.invite.subtitle"
    ttl_minutes = max(1, round(int(ticket.get("ttl_seconds", 300)) / 60))
    return {
        "id": ticket["ticket_id"],
        "kind": "pairing_invite",
        "title": t(title_key, lang, name=ticket.get("display_name") or ticket.get("profile_slug")),
        "subtitle": t(sub_key, lang, minutes=ttl_minutes),
        "pairing": {
            "ticket_id": ticket["ticket_id"],
            "deep_link": ticket["deep_link"],
            "profile_slug": ticket["profile_slug"],
            "display_name": ticket["display_name"],
            "mode": mode,
            "expires_at": ticket["expires_at"],
            "ttl_seconds": ticket["ttl_seconds"],
        },
        "share_label": t("pairing.card.share_label", lang),
        "copy_label": t("pairing.card.copy_label", lang),
    }


def _json_response(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _handle_update_self(caller: str, args: dict[str, Any], root: Path) -> dict[str, Any]:
    updates = {
        key: args[key]
        for key in ("display_name", "preferred_language", "default_model", "badge_color")
        if key in args and args[key] is not None
    }
    if not updates:
        return {"ok": False, "error": "Provide at least one profile field to update.", "code": "empty_update"}
    updated = update_managed_profile(caller_slug=caller, target_slug=caller, updates=updates, root_home=root)
    return {"ok": True, "profile": updated}


def _handle_pair_device(caller: str, args: dict[str, Any], root: Path, lang: str) -> dict[str, Any]:
    caller_admin = is_admin_profile(caller, root)
    admin_cfg = get_admin_settings(root)
    if not caller_admin and not admin_cfg.get("allow_self_device_pairing", True):
        return {
            "ok": False,
            "error": "Device pairing is disabled by your server administrator.",
            "code": "self_pairing_disabled",
        }
    summary = get_profile_summary(caller, root_home=root) or {"display_name": caller}
    ttl = args.get("ttl_seconds", admin_cfg.get("pairing_ttl_seconds"))
    ticket = PairingStore(root).create_ticket(
        profile_slug=caller,
        display_name=str(args.get("display_name") or summary.get("display_name") or caller),
        mode="device",
        created_by=caller,
        ttl_seconds=ttl,
        host=_resolve_host_hint(args.get("host"), root),
    )
    return {
        "ok": True,
        "pairing_ticket": ticket,
        "interactive_card": _build_pairing_card(ticket, lang=lang),
    }


def _handle_create_and_invite(caller: str, args: dict[str, Any], root: Path, lang: str) -> dict[str, Any]:
    if not is_admin_profile(caller, root):
        return {
            "ok": False,
            "error": "Only an admin profile can create new user profiles.",
            "code": "admin_required",
        }
    slug = str(args.get("slug") or "").strip().lower()
    if not slug:
        return {"ok": False, "error": "'slug' is required to create a user profile.", "code": "invalid_profile_slug"}
    if args.get("confirmed") is not True:
        return {
            "ok": False,
            "requires_approval": True,
            "code": "approval_required",
            "approval_prompt": t("pairing.approval.create_user", lang, slug=slug),
            "pending_action": {
                "action": "create_user_profile_and_invite",
                "slug": slug,
                "display_name": str(args.get("display_name") or slug),
                "preferred_language": str(args.get("preferred_language") or "auto"),
                "is_admin": bool(args.get("is_admin", False)),
            },
        }
    created = create_managed_profile(
        caller_slug=caller,
        slug=slug,
        display_name=str(args.get("display_name") or slug),
        preferred_language=str(args.get("preferred_language") or "auto"),
        default_model=args.get("default_model"),
        badge_color=args.get("badge_color"),
        is_admin=bool(args.get("is_admin", False)),
        create_invite=True,
        host=_resolve_host_hint(args.get("host"), root),
        ttl_seconds=args.get("ttl_seconds"),
        root_home=root,
    )
    ticket = created["pairing_ticket"]
    return {
        "ok": True,
        "profile": created["profile"],
        "pairing_ticket": ticket,
        "interactive_card": _build_pairing_card(ticket, lang=lang),
    }


def _handle_rotate_key(caller: str, args: dict[str, Any], root: Path, lang: str) -> dict[str, Any]:
    target = str(args.get("slug") or caller).strip().lower()
    if target != caller and not is_admin_profile(caller, root):
        return {
            "ok": False,
            "error": "You can only rotate your own profile key.",
            "code": "profile_access_denied",
        }
    if args.get("confirmed") is not True:
        return {
            "ok": False,
            "requires_approval": True,
            "code": "approval_required",
            "approval_prompt": t("pairing.approval.rotate_key", lang, slug=target),
            "pending_action": {"action": "rotate_profile_key", "slug": target},
        }
    rotated = rotate_managed_profile_key(
        caller_slug=caller,
        target_slug=target,
        create_invite=True,
        host=_resolve_host_hint(args.get("host"), root),
        ttl_seconds=args.get("ttl_seconds"),
        root_home=root,
    )
    ticket = rotated["pairing_ticket"]
    return {
        "ok": True,
        "profile_slug": rotated["profile_slug"],
        "revoked_tickets": rotated["revoked_tickets"],
        "pairing_ticket": ticket,
        "interactive_card": _build_pairing_card(ticket, lang=lang) if ticket else None,
    }


def manage_profiles_tool(
    action: str = "",
    *,
    caller_profile: Optional[str] = None,
    root_home: Optional[Path] = None,
    **kwargs: Any,
) -> str:
    """Execute a permission-gated ``manage_profiles`` tool action and return structured JSON."""
    clean_action = str(action or "").strip().lower()
    if clean_action not in ALLOWED_ACTIONS:
        return _json_response({
            "ok": False,
            "error": f"Unsupported action '{action}'.",
            "code": "invalid_action",
            "hint": f"Use one of: {', '.join(sorted(ALLOWED_ACTIONS))}.",
        })

    caller = (caller_profile or active_profile_name() or "default").strip().lower()
    root = Path(root_home) if root_home is not None else default_root_home()
    lang = normalize_user_lang(current_user_lang()) or "en"

    try:
        if clean_action == "list_profiles":
            profiles = list_managed_profiles(caller, root_home=root)
            return _json_response({
                "ok": True,
                "caller_profile": caller,
                "is_admin": is_admin_profile(caller, root),
                "profiles": profiles,
            })
        if clean_action == "get_onboarding_status":
            status = build_onboarding_payload(profile_slug=caller, lang=lang, root_home=root)
            return _json_response({"ok": True, "onboarding": status})
        if clean_action == "update_my_profile":
            return _json_response(_handle_update_self(caller, kwargs, root))
        if clean_action == "create_device_pairing_link":
            return _json_response(_handle_pair_device(caller, kwargs, root, lang))
        if clean_action == "create_user_profile_and_invite":
            return _json_response(_handle_create_and_invite(caller, kwargs, root, lang))
        return _json_response(_handle_rotate_key(caller, kwargs, root, lang))
    except (ProfileManagementError, PairingError) as exc:
        logger.warning("manage_profiles action=%s rejected: %s", clean_action, exc.code)
        return _json_response({"ok": False, "error": exc.message, "code": exc.code})


registry.register(
    name="manage_profiles",
    toolset="mobile_profiles",
    schema=MANAGE_PROFILES_SCHEMA,
    handler=lambda args, **kw: manage_profiles_tool(
        action=args.get("action", ""),
        slug=args.get("slug"),
        display_name=args.get("display_name"),
        preferred_language=args.get("preferred_language"),
        default_model=args.get("default_model"),
        badge_color=args.get("badge_color"),
        is_admin=args.get("is_admin", False),
        ttl_seconds=args.get("ttl_seconds"),
        host=args.get("host"),
        confirmed=args.get("confirmed", False),
    ),
)
