"""HTTP routes for frictionless mobile onboarding, pairing tickets, profile CRUD, and admin settings.

Registered through ``APIServerAdapter._http_route_table()`` so every route automatically
mirrors under ``/p/{profile}/...``. Every response sets ``Cache-Control: no-store`` and
never logs secret ticket codes or API keys.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any

from aiohttp import web

from gateway.i18n import resolve_request_lang
from gateway.mobile_pairing import PairingError, PairingStore, normalize_public_host
from gateway.mobile_profiles import (
    ProfileManagementError,
    build_onboarding_payload,
    create_managed_profile,
    default_root_home,
    delete_managed_profile,
    get_admin_settings,
    get_profile_summary,
    is_admin_profile,
    list_managed_profiles,
    read_profile_api_key,
    rotate_managed_profile_key,
    update_admin_settings,
    update_managed_profile,
)
from gateway.platforms.api_server_auth_guard import (
    auth_rate_limited_response,
    extract_client_ip,
)

logger = logging.getLogger("gateway.platforms.api_server")

_NO_STORE = {"Cache-Control": "no-store"}
_PAIRING_CLAIM_BUCKET = "__pairing_claim__"


def _json_no_store(payload: dict[str, Any], *, status: int = 200) -> web.Response:
    return web.json_response(payload, status=status, headers=_NO_STORE)


def _error_no_store(message: str, *, code: str, status: int) -> web.Response:
    from gateway.platforms.api_server import _openai_error

    return web.json_response(_openai_error(message, code=code), status=status, headers=_NO_STORE)


def _auth_fail(adapter, request: web.Request) -> web.Response | None:
    refusal = adapter._check_auth(request)
    if refusal is not None:
        refusal.headers["Cache-Control"] = "no-store"
    return refusal


async def _read_optional_json(adapter, request: web.Request) -> tuple[dict[str, Any], web.Response | None]:
    raw = await request.read()
    if not raw or not raw.strip():
        return {}, None
    return await adapter._read_json_body(request)


def _caller_profile_slug() -> str:
    from gateway.platforms.api_server import _api_request_profile

    return (_api_request_profile.get() or "default").strip().lower()


def _active_runs_for(adapter, slug: str) -> int:
    try:
        return int(adapter.active_profile_agent_work_count(slug))
    except (AttributeError, TypeError, ValueError):
        return 0


def _infer_public_host(request: web.Request, body: dict[str, Any] | None = None) -> str:
    raw = (body or {}).get("host") if isinstance(body, dict) else None
    if isinstance(raw, str) and raw.strip():
        return normalize_public_host(raw)
    forwarded_proto = (request.headers.get("X-Forwarded-Proto") or "").split(",", 1)[0].strip().lower()
    forwarded_host = (request.headers.get("X-Forwarded-Host") or "").split(",", 1)[0].strip()
    scheme = forwarded_proto if forwarded_proto in {"http", "https"} else (request.scheme or "http")
    host = forwarded_host or request.host or ""
    if not host:
        return ""
    return normalize_public_host(f"{scheme}://{host}")


def _capability_counts(adapter) -> tuple[int, int]:
    from gateway.platforms import api_server as _api

    try:
        features = {
            "chat_completions": True,
            "chat_completions_streaming": True,
            "responses_api": True,
            "responses_streaming": True,
            "run_submission": True,
            **_api._STATIC_FEATURE_FLAGS,
            **_api._api_runs._MOBILE_FEATURE_FLAGS,
            **_api._api_memory.CAPABILITY_FEATURES,
            **_api._api_skills.CAPABILITY_FEATURES,
            **_api._api_workspace.CAPABILITY_FEATURES,
            **_api._api_profile.CAPABILITY_FEATURES,
            **_api._api_notes.CAPABILITY_FEATURES,
            **_api._api_sharing.CAPABILITY_FEATURES,
            **_api._api_habits.CAPABILITY_FEATURES,
        }
        endpoints = (
            *_api._CAPABILITY_ENDPOINTS,
            *_api._api_runs._MOBILE_CAPABILITY_ENDPOINTS,
            *_api._api_memory.CAPABILITY_ENDPOINTS,
            *_api._api_skills.CAPABILITY_ENDPOINTS,
            *_api._api_workspace.CAPABILITY_ENDPOINTS,
            *_api._api_profile.CAPABILITY_ENDPOINTS,
            *_api._api_notes.CAPABILITY_ENDPOINTS,
            *_api._api_sharing.CAPABILITY_ENDPOINTS,
            *_api._api_habits.CAPABILITY_ENDPOINTS,
        )
        return sum(1 for v in features.values() if bool(v)), len(endpoints)
    except (AttributeError, TypeError):
        return 0, 0


async def _mint_pairing_for_target(
    adapter, request: web.Request, *, target_override: str | None = None
) -> web.Response:
    if (err := _auth_fail(adapter, request)) is not None:
        return err
    body, parse_err = await _read_optional_json(adapter, request)
    if parse_err is not None:
        return parse_err
    caller = _caller_profile_slug()
    root = default_root_home()
    caller_admin = await asyncio.to_thread(is_admin_profile, caller, root)
    target = (target_override if target_override is not None else str(body.get("profile_slug") or caller)).strip().lower()
    if target != caller and not caller_admin:
        return _error_no_store(
            "Only an admin can create pairing tickets for another profile.",
            code="admin_required",
            status=403,
        )
    admin_cfg = await asyncio.to_thread(get_admin_settings, root)
    if not caller_admin and not admin_cfg.get("allow_self_device_pairing", True):
        return _error_no_store(
            "Device pairing is disabled by your server administrator.",
            code="self_pairing_disabled",
            status=403,
        )
    summary = await asyncio.to_thread(get_profile_summary, target, root_home=root)
    if summary is None:
        return _error_no_store("Profile not found.", code="profile_not_found", status=404)
    mode = body.get("mode") or ("device" if target == caller else "invite")
    display_name = str(body.get("display_name") or summary.get("display_name") or target)
    ttl = body.get("ttl_seconds", admin_cfg.get("pairing_ttl_seconds"))
    try:
        host = _infer_public_host(request, body)
        ticket = await asyncio.to_thread(
            lambda: PairingStore(root).create_ticket(
                profile_slug=target,
                display_name=display_name,
                mode=mode,
                created_by=caller,
                ttl_seconds=ttl,
                host=host,
            )
        )
    except PairingError as exc:
        return _error_no_store(exc.message, code=exc.code, status=exc.status_code)
    return _json_no_store({"object": "hermes.mobile.pairing.ticket", **ticket}, status=201)


async def handle_pairing_create(adapter, request: web.Request) -> web.Response:
    return await _mint_pairing_for_target(adapter, request)


async def handle_invite_profile(adapter, request: web.Request) -> web.Response:
    target = request.match_info["slug"].strip().lower()
    return await _mint_pairing_for_target(adapter, request, target_override=target)


async def handle_pairing_claim(adapter, request: web.Request) -> web.Response:
    limiter = getattr(adapter, "_auth_limiter", None)
    client_ip = extract_client_ip(request) if limiter is not None else ""
    if limiter is not None:
        retry_after = limiter.check_locked_out(client_ip, _PAIRING_CLAIM_BUCKET)
        if retry_after is not None:
            return auth_rate_limited_response(retry_after)
    body, parse_err = await adapter._read_json_body(request)
    if parse_err is not None:
        if limiter is not None:
            limiter.record_failure(client_ip, _PAIRING_CLAIM_BUCKET)
        return parse_err
    root = default_root_home()
    try:
        claimed = await asyncio.to_thread(lambda: PairingStore(root).claim_ticket(body.get("code")))
    except PairingError as exc:
        if limiter is not None:
            limiter.record_failure(client_ip, _PAIRING_CLAIM_BUCKET)
        return _error_no_store(exc.message, code=exc.code, status=exc.status_code)

    if limiter is not None:
        limiter.record_success(client_ip, _PAIRING_CLAIM_BUCKET)

    slug = claimed["profile_slug"]
    api_key = await asyncio.to_thread(read_profile_api_key, slug, root)
    if not api_key and slug == "default":
        api_key = getattr(adapter, "_api_key", "") or ""
    if not api_key:
        return _error_no_store(
            "Target profile has no active API key configured.",
            code="profile_key_unavailable",
            status=503,
        )
    base_host = _infer_public_host(request, body)
    server_url = f"{base_host}/p/{slug}" if (base_host and slug != "default") else base_host
    lang = resolve_request_lang(request.headers)
    active_caps, endpoint_count = _capability_counts(adapter)
    onboarding = await asyncio.to_thread(
        lambda: build_onboarding_payload(
            profile_slug=slug,
            lang=lang,
            active_model=getattr(adapter, "_model_name", None),
            active_capabilities=active_caps,
            endpoint_count=endpoint_count,
            root_home=root,
        )
    )
    profile_info = onboarding.get("profile") or {}
    return _json_no_store({
        "object": "hermes.mobile.pairing.claim",
        "server_url": server_url,
        "base_host": base_host,
        "profile_slug": slug,
        "display_name": claimed.get("display_name") or profile_info.get("display_name") or slug,
        "api_key": api_key,
        "preferred_language": profile_info.get("preferred_language", "auto"),
        "default_model": profile_info.get("default_model") or getattr(adapter, "_model_name", ""),
        "is_admin": bool(onboarding.get("is_admin", False)),
        "role": onboarding.get("role", "member"),
        "capabilities_summary": {
            "active_capabilities": active_caps,
            "endpoint_count": endpoint_count,
            "frictionless_onboarding": True,
            "profile_management": True,
        },
        "onboarding": onboarding,
    })


async def handle_pairing_active(adapter, request: web.Request) -> web.Response:
    if (err := _auth_fail(adapter, request)) is not None:
        return err
    caller = _caller_profile_slug()
    root = default_root_home()
    caller_admin = await asyncio.to_thread(is_admin_profile, caller, root)
    all_flag = (request.query.get("all") or "").strip().lower() in {"1", "true", "yes"}
    scope_slug = None if (caller_admin and all_flag) else caller
    items = await asyncio.to_thread(lambda: PairingStore(root).list_active_tickets(profile_slug=scope_slug))
    return _json_no_store({"object": "list", "data": items})


async def handle_pairing_revoke(adapter, request: web.Request) -> web.Response:
    if (err := _auth_fail(adapter, request)) is not None:
        return err
    ticket_id = request.match_info["ticket_id"].strip()
    caller = _caller_profile_slug()
    root = default_root_home()
    caller_admin = await asyncio.to_thread(is_admin_profile, caller, root)
    try:
        result = await asyncio.to_thread(
            lambda: PairingStore(root).revoke_ticket(
                ticket_id, allowed_profile=caller, is_admin=caller_admin
            )
        )
    except PairingError as exc:
        return _error_no_store(exc.message, code=exc.code, status=exc.status_code)
    return _json_no_store({"object": "hermes.mobile.pairing.revoked", **result})


async def handle_list_profiles(adapter, request: web.Request) -> web.Response:
    if (err := _auth_fail(adapter, request)) is not None:
        return err
    caller = _caller_profile_slug()
    root = default_root_home()
    caller_admin = await asyncio.to_thread(is_admin_profile, caller, root)
    items = await asyncio.to_thread(
        lambda: list_managed_profiles(
            caller, active_runs_fn=lambda s: _active_runs_for(adapter, s), root_home=root
        )
    )
    return _json_no_store({
        "object": "list",
        "caller_profile": caller,
        "is_admin": caller_admin,
        "data": items,
    })


async def handle_create_profile(adapter, request: web.Request) -> web.Response:
    if (err := _auth_fail(adapter, request)) is not None:
        return err
    body, parse_err = await adapter._read_json_body(request)
    if parse_err is not None:
        return parse_err
    caller = _caller_profile_slug()
    root = default_root_home()
    try:
        host = _infer_public_host(request, body)
        created = await asyncio.to_thread(
            lambda: create_managed_profile(
                caller_slug=caller,
                slug=body.get("slug", ""),
                display_name=body.get("display_name", ""),
                preferred_language=body.get("preferred_language", "auto"),
                default_model=body.get("default_model"),
                badge_color=body.get("badge_color"),
                is_admin=bool(body.get("is_admin", False)),
                create_invite=bool(body.get("create_invite", False)),
                host=host,
                ttl_seconds=body.get("ttl_seconds"),
                root_home=root,
            )
        )
    except (ProfileManagementError, PairingError) as exc:
        return _error_no_store(exc.message, code=exc.code, status=exc.status_code)
    return _json_no_store({"object": "hermes.mobile.profile.created", **created}, status=201)


async def handle_update_profile(adapter, request: web.Request) -> web.Response:
    if (err := _auth_fail(adapter, request)) is not None:
        return err
    body, parse_err = await adapter._read_json_body(request)
    if parse_err is not None:
        return parse_err
    caller = _caller_profile_slug()
    target = request.match_info["slug"].strip().lower()
    root = default_root_home()
    try:
        updated = await asyncio.to_thread(
            lambda: update_managed_profile(
                caller_slug=caller, target_slug=target, updates=body, root_home=root
            )
        )
    except ProfileManagementError as exc:
        return _error_no_store(exc.message, code=exc.code, status=exc.status_code)
    return _json_no_store({"object": "hermes.mobile.profile", "profile": updated})


async def handle_rotate_profile_key(adapter, request: web.Request) -> web.Response:
    if (err := _auth_fail(adapter, request)) is not None:
        return err
    body, parse_err = await _read_optional_json(adapter, request)
    if parse_err is not None:
        return parse_err
    caller = _caller_profile_slug()
    target = request.match_info["slug"].strip().lower()
    root = default_root_home()

    def _set_default_key(new_key: str) -> None:
        adapter._api_key = new_key

    try:
        host = _infer_public_host(request, body)
        rotated = await asyncio.to_thread(
            lambda: rotate_managed_profile_key(
                caller_slug=caller,
                target_slug=target,
                create_invite=bool(body.get("create_invite", False)),
                host=host,
                ttl_seconds=body.get("ttl_seconds"),
                on_default_key_rotated=_set_default_key,
                root_home=root,
            )
        )
    except (ProfileManagementError, PairingError) as exc:
        return _error_no_store(exc.message, code=exc.code, status=exc.status_code)
    return _json_no_store({"object": "hermes.mobile.profile.key_rotated", **rotated})


async def handle_delete_profile(adapter, request: web.Request) -> web.Response:
    if (err := _auth_fail(adapter, request)) is not None:
        return err
    body, parse_err = await _read_optional_json(adapter, request)
    if parse_err is not None:
        return parse_err
    caller = _caller_profile_slug()
    target = request.match_info["slug"].strip().lower()
    confirm_slug = str(body.get("confirm_slug") or request.query.get("confirm_slug") or "")
    root = default_root_home()
    try:
        removed = await asyncio.to_thread(
            lambda: delete_managed_profile(
                caller_slug=caller, target_slug=target, confirm_slug=confirm_slug, root_home=root
            )
        )
    except ProfileManagementError as exc:
        return _error_no_store(exc.message, code=exc.code, status=exc.status_code)
    return _json_no_store({"object": "hermes.mobile.profile.deleted", **removed})


async def handle_onboarding(adapter, request: web.Request) -> web.Response:
    if (err := _auth_fail(adapter, request)) is not None:
        return err
    caller = _caller_profile_slug()
    root = default_root_home()
    lang = resolve_request_lang(request.headers)
    active_caps, endpoint_count = _capability_counts(adapter)
    payload = await asyncio.to_thread(
        lambda: build_onboarding_payload(
            profile_slug=caller,
            lang=lang,
            active_model=getattr(adapter, "_model_name", None),
            active_capabilities=active_caps,
            endpoint_count=endpoint_count,
            root_home=root,
        )
    )
    return _json_no_store({"object": "hermes.mobile.onboarding", **payload})


async def handle_get_admin_settings(adapter, request: web.Request) -> web.Response:
    if (err := _auth_fail(adapter, request)) is not None:
        return err
    caller = _caller_profile_slug()
    root = default_root_home()
    if not await asyncio.to_thread(is_admin_profile, caller, root):
        return _error_no_store(
            "Only an admin profile can access server admin settings.",
            code="admin_required",
            status=403,
        )
    settings = await asyncio.to_thread(get_admin_settings, root)
    profiles = await asyncio.to_thread(
        lambda: list_managed_profiles(
            caller, active_runs_fn=lambda s: _active_runs_for(adapter, s), root_home=root
        )
    )
    tickets = await asyncio.to_thread(lambda: PairingStore(root).list_active_tickets(profile_slug=None))
    active_caps, endpoint_count = _capability_counts(adapter)
    try:
        active_runs = int(adapter.active_agent_work_count())
    except (AttributeError, TypeError, ValueError):
        active_runs = sum(int(p.get("active_runs", 0)) for p in profiles)
    return _json_no_store({
        "object": "hermes.mobile.admin.settings",
        "settings": settings,
        "summary": {
            "profile_count": len(profiles),
            "active_runs_count": active_runs,
            "active_pairing_tickets": len(tickets),
            "active_capabilities": active_caps,
            "endpoint_count": endpoint_count,
        },
    })


async def handle_patch_admin_settings(adapter, request: web.Request) -> web.Response:
    if (err := _auth_fail(adapter, request)) is not None:
        return err
    body, parse_err = await adapter._read_json_body(request)
    if parse_err is not None:
        return parse_err
    caller = _caller_profile_slug()
    root = default_root_home()
    try:
        updated = await asyncio.to_thread(
            lambda: update_admin_settings(caller_slug=caller, updates=body, root_home=root)
        )
    except (ProfileManagementError, PairingError) as exc:
        return _error_no_store(exc.message, code=exc.code, status=exc.status_code)
    if "max_concurrent_runs" in updated:
        adapter._max_concurrent_runs = int(updated["max_concurrent_runs"])
    if "per_profile_concurrent_limit" in updated:
        adapter._max_concurrent_runs_per_profile = int(updated["per_profile_concurrent_limit"])
    return _json_no_store({"object": "hermes.mobile.admin.settings", "settings": updated})


def _http_routes(adapter) -> list[tuple[str, str, Any]]:
    return [
        ("POST", "/api/mobile/pairing/create", functools.partial(handle_pairing_create, adapter)),
        ("POST", "/api/mobile/pairing/claim", functools.partial(handle_pairing_claim, adapter)),
        ("GET", "/api/mobile/pairing/active", functools.partial(handle_pairing_active, adapter)),
        ("DELETE", "/api/mobile/pairing/{ticket_id}", functools.partial(handle_pairing_revoke, adapter)),
        ("GET", "/api/mobile/onboarding", functools.partial(handle_onboarding, adapter)),
        ("GET", "/api/mobile/profiles", functools.partial(handle_list_profiles, adapter)),
        ("POST", "/api/mobile/profiles", functools.partial(handle_create_profile, adapter)),
        ("PATCH", "/api/mobile/profiles/{slug}", functools.partial(handle_update_profile, adapter)),
        ("POST", "/api/mobile/profiles/{slug}/invite", functools.partial(handle_invite_profile, adapter)),
        ("POST", "/api/mobile/profiles/{slug}/rotate-key", functools.partial(handle_rotate_profile_key, adapter)),
        ("DELETE", "/api/mobile/profiles/{slug}", functools.partial(handle_delete_profile, adapter)),
        ("GET", "/api/mobile/admin/settings", functools.partial(handle_get_admin_settings, adapter)),
        ("PATCH", "/api/mobile/admin/settings", functools.partial(handle_patch_admin_settings, adapter)),
    ]
