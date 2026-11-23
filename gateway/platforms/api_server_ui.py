"""Authenticated, profile-scoped HTTP routes for Mobile Live UI Modding & Deliveries.

Registered through ``APIServerAdapter._http_route_table()`` so every route automatically
mirrors under ``/p/{profile}/...`` and runs inside ``_profile_scope(profile)``.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import logging
from typing import Any, Iterator

from aiohttp import web

from hermes_constants import get_hermes_home
from gateway.platforms import api_server_profiles as _api_mobile_profiles
from tools.mobile_deliver import MobileDeliveryStore
from tools.mobile_ui_patch import MobileUiStore, RevisionConflictError

logger = logging.getLogger("gateway.platforms.api_server")

_NO_STORE = {"Cache-Control": "no-store"}

MOBILE_FEATURE_FLAGS: dict[str, Any] = {
    "live_ui_modding": True,
    "mobile_deliveries": True,
    "device_bridge": True,
    "run_progress_hooks": True,
    "app_presence_v2": True,
    "interactive_cards": True,
    "deep_android_bridge": True,
    "cursor_pagination": True,
    "bilingual_i18n": True,
    "frictionless_onboarding": True,
    "profile_management": True,
    "admin_settings": True,
}


@contextmanager
def optional_user_lang_scope(headers: Any = None, *, explicit_lang: str | None = None) -> Iterator[str | None]:
    """Bind ``gateway.i18n.user_lang_scope`` when available; no-op fallback if not yet installed."""
    try:
        from gateway.i18n import resolve_request_lang, user_lang_scope
    except ImportError:
        yield None
        return
    lang = resolve_request_lang(headers, explicit_lang=explicit_lang)
    with user_lang_scope(lang) as active:
        yield active


@web.middleware
async def language_scope_middleware(request: web.Request, handler):
    """Bind per-request user language from ``X-Hermes-Lang`` / ``Accept-Language`` for downstream tools."""
    with optional_user_lang_scope(request.headers):
        return await handler(request)


def instructions_with_app_presence(
    run_instructions: str | None, app_presence: dict[str, Any] | None
) -> str | None:
    """Append bounded ``<mobile_app_presence>`` context onto explicit per-run instructions when present."""
    from tools.mobile_app_presence import format_app_presence_context

    presence_ctx = format_app_presence_context(app_presence) if app_presence else ""
    parts = [p for p in (run_instructions, presence_ctx) if isinstance(p, str) and p.strip()]
    return "\n\n".join(parts) or None


def sanitize_optional_app_presence(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Validate optional ``app_presence`` payload on ``POST /v1/runs`` fail-closed."""
    if raw is None:
        return None, None
    if not isinstance(raw, dict):
        return None, "app_presence must be a JSON object."
    import json
    from tools.mobile_app_presence import MAX_PRESENCE_BYTES, sanitize_app_presence

    try:
        encoded = json.dumps(raw, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError):
        return None, "app_presence must be JSON-serializable."
    if len(encoded) > MAX_PRESENCE_BYTES:
        return None, f"app_presence exceeds maximum size of {MAX_PRESENCE_BYTES} bytes."
    return sanitize_app_presence(raw), None

MOBILE_CAPABILITY_ENDPOINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("run_device_result", ("POST", "/v1/runs/{run_id}/device-result", "device_bridge")),
    ("mobile_ui_state", ("GET", "/api/mobile/ui/state", "live_ui_modding")),
    ("mobile_ui_history", ("GET", "/api/mobile/ui/history", "live_ui_modding")),
    ("mobile_ui_apply", ("POST", "/api/mobile/ui/revisions/{revision_id}/apply", "live_ui_modding")),
    ("mobile_ui_revert", ("POST", "/api/mobile/ui/revisions/{revision_id}/revert", "live_ui_modding")),
    ("mobile_deliveries", ("GET", "/api/mobile/deliveries", "mobile_deliveries")),
    ("mobile_delivery", ("GET", "/api/mobile/deliveries/{delivery_id}", "mobile_deliveries")),
    ("mobile_delivery_ack", ("POST", "/api/mobile/deliveries/{delivery_id}/ack", "mobile_deliveries")),
    ("mobile_pairing_create", ("POST", "/api/mobile/pairing/create", "frictionless_onboarding")),
    ("mobile_pairing_inspect", ("POST", "/api/mobile/pairing/inspect", "frictionless_onboarding")),
    ("mobile_pairing_claim", ("POST", "/api/mobile/pairing/claim", "frictionless_onboarding")),
    ("mobile_pairing_active", ("GET", "/api/mobile/pairing/active", "frictionless_onboarding")),
    ("mobile_pairing_revoke", ("DELETE", "/api/mobile/pairing/{ticket_id}", "frictionless_onboarding")),
    ("mobile_onboarding", ("GET", "/api/mobile/onboarding", "frictionless_onboarding")),
    ("mobile_profiles_list", ("GET", "/api/mobile/profiles", "profile_management")),
    ("mobile_profiles_create", ("POST", "/api/mobile/profiles", "profile_management")),
    ("mobile_profile_update", ("PATCH", "/api/mobile/profiles/{slug}", "profile_management")),
    ("mobile_profile_invite", ("POST", "/api/mobile/profiles/{slug}/invite", "profile_management")),
    ("mobile_profile_rotate_key", ("POST", "/api/mobile/profiles/{slug}/rotate-key", "profile_management")),
    ("mobile_profile_delete", ("DELETE", "/api/mobile/profiles/{slug}", "profile_management")),
    ("mobile_admin_settings_get", ("GET", "/api/mobile/admin/settings", "admin_settings")),
    ("mobile_admin_settings_patch", ("PATCH", "/api/mobile/admin/settings", "admin_settings")),
)


def _json_no_store(payload: dict[str, Any], *, status: int = 200) -> web.Response:
    return web.json_response(payload, status=status, headers=_NO_STORE)


def _error_no_store(_openai_error, message: str, *, code: str, status: int) -> web.Response:
    return web.json_response(_openai_error(message, code=code), status=status, headers=_NO_STORE)


def _parse_bool_query(raw: str | None, *, default: bool = False) -> bool:
    if raw is None or not raw.strip():
        return default
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean query value: {raw!r}")


def _parse_int_query(raw: str | None, *, default: int, minimum: int, maximum: int, name: str) -> int:
    if raw is None or not raw.strip():
        return default
    try:
        val = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"'{name}' must be an integer.") from exc
    if not (minimum <= val <= maximum):
        raise ValueError(f"'{name}' must be between {minimum} and {maximum}.")
    return val


def _parse_float_query(raw: str | None, *, name: str) -> float | None:
    if raw is None or not raw.strip():
        return None
    try:
        val = float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"'{name}' must be a number.") from exc
    if val < 0 or val != val:  # reject negative or NaN
        raise ValueError(f"'{name}' must be a non-negative number.")
    return val


def _http_routes(adapter) -> list[tuple[str, str, Any]]:
    from gateway.platforms import api_server as _api_server

    def _auth_fail(request: web.Request) -> web.Response | None:
        refusal = adapter._check_auth(request)
        if refusal is not None:
            refusal.headers["Cache-Control"] = "no-store"
        return refusal

    async def handle_ui_state(request: web.Request) -> web.Response:
        if (err := _auth_fail(request)) is not None:
            return err
        session_id = (request.query.get("session_id") or "").strip() or None
        profile_home = get_hermes_home()
        state = await asyncio.to_thread(lambda: MobileUiStore(profile_home).get_state(session_id=session_id))
        return _json_no_store({"object": "hermes.mobile.ui.state", **state})

    async def handle_ui_history(request: web.Request) -> web.Response:
        if (err := _auth_fail(request)) is not None:
            return err
        session_id = (request.query.get("session_id") or "").strip() or None
        try:
            limit = _parse_int_query(request.query.get("limit"), default=20, minimum=1, maximum=100, name="limit")
        except ValueError as exc:
            return _error_no_store(_api_server._openai_error, str(exc), code="invalid_query", status=400)
        profile_home = get_hermes_home()
        items = await asyncio.to_thread(
            lambda: MobileUiStore(profile_home).list_history(session_id=session_id, limit=limit)
        )
        return _json_no_store({"object": "list", "data": items})

    async def handle_ui_apply(request: web.Request) -> web.Response:
        if (err := _auth_fail(request)) is not None:
            return err
        revision_id = request.match_info["revision_id"].strip()
        profile_home = get_hermes_home()
        try:
            outcome = await asyncio.to_thread(lambda: MobileUiStore(profile_home).apply_revision(revision_id))
        except ValueError as exc:
            return _error_no_store(_api_server._openai_error, str(exc), code="invalid_ui_patch", status=400)
        if outcome is None:
            return _error_no_store(
                _api_server._openai_error, f"UI revision not found: {revision_id}", code="revision_not_found", status=404
            )
        return _json_no_store({"object": "hermes.mobile.ui.apply", **outcome})

    async def handle_ui_revert(request: web.Request) -> web.Response:
        if (err := _auth_fail(request)) is not None:
            return err
        revision_id = request.match_info["revision_id"].strip()
        profile_home = get_hermes_home()
        try:
            outcome = await asyncio.to_thread(lambda: MobileUiStore(profile_home).revert_revision(revision_id))
        except RevisionConflictError as exc:
            return _error_no_store(_api_server._openai_error, str(exc), code="revision_not_active", status=409)
        if outcome is None:
            return _error_no_store(
                _api_server._openai_error, f"UI revision not found: {revision_id}", code="revision_not_found", status=404
            )
        return _json_no_store({"object": "hermes.mobile.ui.revert", **outcome})

    async def handle_list_deliveries(request: web.Request) -> web.Response:
        if (err := _auth_fail(request)) is not None:
            return err
        try:
            since = _parse_float_query(request.query.get("since"), name="since")
            include_acked = _parse_bool_query(request.query.get("include_acked"), default=False)
            limit = _parse_int_query(request.query.get("limit"), default=50, minimum=1, maximum=100, name="limit")
        except ValueError as exc:
            return _error_no_store(_api_server._openai_error, str(exc), code="invalid_query", status=400)
        profile_home = get_hermes_home()
        items = await asyncio.to_thread(
            lambda: MobileDeliveryStore(profile_home).list_deliveries(
                since=since, include_acked=include_acked, limit=limit
            )
        )
        return _json_no_store({"object": "list", "data": items})

    async def handle_get_delivery(request: web.Request) -> web.Response:
        if (err := _auth_fail(request)) is not None:
            return err
        delivery_id = request.match_info["delivery_id"].strip()
        profile_home = get_hermes_home()
        item = await asyncio.to_thread(lambda: MobileDeliveryStore(profile_home).get_delivery(delivery_id))
        if item is None:
            return _error_no_store(
                _api_server._openai_error, f"Delivery not found: {delivery_id}", code="delivery_not_found", status=404
            )
        return _json_no_store({"object": "hermes.mobile.delivery", "delivery": item})

    async def handle_ack_delivery(request: web.Request) -> web.Response:
        if (err := _auth_fail(request)) is not None:
            return err
        delivery_id = request.match_info["delivery_id"].strip()
        profile_home = get_hermes_home()
        item = await asyncio.to_thread(lambda: MobileDeliveryStore(profile_home).ack_delivery(delivery_id))
        if item is None:
            return _error_no_store(
                _api_server._openai_error, f"Delivery not found: {delivery_id}", code="delivery_not_found", status=404
            )
        return _json_no_store({"object": "hermes.mobile.delivery.ack", "delivery": item, "acked": True})

    return [
        ("GET", "/api/mobile/ui/state", handle_ui_state),
        ("GET", "/api/mobile/ui/history", handle_ui_history),
        ("POST", "/api/mobile/ui/revisions/{revision_id}/apply", handle_ui_apply),
        ("POST", "/api/mobile/ui/revisions/{revision_id}/revert", handle_ui_revert),
        ("GET", "/api/mobile/deliveries", handle_list_deliveries),
        ("GET", "/api/mobile/deliveries/{delivery_id}", handle_get_delivery),
        ("POST", "/api/mobile/deliveries/{delivery_id}/ack", handle_ack_delivery),
        *_api_mobile_profiles._http_routes(adapter),
    ]
