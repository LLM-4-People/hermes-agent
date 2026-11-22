"""Authenticated, profile-scoped HTTP routes for Mobile Live UI Modding & Deliveries.

Registered through ``APIServerAdapter._http_route_table()`` so every route automatically
mirrors under ``/p/{profile}/...`` and runs inside ``_profile_scope(profile)``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from aiohttp import web

from hermes_constants import get_hermes_home
from tools.mobile_deliver import MobileDeliveryStore
from tools.mobile_ui_patch import MobileUiStore, RevisionConflictError

logger = logging.getLogger("gateway.platforms.api_server")

_NO_STORE = {"Cache-Control": "no-store"}


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
    ]
