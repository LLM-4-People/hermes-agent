"""Universal Resource Sharing REST API routes for ``APIServerAdapter``.

Thin HTTP adapter over ``tools.mobile_sharing`` (Single Source of Truth ACL engine).
Supports ``note``, ``surface``, ``widget``, ``tool``, ``skill``, ``artifact``, and
``session`` under ``/api/shares/*`` (automatically mirrored under ``/p/{profile}/...``).
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]

from tools.mobile_sharing import (
    ShareError,
    grant_share,
    list_incoming_shares,
    list_outgoing_shares,
    read_incoming_share_content,
    revoke_or_dismiss_share,
)

logger = logging.getLogger("gateway.platforms.api_server")

CAPABILITY_FEATURES: dict[str, Any] = {
    "universal_sharing": True,
}

CAPABILITY_ENDPOINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("shares", ("POST", "/api/shares", "universal_sharing")),
    ("shares_outgoing", ("GET", "/api/shares/outgoing", "universal_sharing")),
    ("shares_incoming", ("GET", "/api/shares/incoming", "universal_sharing")),
    ("share_incoming_content", ("GET", "/api/shares/incoming/{share_id}/content", "universal_sharing")),
    ("share_delete", ("DELETE", "/api/shares/{share_id}", "universal_sharing")),
)


def _share_error_response(exc: ShareError) -> web.Response:
    from gateway.platforms.api_server import _error_response

    return _error_response(exc.message, exc.status_code, code=exc.code)


async def handle_create_share(adapter, request: web.Request) -> web.Response:
    """POST /api/shares — owner-only explicit resource grant to another profile on this server."""
    if auth_err := adapter._check_auth(request):
        return auth_err
    body, err = await adapter._read_json_body(request)
    if err is not None:
        return err
    try:
        share = await asyncio.to_thread(
            grant_share,
            body.get("resource_kind", ""),
            body.get("resource_id", ""),
            body.get("target_profile", ""),
            body.get("permission", "read"),
            label=body.get("label", ""),
        )
    except ShareError as exc:
        return _share_error_response(exc)
    return web.json_response({"object": "hermes.share", "share": share}, status=201)


async def handle_list_outgoing_shares(adapter, request: web.Request) -> web.Response:
    """GET /api/shares/outgoing — resources shared by the active profile."""
    if auth_err := adapter._check_auth(request):
        return auth_err
    kind = (request.query.get("resource_kind") or "").strip() or None
    res_id = (request.query.get("resource_id") or "").strip() or None
    try:
        items = await asyncio.to_thread(list_outgoing_shares, resource_kind=kind, resource_id=res_id)
    except ShareError as exc:
        return _share_error_response(exc)
    return web.json_response({"object": "list", "data": items})


async def handle_list_incoming_shares(adapter, request: web.Request) -> web.Response:
    """GET /api/shares/incoming — resources explicitly shared with the active profile."""
    if auth_err := adapter._check_auth(request):
        return auth_err
    kind = (request.query.get("resource_kind") or "").strip() or None
    try:
        items = await asyncio.to_thread(list_incoming_shares, resource_kind=kind)
    except ShareError as exc:
        return _share_error_response(exc)
    return web.json_response({"object": "list", "data": items})


async def handle_get_incoming_share_content(adapter, request: web.Request) -> web.Response:
    """GET /api/shares/incoming/{share_id}/content — authorized snapshot respecting instant revoke."""
    if auth_err := adapter._check_auth(request):
        return auth_err
    share_id = request.match_info["share_id"]
    try:
        payload = await asyncio.to_thread(read_incoming_share_content, share_id)
    except ShareError as exc:
        return _share_error_response(exc)
    return web.json_response({"object": "hermes.share.content", **payload})


async def handle_delete_share(adapter, request: web.Request) -> web.Response:
    """DELETE /api/shares/{share_id} — owner instant revoke OR recipient dismiss."""
    if auth_err := adapter._check_auth(request):
        return auth_err
    share_id = request.match_info["share_id"]
    try:
        result = await asyncio.to_thread(revoke_or_dismiss_share, share_id)
    except ShareError as exc:
        return _share_error_response(exc)
    return web.json_response({"object": "hermes.share.deleted", **result})


def _http_routes(adapter) -> list[tuple]:
    return [
        ("POST", "/api/shares", functools.partial(handle_create_share, adapter)),
        ("GET", "/api/shares/outgoing", functools.partial(handle_list_outgoing_shares, adapter)),
        ("GET", "/api/shares/incoming/{share_id}/content", functools.partial(handle_get_incoming_share_content, adapter)),
        ("GET", "/api/shares/incoming", functools.partial(handle_list_incoming_shares, adapter)),
        ("DELETE", "/api/shares/{share_id}", functools.partial(handle_delete_share, adapter)),
    ]
