"""Smart Notes & Explicit Same-Server Sharing REST API routes for ``APIServerAdapter``.

Thin HTTP adapter over ``tools.mobile_notes`` (Single Source of Truth). Every
route runs inside ``_profile_scope(profile)`` with ``_check_auth`` and
``Cache-Control: no-store``.
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

from tools.mobile_notes import (
    NotesError,
    create_note,
    delete_note,
    get_note,
    list_notes,
    list_shared_with_me,
    share_note,
    unshare_note,
    update_note,
)

logger = logging.getLogger("gateway.platforms.api_server")

CAPABILITY_FEATURES: dict[str, Any] = {
    "smart_notes": True,
    "note_sharing": True,
}

CAPABILITY_ENDPOINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("notes", ("GET", "/api/notes", "smart_notes")),
    ("note_create", ("POST", "/api/notes", "smart_notes")),
    ("notes_shared_with_me", ("GET", "/api/notes/shared-with-me", "note_sharing")),
    ("note_detail", ("GET", "/api/notes/{note_id}", "smart_notes")),
    ("note_update", ("PATCH", "/api/notes/{note_id}", "smart_notes")),
    ("note_delete", ("DELETE", "/api/notes/{note_id}", "smart_notes")),
    ("note_share", ("POST", "/api/notes/{note_id}/shares", "note_sharing")),
    ("note_unshare", ("DELETE", "/api/notes/{note_id}/shares/{target_profile}", "note_sharing")),
)


def _notes_error_response(exc: NotesError) -> web.Response:
    from gateway.platforms.api_server import _error_response

    return _error_response(exc.message, exc.status_code, code=exc.code)


async def handle_list_notes(adapter, request: web.Request) -> web.Response:
    """GET /api/notes — list/search the active profile's own notes."""
    from gateway.platforms.api_server import _coerce_request_bool, _error_response

    if auth_err := adapter._check_auth(request):
        return auth_err
    query = (request.query.get("q") or "").strip() or None
    tag = (request.query.get("tag") or "").strip() or None
    pinned_only = _coerce_request_bool(request.query.get("pinned"), default=False)
    include_archived = _coerce_request_bool(request.query.get("include_archived"), default=False)
    try:
        limit = int(request.query.get("limit", "50"))
        offset = int(request.query.get("offset", "0"))
    except ValueError:
        return _error_response("limit and offset must be integers.", 400, code="invalid_pagination")
    if limit < 1 or limit > 200 or offset < 0:
        return _error_response("limit must be 1..200 and offset >= 0.", 400, code="invalid_pagination")
    try:
        items = await asyncio.to_thread(
            list_notes,
            query=query,
            tag=tag,
            pinned_only=pinned_only,
            include_archived=include_archived,
            limit=limit,
            offset=offset,
        )
    except NotesError as exc:
        return _notes_error_response(exc)
    return web.json_response({"object": "list", "data": items, "limit": limit, "offset": offset})


async def handle_list_shared_with_me(adapter, request: web.Request) -> web.Response:
    """GET /api/notes/shared-with-me — notes explicitly shared with the active profile."""
    from gateway.platforms.api_server import _error_response

    if auth_err := adapter._check_auth(request):
        return auth_err
    try:
        limit = int(request.query.get("limit", "50"))
    except ValueError:
        return _error_response("limit must be an integer.", 400, code="invalid_pagination")
    if limit < 1 or limit > 200:
        return _error_response("limit must be 1..200.", 400, code="invalid_pagination")
    items = await asyncio.to_thread(list_shared_with_me, limit=limit)
    return web.json_response({"object": "list", "data": items, "limit": limit})


async def handle_create_note(adapter, request: web.Request) -> web.Response:
    """POST /api/notes — create a note in the active profile."""
    if auth_err := adapter._check_auth(request):
        return auth_err
    body, err = await adapter._read_json_body(request)
    if err is not None:
        return err
    try:
        note = await asyncio.to_thread(
            create_note,
            title=body.get("title"),
            content=body.get("content", ""),
            tags=body.get("tags"),
            pinned=body.get("pinned", False),
            note_id=body.get("id") or body.get("note_id"),
        )
    except NotesError as exc:
        return _notes_error_response(exc)
    return web.json_response({"object": "hermes.note", "note": note}, status=201)


async def handle_get_note(adapter, request: web.Request) -> web.Response:
    """GET /api/notes/{note_id} — read an owned or explicitly shared note."""
    if auth_err := adapter._check_auth(request):
        return auth_err
    note_id = request.match_info["note_id"]
    owner_hint = (request.query.get("owner_profile") or "").strip() or None
    try:
        note = await asyncio.to_thread(get_note, note_id, owner_profile=owner_hint)
    except NotesError as exc:
        return _notes_error_response(exc)
    return web.json_response({"object": "hermes.note", "note": note})


async def handle_update_note(adapter, request: web.Request) -> web.Response:
    """PATCH /api/notes/{note_id} — edit an owned note or an edit-shared note."""
    from gateway.platforms.api_server import _error_response

    if auth_err := adapter._check_auth(request):
        return auth_err
    note_id = request.match_info["note_id"]
    body, err = await adapter._read_json_body(request)
    if err is not None:
        return err
    allowed = {"title", "content", "append", "tags", "pinned", "archived", "owner_profile"}
    unknown = sorted(set(body) - allowed)
    if unknown:
        return _error_response(f"Unsupported note field(s): {', '.join(unknown)}.", 400, code="unsupported_note_field")
    if not any(k in body for k in ("title", "content", "append", "tags", "pinned", "archived")):
        return _error_response("Provide at least one field to update.", 400, code="empty_note_update")
    try:
        note = await asyncio.to_thread(
            update_note,
            note_id,
            title=body.get("title"),
            content=body.get("content"),
            append=body.get("append"),
            tags=body.get("tags"),
            has_tags="tags" in body,
            pinned=body.get("pinned"),
            archived=body.get("archived"),
            owner_profile=(str(body.get("owner_profile") or "").strip() or None),
        )
    except NotesError as exc:
        return _notes_error_response(exc)
    return web.json_response({"object": "hermes.note", "note": note})


async def handle_delete_note(adapter, request: web.Request) -> web.Response:
    """DELETE /api/notes/{note_id} — delete an owned note (owner only)."""
    if auth_err := adapter._check_auth(request):
        return auth_err
    note_id = request.match_info["note_id"]
    try:
        result = await asyncio.to_thread(delete_note, note_id)
    except NotesError as exc:
        return _notes_error_response(exc)
    return web.json_response({"object": "hermes.note.deleted", **result})


async def handle_share_note(adapter, request: web.Request) -> web.Response:
    """POST /api/notes/{note_id}/shares — explicitly grant another profile read/edit access."""
    if auth_err := adapter._check_auth(request):
        return auth_err
    note_id = request.match_info["note_id"]
    body, err = await adapter._read_json_body(request)
    if err is not None:
        return err
    try:
        result = await asyncio.to_thread(
            share_note,
            note_id,
            body.get("target_profile", ""),
            body.get("permission", "read"),
        )
    except NotesError as exc:
        return _notes_error_response(exc)
    return web.json_response({"object": "hermes.note.share", **result}, status=201)


async def handle_unshare_note(adapter, request: web.Request) -> web.Response:
    """DELETE /api/notes/{note_id}/shares/{target_profile} — immediately revoke access."""
    if auth_err := adapter._check_auth(request):
        return auth_err
    note_id = request.match_info["note_id"]
    target_profile = request.match_info["target_profile"]
    try:
        result = await asyncio.to_thread(unshare_note, note_id, target_profile)
    except NotesError as exc:
        return _notes_error_response(exc)
    return web.json_response({"object": "hermes.note.share.revoked", **result})


def _http_routes(adapter) -> list[tuple]:
    return [
        ("GET", "/api/notes/shared-with-me", functools.partial(handle_list_shared_with_me, adapter)),
        ("GET", "/api/notes", functools.partial(handle_list_notes, adapter)),
        ("POST", "/api/notes", functools.partial(handle_create_note, adapter)),
        ("POST", "/api/notes/{note_id}/shares", functools.partial(handle_share_note, adapter)),
        ("DELETE", "/api/notes/{note_id}/shares/{target_profile}", functools.partial(handle_unshare_note, adapter)),
        ("GET", "/api/notes/{note_id}", functools.partial(handle_get_note, adapter)),
        ("PATCH", "/api/notes/{note_id}", functools.partial(handle_update_note, adapter)),
        ("DELETE", "/api/notes/{note_id}", functools.partial(handle_delete_note, adapter)),
    ]
