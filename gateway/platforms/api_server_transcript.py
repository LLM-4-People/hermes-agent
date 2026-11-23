"""The thin client transcript page and its logical thread metadata."""

import asyncio
from typing import Optional

from aiohttp import web
from persistence.conversation.messages import TranscriptRead, TranscriptView
from persistence.core.identity import MessageId


def _parse_optional_cursor(raw: Optional[str]) -> tuple[Optional[int], bool]:
    if raw is None:
        return None, True
    if not isinstance(raw, str) or not raw or not raw.isascii() or not raw.isdigit():
        return None, False
    return int(raw), True


def _read(store, session_id, page_request, visible_only, before_id=None, after_id=None):
    for _ in range(3):
        with store.messages.read_snapshot() as consistent:
            sid = store.lineage.resolve_resume_target(session_id)
            page = store.messages.read_transcript_page(
                sid,
                page_request,
                visible_only=visible_only,
                before_id=None if before_id is None else MessageId(before_id),
                after_id=None if after_id is None else MessageId(after_id),
            )
            title = store.sessions.get_title(sid)
            if not (title or "").strip() and page.root_id != sid:
                title = store.sessions.get_title(page.root_id)
            if consistent():
                return sid, title if (title or "").strip() else None, page
    return None


def _public_message(adapter, row):
    message = adapter._message_response(row.as_wire_row())
    if row.api_admission_id is not None and message["role"] == "user" and message.get("display_kind") != "hidden":
        message["api_admission_id"] = row.api_admission_id
    return message


async def session_messages(adapter, request):
    from gateway.platforms.api_server import _coerce_request_bool, _error_response

    session_id = request.match_info["session_id"]
    _, error = await adapter._get_existing_session_or_404(session_id)
    if error:
        return error
    store = adapter._store_beside(await adapter._ensure_store_async())
    raw_limit, raw_offset = request.query.get("limit"), request.query.get("offset", "0")
    order = request.query.get("order")
    if order not in (None, "oldest", "latest"):
        return _error_response("order must be one of: oldest, latest", 400, code="invalid_pagination")
    try:
        offset = int(raw_offset)
        limit = min(int(raw_limit), 500) if raw_limit is not None else 500
    except (TypeError, ValueError):
        offset = limit = -1
    if offset < 0 or limit < 0:
        return _error_response("limit and offset must be non-negative integers", 400, code="invalid_pagination")

    before_id, before_ok = _parse_optional_cursor(request.query.get("before_id"))
    after_id, after_ok = _parse_optional_cursor(request.query.get("after_id"))
    if not before_ok or not after_ok:
        return _error_response("before_id and after_id must be non-negative integers", 400, code="invalid_pagination")
    if before_id is not None and after_id is not None:
        return _error_response("before_id and after_id are mutually exclusive", 400, code="invalid_pagination")
    if (before_id is not None or after_id is not None) and offset > 0:
        return _error_response("before_id and after_id cannot be combined with a non-zero offset", 400, code="invalid_pagination")

    if order is None:
        if before_id is not None:
            order = "latest"
        elif after_id is not None:
            order = "oldest"
        else:
            order = "latest" if raw_limit is None else "oldest"

    compacted = _coerce_request_bool(request.query.get("include_compacted"), default=False)
    visible_only = _coerce_request_bool(request.query.get("visible_only"), default=False)
    page_request = TranscriptRead(
        view=TranscriptView.DISPLAY if compacted else TranscriptView.ACTIVE,
        limit=limit,
        offset=offset,
        latest=order == "latest",
        include_lineage=True,
    )
    result = await asyncio.to_thread(_read, store, session_id, page_request, visible_only, before_id, after_id)
    if result is None:
        return _error_response("Session changed during transcript read; retry", 503, code="transcript_changed")
    sid, title, page = result
    has_more = page.has_more if page.has_more is not None else ((offset + len(page.rows)) < page.total)
    oldest_id = int(page.rows[0].id) if page.rows else None
    newest_id = int(page.rows[-1].id) if page.rows else None
    return web.json_response({
        "object": "list",
        "session_id": sid,
        "thread_id": page.root_id,
        "thread_title": title,
        "data": [_public_message(adapter, row) for row in page.rows],
        "pagination": {
            "limit": limit,
            "offset": offset,
            "order": order,
            "returned": len(page.rows),
            "total": page.total,
            "visible_only": visible_only,
            "has_more": bool(has_more),
            "oldest_id": oldest_id,
            "newest_id": newest_id,
            "before_id": before_id,
            "after_id": after_id,
        },
    })
