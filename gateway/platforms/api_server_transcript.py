"""The thin client transcript page and its logical thread metadata."""

import asyncio

from aiohttp import web
from persistence.conversation.messages import TranscriptRead, TranscriptView


def _read(store, session_id, page_request, visible_only):
    for _ in range(3):
        with store.messages.read_snapshot() as consistent:
            sid = store.lineage.resolve_resume_target(session_id)
            page = store.messages.read_transcript_page(sid, page_request, visible_only=visible_only)
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
    order = order or ("latest" if raw_limit is None else "oldest")
    compacted = _coerce_request_bool(request.query.get("include_compacted"), default=False)
    visible_only = _coerce_request_bool(request.query.get("visible_only"), default=False)
    page_request = TranscriptRead(view=TranscriptView.DISPLAY if compacted else TranscriptView.ACTIVE,
                                  limit=limit, offset=offset, latest=order == "latest", include_lineage=True)
    result = await asyncio.to_thread(_read, store, session_id, page_request, visible_only)
    if result is None:
        return _error_response("Session changed during transcript read; retry", 503, code="transcript_changed")
    sid, title, page = result
    return web.json_response({
        "object": "list", "session_id": sid, "thread_id": page.root_id, "thread_title": title,
        "data": [_public_message(adapter, row) for row in page.rows],
        "pagination": {"limit": limit, "offset": offset, "order": order, "returned": len(page.rows),
                       "total": page.total, "visible_only": visible_only},
    })
