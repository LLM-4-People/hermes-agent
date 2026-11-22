"""Hermes Brain API routes: Memory CRUD, Cross-Thread FTS5 Search, Session Fork & Export.

All handlers run inside the active profile's ``_profile_scope`` and reuse
``tools.memory_tool`` / ``MemoryStore``, ``persistence`` FTS5 search, and
``api_server_transcript`` as their single sources of truth.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime
import functools
import hashlib
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Optional
import uuid

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]

from agent.redact import redact_sensitive_text
from persistence.conversation.messages import NewMessage, ReplaceMode, TranscriptRead, TranscriptView
from persistence.conversation.search import SearchSort
from persistence.conversation.sessions import NewSession, SessionLineageKind
from tools.memory_tool import get_memory_dir, load_on_disk_store
from tools.memory_tool_store import ENTRY_DELIMITER, MemoryStore, _scan_memory_content
from utils import atomic_write_text

logger = logging.getLogger("gateway.platforms.api_server")

_VALID_MEMORY_TARGETS = frozenset({"memory", "user"})
_VALID_SEARCH_ROLES = frozenset({"user", "assistant", "tool"})
_VALID_SEARCH_SORTS = {
    "relevance": SearchSort.RELEVANCE,
    "newest": SearchSort.NEWEST,
    "oldest": SearchSort.OLDEST,
}
_MAX_SEARCH_QUERY_CHARS = 500
_MAX_SEARCH_LIMIT = 50
_PINS_FILENAME = ".pins.json"
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

CAPABILITY_FEATURES: dict[str, Any] = {
    "memory_api": True,
    "session_search": True,
    "session_fork_at_turn": True,
    "session_export": True,
}

CAPABILITY_ENDPOINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("memory", ("GET", "/api/memory", "memory_api")),
    ("memory_add", ("POST", "/api/memory/{target}", "memory_api")),
    ("memory_update", ("PATCH", "/api/memory/{target}/{entry_id}", "memory_api")),
    ("memory_delete", ("DELETE", "/api/memory/{target}/{entry_id}", "memory_api")),
    ("session_search", ("GET", "/api/sessions/search", "session_search")),
    ("session_export", ("GET", "/api/sessions/{session_id}/export", "session_export")),
)


def memory_entry_id(text: str) -> str:
    """Deterministic SHA-256 entry ID for race-free PATCH/DELETE operations."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _validate_utf8_text(value: Any, field: str) -> Optional[str]:
    if not isinstance(value, str):
        return f"'{field}' must be a string."
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return f"'{field}' must be valid UTF-8 text."
    return None


def _pins_path() -> Path:
    return get_memory_dir() / _PINS_FILENAME


def _load_pins() -> dict[str, list[str]]:
    path = _pins_path()
    if not path.is_file():
        return {"memory": [], "user": []}
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {"memory": [], "user": []}
    if not isinstance(raw, dict):
        return {"memory": [], "user": []}
    return {
        target: [str(item) for item in (raw.get(target) or []) if isinstance(item, str)]
        for target in ("memory", "user")
    }


def _save_pins(pins: dict[str, list[str]]) -> None:
    path = _pins_path()
    from hermes_constants import mkdir_under_hermes_home

    mkdir_under_hermes_home(path.parent)
    cleaned = {target: sorted(set(pins.get(target) or [])) for target in ("memory", "user")}
    atomic_write_text(path, json.dumps(cleaned, indent=2, sort_keys=True) + "\n", preserve_mode=True, create_mode=0o600)


def _snapshot_target(store: MemoryStore, target: str, pinned_ids: set[str]) -> dict[str, Any]:
    entries = store._entries_for(target)
    return {
        "entries": [
            {"id": (eid := memory_entry_id(text)), "text": text, "pinned": eid in pinned_ids}
            for text in entries
        ],
        "chars": len(ENTRY_DELIMITER.join(entries)),
        "limit": store._char_limit(target),
        "enabled": store.target_enabled(target),
    }


def _build_memory_snapshot(store: MemoryStore, pins: Optional[dict[str, list[str]]] = None) -> dict[str, Any]:
    pin_map = pins if pins is not None else _load_pins()
    return {
        "object": "hermes.memory",
        "memory": _snapshot_target(store, "memory", set(pin_map.get("memory") or ())),
        "user": _snapshot_target(store, "user", set(pin_map.get("user") or ())),
    }


def _sync_get_memory() -> dict[str, Any]:
    store = load_on_disk_store()
    return _build_memory_snapshot(store)


def _sync_add_memory(target: str, text: str, pin: bool) -> tuple[int, dict[str, Any]]:
    cleaned = text.strip()
    if not cleaned:
        return 400, {"error": "Content cannot be empty.", "code": "empty_memory_content"}
    if threat := _scan_memory_content(cleaned):
        logger.warning("Memory API rejected threat content on %s: %s", target, threat)
        return 400, {"error": threat, "code": "memory_content_blocked"}
    store = load_on_disk_store()
    if not store.target_enabled(target):
        return 403, {"error": f"Memory target '{target}' is disabled.", "code": "memory_target_disabled"}
    result = store.add(target, cleaned)
    if not result.get("success"):
        msg = str(result.get("error") or "Failed to add memory entry.")
        status = 409 if "exceed" in msg.lower() or "budget" in msg.lower() else 400
        return status, {"error": msg, "code": "memory_add_rejected"}
    entry_id = memory_entry_id(cleaned)
    pins = _load_pins()
    if pin:
        pins.setdefault(target, [])
        if entry_id not in pins[target]:
            pins[target].append(entry_id)
        _save_pins(pins)
    snapshot = _build_memory_snapshot(store, pins)
    snapshot["target"] = target
    snapshot["entry"] = {"id": entry_id, "text": cleaned, "pinned": entry_id in set(pins.get(target) or ())}
    return 201, snapshot


def _sync_patch_memory(
    target: str,
    entry_id: str,
    *,
    new_text: Optional[str],
    pinned: Optional[bool],
) -> tuple[int, dict[str, Any]]:
    store = load_on_disk_store()
    if not store.target_enabled(target):
        return 403, {"error": f"Memory target '{target}' is disabled.", "code": "memory_target_disabled"}
    cleaned_new: Optional[str] = None
    if new_text is not None:
        cleaned_new = new_text.strip()
        if not cleaned_new:
            return 400, {"error": "Replacement text cannot be empty.", "code": "empty_memory_content"}
        if threat := _scan_memory_content(cleaned_new):
            logger.warning("Memory API PATCH rejected threat content on %s: %s", target, threat)
            return 400, {"error": threat, "code": "memory_content_blocked"}

    matched_entry = next((e for e in store._entries_for(target) if memory_entry_id(e) == entry_id), None)
    if matched_entry is None:
        return 409, {"error": f"Memory entry '{entry_id}' not found or changed.", "code": "stale_entry"}

    final_text = matched_entry
    if cleaned_new is not None and cleaned_new != matched_entry:
        result = store.replace(target, matched_entry, cleaned_new, matched_entry=matched_entry)
        if not result.get("success"):
            msg = str(result.get("error") or "Failed to update memory entry.")
            status = 409 if ("changed" in msg.lower() or "exceed" in msg.lower() or "limit" in msg.lower()) else 400
            return status, {"error": msg, "code": "memory_update_rejected"}
        final_text = cleaned_new

    new_id = memory_entry_id(final_text)
    pins = _load_pins()
    target_pins = set(pins.get(target) or ())
    was_pinned = entry_id in target_pins
    target_pins.discard(entry_id)
    final_pinned = was_pinned if pinned is None else pinned
    if final_pinned:
        target_pins.add(new_id)
    pins[target] = sorted(target_pins)
    _save_pins(pins)
    snapshot = _build_memory_snapshot(store, pins)
    snapshot["target"] = target
    snapshot["entry"] = {"id": new_id, "text": final_text, "pinned": final_pinned}
    return 200, snapshot


def _sync_delete_memory(target: str, entry_id: str) -> tuple[int, dict[str, Any]]:
    store = load_on_disk_store()
    if not store.target_enabled(target):
        return 403, {"error": f"Memory target '{target}' is disabled.", "code": "memory_target_disabled"}
    matched_entry = next((e for e in store._entries_for(target) if memory_entry_id(e) == entry_id), None)
    if matched_entry is None:
        return 404, {"error": f"Memory entry '{entry_id}' not found.", "code": "memory_entry_not_found"}
    result = store.remove(target, matched_entry, matched_entry=matched_entry)
    if not result.get("success"):
        msg = str(result.get("error") or "Failed to delete memory entry.")
        status = 409 if "changed" in msg.lower() else 400
        return status, {"error": msg, "code": "memory_delete_rejected"}
    pins = _load_pins()
    target_pins = set(pins.get(target) or ())
    if entry_id in target_pins:
        target_pins.discard(entry_id)
        pins[target] = sorted(target_pins)
        _save_pins(pins)
    snapshot = _build_memory_snapshot(store, pins)
    snapshot["target"] = target
    snapshot["entry_id"] = entry_id
    snapshot["deleted"] = True
    return 200, snapshot


def _parse_bool_flag(body: dict[str, Any], *keys: str) -> tuple[Optional[bool], Optional[str]]:
    for key in keys:
        if key in body:
            value = body[key]
            if not isinstance(value, bool):
                return None, f"'{key}' must be a boolean."
            return value, None
    return None, None


async def handle_get_memory(adapter, request: web.Request) -> web.Response:
    from gateway.platforms.api_server import _error_response

    if auth_err := adapter._check_auth(request):
        return auth_err
    try:
        payload = await asyncio.to_thread(_sync_get_memory)
    except OSError:
        logger.exception("[%s] GET /api/memory failed", adapter.name)
        return _error_response("Failed to read memory store.", 500, code="memory_read_failed")
    return web.json_response(payload)


async def handle_add_memory(adapter, request: web.Request) -> web.Response:
    from gateway.platforms.api_server import _error_response

    if auth_err := adapter._check_auth(request):
        return auth_err
    target = request.match_info["target"]
    if target not in _VALID_MEMORY_TARGETS:
        return _error_response("Invalid memory target. Use 'memory' or 'user'.", 400, code="invalid_memory_target")
    body, err = await adapter._read_json_body(request)
    if err is not None:
        return err
    if text_err := _validate_utf8_text(body.get("text"), "text"):
        return _error_response(text_err, 400, code="invalid_memory_text")
    pin_val, pin_err = _parse_bool_flag(body, "pin", "pinned")
    if pin_err is not None:
        return _error_response(pin_err, 400, code="invalid_memory_pin")
    status, payload = await asyncio.to_thread(_sync_add_memory, target, body["text"], bool(pin_val))
    if status >= 400:
        return _error_response(payload["error"], status, code=payload.get("code"))
    return web.json_response(payload, status=status)


async def handle_patch_memory(adapter, request: web.Request) -> web.Response:
    from gateway.platforms.api_server import _error_response

    if auth_err := adapter._check_auth(request):
        return auth_err
    target = request.match_info["target"]
    entry_id = request.match_info["entry_id"].strip()
    if target not in _VALID_MEMORY_TARGETS:
        return _error_response("Invalid memory target. Use 'memory' or 'user'.", 400, code="invalid_memory_target")
    if not entry_id:
        return _error_response("Memory entry_id is required.", 400, code="invalid_entry_id")
    body, err = await adapter._read_json_body(request)
    if err is not None:
        return err
    if "text" not in body and "pinned" not in body and "pin" not in body:
        return _error_response("Provide at least one of 'text' or 'pinned'.", 400, code="empty_patch")
    new_text: Optional[str] = None
    if "text" in body:
        if text_err := _validate_utf8_text(body["text"], "text"):
            return _error_response(text_err, 400, code="invalid_memory_text")
        new_text = body["text"]
    pin_val, pin_err = _parse_bool_flag(body, "pinned", "pin")
    if pin_err is not None:
        return _error_response(pin_err, 400, code="invalid_memory_pin")
    status, payload = await asyncio.to_thread(
        _sync_patch_memory, target, entry_id, new_text=new_text, pinned=pin_val
    )
    if status >= 400:
        return _error_response(payload["error"], status, code=payload.get("code"))
    return web.json_response(payload, status=status)


async def handle_delete_memory(adapter, request: web.Request) -> web.Response:
    from gateway.platforms.api_server import _error_response

    if auth_err := adapter._check_auth(request):
        return auth_err
    target = request.match_info["target"]
    entry_id = request.match_info["entry_id"].strip()
    if target not in _VALID_MEMORY_TARGETS:
        return _error_response("Invalid memory target. Use 'memory' or 'user'.", 400, code="invalid_memory_target")
    if not entry_id:
        return _error_response("Memory entry_id is required.", 400, code="invalid_entry_id")
    status, payload = await asyncio.to_thread(_sync_delete_memory, target, entry_id)
    if status >= 400:
        return _error_response(payload["error"], status, code=payload.get("code"))
    return web.json_response(payload, status=status)


# ── Cross-thread FTS5 Search ─────────────────────────────────────────────────


def _sync_search_sessions(
    store,
    query: str,
    *,
    limit: int,
    offset: int,
    role: Optional[str],
    session_id: Optional[str],
    source: str,
    sort: SearchSort,
) -> dict[str, Any]:
    source_filter = None if source == "all" else [source]
    role_filter = [role] if role else None
    fetch_limit = min(200, max(limit + offset, 50)) if session_id else limit
    fetch_offset = 0 if session_id else offset
    hits = store.search.search_messages(
        query,
        source_filter=source_filter,
        role_filter=role_filter,
        limit=fetch_limit,
        offset=fetch_offset,
        sort=sort,
    )
    if session_id:
        filtered = [h for h in hits if str(h.session_id or "") == session_id]
        total = len(filtered)
        page_hits = filtered[offset : offset + limit]
    else:
        page_hits = list(hits)
        total = len(page_hits)

    root_cache: dict[str, tuple[str, Optional[str]]] = {}
    items: list[dict[str, Any]] = []
    for hit in page_hits:
        sid = str(hit.session_id or "")
        if sid not in root_cache:
            root_id = sid
            with suppress(Exception):
                root_id = str(store.lineage.conversation_root(sid) or sid)
            title = None
            with suppress(Exception):
                title = store.sessions.get_title(sid) or (store.sessions.get_title(root_id) if root_id != sid else None)
            root_cache[sid] = (root_id, title if (title or "").strip() else None)
        thread_id, thread_title = root_cache[sid]
        snippet = redact_sensitive_text(str(hit.snippet or ""), force=True)
        items.append({
            "session_id": sid,
            "thread_id": thread_id,
            "thread_title": thread_title,
            "message_id": int(hit.message_id) if hit.message_id is not None else None,
            "role": hit.role,
            "snippet": snippet,
            "timestamp": hit.timestamp,
            "source": hit.source,
        })
    return {
        "object": "list",
        "query": query,
        "data": items,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


async def handle_search_sessions(adapter, request: web.Request) -> web.Response:
    from gateway.platforms.api_server import _error_response

    if auth_err := adapter._check_auth(request):
        return auth_err
    query = (request.query.get("q") or "").strip()
    if not query:
        return _error_response("Query parameter 'q' is required.", 400, code="missing_query")
    if len(query) > _MAX_SEARCH_QUERY_CHARS:
        return _error_response(f"Query exceeds {_MAX_SEARCH_QUERY_CHARS} characters.", 400, code="query_too_long")
    try:
        limit = int(request.query.get("limit", "20"))
        offset = int(request.query.get("offset", "0"))
    except ValueError:
        return _error_response("limit and offset must be integers.", 400, code="invalid_pagination")
    if limit < 1 or limit > _MAX_SEARCH_LIMIT or offset < 0:
        return _error_response(f"limit must be 1..{_MAX_SEARCH_LIMIT} and offset >= 0.", 400, code="invalid_pagination")
    role = (request.query.get("role") or "").strip() or None
    if role is not None and role not in _VALID_SEARCH_ROLES:
        return _error_response("role must be one of: user, assistant, tool.", 400, code="invalid_role")
    sort_name = (request.query.get("sort") or "relevance").strip().lower()
    if sort_name not in _VALID_SEARCH_SORTS:
        return _error_response("sort must be one of: relevance, newest, oldest.", 400, code="invalid_sort")
    session_id = (request.query.get("session_id") or "").strip() or None
    source = (request.query.get("source") or "api_server").strip() or "api_server"

    store = adapter._store_beside(await adapter._ensure_store_async())
    if store is None:
        return adapter._session_db_unavailable()
    try:
        payload = await asyncio.to_thread(
            _sync_search_sessions,
            store,
            query,
            limit=limit,
            offset=offset,
            role=role,
            session_id=session_id,
            source=source,
            sort=_VALID_SEARCH_SORTS[sort_name],
        )
    except Exception:
        logger.exception("[%s] GET /api/sessions/search failed", adapter.name)
        return _error_response("Session search failed.", 500, code="session_search_failed")
    return web.json_response(payload)


# ── Session Fork (Branch at Any Turn / Message ID) ──────────────────────────


def _parse_fork_slice_params(body: dict[str, Any]) -> tuple[Optional[int], Optional[int], bool, Optional[str]]:
    up_to_message_id = body.get("up_to_message_id")
    turn_index = body.get("turn_index")
    keep_source_open = body.get("keep_source_open", False)
    if not isinstance(keep_source_open, bool):
        return None, None, False, "'keep_source_open' must be a boolean."
    if up_to_message_id is not None and turn_index is not None:
        return None, None, False, "Specify either 'up_to_message_id' or 'turn_index', not both."
    if up_to_message_id is not None:
        if isinstance(up_to_message_id, bool) or not isinstance(up_to_message_id, int) or up_to_message_id < 1:
            return None, None, False, "'up_to_message_id' must be a positive integer."
    if turn_index is not None:
        if isinstance(turn_index, bool) or not isinstance(turn_index, int) or turn_index < 0:
            return None, None, False, "'turn_index' must be a non-negative integer."
    return up_to_message_id, turn_index, keep_source_open, None


def _slice_fork_rows(
    rows: list[Any],
    *,
    up_to_message_id: Optional[int],
    turn_index: Optional[int],
) -> tuple[Optional[list[Any]], Optional[tuple[int, str, str]]]:
    if up_to_message_id is not None:
        matched_index = next((idx for idx, row in enumerate(rows) if int(row.id) == up_to_message_id), None)
        if matched_index is None:
            return None, (404, "message_not_found", f"Message {up_to_message_id} was not found in this session.")
        return rows[: matched_index + 1], None
    if turn_index is not None:
        if turn_index >= len(rows):
            return None, (400, "invalid_turn_index", f"turn_index {turn_index} is out of range (0..{max(0, len(rows) - 1)}).")
        return rows[: turn_index + 1], None
    return rows, None


async def handle_fork_session(adapter, request: web.Request) -> web.Response:
    """POST /api/sessions/{session_id}/fork — branch a session at the tip or any turn/message ID."""
    from gateway.platforms.api_server import _error_response, _session_row

    if auth_err := adapter._check_auth(request):
        return auth_err
    source_id = request.match_info["session_id"]
    source, err = await adapter._get_existing_session_or_404(source_id)
    if err is not None:
        return err
    body, err = await adapter._read_json_body(request)
    if err is not None:
        return err
    up_to_message_id, turn_index, keep_source_open, slice_param_err = _parse_fork_slice_params(body)
    if slice_param_err is not None:
        return _error_response(slice_param_err, 400, code="invalid_fork_params")

    store = adapter._store_beside(await adapter._ensure_store_async())
    if store is None:
        return adapter._session_db_unavailable()
    sessions = store.aio.sessions

    fork_id = str(body.get("id") or body.get("session_id") or f"api_{int(time.time())}_{uuid.uuid4().hex[:8]}").strip()
    if not fork_id or re.search(r"[\r\n\x00]", fork_id):
        return _error_response("Invalid session ID", 400, code="invalid_session_id")
    if await sessions.get(fork_id) is not None:
        return _error_response(f"Session already exists: {fork_id}", 409, code="session_exists")

    rows = list(await store.aio.messages.read_transcript(source_id))
    selected_rows, slice_err = _slice_fork_rows(
        rows, up_to_message_id=up_to_message_id, turn_index=turn_index
    )
    if slice_err is not None:
        status, code, msg = slice_err
        return _error_response(msg, status, code=code)
    assert selected_rows is not None

    await sessions.create(NewSession(
        session_id=fork_id,
        source="api_server",
        model=source.get("model"),
        system_prompt=source.get("system_prompt"),
        parent_session_id=source_id,
        cwd=source.get("cwd"),
        lineage_kind=SessionLineageKind.BRANCH,
    ))
    if not keep_source_open:
        await sessions.end(source_id, "branched")
    await store.aio.messages.replace_transcript(
        fork_id,
        [NewMessage.from_live(row.as_wire_row()) for row in selected_rows],
        ReplaceMode.DESTRUCTIVE,
    )
    title = body.get("title")
    if title is None:
        base = source.get("title") or "fork"
        title = f"{base} fork"
        with suppress(Exception):
            title = await sessions.next_title_in_lineage(base)
    try:
        await sessions.set_title(fork_id, str(title))
    except ValueError as exc:
        return _error_response(str(exc), 400, code="invalid_title")
    fork = _session_row(f) if (f := await sessions.get(fork_id)) else {"id": fork_id, "parent_session_id": source_id}
    return web.json_response(
        {
            "object": "hermes.session",
            "session": adapter._session_response(fork),
            "forked_from": source_id,
            "copied_messages": len(selected_rows),
            "keep_source_open": keep_source_open,
        },
        status=201,
    )


# ── Session Transcript Export ────────────────────────────────────────────────


def _render_markdown_export(session: dict[str, Any], title: Optional[str], messages: list[dict[str, Any]]) -> str:
    heading = title or session.get("title") or session.get("id") or "Hermes Conversation"
    lines = [
        f"# {heading}",
        "",
        f"- **Session ID:** `{session.get('id', '')}`",
        f"- **Model:** `{session.get('model') or 'default'}`",
        f"- **Exported:** {datetime.now(UTC).isoformat()}",
        "",
        "---",
        "",
    ]
    for msg in messages:
        role = str(msg.get("role") or "unknown").capitalize()
        content = msg.get("content")
        if isinstance(content, list):
            parts = [str(item.get("text") or "") for item in content if isinstance(item, dict) and item.get("text")]
            body = "\n".join(parts)
        else:
            body = "" if content is None else str(content)
        lines.append(f"## {role}")
        lines.append("")
        lines.append(body)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


async def handle_export_session(adapter, request: web.Request) -> web.Response:
    """GET /api/sessions/{session_id}/export?format=markdown|json."""
    from gateway.platforms.api_server import _coerce_request_bool, _error_response
    from gateway.platforms.api_server_transcript import _public_message, _read

    if auth_err := adapter._check_auth(request):
        return auth_err
    session_id = request.match_info["session_id"]
    session, err = await adapter._get_existing_session_or_404(session_id)
    if err is not None:
        return err
    fmt = (request.query.get("format") or "markdown").strip().lower()
    if fmt not in ("markdown", "json"):
        return _error_response("format must be 'markdown' or 'json'.", 400, code="invalid_export_format")
    visible_only = _coerce_request_bool(request.query.get("visible_only"), default=True)
    compacted = _coerce_request_bool(request.query.get("include_compacted"), default=False)

    store = adapter._store_beside(await adapter._ensure_store_async())
    if store is None:
        return adapter._session_db_unavailable()
    page_request = TranscriptRead(
        view=TranscriptView.DISPLAY if compacted else TranscriptView.ACTIVE,
        limit=10_000,
        offset=0,
        latest=False,
        include_lineage=True,
    )
    result = await asyncio.to_thread(_read, store, session_id, page_request, visible_only)
    if result is None:
        return _error_response("Session changed during export; retry.", 503, code="transcript_changed")
    sid, title, page = result
    messages = [_public_message(adapter, row) for row in page.rows]
    safe_stem = _SAFE_FILENAME_RE.sub("-", sid).strip(".-") or "session"

    if fmt == "json":
        payload = {
            "object": "hermes.session.export",
            "session": adapter._session_response(session),
            "thread_id": page.root_id,
            "thread_title": title,
            "exported_at": datetime.now(UTC).isoformat(),
            "messages": messages,
        }
        return web.json_response(
            payload,
            headers={
                "Content-Disposition": f'attachment; filename="{safe_stem}.json"',
                "Cache-Control": "no-store",
            },
        )
    md_text = _render_markdown_export(session, title, messages)
    return web.Response(
        text=md_text,
        content_type="text/markdown",
        charset="utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_stem}.md"',
            "Cache-Control": "no-store",
        },
    )


def _http_routes(adapter) -> list[tuple]:
    return [
        ("GET", "/api/memory", functools.partial(handle_get_memory, adapter)),
        ("POST", "/api/memory/{target}", functools.partial(handle_add_memory, adapter)),
        ("PATCH", "/api/memory/{target}/{entry_id}", functools.partial(handle_patch_memory, adapter)),
        ("DELETE", "/api/memory/{target}/{entry_id}", functools.partial(handle_delete_memory, adapter)),
        ("GET", "/api/sessions/search", functools.partial(handle_search_sessions, adapter)),
        ("GET", "/api/sessions/{session_id}/export", functools.partial(handle_export_session, adapter)),
    ]
