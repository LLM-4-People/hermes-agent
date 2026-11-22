"""Workspace Explorer, Usage Analytics, Cron Run History & Session Uploads API routes.

Reuses ``hermes_cli.workspace_ops``, ``persistence`` ``usage_analytics``,
``hermes_cli.web_routers.cron`` run reconciliation, and ``hermes_constants``
``WEBAPP_ATTACHMENT_MAX_BYTES`` as single sources of truth.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import functools
import hashlib
import json
import logging
import mimetypes
import os
from pathlib import Path
import re
import secrets
import time
from typing import Any, Optional

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]

from hermes_cli.staged_uploads import _safe_filename
from hermes_cli.workspace_ops import (
    WORKSPACE_TREE_MAX_DEPTH,
    WORKSPACE_TREE_MAX_ENTRIES,
    WorkspaceError,
    list_workspace_tree,
    path_is_under,
    read_workspace_diff,
    read_workspace_file,
)
from hermes_constants import (
    WEBAPP_ATTACHMENT_MAX_BYTES,
    assert_named_profile_home_live,
    get_hermes_home,
    mkdir_under_hermes_home,
)
from persistence.conversation.sessions import SessionWindow

logger = logging.getLogger("gateway.platforms.api_server")

_MAX_ANALYTICS_DAYS = 365
_MAX_CRON_RUNS_LIMIT = 100
_UPLOAD_CHUNK_BYTES = 64 * 1024
_MAX_ATTACHMENTS_PER_RUN = 10
_UPLOAD_ID_RE = re.compile(r"^upl_[0-9a-f]{24}$")

CAPABILITY_FEATURES: dict[str, Any] = {
    "workspace_browser": True,
    "usage_analytics": True,
    "session_uploads": True,
}

CAPABILITY_ENDPOINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("workspace_tree", ("GET", "/api/sessions/{session_id}/workspace/tree", "workspace_browser")),
    ("workspace_file", ("GET", "/api/sessions/{session_id}/workspace/file", "workspace_browser")),
    ("workspace_diff", ("GET", "/api/sessions/{session_id}/workspace/diff", "workspace_browser")),
    ("analytics_usage", ("GET", "/api/analytics/usage", "usage_analytics")),
    ("job_runs", ("GET", "/api/jobs/{job_id}/runs", "cron_run_history")),
    ("session_uploads", ("POST", "/api/sessions/{session_id}/uploads", "session_uploads")),
)


def _resolve_session_cwd(session_id: str, session: dict[str, Any]) -> Optional[Path]:
    raw_cwd = str(session.get("cwd") or "").strip()
    if not raw_cwd:
        with suppress(ImportError, RuntimeError):
            from tools.terminal_tool import resolve_task_overrides

            bound = resolve_task_overrides(session_id).get("cwd")
            raw_cwd = str(bound or "").strip()
    if not raw_cwd or "\x00" in raw_cwd:
        return None
    try:
        resolved = Path(raw_cwd).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    return resolved if resolved.is_dir() else None


def _parse_int_query(request: web.Request, name: str, default: int, *, minimum: int, maximum: int) -> tuple[Optional[int], Optional[str]]:
    raw = request.query.get(name)
    if raw is None or not raw.strip():
        return default, None
    try:
        value = int(raw)
    except ValueError:
        return None, f"'{name}' must be an integer."
    if value < minimum or value > maximum:
        return None, f"'{name}' must be between {minimum} and {maximum}."
    return value, None


async def handle_workspace_tree(adapter, request: web.Request) -> web.Response:
    """GET /api/sessions/{session_id}/workspace/tree."""
    from gateway.platforms.api_server import _error_response

    if auth_err := adapter._check_auth(request):
        return auth_err
    session_id = request.match_info["session_id"]
    session, err = await adapter._get_existing_session_or_404(session_id)
    if err is not None:
        return err
    cwd = await asyncio.to_thread(_resolve_session_cwd, session_id, session)
    if cwd is None:
        return _error_response("Session has no accessible workspace directory.", 404, code="session_workspace_unavailable")
    depth, depth_err = _parse_int_query(request, "depth", 2, minimum=1, maximum=WORKSPACE_TREE_MAX_DEPTH)
    if depth_err is not None:
        return _error_response(depth_err, 400, code="invalid_depth")
    limit, limit_err = _parse_int_query(request, "limit", WORKSPACE_TREE_MAX_ENTRIES, minimum=1, maximum=WORKSPACE_TREE_MAX_ENTRIES)
    if limit_err is not None:
        return _error_response(limit_err, 400, code="invalid_limit")
    rel_path = request.query.get("path", "")
    assert depth is not None and limit is not None
    try:
        payload = await asyncio.to_thread(list_workspace_tree, cwd, rel_path, depth=depth, limit=limit)
    except WorkspaceError as exc:
        logger.warning("[%s] Workspace tree blocked for %s: %s (%s)", adapter.name, session_id, exc.code, exc.message)
        return _error_response(exc.message, exc.status_code, code=exc.code)
    return web.json_response({"object": "hermes.workspace.tree", "session_id": session_id, **payload})


async def handle_workspace_file(adapter, request: web.Request) -> web.Response:
    """GET /api/sessions/{session_id}/workspace/file."""
    from gateway.platforms.api_server import _error_response

    if auth_err := adapter._check_auth(request):
        return auth_err
    session_id = request.match_info["session_id"]
    session, err = await adapter._get_existing_session_or_404(session_id)
    if err is not None:
        return err
    cwd = await asyncio.to_thread(_resolve_session_cwd, session_id, session)
    if cwd is None:
        return _error_response("Session has no accessible workspace directory.", 404, code="session_workspace_unavailable")
    rel_path = request.query.get("path", "")
    start_line, start_err = _parse_int_query(request, "start_line", 1, minimum=1, maximum=1_000_000)
    if start_err is not None:
        return _error_response(start_err, 400, code="invalid_line_range")
    raw_end = request.query.get("end_line")
    end_line: Optional[int] = None
    if raw_end is not None and raw_end.strip():
        end_line, end_err = _parse_int_query(request, "end_line", 1, minimum=1, maximum=1_000_000)
        if end_err is not None:
            return _error_response(end_err, 400, code="invalid_line_range")
    assert start_line is not None
    try:
        payload = await asyncio.to_thread(read_workspace_file, cwd, rel_path, start_line=start_line, end_line=end_line)
    except WorkspaceError as exc:
        logger.warning("[%s] Workspace file blocked for %s: %s (%s)", adapter.name, session_id, exc.code, exc.message)
        return _error_response(exc.message, exc.status_code, code=exc.code)
    return web.json_response({"object": "hermes.workspace.file", "session_id": session_id, **payload})


async def handle_workspace_diff(adapter, request: web.Request) -> web.Response:
    """GET /api/sessions/{session_id}/workspace/diff."""
    from gateway.platforms.api_server import _error_response

    if auth_err := adapter._check_auth(request):
        return auth_err
    session_id = request.match_info["session_id"]
    session, err = await adapter._get_existing_session_or_404(session_id)
    if err is not None:
        return err
    cwd = await asyncio.to_thread(_resolve_session_cwd, session_id, session)
    if cwd is None:
        return _error_response("Session has no accessible workspace directory.", 404, code="session_workspace_unavailable")
    file_path = request.query.get("path")
    try:
        payload = await asyncio.to_thread(read_workspace_diff, cwd, file_path=file_path)
    except WorkspaceError as exc:
        logger.warning("[%s] Workspace diff blocked for %s: %s (%s)", adapter.name, session_id, exc.code, exc.message)
        return _error_response(exc.message, exc.status_code, code=exc.code)
    return web.json_response({"object": "hermes.workspace.diff", "session_id": session_id, **payload})


# ── Usage Analytics ──────────────────────────────────────────────────────────


def _sync_usage_analytics(store, *, days: int, session: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    if session is not None:
        input_tok = int(session.get("input_tokens") or 0)
        output_tok = int(session.get("output_tokens") or 0)
        cache_read = int(session.get("cache_read_tokens") or 0)
        cache_write = int(session.get("cache_write_tokens") or 0)
        reasoning = int(session.get("reasoning_tokens") or 0)
        est_cost = float(session.get("estimated_cost_usd") or 0.0)
        act_cost = float(session.get("actual_cost_usd") or 0.0)
        api_calls = int(session.get("api_call_count") or 0)
        model = session.get("model") or "unknown"
        return {
            "object": "hermes.analytics.usage",
            "period_days": days,
            "session_id": session.get("id"),
            "daily": [],
            "by_model": [{
                "model": model,
                "input_tokens": input_tok,
                "output_tokens": output_tok,
                "cache_read_tokens": cache_read,
                "cache_write_tokens": cache_write,
                "reasoning_tokens": reasoning,
                "estimated_cost": est_cost,
                "actual_cost": act_cost,
                "sessions": 1,
                "api_calls": api_calls,
            }],
            "totals": {
                "total_input": input_tok,
                "total_output": output_tok,
                "total_cache_read": cache_read,
                "total_cache_write": cache_write,
                "total_reasoning": reasoning,
                "total_estimated_cost": est_cost,
                "total_actual_cost": act_cost,
                "total_sessions": 1,
                "total_api_calls": api_calls,
            },
        }

    cutoff = time.time() - (days * 86400)
    analytics = store.usage_analytics
    daily = analytics.daily_usage(started_after=cutoff)
    by_model = analytics.session_usage_by_model(started_after=cutoff)
    with suppress(ImportError, RuntimeError):
        from hermes_cli.web_server_profiles import _aux_usage_rows, _merge_aux_into_by_model

        by_model = _merge_aux_into_by_model(by_model, _aux_usage_rows(store, cutoff))
    totals = analytics.session_usage_totals(started_after=cutoff)
    return {
        "object": "hermes.analytics.usage",
        "period_days": days,
        "session_id": None,
        "daily": daily,
        "by_model": by_model,
        "totals": totals,
    }


async def handle_usage_analytics(adapter, request: web.Request) -> web.Response:
    """GET /api/analytics/usage — profile-scoped usage & cost analytics."""
    from gateway.platforms.api_server import _error_response

    if auth_err := adapter._check_auth(request):
        return auth_err
    days, days_err = _parse_int_query(request, "days", 30, minimum=1, maximum=_MAX_ANALYTICS_DAYS)
    if days_err is not None:
        return _error_response(days_err, 400, code="invalid_days")
    session_id = (request.query.get("session_id") or "").strip() or None
    session: Optional[dict[str, Any]] = None
    if session_id is not None:
        session, err = await adapter._get_existing_session_or_404(session_id)
        if err is not None:
            return err
    store = adapter._store_beside(await adapter._ensure_store_async())
    if store is None:
        return adapter._session_db_unavailable()
    assert days is not None
    try:
        payload = await asyncio.to_thread(_sync_usage_analytics, store, days=days, session=session)
    except Exception:
        logger.exception("[%s] GET /api/analytics/usage failed", adapter.name)
        return _error_response("Failed to compute usage analytics.", 500, code="analytics_failed")
    return web.json_response(payload)


# ── Cron Run History ─────────────────────────────────────────────────────────


def _sync_job_runs(store, job: dict[str, Any], limit: int) -> dict[str, Any]:
    from hermes_cli.web_routers.cron import (
        _list_cron_output_runs,
        _live_inflight_execution,
        _reconcile_cron_runs,
        _run_owned_by,
    )

    canonical_id = str(job.get("id") or "")
    session_runs = [
        run.as_cron_row()
        for run in store.sessions.list_cron_job_runs(canonical_id, SessionWindow(limit=limit))
    ]
    now = time.time()
    inflight = _live_inflight_execution(canonical_id)
    for row in session_runs:
        row["is_active"] = row.get("ended_at") is None and (now - row.get("last_active", row.get("started_at", 0))) < 300
        row["scheduler_owned"] = _run_owned_by(row, inflight)
        row["archived"] = bool(row.get("archived"))
    docs, attempts = _list_cron_output_runs(job, canonical_id, profile=None, limit=limit)
    reconciled = _reconcile_cron_runs(session_runs, docs, attempts, canonical_id, limit)
    runs = []
    for item in reconciled:
        is_output = str(item.get("source") or "") == "cron_output"
        status = "running" if item.get("is_active") else ("completed" if item.get("ended_at") is not None else "unknown")
        title = str(item.get("title") or "")
        for prefix, mapped in (("FAILED", "failed"), ("CANCELLED", "cancelled"), ("MISSED", "missed")):
            if title.startswith(prefix):
                status = mapped
                break
        runs.append({
            "id": item.get("id"),
            "session_id": None if is_output else item.get("id"),
            "source": item.get("source"),
            "started_at": item.get("started_at"),
            "ended_at": item.get("ended_at"),
            "status": status,
            "title": title,
            "preview": item.get("preview"),
            "is_active": bool(item.get("is_active")),
            "input_tokens": int(item.get("input_tokens") or 0),
            "output_tokens": int(item.get("output_tokens") or 0),
        })
    return {"object": "list", "job_id": canonical_id, "runs": runs, "limit": limit}


async def handle_job_runs(adapter, request: web.Request) -> web.Response:
    """GET /api/jobs/{job_id}/runs — profile-scoped execution history for a cron job."""
    from gateway.platforms import api_server as facade

    job_id, err = adapter._cron_request_guard(request, need_job_id=True)
    if err is not None:
        return err
    limit, limit_err = _parse_int_query(request, "limit", 20, minimum=1, maximum=_MAX_CRON_RUNS_LIMIT)
    if limit_err is not None:
        return facade._error_response(limit_err, 400, code="invalid_limit")
    job = await asyncio.to_thread(facade._cron_get, job_id)
    if not job:
        return facade._error_response("Job not found", 404, code="job_not_found")
    store = adapter._store_beside(await adapter._ensure_store_async())
    if store is None:
        return adapter._session_db_unavailable()
    assert limit is not None
    try:
        payload = await asyncio.to_thread(_sync_job_runs, store, job, limit)
    except Exception:
        logger.exception("[%s] GET /api/jobs/%s/runs failed", adapter.name, job_id)
        return facade._error_response("Failed to load job runs.", 500, code="job_runs_failed")
    return web.json_response(payload)


# ── Multi-File Session Uploads & Run Attachment Resolution ───────────────────


def _session_upload_dir(session_id: str) -> Path:
    home = get_hermes_home()
    assert_named_profile_home_live(home)
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]
    return home / "uploads" / f"session_{digest}"


def _sniff_mime_type(sample: bytes, filename: str, declared: Optional[str]) -> str:
    if sample.startswith(b"%PDF-"):
        return "application/pdf"
    if sample.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if sample.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if sample.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(sample) >= 12 and sample[:4] == b"RIFF" and sample[8:12] == b"WEBP":
        return "image/webp"
    if sample.startswith(b"PK\x03\x04"):
        guessed, _ = mimetypes.guess_type(filename)
        return guessed or "application/zip"
    if sample.startswith(b"\x1f\x8b"):
        return "application/gzip"
    guessed, _ = mimetypes.guess_type(filename)
    if guessed:
        return guessed
    if sample:
        try:
            sample.decode("utf-8")
            return "text/plain"
        except UnicodeDecodeError:
            pass
    if declared and "/" in declared and not any(ch in declared for ch in ("\r", "\n", ";")):
        return declared.strip().lower()
    return "application/octet-stream"


def _prepare_upload_target(session_id: str, filename: str) -> tuple[str, Path, Path, str]:
    upload_dir = _session_upload_dir(session_id)
    mkdir_under_hermes_home(upload_dir)
    upload_dir.chmod(0o700)
    upload_id = f"upl_{secrets.token_hex(12)}"
    safe_name = _safe_filename(filename)
    stored_name = f"{upload_id}-{safe_name}"
    target = upload_dir / stored_name
    meta_path = upload_dir / f"{upload_id}.json"
    return upload_id, target, meta_path, safe_name


def _write_upload_metadata(meta_path: Path, record: dict[str, Any]) -> None:
    fd = os.open(meta_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())


def _open_upload_file(target: Path):
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    return os.fdopen(fd, "wb")


def _unlink_quietly(path: Path) -> None:
    with suppress(OSError):
        path.unlink(missing_ok=True)


async def _canonical_upload_session_id(adapter, session_id: str) -> str:
    """Resolve the stable lineage root for ``session_id`` so uploads survive compaction."""
    store = adapter._store_beside(await adapter._ensure_store_async())
    if store is None:
        return session_id
    try:
        root_id = await asyncio.to_thread(store.lineage.conversation_root, session_id)
        return str(root_id or session_id)
    except Exception:
        logger.debug("Lineage root lookup skipped for upload session %s", session_id, exc_info=True)
        return session_id


async def handle_session_upload(adapter, request: web.Request) -> web.Response:
    """POST /api/sessions/{session_id}/uploads — stage a profile/session-scoped artifact."""
    from gateway.platforms.api_server import _error_response

    if auth_err := adapter._check_auth(request):
        return auth_err
    session_id = request.match_info["session_id"]
    _, err = await adapter._get_existing_session_or_404(session_id)
    if err is not None:
        return err
    canonical_session_id = await _canonical_upload_session_id(adapter, session_id)

    content_type = (request.content_type or "").lower()
    if not content_type.startswith("multipart/"):
        return _error_response("Expected multipart/form-data upload.", 400, code="invalid_content_type")
    try:
        reader = await request.multipart()
    except (ValueError, KeyError, RuntimeError):
        return _error_response("Malformed multipart request.", 400, code="invalid_multipart")

    file_part = None
    while (part := await reader.next()) is not None:
        if part.filename is not None or part.name in ("file", "attachment"):
            file_part = part
            break
    if file_part is None:
        return _error_response("Missing 'file' part in multipart upload.", 400, code="missing_upload_file")

    raw_filename = file_part.filename or "attachment"
    declared_mime = file_part.headers.get("Content-Type")
    try:
        upload_id, target, meta_path, safe_name = await asyncio.to_thread(
            _prepare_upload_target, canonical_session_id, raw_filename
        )
    except FileNotFoundError:
        return _error_response("Profile home is unavailable.", 404, code="profile_unavailable")
    except OSError:
        logger.exception("[%s] Failed to prepare session upload directory", adapter.name)
        return _error_response("Could not stage upload.", 500, code="upload_stage_failed")

    hasher = hashlib.sha256()
    sample = bytearray()
    total_bytes = 0
    handle = None
    committed = False
    try:
        handle = await asyncio.to_thread(_open_upload_file, target)
        while chunk := await file_part.read_chunk(_UPLOAD_CHUNK_BYTES):
            total_bytes += len(chunk)
            if total_bytes > WEBAPP_ATTACHMENT_MAX_BYTES:
                raise web.HTTPRequestEntityTooLarge(
                    max_size=WEBAPP_ATTACHMENT_MAX_BYTES,
                    actual_size=total_bytes,
                )
            hasher.update(chunk)
            if len(sample) < 4096:
                sample.extend(chunk[: 4096 - len(sample)])
            await asyncio.to_thread(handle.write, chunk)
        await asyncio.to_thread(handle.flush)
        await asyncio.to_thread(os.fsync, handle.fileno())
        await asyncio.to_thread(handle.close)
        handle = None
        if total_bytes == 0:
            return _error_response("Uploaded file cannot be empty.", 400, code="empty_upload")
        mime_type = _sniff_mime_type(bytes(sample), safe_name, declared_mime)
        sha256_hex = hasher.hexdigest()
        record = {
            "upload_id": upload_id,
            "session_id": canonical_session_id,
            "filename": safe_name,
            "stored_name": target.name,
            "mime_type": mime_type,
            "size": total_bytes,
            "sha256": sha256_hex,
            "created_at": time.time(),
        }
        await asyncio.to_thread(_write_upload_metadata, meta_path, record)
        committed = True
    except web.HTTPRequestEntityTooLarge:
        return _error_response(
            f"Upload exceeds maximum artifact size of {WEBAPP_ATTACHMENT_MAX_BYTES} bytes.",
            413,
            code="artifact_too_large",
        )
    except OSError:
        logger.exception("[%s] Failed writing session upload", adapter.name)
        return _error_response("Could not store upload.", 500, code="upload_write_failed")
    finally:
        if handle is not None:
            with suppress(OSError):
                await asyncio.to_thread(handle.close)
        if not committed:
            await asyncio.to_thread(_unlink_quietly, target)

    return web.json_response(
        {
            "object": "hermes.session.upload",
            "upload_id": upload_id,
            "session_id": session_id,
            "thread_id": canonical_session_id,
            "filename": safe_name,
            "mime_type": mime_type,
            "size": total_bytes,
            "sha256": sha256_hex,
            "max_artifact_bytes": WEBAPP_ATTACHMENT_MAX_BYTES,
        },
        status=201,
    )


def resolve_session_attachments(session_id: str, raw_attachments: Any) -> tuple[list[dict[str, Any]], Optional[tuple[int, str, str]]]:
    """Validate and resolve ``attachments: [upload_id]`` strictly inside ``session_id``'s staging dir."""
    if raw_attachments is None:
        return [], None
    if not isinstance(raw_attachments, list) or not raw_attachments:
        return [], (400, "invalid_attachments", "'attachments' must be a non-empty array of upload_ids.")
    if len(raw_attachments) > _MAX_ATTACHMENTS_PER_RUN:
        return [], (400, "too_many_attachments", f"At most {_MAX_ATTACHMENTS_PER_RUN} attachments are allowed per run.")

    try:
        upload_dir = _session_upload_dir(session_id).resolve(strict=True)
    except (FileNotFoundError, OSError):
        return [], (404, "attachment_not_found", "Referenced attachment was not found in this session.")

    resolved_items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_id in raw_attachments:
        if not isinstance(raw_id, str) or not _UPLOAD_ID_RE.fullmatch(raw_id):
            return [], (400, "invalid_attachment_id", f"Invalid attachment id: {raw_id!r}.")
        if raw_id in seen:
            continue
        seen.add(raw_id)
        meta_path = upload_dir / f"{raw_id}.json"
        if meta_path.is_symlink() or not meta_path.is_file():
            return [], (404, "attachment_not_found", f"Attachment '{raw_id}' not found in this session.")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return [], (404, "attachment_not_found", f"Attachment '{raw_id}' metadata is unreadable.")
        if meta.get("session_id") != session_id or meta.get("upload_id") != raw_id:
            return [], (403, "attachment_scope_mismatch", f"Attachment '{raw_id}' does not belong to this session.")
        stored_name = str(meta.get("stored_name") or "")
        file_path = upload_dir / stored_name
        if not stored_name or file_path.is_symlink():
            return [], (403, "attachment_scope_mismatch", f"Attachment '{raw_id}' is invalid.")
        try:
            real_file = file_path.resolve(strict=True)
        except OSError:
            return [], (404, "attachment_not_found", f"Attachment '{raw_id}' file is missing.")
        if not path_is_under(upload_dir, real_file) or not real_file.is_file():
            return [], (403, "attachment_scope_mismatch", f"Attachment '{raw_id}' escapes session staging.")
        resolved_items.append({
            "upload_id": raw_id,
            "filename": str(meta.get("filename") or stored_name),
            "mime_type": str(meta.get("mime_type") or "application/octet-stream"),
            "size": int(meta.get("size") or 0),
            "sha256": str(meta.get("sha256") or ""),
            "path": str(real_file),
        })
    return resolved_items, None


def format_attachments_for_prompt(attachments: list[dict[str, Any]]) -> str:
    """Build a compact structured prompt appendix for staged session uploads."""
    if not attachments:
        return ""
    lines = ["[Attached session files]"]
    for item in attachments:
        lines.append(
            f"- {item['filename']} ({item['mime_type']}, {item['size']} bytes, "
            f"upload_id={item['upload_id']}, path={item['path']})"
        )
    return "\n".join(lines)


async def apply_run_attachments(
    adapter,
    session_id: str,
    body: dict[str, Any],
    user_message: Any,
    openai_error,
) -> tuple[Any, Optional[web.Response]]:
    """Resolve ``body['attachments']`` for ``POST /v1/runs`` (compaction-lineage & multimodal safe)."""
    raw = body.get("attachments")
    if raw is None:
        return user_message, None
    canonical_sid = await _canonical_upload_session_id(adapter, session_id)
    resolved, err = await asyncio.to_thread(resolve_session_attachments, canonical_sid, raw)
    if err is not None:
        return user_message, web.json_response(openai_error(err[2], code=err[1]), status=err[0])
    if not resolved:
        return user_message, None
    appendix = format_attachments_for_prompt(resolved)
    if isinstance(user_message, list):
        return [*user_message, {"type": "text", "text": appendix}], None
    return f"{user_message}\n\n{appendix}", None


def _http_routes(adapter) -> list[tuple]:
    return [
        ("GET", "/api/sessions/{session_id}/workspace/tree", functools.partial(handle_workspace_tree, adapter)),
        ("GET", "/api/sessions/{session_id}/workspace/file", functools.partial(handle_workspace_file, adapter)),
        ("GET", "/api/sessions/{session_id}/workspace/diff", functools.partial(handle_workspace_diff, adapter)),
        ("POST", "/api/sessions/{session_id}/uploads", functools.partial(handle_session_upload, adapter)),
        ("GET", "/api/analytics/usage", functools.partial(handle_usage_analytics, adapter)),
        ("GET", "/api/jobs/{job_id}/runs", functools.partial(handle_job_runs, adapter)),
    ]
