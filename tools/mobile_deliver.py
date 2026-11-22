"""Universal Hermes-to-Phone Delivery Tool (``mobile_deliver``) & ``run_progress`` Hook Registry.

Works uniformly across interactive turns, background subagents, and cron jobs.
Persists full structured deliveries inside the active profile's ``mobile_state.db`` and
invokes zero-leak wakeup hooks (carrying ONLY opaque ``event_id``, ``kind``, ``urgency``,
and ``session_id`` — never private title/payload text) so ``server/mobile_push`` can wake
a closed phone safely.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from contextlib import closing, contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_constants import get_hermes_home
from tools.mobile_ui_patch import (
    _lock_for,
    _scan_strings_safe,
    _valid_utf8_text,
    mobile_state_db_path,
    validate_ui_patch_ops,
    validate_widget_spec,
)
from tools.registry import registry

logger = logging.getLogger("hermes.mobile_deliver")

# ── Single Source of Truth Constants ─────────────────────────────────────────

ALLOWED_DELIVERY_KINDS = frozenset({
    "notice",
    "card",
    "file",
    "ui_patch",
    "widget_update",
    "device_action",
    "reminder_result",
    "briefing",
})

DELIVERY_KIND_ALIASES: dict[str, str] = {
    "alert": "notice",
    "notification": "notice",
    "message": "notice",
    "summary": "briefing",
    "daily": "briefing",
    "digest": "briefing",
    "widget": "widget_update",
    "reminder": "reminder_result",
    "action": "device_action",
    "handoff": "device_action",
    "ui": "ui_patch",
    "patch": "ui_patch",
    "attachment": "file",
    "document": "file",
}

ALLOWED_URGENCIES = frozenset({"silent", "normal", "high", "urgent"})
URGENCY_ALIASES: dict[str, str] = {
    "low": "silent",
    "quiet": "silent",
    "mute": "silent",
    "muted": "silent",
    "default": "normal",
    "medium": "normal",
    "critical": "urgent",
    "important": "high",
}
MAX_DELIVERY_TITLE_LEN = 200
MAX_DELIVERY_PAYLOAD_BYTES = 131_072

_DELIVERY_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS mobile_deliveries (
    delivery_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    urgency TEXT NOT NULL,
    session_id TEXT,
    origin TEXT NOT NULL,
    created_at REAL NOT NULL,
    acked_at REAL
);
CREATE INDEX IF NOT EXISTS idx_mobile_deliveries_created ON mobile_deliveries (created_at DESC);
"""

# ── Zero-Leak Wakeup & Live Run Progress Hook Registries ─────────────────────

_HOOKS_LOCK = threading.RLock()
_DELIVERY_WAKEUP_HOOKS: list[Callable[[dict[str, Any]], None]] = []
_RUN_PROGRESS_HOOKS: list[Callable[[dict[str, Any]], None]] = []

_delivery_emitter: ContextVar[Optional[Callable[[dict[str, Any]], None]]] = ContextVar(
    "mobile_delivery_emitter", default=None
)


def register_delivery_wakeup_hook(callback: Callable[[dict[str, Any]], None]) -> None:
    """Register a plugin callback invoked with zero-leak wakeup metadata on new deliveries."""
    with _HOOKS_LOCK:
        if callback not in _DELIVERY_WAKEUP_HOOKS:
            _DELIVERY_WAKEUP_HOOKS.append(callback)


def unregister_delivery_wakeup_hook(callback: Callable[[dict[str, Any]], None]) -> None:
    with _HOOKS_LOCK:
        if callback in _DELIVERY_WAKEUP_HOOKS:
            _DELIVERY_WAKEUP_HOOKS.remove(callback)


def register_run_progress_hook(callback: Callable[[dict[str, Any]], None]) -> None:
    """Register a plugin callback for live Android 16 ongoing run progress updates."""
    with _HOOKS_LOCK:
        if callback not in _RUN_PROGRESS_HOOKS:
            _RUN_PROGRESS_HOOKS.append(callback)


def unregister_run_progress_hook(callback: Callable[[dict[str, Any]], None]) -> None:
    with _HOOKS_LOCK:
        if callback in _RUN_PROGRESS_HOOKS:
            _RUN_PROGRESS_HOOKS.remove(callback)


def emit_run_progress(
    *,
    run_id: str,
    status: str,
    session_id: Optional[str] = None,
    profile: Optional[str] = None,
    tool_name: Optional[str] = None,
    stage: Optional[str] = None,
    pending_approval: bool = False,
    pending_device: bool = False,
) -> None:
    """Broadcast safe, non-sensitive run progress metadata to registered push hooks."""
    with _HOOKS_LOCK:
        hooks = list(_RUN_PROGRESS_HOOKS)
    if not hooks:
        return
    event = {
        "event": "run.progress",
        "run_id": str(run_id),
        "session_id": str(session_id) if session_id else None,
        "profile": str(profile or "default"),
        "status": str(status),
        "tool_name": str(tool_name)[:64] if tool_name else None,
        "stage": str(stage)[:64] if stage else None,
        "pending_approval": bool(pending_approval),
        "pending_device": bool(pending_device),
        "timestamp": time.time(),
    }
    for hook in hooks:
        try:
            hook(event)
        except Exception:
            logger.warning("run_progress hook failed for run=%s", run_id, exc_info=True)


@contextmanager
def bind_delivery_emitter(callback: Optional[Callable[[dict[str, Any]], None]]):
    """Bind a live run SSE emitter for ``mobile.delivery`` events during an active turn."""
    token = _delivery_emitter.set(callback)
    try:
        yield
    finally:
        _delivery_emitter.reset(token)


def _fire_wakeup_hooks(wakeup_meta: dict[str, Any]) -> None:
    with _HOOKS_LOCK:
        hooks = list(_DELIVERY_WAKEUP_HOOKS)
    for hook in hooks:
        try:
            hook(dict(wakeup_meta))
        except Exception:
            logger.warning("delivery wakeup hook failed for event_id=%s", wakeup_meta.get("event_id"), exc_info=True)


# ── Validation & SQLite Store ────────────────────────────────────────────────

def normalize_delivery_input(
    kind: Any, title: Any, payload: Any, urgency: Any = "normal"
) -> tuple[Optional[tuple[str, str, dict[str, Any], str]], Optional[str]]:
    """Normalize small-LLM friendly aliases and validate delivery fields strictly."""
    norm_kind = (
        DELIVERY_KIND_ALIASES.get(kind.strip().lower(), kind.strip().lower())
        if isinstance(kind, str)
        else kind
    )
    if norm_kind not in ALLOWED_DELIVERY_KINDS:
        return None, f"kind must be one of {sorted(ALLOWED_DELIVERY_KINDS)}."
    norm_urgency = (
        URGENCY_ALIASES.get(urgency.strip().lower(), urgency.strip().lower())
        if isinstance(urgency, str)
        else urgency
    )
    if norm_urgency not in ALLOWED_URGENCIES:
        return None, f"urgency must be one of {sorted(ALLOWED_URGENCIES)} (alias 'low' maps to 'silent')."
    if not _valid_utf8_text(title, max_len=MAX_DELIVERY_TITLE_LEN):
        return None, f"title must be non-empty UTF-8 text (<= {MAX_DELIVERY_TITLE_LEN} chars)."
    norm_payload: Any = payload
    if isinstance(norm_payload, str) and norm_kind in {"notice", "card", "briefing", "reminder_result"}:
        norm_payload = {"text": norm_payload}
    if not isinstance(norm_payload, dict):
        return None, "payload must be a JSON object."
    scan_err = _scan_strings_safe(norm_payload)
    if scan_err:
        return None, scan_err
    try:
        encoded = json.dumps(norm_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        return None, f"payload must be JSON-serializable: {exc}"
    if len(encoded) > MAX_DELIVERY_PAYLOAD_BYTES:
        return None, f"payload exceeds maximum size of {MAX_DELIVERY_PAYLOAD_BYTES} bytes."
    if norm_kind == "ui_patch":
        norm_ops, ops_err = validate_ui_patch_ops(norm_payload.get("ops"))
        if ops_err:
            return None, f"ui_patch payload invalid: {ops_err}"
        norm_payload = {**norm_payload, "ops": norm_ops}
    elif norm_kind == "widget_update":
        norm_widget, widget_err = validate_widget_spec(norm_payload, prefix="widget_update")
        if widget_err:
            return None, f"widget_update payload invalid: {widget_err}"
        norm_payload = norm_widget
    return (norm_kind, title.strip(), norm_payload, norm_urgency), None


def validate_delivery_input(kind: Any, title: Any, payload: Any, urgency: Any = "normal") -> Optional[str]:
    _, err = normalize_delivery_input(kind, title, payload, urgency)
    return err


class MobileDeliveryStore:
    """Profile-scoped SQLite inbox for Hermes-to-Phone deliveries."""

    def __init__(self, profile_home: Optional[Path] = None) -> None:
        self._profile_home = Path(profile_home) if profile_home is not None else get_hermes_home()
        self._db_path = mobile_state_db_path(self._profile_home)
        self._lock = _lock_for(self._db_path)

    def _connect(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), timeout=10.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.executescript(_DELIVERY_SCHEMA_SQL)
        try:
            os.chmod(self._db_path, 0o600)
        except OSError:
            pass
        return conn

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "delivery_id": row["delivery_id"],
            "kind": row["kind"],
            "title": row["title"],
            "payload": json.loads(row["payload_json"]),
            "urgency": row["urgency"],
            "session_id": row["session_id"],
            "origin": row["origin"],
            "created_at": row["created_at"],
            "acked_at": row["acked_at"],
        }

    def deliver(
        self,
        kind: str,
        title: str,
        payload: dict[str, Any],
        *,
        urgency: str = "normal",
        session_id: Optional[str] = None,
        origin: str = "agent",
    ) -> dict[str, Any]:
        normalized, err = normalize_delivery_input(kind, title, payload, urgency)
        if err or normalized is None:
            raise ValueError(err or "Invalid delivery payload.")
        kind, clean_title, payload, urgency = normalized

        now = time.time()
        delivery_id = f"dlv_{uuid.uuid4().hex[:16]}"
        clean_origin = str(origin or "agent").strip()[:48] or "agent"
        with self._lock, closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    """INSERT INTO mobile_deliveries
                       (delivery_id, kind, title, payload_json, urgency, session_id, origin, created_at, acked_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)""",
                    (
                        delivery_id,
                        kind,
                        clean_title,
                        json.dumps(payload, ensure_ascii=False),
                        urgency,
                        session_id,
                        clean_origin,
                        now,
                    ),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

        delivery = {
            "delivery_id": delivery_id,
            "kind": kind,
            "title": clean_title,
            "payload": payload,
            "urgency": urgency,
            "session_id": session_id,
            "origin": clean_origin,
            "created_at": now,
            "acked_at": None,
        }
        logger.info("mobile_deliver id=%s kind=%s urgency=%s origin=%s", delivery_id, kind, urgency, clean_origin)

        # Live SSE delivery for connected clients.
        emitter = _delivery_emitter.get()
        if emitter is not None:
            try:
                emitter(delivery)
            except Exception:
                logger.warning("mobile.delivery SSE emitter failed for id=%s", delivery_id, exc_info=True)

        # Zero-leak wakeup hook for closed-app encrypted push (NO title/payload text).
        _fire_wakeup_hooks({
            "event_id": delivery_id,
            "profile_home": str(self._profile_home),
            "kind": kind,
            "urgency": urgency,
            "session_id": session_id,
            "created_at": now,
        })
        return delivery

    def list_deliveries(
        self,
        *,
        since: Optional[float] = None,
        include_acked: bool = False,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        capped = max(1, min(int(limit), 100))
        clauses: list[str] = []
        params: list[Any] = []
        if not include_acked:
            clauses.append("acked_at IS NULL")
        if since is not None:
            clauses.append("created_at > ?")
            params.append(float(since))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(capped)
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT * FROM mobile_deliveries {where} ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_delivery(self, delivery_id: str) -> Optional[dict[str, Any]]:
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM mobile_deliveries WHERE delivery_id = ?", (delivery_id,)).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def ack_delivery(self, delivery_id: str) -> Optional[dict[str, Any]]:
        now = time.time()
        with self._lock, closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute("SELECT * FROM mobile_deliveries WHERE delivery_id = ?", (delivery_id,)).fetchone()
                if row is None:
                    conn.execute("ROLLBACK")
                    return None
                if row["acked_at"] is None:
                    conn.execute(
                        "UPDATE mobile_deliveries SET acked_at = ? WHERE delivery_id = ?",
                        (now, delivery_id),
                    )
                    row = conn.execute("SELECT * FROM mobile_deliveries WHERE delivery_id = ?", (delivery_id,)).fetchone()
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return self._row_to_dict(row)


# ── Agent Tool Implementation & Registration ─────────────────────────────────

MOBILE_DELIVER_SCHEMA = {
    "name": "mobile_deliver",
    "description": (
        "Deliver a structured item directly to the user's Hermes Mobile app (works in interactive turns, "
        "subagents, and scheduled cron jobs). Supports notices, rich cards, published files, live UI patches, "
        "OS handoff proposals, reminder results, and daily briefings. Closed phones receive only a zero-leak "
        "encrypted wakeup until unlocked."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": sorted(ALLOWED_DELIVERY_KINDS),
                "description": "Structured delivery kind.",
            },
            "title": {
                "type": "string",
                "description": "Short human-readable heading shown inside the unlocked Activité inbox.",
            },
            "payload": {
                "type": "object",
                "description": "Structured JSON payload matching the chosen kind.",
            },
            "urgency": {
                "type": "string",
                "enum": ["silent", "low", "normal", "high", "urgent"],
                "description": "Delivery priority (default: normal; 'low' normalizes to 'silent').",
            },
            "session_id": {
                "type": "string",
                "description": "Optional discussion session ID to link this delivery to.",
            },
        },
        "required": ["kind", "title", "payload"],
    },
}


def _detect_delivery_origin() -> str:
    from agent.delegation_context import is_delegated_child_context
    from gateway.session_context import get_session_env

    if get_session_env("HERMES_CRON_SESSION") == "1":
        return "cron"
    if is_delegated_child_context():
        return "subagent"
    return "agent"


_DELIVERY_EXAMPLE = {
    "kind": "briefing",
    "title": "Briefing du matin",
    "payload": {"summary": "3 tâches prioritaires aujourd'hui"},
    "urgency": "normal",
}


def mobile_deliver_tool(
    kind: str,
    title: str,
    payload: Any,
    *,
    urgency: str = "normal",
    session_id: Optional[str] = None,
    origin: Optional[str] = None,
) -> str:
    """Execute ``mobile_deliver`` tool call and return JSON receipt (with self-correcting hints on error)."""
    if not session_id:
        from gateway.session_context import get_session_env
        session_id = get_session_env("HERMES_SESSION_ID") or None
    store = MobileDeliveryStore()
    try:
        delivery = store.deliver(
            kind,
            title,
            payload,
            urgency=urgency,
            session_id=session_id,
            origin=origin or _detect_delivery_origin(),
        )
    except ValueError as exc:
        logger.warning("mobile_deliver rejected: %s", exc)
        return json.dumps(
            {
                "ok": False,
                "error": str(exc),
                "hint": (
                    f"Use kind in {sorted(ALLOWED_DELIVERY_KINDS)} (alias 'alert'->'notice'), "
                    f"urgency in {sorted(ALLOWED_URGENCIES)} (alias 'low'->'silent'), non-empty title, and JSON payload."
                ),
                "example": _DELIVERY_EXAMPLE,
            },
            ensure_ascii=False,
        )
    return json.dumps({"ok": True, "delivery": delivery}, ensure_ascii=False)


registry.register(
    name="mobile_deliver",
    toolset="mobile_deliver",
    schema=MOBILE_DELIVER_SCHEMA,
    handler=lambda args, **kw: mobile_deliver_tool(
        args.get("kind", ""),
        args.get("title", ""),
        args.get("payload"),
        urgency=args.get("urgency", "normal"),
        session_id=args.get("session_id") or kw.get("session_id"),
    ),
    check_fn=lambda: True,
)
