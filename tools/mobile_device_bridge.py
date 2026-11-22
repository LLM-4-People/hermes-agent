"""Approved Bidirectional Android Device Capability Bridge (``request_device_action``).

Provides per-run scoped capability advertisement (mirroring ``agent/run_instructions.py``
expiry semantics so copied execution contexts or later turns never inherit prior capabilities)
and a thread-safe pause/resume broker keyed by ``(profile_key, run_id)``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Optional

from tools.mobile_ui_patch import _scan_strings_safe, _valid_utf8_text
from tools.registry import no_cache_check_fn, registry

logger = logging.getLogger("hermes.mobile_device_bridge")

# ── Single Source of Truth Constants ─────────────────────────────────────────

ALLOWED_DEVICE_CAPABILITIES = frozenset({
    "camera",
    "location_coarse",
    "nearby_osm",
    "calendar_draft",
    "alarm_timer",
    "maps_handoff",
    "map_navigation",
    "share_sheet",
    "clipboard_read",
    "clipboard_write",
    "dial_draft",
    "sms_draft",
    "email_draft",
    "contacts_pick",
    "contact_draft",
    "ui_surfaces_v2",
})

DEVICE_ACTION_ALIASES: dict[str, str] = {
    "photo": "camera",
    "picture": "camera",
    "take_photo": "camera",
    "location": "location_coarse",
    "gps": "location_coarse",
    "where": "location_coarse",
    "nearby": "nearby_osm",
    "places": "nearby_osm",
    "calendar": "calendar_draft",
    "event": "calendar_draft",
    "alarm": "alarm_timer",
    "timer": "alarm_timer",
    "countdown": "alarm_timer",
    "maps": "maps_handoff",
    "directions": "map_navigation",
    "navigate": "map_navigation",
    "nav": "map_navigation",
    "share": "share_sheet",
    "clipboard": "clipboard_read",
    "copy": "clipboard_write",
    "call": "dial_draft",
    "phone": "dial_draft",
    "dial": "dial_draft",
    "sms": "sms_draft",
    "text_message": "sms_draft",
    "email": "email_draft",
    "mail": "email_draft",
    "contact": "contacts_pick",
    "contacts": "contacts_pick",
    "add_contact": "contact_draft",
}

ALLOWED_DEVICE_RESULT_STATUSES = frozenset({"completed", "denied", "unavailable"})
MAX_DEVICE_PARAMS_BYTES = 32_768
MAX_DEVICE_RESULT_BYTES = 131_072
MAX_REASON_LEN = 500
MIN_DEVICE_TIMEOUT_SECONDS = 1.0
MAX_DEVICE_TIMEOUT_SECONDS = 300.0
DEFAULT_DEVICE_TIMEOUT_SECONDS = 120.0


def configured_device_timeout(override: Optional[float] = None) -> float:
    raw = override
    if raw is None:
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
            mobile_cfg = cfg.get("mobile") if isinstance(cfg, dict) else None
            if isinstance(mobile_cfg, dict) and mobile_cfg.get("device_bridge_timeout_seconds") is not None:
                raw = mobile_cfg["device_bridge_timeout_seconds"]
        except (ImportError, OSError, ValueError) as exc:
            logger.debug("Falling back to default device bridge timeout: %s", exc)
            raw = None
    if raw is None:
        raw = DEFAULT_DEVICE_TIMEOUT_SECONDS
    try:
        val = float(raw)
    except (TypeError, ValueError):
        val = DEFAULT_DEVICE_TIMEOUT_SECONDS
    return max(MIN_DEVICE_TIMEOUT_SECONDS, min(val, MAX_DEVICE_TIMEOUT_SECONDS))


def validate_device_capabilities(value: Any) -> tuple[Optional[list[str]], Optional[str]]:
    """Validate optional ``device_capabilities`` array on ``POST /v1/runs``."""
    if value is None:
        return None, None
    if not isinstance(value, list):
        return None, "device_capabilities must be an array of capability strings."
    seen: list[str] = []
    seen_set: set[str] = set()
    for idx, item in enumerate(value):
        if not isinstance(item, str) or item not in ALLOWED_DEVICE_CAPABILITIES:
            return None, (
                f"device_capabilities[{idx}] must be one of {sorted(ALLOWED_DEVICE_CAPABILITIES)}."
            )
        if item not in seen_set:
            seen_set.add(item)
            seen.append(item)
    return seen, None


# ── Per-Run Isolated Scope (Expires in Copied Contexts) ──────────────────────

@dataclass
class _BoundDeviceScope:
    owner: Any
    capabilities: frozenset[str]
    run_id: str
    profile_key: str
    broker: Optional[DeviceBridgeBroker]
    timeout_seconds: Optional[float] = None


_active_device_scope: ContextVar[Optional[_BoundDeviceScope]] = ContextVar(
    "mobile_device_bridge_scope", default=None
)


@contextmanager
def device_bridge_scope(
    owner: Any,
    *,
    capabilities: Optional[list[str] | frozenset[str]],
    run_id: str,
    profile_key: str,
    broker: Optional[DeviceBridgeBroker],
    timeout_seconds: Optional[float] = None,
):
    """Bind per-run Android device capabilities; expires cleanly even in copied contexts."""
    caps = frozenset(c for c in (capabilities or ()) if c in ALLOWED_DEVICE_CAPABILITIES)
    bound = (
        _BoundDeviceScope(
            owner=owner,
            capabilities=caps,
            run_id=run_id,
            profile_key=profile_key,
            broker=broker,
            timeout_seconds=timeout_seconds,
        )
        if caps and broker is not None
        else None
    )
    token = _active_device_scope.set(bound)
    try:
        yield
    finally:
        if bound is not None:
            # Copied execution contexts share this object; scrub it so delayed tasks cannot reuse it.
            bound.capabilities = frozenset()
            bound.broker = None
            bound.owner = None
            bound.run_id = ""
            bound.profile_key = ""
        _active_device_scope.reset(token)


def current_device_capabilities() -> frozenset[str]:
    active = _active_device_scope.get()
    return active.capabilities if active is not None and active.broker is not None else frozenset()


# ── Thread-Safe Pause/Resume Broker ──────────────────────────────────────────

@dataclass
class _PendingDeviceRequest:
    request_id: str
    run_id: str
    profile_key: str
    action_type: str
    params: dict[str, Any]
    reason: str
    created_at: float
    expires_at: float
    event: threading.Event
    result_status: Optional[str] = None
    result_data: Optional[dict[str, Any]] = None
    settled: bool = False

    def public_view(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "run_id": self.run_id,
            "action_type": self.action_type,
            "params": self.params,
            "reason": self.reason,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }


class DeviceBridgeBroker:
    """Thread-safe pause/resume coordinator isolated by ``(profile_key, run_id)``."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._pending: dict[tuple[str, str], dict[str, _PendingDeviceRequest]] = {}
        self._listeners: dict[tuple[str, str], Callable[[str, dict[str, Any]], None]] = {}

    def register_run_listener(
        self,
        profile_key: str,
        run_id: str,
        callback: Callable[[str, dict[str, Any]], None],
    ) -> None:
        with self._lock:
            self._listeners[(profile_key, run_id)] = callback

    def unregister_run(self, profile_key: str, run_id: str) -> None:
        self.cancel_run(profile_key, run_id)
        with self._lock:
            self._listeners.pop((profile_key, run_id), None)
            self._pending.pop((profile_key, run_id), None)

    def list_pending(self, profile_key: str, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            bucket = self._pending.get((profile_key, run_id), {})
            return [entry.public_view() for entry in bucket.values() if not entry.settled]

    def _emit(self, profile_key: str, run_id: str, event_name: str, payload: dict[str, Any]) -> None:
        with self._lock:
            listener = self._listeners.get((profile_key, run_id))
        if listener is None:
            return
        try:
            listener(event_name, payload)
        except Exception:
            logger.warning("device bridge listener failed for run=%s event=%s", run_id, event_name, exc_info=True)

    def request_action(
        self,
        *,
        profile_key: str,
        run_id: str,
        action_type: str,
        params: dict[str, Any],
        reason: str,
        timeout_seconds: Optional[float] = None,
    ) -> dict[str, Any]:
        timeout = configured_device_timeout(timeout_seconds)
        now = time.time()
        request_id = f"dev_{uuid.uuid4().hex[:16]}"
        entry = _PendingDeviceRequest(
            request_id=request_id,
            run_id=run_id,
            profile_key=profile_key,
            action_type=action_type,
            params=params,
            reason=reason,
            created_at=now,
            expires_at=now + timeout,
            event=threading.Event(),
        )
        key = (profile_key, run_id)
        with self._lock:
            self._pending.setdefault(key, {})[request_id] = entry

        logger.info("device_bridge requested run=%s req=%s action=%s", run_id, request_id, action_type)
        self._emit(profile_key, run_id, "device.requested", entry.public_view())

        signaled = entry.event.wait(timeout=timeout)
        with self._lock:
            bucket = self._pending.get(key)
            if bucket is not None:
                bucket.pop(request_id, None)
                if not bucket:
                    self._pending.pop(key, None)
            if not signaled and not entry.settled:
                entry.settled = True
                entry.result_status = "timed_out"
                timed_out = True
            else:
                timed_out = False
            status = entry.result_status or "timed_out"
            data = entry.result_data

        if timed_out:
            logger.info("device_bridge timed_out run=%s req=%s action=%s", run_id, request_id, action_type)
            self._emit(
                profile_key,
                run_id,
                "device.settled",
                {"request_id": request_id, "action_type": action_type, "status": "timed_out"},
            )
            return {
                "ok": False,
                "request_id": request_id,
                "action_type": action_type,
                "status": "timed_out",
                "error": "device_request_timed_out",
            }

        return {
            "ok": status == "completed",
            "request_id": request_id,
            "action_type": action_type,
            "status": status,
            **({"data": data} if data is not None else {}),
            **({"error": f"device_action_{status}"} if status != "completed" else {}),
        }

    def resolve_request(
        self,
        *,
        profile_key: str,
        run_id: str,
        request_id: str,
        status: str,
        data: Optional[dict[str, Any]] = None,
    ) -> tuple[str, Optional[dict[str, Any]]]:
        """Resolve one pending request. Returns ``(outcome, public_summary)`` where outcome is
        ``'resolved'``, ``'not_found'``, or ``'already_settled'``."""
        key = (profile_key, run_id)
        with self._lock:
            bucket = self._pending.get(key)
            if not bucket or request_id not in bucket:
                return "not_found", None
            entry = bucket[request_id]
            if entry.settled:
                return "already_settled", entry.public_view()
            entry.settled = True
            entry.result_status = status
            entry.result_data = data
            bucket.pop(request_id, None)
            if not bucket:
                self._pending.pop(key, None)
            action_type = entry.action_type
            entry.event.set()

        logger.info("device_bridge resolved run=%s req=%s action=%s status=%s", run_id, request_id, action_type, status)
        settled_payload = {"request_id": request_id, "action_type": action_type, "status": status}
        self._emit(profile_key, run_id, "device.settled", settled_payload)
        return "resolved", settled_payload

    def cancel_run(self, profile_key: str, run_id: str) -> int:
        key = (profile_key, run_id)
        cancelled: list[_PendingDeviceRequest] = []
        with self._lock:
            bucket = self._pending.pop(key, None) or {}
            for entry in bucket.values():
                if not entry.settled:
                    entry.settled = True
                    entry.result_status = "cancelled"
                    entry.event.set()
                    cancelled.append(entry)
        for entry in cancelled:
            self._emit(
                profile_key,
                run_id,
                "device.settled",
                {"request_id": entry.request_id, "action_type": entry.action_type, "status": "cancelled"},
            )
        return len(cancelled)


# ── Agent Tool Implementation & Registration ─────────────────────────────────

REQUEST_DEVICE_ACTION_SCHEMA = {
    "name": "request_device_action",
    "description": (
        "Request an explicitly user-approved Android device capability for the current run "
        "(e.g., camera capture, coarse location, calendar draft, alarm/timer, maps handoff, share sheet, "
        "clipboard read, dial/email draft). Pauses the run until the user approves, declines, or times out."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action_type": {
                "type": "string",
                "enum": sorted(ALLOWED_DEVICE_CAPABILITIES),
                "description": "Device capability to invoke (must be advertised by the current run).",
            },
            "reason": {
                "type": "string",
                "description": "Clear user-facing explanation of why this device action is requested.",
            },
            "params": {
                "type": "object",
                "description": "Structured parameters for the requested device action.",
            },
        },
        "required": ["action_type", "reason"],
    },
}


_DEVICE_ACTION_EXAMPLE = {
    "action_type": "alarm_timer",
    "reason": "Régler un minuteur de 10 minutes",
    "params": {"duration_seconds": 600, "label": "Pause"},
}


def request_device_action_tool(
    action_type: Any,
    reason: Any,
    params: Any = None,
) -> str:
    """Execute ``request_device_action`` against the current run's isolated device scope."""
    scope = _active_device_scope.get()
    if scope is None or scope.broker is None:
        return json.dumps({
            "ok": False,
            "error": "device_bridge_unavailable",
            "message": "No active mobile device bridge is bound to this run.",
            "hint": "Only call request_device_action when the client advertised device_capabilities on this run.",
        }, ensure_ascii=False)
    norm_action = (
        DEVICE_ACTION_ALIASES.get(action_type.strip().lower(), action_type.strip().lower())
        if isinstance(action_type, str)
        else action_type
    )
    if not isinstance(norm_action, str) or norm_action not in ALLOWED_DEVICE_CAPABILITIES:
        return json.dumps({
            "ok": False,
            "error": "invalid_action_type",
            "allowed": sorted(ALLOWED_DEVICE_CAPABILITIES),
            "hint": f"Choose action_type from advertised capabilities: {sorted(scope.capabilities)}.",
            "example": _DEVICE_ACTION_EXAMPLE,
        }, ensure_ascii=False)
    if norm_action not in scope.capabilities:
        return json.dumps({
            "ok": False,
            "error": "capability_not_advertised",
            "action_type": norm_action,
            "advertised": sorted(scope.capabilities),
            "hint": f"This phone run only advertised {sorted(scope.capabilities)}; use one of those or answer without device access.",
        }, ensure_ascii=False)
    if not _valid_utf8_text(reason, max_len=MAX_REASON_LEN):
        return json.dumps({
            "ok": False,
            "error": f"reason must be non-empty UTF-8 text (<= {MAX_REASON_LEN} chars).",
            "hint": "Provide a short, plain-language reason shown to the user before they approve.",
            "example": _DEVICE_ACTION_EXAMPLE,
        }, ensure_ascii=False)
    clean_params = {} if params is None else params
    if not isinstance(clean_params, dict):
        return json.dumps({
            "ok": False,
            "error": "params must be a JSON object.",
            "hint": "Pass params as a flat JSON object (or omit it).",
            "example": _DEVICE_ACTION_EXAMPLE,
        }, ensure_ascii=False)
    scan_err = _scan_strings_safe(clean_params)
    if scan_err:
        return json.dumps({
            "ok": False,
            "error": scan_err,
            "hint": "Remove any script tags or unsafe URIs from params and retry.",
            "example": _DEVICE_ACTION_EXAMPLE,
        }, ensure_ascii=False)
    encoded = json.dumps(clean_params, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_DEVICE_PARAMS_BYTES:
        return json.dumps({
            "ok": False,
            "error": f"params exceeds maximum size of {MAX_DEVICE_PARAMS_BYTES} bytes.",
            "hint": "Trim params to only the essential fields for the native Android confirmation screen.",
        }, ensure_ascii=False)

    outcome = scope.broker.request_action(
        profile_key=scope.profile_key,
        run_id=scope.run_id,
        action_type=norm_action,
        params=clean_params,
        reason=reason.strip(),
        timeout_seconds=scope.timeout_seconds,
    )
    return json.dumps(outcome, ensure_ascii=False)


@no_cache_check_fn
def _device_bridge_available() -> bool:
    return bool(current_device_capabilities())


registry.register(
    name="request_device_action",
    toolset="mobile_device",
    schema=REQUEST_DEVICE_ACTION_SCHEMA,
    handler=lambda args, **kw: request_device_action_tool(
        args.get("action_type"),
        args.get("reason"),
        args.get("params"),
    ),
    check_fn=_device_bridge_available,
)
