"""Authenticated HTTP route for resolving pending Android device bridge requests.

Registered through ``APIServerAdapter._http_route_table()`` so ``/p/{profile}/...``
mirroring and profile isolation apply uniformly.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from aiohttp import web

from tools.mobile_device_bridge import (
    ALLOWED_DEVICE_RESULT_STATUSES,
    MAX_DEVICE_RESULT_BYTES,
    DeviceBridgeBroker,
)
from tools.mobile_ui_patch import _scan_strings_safe

logger = logging.getLogger("gateway.platforms.api_server")

_NO_STORE = {"Cache-Control": "no-store"}
_ALLOWED_RESULT_KEYS = frozenset({"request_id", "status", "data"})


def ensure_device_bridge_broker(adapter) -> DeviceBridgeBroker:
    """Return the adapter's ``DeviceBridgeBroker``, initializing lazily if needed."""
    broker = getattr(adapter, "_device_bridge_broker", None)
    if broker is None:
        broker = DeviceBridgeBroker()
        adapter._device_bridge_broker = broker
    return broker


def resolve_device_profile_key(adapter, request_profile: str | None = None) -> str:
    """Single source of truth for the ``DeviceBridgeBroker`` profile isolation key."""
    if request_profile:
        return str(request_profile)
    from gateway.session_authorities import active_authority
    authority = active_authority(getattr(adapter, "gateway_runner", None))
    pid = getattr(authority, "profile_id", None)
    return str(pid) if pid else "default"


def _error_no_store(_openai_error, message: str, *, code: str, status: int) -> web.Response:
    return web.json_response(_openai_error(message, code=code), status=status, headers=_NO_STORE)


def _http_routes(adapter) -> list[tuple[str, str, Any]]:
    from gateway.platforms import api_server as _api_server
    from gateway.platforms.api_server_runs import _load_owned_run

    async def handle_run_device_result(request: web.Request) -> web.Response:
        _openai_error = _api_server._openai_error
        run_id, _status, _agent, _task, err = _load_owned_run(
            adapter, request, _api_server=_api_server, permission="approve", active_fallback=True
        )
        if err is not None:
            err.headers["Cache-Control"] = "no-store"
            return err

        body, body_err = await adapter._read_json_body(request)
        if body_err is not None:
            body_err.headers["Cache-Control"] = "no-store"
            return body_err
        if not isinstance(body, dict):
            return _error_no_store(_openai_error, "Body must be a JSON object.", code="invalid_device_result", status=400)

        unknown = sorted(set(body) - _ALLOWED_RESULT_KEYS)
        if unknown:
            return _error_no_store(
                _openai_error,
                f"Unsupported fields: {', '.join(unknown)}",
                code="invalid_device_result",
                status=400,
            )

        request_id = body.get("request_id")
        if not isinstance(request_id, str) or not request_id.strip():
            return _error_no_store(
                _openai_error, "request_id must be a non-empty string.", code="invalid_device_result", status=400
            )
        status = body.get("status")
        if status not in ALLOWED_DEVICE_RESULT_STATUSES:
            return _error_no_store(
                _openai_error,
                f"status must be one of {sorted(ALLOWED_DEVICE_RESULT_STATUSES)}.",
                code="invalid_device_result",
                status=400,
            )
        data = body.get("data")
        if data is not None:
            if not isinstance(data, dict):
                return _error_no_store(
                    _openai_error, "data must be a JSON object when provided.", code="invalid_device_result", status=400
                )
            scan_err = _scan_strings_safe(data)
            if scan_err:
                return _error_no_store(_openai_error, scan_err, code="invalid_device_result", status=400)
            encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            if len(encoded) > MAX_DEVICE_RESULT_BYTES:
                return _error_no_store(
                    _openai_error,
                    f"data exceeds maximum size of {MAX_DEVICE_RESULT_BYTES} bytes.",
                    code="invalid_device_result",
                    status=400,
                )

        profile_key = resolve_device_profile_key(adapter, _api_server._api_request_profile.get())
        broker = ensure_device_bridge_broker(adapter)
        outcome, summary = broker.resolve_request(
            profile_key=profile_key,
            run_id=run_id,
            request_id=request_id.strip(),
            status=status,
            data=data,
        )
        if outcome == "not_found":
            return _error_no_store(
                _openai_error,
                f"Pending device request not found: {request_id}",
                code="device_request_not_found",
                status=404,
            )
        if outcome == "already_settled":
            return _error_no_store(
                _openai_error,
                f"Device request already settled: {request_id}",
                code="device_request_already_settled",
                status=409,
            )
        return web.json_response(
            {
                "object": "hermes.run.device_result",
                "run_id": run_id,
                "request_id": request_id.strip(),
                "status": status,
                "accepted": True,
                **({"action_type": summary["action_type"]} if summary and "action_type" in summary else {}),
            },
            headers=_NO_STORE,
        )

    return [
        ("POST", "/v1/runs/{run_id}/device-result", handle_run_device_result),
    ]
