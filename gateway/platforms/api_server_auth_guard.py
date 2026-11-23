"""Per-client, per-profile authentication rate limiter for the API server.

Provides bounded-memory sliding-window brute-force protection for Bearer key
verification without ever logging or storing secret material.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from contextlib import suppress
import ipaddress
import math
import threading
import time
from typing import Any, Callable, Optional

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]


DEFAULT_AUTH_FAILURE_MAX_ATTEMPTS = 20
DEFAULT_AUTH_FAILURE_WINDOW_SECONDS = 60
DEFAULT_AUTH_FAILURE_MAX_BUCKETS = 4096


def _is_loopback_ip(raw: str) -> bool:
    if not raw:
        return False
    cleaned = raw.strip().split("%", 1)[0]
    try:
        return ipaddress.ip_address(cleaned).is_loopback
    except ValueError:
        return False


def extract_client_ip(request: Any) -> str:
    """Resolve a stable, non-spoofable client IP bucket key for rate limiting.

    Only trusts ``X-Forwarded-For`` / ``X-Real-IP`` when the direct peer is
    loopback (e.g. Caddy/nginx reverse proxy on the same host), so direct
    callers cannot spoof headers to evade rate limits or lock out others.
    """
    peer_ip = ""
    with suppress(Exception):
        transport = getattr(request, "transport", None)
        peer = transport.get_extra_info("peername") if transport else None
        if isinstance(peer, (tuple, list)) and peer:
            peer_ip = str(peer[0]).strip()
    remote = str(getattr(request, "remote", "") or peer_ip).strip()
    if not remote or _is_loopback_ip(remote):
        headers = getattr(request, "headers", None) or {}
        forwarded = str(headers.get("X-Forwarded-For", "") or "").strip()
        if forwarded:
            first_hop = forwarded.split(",", 1)[0].strip()
            if first_hop:
                return first_hop[:128]
        real_ip = str(headers.get("X-Real-IP", "") or "").strip()
        if real_ip:
            return real_ip[:128]
    return (remote or "unknown")[:128]


class AuthFailureLimiter:
    """Bounded-memory sliding-window limiter scoped per ``(client_ip, profile)``."""

    def __init__(
        self,
        *,
        max_attempts: int = DEFAULT_AUTH_FAILURE_MAX_ATTEMPTS,
        window_seconds: float = DEFAULT_AUTH_FAILURE_WINDOW_SECONDS,
        max_buckets: int = DEFAULT_AUTH_FAILURE_MAX_BUCKETS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_attempts = max(0, int(max_attempts))
        self._window_seconds = max(1.0, float(window_seconds))
        self._max_buckets = max(64, int(max_buckets))
        self._clock = clock
        self._lock = threading.Lock()
        self._buckets: OrderedDict[tuple[str, str], deque[float]] = OrderedDict()

    @property
    def max_attempts(self) -> int:
        return self._max_attempts

    @property
    def window_seconds(self) -> float:
        return self._window_seconds

    def _prune(self, bucket: deque[float], now: float) -> None:
        cutoff = now - self._window_seconds
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

    def _retry_after_unlocked(self, bucket: deque[float], now: float) -> Optional[int]:
        if len(bucket) < self._max_attempts:
            return None
        remaining = self._window_seconds - (now - bucket[0])
        return max(1, math.ceil(remaining))

    def check_locked_out(self, client_ip: str, profile_key: str) -> Optional[int]:
        """Return ``Retry-After`` seconds if locked out, else ``None``."""
        if self._max_attempts <= 0:
            return None
        key = (client_ip or "unknown", profile_key or "default")
        now = self._clock()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                return None
            self._prune(bucket, now)
            if not bucket:
                self._buckets.pop(key, None)
                return None
            self._buckets.move_to_end(key)
            return self._retry_after_unlocked(bucket, now)

    def record_failure(self, client_ip: str, profile_key: str) -> Optional[int]:
        """Record one failed auth attempt; return ``Retry-After`` if now locked out."""
        if self._max_attempts <= 0:
            return None
        key = (client_ip or "unknown", profile_key or "default")
        now = self._clock()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                while len(self._buckets) >= self._max_buckets:
                    self._buckets.popitem(last=False)
                bucket = deque()
                self._buckets[key] = bucket
            else:
                self._prune(bucket, now)
                self._buckets.move_to_end(key)
            bucket.append(now)
            return self._retry_after_unlocked(bucket, now)

    def record_success(self, client_ip: str, profile_key: str) -> None:
        """Clear accumulated auth failures for ``(client_ip, profile_key)``."""
        key = (client_ip or "unknown", profile_key or "default")
        with self._lock:
            self._buckets.pop(key, None)

    def reset(self) -> None:
        """Clear all tracked failure buckets."""
        with self._lock:
            self._buckets.clear()


def auth_rate_limited_response(retry_after: int) -> web.Response:
    """Build a fail-closed 429 response for authentication brute-force throttling."""
    retry_value = str(max(1, int(retry_after)))
    return web.json_response(
        {
            "error": {
                "message": "Too many failed authentication attempts; retry later.",
                "type": "rate_limit_error",
                "code": "auth_rate_limited",
            }
        },
        status=429,
        headers={"Retry-After": retry_value, "Cache-Control": "no-store"},
    )


DEFAULT_MAX_CONCURRENT_RUNS = 32
DEFAULT_MAX_CONCURRENT_RUNS_PER_PROFILE = 12


def init_adapter_security_and_concurrency(adapter: Any) -> None:
    """Initialize per-profile auth limiting, model cache, and fair-share concurrency state."""
    adapter._last_resolved_model_by_profile = {}
    adapter._max_concurrent_runs_per_profile = adapter._resolve_api_server_int(
        "max_concurrent_runs_per_profile", default=DEFAULT_MAX_CONCURRENT_RUNS_PER_PROFILE
    )
    adapter._auth_limiter = AuthFailureLimiter(
        max_attempts=adapter._resolve_api_server_int(
            "auth_failure_max_attempts", default=DEFAULT_AUTH_FAILURE_MAX_ATTEMPTS
        ),
        window_seconds=adapter._resolve_api_server_int(
            "auth_failure_window_seconds", default=DEFAULT_AUTH_FAILURE_WINDOW_SECONDS
        ),
    )
    adapter._inflight_agent_runs_by_profile = {}
    adapter._pending_agent_requests_by_profile = {}


def adjust_profile_counter(table: Any, profile_key: Any, delta: int) -> None:
    """Increment or decrement a profile-scoped counter dict without leaving zero keys behind."""
    if not isinstance(table, dict) or not isinstance(profile_key, str):
        return
    updated = table.get(profile_key, 0) + delta
    if updated > 0:
        table[profile_key] = updated
    else:
        table.pop(profile_key, None)


def active_profile_agent_work_count(adapter: Any, profile: Optional[str], profile_var: Any) -> int:
    """Live agent work scoped to one profile (fair-share isolation under multi-user load)."""
    profile_key = (profile if profile is not None else profile_var.get()) or "default"
    try:
        pending = int(getattr(adapter, "_pending_agent_requests_by_profile", {}).get(profile_key, 0))
        inflight = int(getattr(adapter, "_inflight_agent_runs_by_profile", {}).get(profile_key, 0))
        run_profiles = getattr(adapter, "_run_profiles", {})
        runs = sum(
            1 for run_id, task in adapter._active_run_tasks.items()
            if not task.done() and run_profiles.get(run_id, "default") == profile_key
        )
        return pending + inflight + runs
    except (AttributeError, TypeError, ValueError):
        return 0


def check_adapter_auth(adapter: Any, request: Any, *, profile_var: Any, secrets_match: Callable[[str, str], bool], logger: Any) -> Optional[web.Response]:
    """Validate Bearer auth with fail-closed named profiles and per-(IP, profile) rate limiting."""
    profile = profile_var.get()
    profile_key = profile or "default"
    expected_key = adapter._expected_api_key()
    if not expected_key and not (profile and profile != "default"):
        return None
    limiter = getattr(adapter, "_auth_limiter", None)
    client_ip = extract_client_ip(request) if limiter is not None else ""
    if limiter is not None:
        retry_after = limiter.check_locked_out(client_ip, profile_key)
        if retry_after is not None:
            logger.warning(
                "API server rate-limited repeated auth failures for profile %r: %s",
                profile_key, adapter._request_audit_log_suffix(request),
            )
            return auth_rate_limited_response(retry_after)
    if not expected_key:
        if limiter is not None:
            limiter.record_failure(client_ip, profile_key)
        logger.warning(
            "API server rejected request for profile %r: no profile-scoped "
            "API_SERVER_KEY is configured; %s",
            profile, adapter._request_audit_log_suffix(request),
        )
        return adapter._auth_failed_response()
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        if secrets_match(token, expected_key):
            if limiter is not None:
                limiter.record_success(client_ip, profile_key)
            return None
    if limiter is not None:
        limiter.record_failure(client_ip, profile_key)
    logger.warning("API server rejected invalid API key: %s", adapter._request_audit_log_suffix(request))
    return adapter._auth_failed_response()


def concurrency_limited_response(
    adapter: Any, *, profile_var: Any, reservation_var: Any, error_response: Callable[..., web.Response]
) -> Optional[web.Response]:
    """Return 429 when global or per-profile concurrent-run cap is reached (0 disables)."""
    reservation = reservation_var.get()
    limit = adapter._max_concurrent_runs
    if limit > 0:
        inflight = adapter.active_agent_work_count()
        if reservation and reservation["active"]:
            inflight -= 1
        if inflight >= limit:
            return error_response(
                f"Too many concurrent runs (max {limit})", 429, err_type="rate_limit_error",
                code="rate_limit_exceeded", headers={"Retry-After": "1", "Cache-Control": "no-store"},
            )
    profile_limit = getattr(adapter, "_max_concurrent_runs_per_profile", 0)
    if profile_limit > 0:
        profile_key = profile_var.get() or "default"
        profile_inflight = adapter.active_profile_agent_work_count(profile_key)
        if reservation and reservation["active"] and reservation.get("profile", "default") == profile_key:
            profile_inflight -= 1
        if profile_inflight >= profile_limit:
            return error_response(
                f"Too many concurrent runs for profile '{profile_key}' (max {profile_limit})",
                429, err_type="rate_limit_error", code="rate_limit_exceeded",
                headers={"Retry-After": "1", "Cache-Control": "no-store"},
            )
    return None


def recover_or_record_profile_model(
    adapter: Any, model: str, gateway_session_key: Any, *, profile_var: Any, logger: Any
) -> str:
    """Profile-isolated last-known-good model recovery and recording."""
    profile = profile_var.get()
    if not profile or profile == "default":
        model_cache = adapter._last_resolved_model
    else:
        by_profile = getattr(adapter, "_last_resolved_model_by_profile", None)
        if not isinstance(by_profile, dict):
            by_profile = adapter._last_resolved_model_by_profile = {}
        model_cache = by_profile.setdefault(profile, {})
    resolved_key = gateway_session_key or ""
    if not model:
        recovered = model_cache.get(resolved_key) or model_cache.get("*")
        if recovered and recovered != adapter._model_name:
            logger.warning(
                "Empty model resolved for session=%s — recovering "
                "last-known-good model %s (config read likely returned empty; see #35314)",
                resolved_key, recovered,
            )
            model = recovered
    elif model != adapter._model_name:
        if resolved_key:
            model_cache[resolved_key] = model
        model_cache["*"] = model
    return model
