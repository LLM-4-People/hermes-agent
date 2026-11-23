#!/usr/bin/env python3
"""Zero-dependency CLI self-check for the LLM-4-People Hermes Mobile companion server.

Verifies API server health, `/v1/capabilities` companion feature count, `/v1/toolsets`,
active `/p/<profile>` profiles, redacted `API_SERVER_KEY` presence, model/provider
readiness, and prints the exact copy-paste connection URL(s) for Hermes Mobile.

Security invariant: NEVER prints, echoes, or serializes any API key or secret value.

Usage:
    python3 scripts/mobile_doctor.py
    python3 scripts/mobile_doctor.py --hermes-home ~/.hermes --public-url https://hermes.example.com
    python3 scripts/mobile_doctor.py --profile alice --json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

# Core companion capability flags expected on LLM-4-People/hermes-agent.
COMPANION_REQUIRED_FEATURES: tuple[str, ...] = (
    "run_submission",
    "run_events_sse",
    "run_scoped_instructions",
    "live_ui_modding",
    "mobile_deliveries",
    "device_bridge",
    "memory_api",
    "skill_studio",
    "workspace_browser",
    "profile_defaults",
    "smart_notes",
    "universal_sharing",
    "habit_suggestions",
)

MOBILE_TOOLSETS: tuple[str, ...] = (
    "mobile_ui",
    "mobile_deliver",
    "mobile_device",
)

EXPECTED_ACTIVE_CAPABILITIES = 51
MIN_RECOMMENDED_KEY_LEN = 16
_PLACEHOLDER_KEY_MARKERS = ("change-me", "changeme", "example", "placeholder", "test-key", "your-api-key")
_PROFILE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_VALID_SCHEMES = frozenset({"http", "https"})
_DEFAULT_PORT = 8642
_DEFAULT_TIMEOUT = 8.0


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str  # "ok" | "warn" | "fail"
    summary: str
    details: dict[str, Any]


@dataclass(frozen=True)
class HttpResponse:
    status: int | None
    latency_ms: int
    body: Any
    error: str | None = None


def validate_http_url(raw: str, *, label: str = "URL") -> str:
    """Validate and normalize an http(s) base URL (strips trailing slash, rejects unsafe schemes)."""
    cleaned = raw.strip()
    if not cleaned:
        raise ValueError(f"{label} must not be empty.")
    parsed = urlparse(cleaned)
    if parsed.scheme.lower() not in _VALID_SCHEMES or not parsed.netloc:
        raise ValueError(f"{label} must be an http:// or https:// URL with a host.")
    if parsed.username or parsed.password:
        raise ValueError(f"{label} must not embed credentials in the URL.")
    return cleaned.rstrip("/")


def normalize_profile_name(raw: str | None) -> str | None:
    """Validate an optional profile name (lowercase slug; None means default profile)."""
    if raw is None:
        return None
    cleaned = raw.strip().lower()
    if not cleaned or cleaned == "default":
        return None
    if not _PROFILE_NAME_RE.match(cleaned):
        raise ValueError(
            f"Invalid profile name {raw!r}: use 1-64 lowercase letters, digits, '-' or '_'."
        )
    return cleaned


def parse_dotenv(path: Path) -> dict[str, str]:
    """Read key=value pairs from a .env file without mutating os.environ."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return values
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if key:
            values[key] = val
    return values


def _env_path_for_profile(hermes_home: Path, profile: str | None) -> Path:
    if profile:
        return hermes_home / "profiles" / profile / ".env"
    return hermes_home / ".env"


def _resolve_api_key(
    hermes_home: Path,
    profile: str | None,
    api_key_env: str,
) -> tuple[str | None, str]:
    """Resolve the profile's API_SERVER_KEY strictly in memory; returns (key, source_label)."""
    env_path = _env_path_for_profile(hermes_home, profile)
    file_env = parse_dotenv(env_path)
    file_key = file_env.get("API_SERVER_KEY", "").strip()
    if file_key:
        return file_key, str(env_path)
    if profile is None:
        env_key = os.environ.get(api_key_env, "").strip()
        if env_key:
            return env_key, f"env:{api_key_env}"
    return None, str(env_path)


def _is_placeholder_key(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in _PLACEHOLDER_KEY_MARKERS)


def _default_base_url(hermes_home: Path) -> str:
    root_env = parse_dotenv(hermes_home / ".env")
    raw_host = root_env.get("API_SERVER_HOST") or os.environ.get("API_SERVER_HOST") or "127.0.0.1"
    host = "127.0.0.1" if raw_host.strip() in ("", "0.0.0.0", "::", "[::]") else raw_host.strip()
    raw_port = root_env.get("API_SERVER_PORT") or os.environ.get("API_SERVER_PORT") or str(_DEFAULT_PORT)
    try:
        port = int(raw_port)
        if not (1 <= port <= 65535):
            port = _DEFAULT_PORT
    except ValueError:
        port = _DEFAULT_PORT
    return f"http://{host}:{port}"


def build_mobile_connection_url(base_url: str, profile: str | None) -> str:
    """Return the exact copy-paste Hermes Mobile connection URL for a profile."""
    normalized = validate_http_url(base_url, label="Base URL")
    if profile:
        return f"{normalized}/p/{profile}"
    return normalized


def _http_get_json(url: str, *, api_key: str | None, timeout: float) -> HttpResponse:
    validate_http_url(url, label="Request URL")
    headers = {"Accept": "application/json", "User-Agent": "hermes-mobile-doctor/1"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = Request(url, headers=headers, method="GET")
    started = time.monotonic()
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            elapsed = int((time.monotonic() - started) * 1000)
            try:
                body = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError as exc:
                return HttpResponse(resp.status, elapsed, None, f"invalid JSON: {exc.msg}")
            return HttpResponse(resp.status, elapsed, body, None)
    except HTTPError as exc:
        elapsed = int((time.monotonic() - started) * 1000)
        return HttpResponse(exc.code, elapsed, None, f"HTTP {exc.code} {exc.reason}")
    except (URLError, OSError, TimeoutError) as exc:
        elapsed = int((time.monotonic() - started) * 1000)
        reason = getattr(exc, "reason", exc)
        return HttpResponse(None, elapsed, None, str(reason))


def _check_api_key(key: str | None, source: str, profile: str | None) -> CheckResult:
    label = f"profile '{profile}'" if profile else "default profile"
    if not key:
        return CheckResult(
            name="api_key",
            status="fail",
            summary=f"API_SERVER_KEY not found for {label} ({source})",
            details={"configured": False, "source": source},
        )
    length = len(key)
    if _is_placeholder_key(key) or length < MIN_RECOMMENDED_KEY_LEN:
        return CheckResult(
            name="api_key",
            status="warn",
            summary=f"API_SERVER_KEY present for {label} (redacted, {length} chars) — use a random key >= {MIN_RECOMMENDED_KEY_LEN} chars",
            details={"configured": True, "length": length, "weak": True, "source": source},
        )
    return CheckResult(
        name="api_key",
        status="ok",
        summary=f"API_SERVER_KEY configured for {label} (redacted, {length} chars)",
        details={"configured": True, "length": length, "weak": False, "source": source},
    )


def _check_health(target_base: str, api_key: str | None, timeout: float) -> CheckResult:
    resp = _http_get_json(f"{target_base}/health", api_key=api_key, timeout=timeout)
    if resp.status == 200:
        return CheckResult(
            name="api_health",
            status="ok",
            summary=f"API server reachable ({resp.latency_ms} ms)",
            details={"status_code": 200, "latency_ms": resp.latency_ms},
        )
    detail_msg = resp.error or f"HTTP {resp.status}"
    return CheckResult(
        name="api_health",
        status="fail",
        summary=f"API server health check failed at {target_base}/health ({detail_msg})",
        details={"status_code": resp.status, "latency_ms": resp.latency_ms, "error": detail_msg},
    )


def count_active_capabilities(features: dict[str, Any]) -> int:
    """Count active /v1/capabilities feature entries (True bools, dicts, and string headers)."""
    count = 0
    for val in features.values():
        is_active_dict = (
            isinstance(val, dict)
            and bool(val)
            and val.get("supported") is not False
            and val.get("enabled") is not False
        )
        if val is True or (isinstance(val, str) and val.strip()) or is_active_dict:
            count += 1
    return count


def _check_capabilities(
    target_base: str,
    api_key: str | None,
    timeout: float,
) -> tuple[CheckResult, dict[str, Any] | None]:
    resp = _http_get_json(f"{target_base}/v1/capabilities", api_key=api_key, timeout=timeout)
    if resp.status == 401:
        return (
            CheckResult(
                name="capabilities",
                status="fail",
                summary="Authentication rejected (HTTP 401) — verify API_SERVER_KEY matches this profile",
                details={"status_code": 401},
            ),
            None,
        )
    if resp.status == 404:
        return (
            CheckResult(
                name="capabilities",
                status="fail",
                summary="Endpoint returned HTTP 404 — check profile name and gateway.multiplex_profiles",
                details={"status_code": 404},
            ),
            None,
        )
    if resp.status != 200 or not isinstance(resp.body, dict):
        return (
            CheckResult(
                name="capabilities",
                status="fail",
                summary=f"Failed to read /v1/capabilities ({resp.error or f'HTTP {resp.status}'})",
                details={"status_code": resp.status, "error": resp.error},
            ),
            None,
        )

    features = resp.body.get("features") if isinstance(resp.body.get("features"), dict) else {}
    endpoints = resp.body.get("endpoints") if isinstance(resp.body.get("endpoints"), dict) else {}
    active_count = count_active_capabilities(features)
    missing = [flag for flag in COMPANION_REQUIRED_FEATURES if not features.get(flag)]
    details = {
        "active_capabilities": active_count,
        "expected_capabilities": EXPECTED_ACTIVE_CAPABILITIES,
        "endpoints": len(endpoints),
        "missing_companion_features": missing,
        "auth_required": bool((resp.body.get("auth") or {}).get("required")),
    }
    if missing:
        return (
            CheckResult(
                name="capabilities",
                status="warn",
                summary=(
                    f"{active_count} active capabilities ({len(endpoints)} endpoints), missing "
                    f"{len(missing)} companion fork flag(s): {', '.join(missing[:4])}"
                    + ("..." if len(missing) > 4 else "")
                    + " — is this stock upstream instead of LLM-4-People/hermes-agent?"
                ),
                details=details,
            ),
            resp.body,
        )
    return (
        CheckResult(
            name="capabilities",
            status="ok",
            summary=f"Companion fork verified: {active_count} active capabilities, {len(endpoints)} endpoints",
            details=details,
        ),
        resp.body,
    )


def _check_toolsets(target_base: str, api_key: str | None, timeout: float) -> CheckResult:
    resp = _http_get_json(f"{target_base}/v1/toolsets", api_key=api_key, timeout=timeout)
    if resp.status != 200 or not isinstance(resp.body, dict):
        return CheckResult(
            name="toolsets",
            status="warn" if resp.status in (401, 404) else "fail",
            summary=f"Could not inspect /v1/toolsets ({resp.error or f'HTTP {resp.status}'})",
            details={"status_code": resp.status},
        )
    rows = resp.body.get("data")
    if not isinstance(rows, list):
        return CheckResult(
            name="toolsets",
            status="warn",
            summary="/v1/toolsets returned unexpected payload format",
            details={"status_code": 200},
        )
    enabled = [r.get("name") for r in rows if isinstance(r, dict) and r.get("enabled") and r.get("name")]
    missing_mobile = [name for name in MOBILE_TOOLSETS if name not in enabled]
    details = {
        "enabled_count": len(enabled),
        "total_count": len(rows),
        "missing_mobile_toolsets": missing_mobile,
    }
    if missing_mobile:
        return CheckResult(
            name="toolsets",
            status="warn",
            summary=f"{len(enabled)}/{len(rows)} toolsets enabled; missing mobile toolset(s): {', '.join(missing_mobile)}",
            details=details,
        )
    return CheckResult(
        name="toolsets",
        status="ok",
        summary=f"{len(enabled)}/{len(rows)} toolsets enabled (including mobile_ui, mobile_deliver, mobile_device)",
        details=details,
    )


def _check_models(
    target_base: str,
    api_key: str | None,
    timeout: float,
    capabilities_body: dict[str, Any] | None,
) -> CheckResult:
    cap_model = ""
    if isinstance(capabilities_body, dict):
        cap_model = str(capabilities_body.get("model") or "").strip()
    resp = _http_get_json(f"{target_base}/v1/models", api_key=api_key, timeout=timeout)
    models: list[str] = []
    if resp.status == 200 and isinstance(resp.body, dict):
        raw_data = resp.body.get("data")
        if isinstance(raw_data, list):
            models = [str(m.get("id")) for m in raw_data if isinstance(m, dict) and m.get("id")]
    if models or cap_model:
        label = cap_model or models[0]
        return CheckResult(
            name="models",
            status="ok",
            summary=f"Model/provider ready (active: {label}, advertised models: {len(models)})",
            details={"active_model": label, "model_count": len(models)},
        )
    return CheckResult(
        name="models",
        status="warn",
        summary="No active model advertised yet — run `hermes model` to configure a provider",
        details={"active_model": None, "model_count": 0, "status_code": resp.status},
    )


def _discover_profiles(hermes_home: Path) -> tuple[CheckResult, list[dict[str, Any]]]:
    state_path = hermes_home / "gateway_state.json"
    served: list[str] = []
    gateway_state = None
    code_sha = None
    if state_path.is_file():
        try:
            raw_state = json.loads(state_path.read_text(encoding="utf-8-sig", errors="replace"))
            if isinstance(raw_state, dict):
                gateway_state = raw_state.get("gateway_state")
                code_sha = raw_state.get("code_sha")
                raw_served = raw_state.get("served_profiles")
                if isinstance(raw_served, list):
                    served = [str(p).strip().lower() for p in raw_served if str(p).strip()]
        except (OSError, json.JSONDecodeError):
            pass

    disk_profiles: list[str] = []
    profiles_dir = hermes_home / "profiles"
    if profiles_dir.is_dir():
        for entry in sorted(profiles_dir.iterdir()):
            if entry.is_dir() and _PROFILE_NAME_RE.match(entry.name.lower()):
                disk_profiles.append(entry.name.lower())

    ordered = list(dict.fromkeys(["default", *served, *disk_profiles]))
    profile_rows: list[dict[str, Any]] = []
    missing_keys: list[str] = []
    for name in ordered:
        prof_arg = None if name == "default" else name
        env_path = _env_path_for_profile(hermes_home, prof_arg)
        has_key = bool(parse_dotenv(env_path).get("API_SERVER_KEY", "").strip())
        is_served = (name in served) if served else (name == "default")
        if is_served and not has_key:
            missing_keys.append(name)
        profile_rows.append(
            {
                "name": name,
                "served": is_served,
                "api_key_configured": has_key,
            }
        )

    details = {
        "gateway_state": gateway_state,
        "code_sha": str(code_sha)[:12] if code_sha else None,
        "profiles": profile_rows,
    }
    if missing_keys:
        return (
            CheckResult(
                name="profiles",
                status="warn",
                summary=f"{len(profile_rows)} profile(s) discovered; missing API_SERVER_KEY for served profile(s): {', '.join(missing_keys)}",
                details=details,
            ),
            profile_rows,
        )
    served_names = [r["name"] for r in profile_rows if r["served"]]
    summary = f"Active profiles ready: {', '.join(served_names) if served_names else 'default'}"
    if code_sha:
        summary += f" (rev {str(code_sha)[:8]})"
    return CheckResult(name="profiles", status="ok", summary=summary, details=details), profile_rows


def run_doctor(
    *,
    hermes_home: Path,
    base_url: str | None = None,
    public_url: str | None = None,
    profile: str | None = None,
    api_key_env: str = "API_SERVER_KEY",
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Run all companion diagnostics and return a strictly secret-free report dictionary."""
    norm_profile = normalize_profile_name(profile)
    resolved_base = validate_http_url(base_url or _default_base_url(hermes_home), label="Server URL")
    resolved_public = validate_http_url(public_url, label="Public URL") if public_url else resolved_base
    target_base = build_mobile_connection_url(resolved_base, norm_profile)

    api_key, key_source = _resolve_api_key(hermes_home, norm_profile, api_key_env)
    key_check = _check_api_key(api_key, key_source, norm_profile)
    health_check = _check_health(target_base, api_key, timeout)
    cap_check, cap_body = _check_capabilities(target_base, api_key, timeout)
    toolsets_check = _check_toolsets(target_base, api_key, timeout)
    models_check = _check_models(target_base, api_key, timeout, cap_body)
    profiles_check, profile_rows = _discover_profiles(hermes_home)

    checks = [key_check, health_check, cap_check, toolsets_check, models_check, profiles_check]

    if norm_profile:
        connection_urls = {norm_profile: build_mobile_connection_url(resolved_public, norm_profile)}
    else:
        connection_urls = {
            row["name"]: build_mobile_connection_url(
                resolved_public, None if row["name"] == "default" else row["name"]
            )
            for row in profile_rows
        }

    ok = all(item.status != "fail" for item in checks)
    return {
        "ok": ok,
        "profile": norm_profile or "default",
        "server_url": target_base,
        "connection_urls": connection_urls,
        "checks": [asdict(item) for item in checks],
    }


def format_report_text(report: dict[str, Any]) -> str:
    """Format a secret-free human-readable diagnostic report."""
    badges = {"ok": "[OK]  ", "warn": "[WARN]", "fail": "[FAIL]"}
    lines = [
        "Hermes Mobile Companion Doctor",
        f"  Target: {report['server_url']} (profile: {report['profile']})",
        "",
    ]
    for item in report["checks"]:
        badge = badges.get(item["status"], "[INFO]")
        lines.append(f"  {badge} {item['name']}: {item['summary']}")

    lines.append("")
    lines.append("Hermes Mobile connection URL(s) (paste into the Android app with that profile's API_SERVER_KEY):")
    for prof_name, url in report["connection_urls"].items():
        lines.append(f"  - {prof_name}: {url}")

    verdict = "READY" if report["ok"] else "ACTION REQUIRED"
    lines.append("")
    lines.append(f"Result: {verdict}")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mobile_doctor.py",
        description="Verify a Hermes Agent server for Hermes Mobile (never prints secrets).",
    )
    parser.add_argument(
        "--url",
        help="Local or reachable API server base URL (default: from API_SERVER_HOST/API_SERVER_PORT or http://127.0.0.1:8642)",
    )
    parser.add_argument(
        "--public-url",
        help="Public HTTPS reverse-proxy URL to format copy-paste Hermes Mobile connection URLs (e.g. https://hermes.example.com)",
    )
    parser.add_argument(
        "--hermes-home",
        type=Path,
        default=Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))),
        help="Path to the Hermes data directory (default: $HERMES_HOME or ~/.hermes)",
    )
    parser.add_argument(
        "--profile",
        help="Optional profile name to verify via /p/<profile>/ (default: default profile)",
    )
    parser.add_argument(
        "--api-key-env",
        default="API_SERVER_KEY",
        help="Environment variable fallback for the default profile bearer key (default: API_SERVER_KEY)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=_DEFAULT_TIMEOUT,
        help=f"Per-request HTTP timeout in seconds (default: {_DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable, secret-free JSON report",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        print("mobile_doctor: --timeout must be positive", file=sys.stderr)
        return 2
    try:
        report = run_doctor(
            hermes_home=args.hermes_home,
            base_url=args.url,
            public_url=args.public_url,
            profile=args.profile,
            api_key_env=args.api_key_env,
            timeout=args.timeout,
        )
    except ValueError as exc:
        print(f"mobile_doctor: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report_text(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
