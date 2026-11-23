"""Unit and mock-HTTP tests for scripts/mobile_doctor.py."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import sys
import threading
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "mobile_doctor.py"


@pytest.fixture(scope="module")
def doctor():
    spec = importlib.util.spec_from_file_location("mobile_doctor", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["mobile_doctor"] = module
    spec.loader.exec_module(module)
    return module


def _full_companion_features(doctor_mod) -> dict[str, Any]:
    features: dict[str, Any] = {flag: True for flag in doctor_mod.COMPANION_REQUIRED_FEATURES}
    # Add additional flags to reach 51 active entries (48 bools + 1 dict + 2 headers).
    for idx in range(48 - len(features)):
        features[f"extra_feature_{idx}"] = True
    features["runs_idempotency"] = {"supported": True, "header": "Idempotency-Key"}
    features["session_continuity_header"] = "X-Hermes-Session-Id"
    features["session_key_header"] = "X-Hermes-Session-Key"
    features["admin_config_rw"] = False
    features["disabled_structured"] = {"supported": False}
    return features


def _start_mock_server(routes: dict[str, tuple[int, dict[str, Any]]], expected_keys: dict[str, str]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            path = self.path
            expected_key = expected_keys.get(path)
            if expected_key is not None:
                auth = self.headers.get("Authorization")
                if auth != f"Bearer {expected_key}":
                    self.send_response(401)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"error":"unauthorized"}')
                    return
            entry = routes.get(path)
            if entry is None:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":"not_found"}')
                return
            status, payload = entry
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_port}"


def test_healthy_companion_fork_and_zero_secret_leakage(doctor, tmp_path: Path, capsys):
    sentinel_default = "sk-sentinel-default-never-leak-99887766"
    sentinel_alice = "sk-sentinel-alice-never-leak-11223344"

    (tmp_path / ".env").write_text(f"API_SERVER_KEY={sentinel_default}\n", encoding="utf-8")
    alice_dir = tmp_path / "profiles" / "alice"
    alice_dir.mkdir(parents=True)
    (alice_dir / ".env").write_text(f'API_SERVER_KEY="{sentinel_alice}"\n', encoding="utf-8")
    (tmp_path / "gateway_state.json").write_text(
        json.dumps(
            {
                "gateway_state": "running",
                "code_sha": "6601e920be2461eb832caeea70299f20d0f0ea7f",
                "served_profiles": ["default", "alice"],
            }
        ),
        encoding="utf-8",
    )

    features = _full_companion_features(doctor)
    routes = {
        "/health": (200, {"status": "ok"}),
        "/v1/capabilities": (
            200,
            {
                "model": "hermes-agent",
                "auth": {"type": "bearer", "required": True},
                "features": features,
                "endpoints": {f"ep_{i}": {"method": "GET", "path": f"/x/{i}"} for i in range(71)},
            },
        ),
        "/v1/toolsets": (
            200,
            {
                "object": "list",
                "data": [
                    {"name": name, "enabled": True}
                    for name in ("web", "file", "mobile_ui", "mobile_deliver", "mobile_device")
                ]
                + [{"name": "discord", "enabled": False}],
            },
        ),
        "/v1/models": (200, {"object": "list", "data": [{"id": "hermes-agent"}]}),
    }
    expected_keys = {
        "/v1/capabilities": sentinel_default,
        "/v1/toolsets": sentinel_default,
        "/v1/models": sentinel_default,
    }
    server, base_url = _start_mock_server(routes, expected_keys)
    try:
        rc = doctor.main(
            [
                "--hermes-home",
                str(tmp_path),
                "--url",
                base_url,
                "--public-url",
                "https://hermes.example.com/",
            ]
        )
        text_out = capsys.readouterr()
        assert rc == 0
        assert "Result: READY" in text_out.out
        assert "51 active capabilities, 71 endpoints" in text_out.out
        assert "- default: https://hermes.example.com" in text_out.out
        assert "- alice: https://hermes.example.com/p/alice" in text_out.out
        assert sentinel_default not in text_out.out
        assert sentinel_default not in text_out.err
        assert sentinel_alice not in text_out.out
        assert sentinel_alice not in text_out.err

        rc_json = doctor.main(
            [
                "--hermes-home",
                str(tmp_path),
                "--url",
                base_url,
                "--public-url",
                "https://hermes.example.com",
                "--json",
            ]
        )
        json_out = capsys.readouterr()
        assert rc_json == 0
        assert sentinel_default not in json_out.out
        assert sentinel_alice not in json_out.out
        payload = json.loads(json_out.out)
        assert payload["ok"] is True
        assert payload["connection_urls"] == {
            "default": "https://hermes.example.com",
            "alice": "https://hermes.example.com/p/alice",
        }
        assert all(c["status"] == "ok" for c in payload["checks"])
    finally:
        server.shutdown()
        server.server_close()


def test_profile_target_and_upstream_non_fork_warning(doctor, tmp_path: Path):
    alice_key = "sk-alice-strong-random-key-12345678"
    alice_dir = tmp_path / "profiles" / "alice"
    alice_dir.mkdir(parents=True)
    (alice_dir / ".env").write_text(f"API_SERVER_KEY={alice_key}\n", encoding="utf-8")

    routes = {
        "/p/alice/health": (200, {"status": "ok"}),
        "/p/alice/v1/capabilities": (
            200,
            {
                "model": "upstream-model",
                "auth": {"type": "bearer", "required": True},
                "features": {"chat_completions": True, "run_submission": True},
                "endpoints": {"health": {"method": "GET", "path": "/health"}},
            },
        ),
        "/p/alice/v1/toolsets": (200, {"object": "list", "data": [{"name": "web", "enabled": True}]}),
        "/p/alice/v1/models": (200, {"object": "list", "data": [{"id": "upstream-model"}]}),
    }
    expected_keys = {
        "/p/alice/v1/capabilities": alice_key,
        "/p/alice/v1/toolsets": alice_key,
        "/p/alice/v1/models": alice_key,
    }
    server, base_url = _start_mock_server(routes, expected_keys)
    try:
        report = doctor.run_doctor(
            hermes_home=tmp_path,
            base_url=base_url,
            public_url="https://hermes.example.com/",
            profile="Alice",
        )
        assert report["ok"] is True
        assert report["profile"] == "alice"
        assert report["connection_urls"] == {"alice": "https://hermes.example.com/p/alice"}
        by_name = {c["name"]: c for c in report["checks"]}
        assert by_name["capabilities"]["status"] == "warn"
        assert "stock upstream" in by_name["capabilities"]["summary"]
        assert by_name["toolsets"]["status"] == "warn"
        assert "mobile_device" in by_name["toolsets"]["details"]["missing_mobile_toolsets"]
        assert alice_key not in json.dumps(report)
    finally:
        server.shutdown()
        server.server_close()


def test_wrong_key_and_missing_key_fail_closed(doctor, tmp_path: Path, monkeypatch):
    monkeypatch.delenv("API_SERVER_KEY", raising=False)
    routes = {
        "/health": (200, {"status": "ok"}),
        "/v1/capabilities": (200, {"features": {}, "endpoints": {}}),
        "/v1/toolsets": (200, {"data": []}),
        "/v1/models": (200, {"data": []}),
    }
    expected_keys = {
        "/v1/capabilities": "correct-server-key-9999999999999999",
        "/v1/toolsets": "correct-server-key-9999999999999999",
        "/v1/models": "correct-server-key-9999999999999999",
    }
    server, base_url = _start_mock_server(routes, expected_keys)
    try:
        report = doctor.run_doctor(hermes_home=tmp_path, base_url=base_url)
        assert report["ok"] is False
        by_name = {c["name"]: c for c in report["checks"]}
        assert by_name["api_key"]["status"] == "fail"
        assert by_name["capabilities"]["status"] == "fail"
        assert by_name["capabilities"]["details"]["status_code"] == 401
    finally:
        server.shutdown()
        server.server_close()


def test_weak_placeholder_key_warns_without_echoing(doctor, tmp_path: Path):
    weak_key = "change-me-local-dev"
    (tmp_path / ".env").write_text(f"API_SERVER_KEY={weak_key}\n", encoding="utf-8")
    check = doctor._check_api_key(weak_key, str(tmp_path / ".env"), None)
    assert check.status == "warn"
    assert check.details["weak"] is True
    assert weak_key not in check.summary


@pytest.mark.parametrize(
    "bad_url",
    [
        "",
        "file:///etc/passwd",
        "ftp://hermes.example.com",
        "https://user:pass@hermes.example.com",
        "not-a-url",
    ],
)
def test_validate_http_url_rejects_unsafe_inputs(doctor, bad_url: str):
    with pytest.raises(ValueError):
        doctor.validate_http_url(bad_url)


@pytest.mark.parametrize("bad_profile", ["../etc", "alice/bob", "bad space", "-leading-dash"])
def test_normalize_profile_name_rejects_traversal(doctor, bad_profile: str):
    with pytest.raises(ValueError):
        doctor.normalize_profile_name(bad_profile)


def test_main_rejects_invalid_timeout_and_url(doctor, tmp_path: Path):
    assert doctor.main(["--hermes-home", str(tmp_path), "--timeout", "0"]) == 2
    assert doctor.main(["--hermes-home", str(tmp_path), "--url", "file:///etc/hosts"]) == 2
