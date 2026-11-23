"""Brutal multi-user concurrency, fair-share, SQLite WAL, and profile isolation tests.

Proves under simultaneous multi-profile + multi-thread load:
- Fail-closed per-profile auth (`401` + `Cache-Control: no-store`, zero fallback)
- Per-(IP, profile) auth brute-force rate limiting (`429` + `Retry-After`, zero secret logs)
- 15+ simultaneous runs across 3 isolated profiles (5 true-parallel discussions per user)
- Zero cross-profile leakage across ContextVars, secrets, runs, SSE, models, notes, shares, UI, deliveries, memory
- Fair-share per-profile concurrency cap prevents noisy-neighbor starvation
- Thread-safe `_set_run_status` with zero terminal-status regression
- Zero `SQLITE_BUSY` or lost updates across concurrent WAL reads & writes
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import threading
import time
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent import secret_scope
from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _api_request_profile,
    security_headers_middleware,
)
from gateway.platforms.api_server_auth_guard import AuthFailureLimiter, extract_client_ip
from gateway.platforms.api_server_habits import list_habit_suggestions, record_habit_event
from gateway.platforms.api_server_memory import _sync_add_memory, _sync_get_memory
from gateway.platforms.api_server_response_store import ResponseStore
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from hermes_constants import get_hermes_home
from tools.mobile_deliver import MobileDeliveryStore
from tools.mobile_notes import create_note, list_notes
from tools.mobile_sharing import grant_share, list_incoming_shares, list_outgoing_shares
from tools.mobile_ui_patch import MobileUiStore


DEFAULT_KEY = "default-key-0123456789abcdef0123"
ALICE_KEY = "alice-key-0123456789abcdef012345"
BOB_KEY = "bob-key-0123456789abcdef01234567"
CHARLIE_KEY = "charlie-key-0123456789abcdef0123"

PROFILE_KEYS = {
    "alice": ALICE_KEY,
    "bob": BOB_KEY,
    "charlie": CHARLIE_KEY,
}


@pytest.fixture(autouse=True)
def _reset_multiplex_flag():
    secret_scope.set_multiplex_active(False)
    yield
    secret_scope.set_multiplex_active(False)


def _setup_multiplex_homes(tmp_path: Path, monkeypatch, *, include_unconfigured: bool = False) -> dict[str, Path]:
    root = tmp_path / "hermes"
    root.mkdir(parents=True)
    (root / "config.yaml").write_text("{}\n", encoding="utf-8")
    (root / ".env").write_text(f"API_SERVER_KEY={DEFAULT_KEY}\n", encoding="utf-8")
    homes: dict[str, Path] = {"default": root}
    for name, key in PROFILE_KEYS.items():
        home = root / "profiles" / name
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
        (home / ".env").write_text(f"API_SERVER_KEY={key}\n", encoding="utf-8")
        homes[name] = home
    if include_unconfigured:
        unconf = root / "profiles" / "unconfigured"
        unconf.mkdir(parents=True)
        (unconf / "config.yaml").write_text("{}\n", encoding="utf-8")
        homes["unconfigured"] = unconf
    monkeypatch.setenv("HERMES_HOME", str(root))
    return homes


def _close_adapter(adapter: APIServerAdapter) -> None:
    for store in (getattr(adapter, "_response_store", None), getattr(adapter, "_run_idempotency_store", None)):
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
    for store in list(getattr(adapter, "_response_stores", {}).values()):
        try:
            store.close()
        except Exception:
            pass


_CREATED_ADAPTERS: list[APIServerAdapter] = []


@pytest.fixture(autouse=True)
def _close_created_adapters():
    yield
    while _CREATED_ADAPTERS:
        _close_adapter(_CREATED_ADAPTERS.pop())


def _make_adapter(key: str = DEFAULT_KEY) -> APIServerAdapter:
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": key}))
    _CREATED_ADAPTERS.append(adapter)

    class _Runner:
        config = GatewayConfig(multiplex_profiles=True)

    adapter.gateway_runner = _Runner()
    secret_scope.set_multiplex_active(True)
    return adapter


def _make_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(
        middlewares=[adapter._make_profile_prefix_middleware(), security_headers_middleware]
    )
    app["api_server_adapter"] = adapter
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
        app.router.add_route(method, f"/p/{{profile}}{path}", handler)
    return app


def _auth_headers(key: str, *, forwarded_for: str | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {key}"}
    if forwarded_for:
        headers["X-Forwarded-For"] = forwarded_for
    return headers


@pytest.mark.asyncio
async def test_cross_profile_auth_matrix_fail_closed_and_no_store(tmp_path, monkeypatch):
    _setup_multiplex_homes(tmp_path, monkeypatch, include_unconfigured=True)
    adapter = _make_adapter()

    async with TestClient(TestServer(_make_app(adapter))) as cli:
        # Every profile rejects every other profile's key with 401 + Cache-Control: no-store
        for target_profile, own_key in PROFILE_KEYS.items():
            ok_resp = await cli.get(f"/p/{target_profile}/v1/capabilities", headers=_auth_headers(own_key))
            assert ok_resp.status == 200

            for wrong_key in [DEFAULT_KEY, *[k for p, k in PROFILE_KEYS.items() if p != target_profile], "wrong-secret-key-999999"]:
                bad_resp = await cli.get(f"/p/{target_profile}/v1/capabilities", headers=_auth_headers(wrong_key))
                assert bad_resp.status == 401
                assert bad_resp.headers.get("Cache-Control") == "no-store"

        # Named profile with NO API_SERVER_KEY fails closed — never falls back to default key
        for candidate_key in (DEFAULT_KEY, ALICE_KEY, ""):
            headers = _auth_headers(candidate_key) if candidate_key else {}
            resp = await cli.get("/p/unconfigured/v1/capabilities", headers=headers)
            assert resp.status == 401
            assert resp.headers.get("Cache-Control") == "no-store"


@pytest.mark.asyncio
async def test_auth_failure_rate_limiting_isolated_per_ip_and_profile_no_secret_logged(
    tmp_path, monkeypatch, caplog
):
    _setup_multiplex_homes(tmp_path, monkeypatch)
    adapter = _make_adapter()
    adapter._auth_limiter = AuthFailureLimiter(max_attempts=4, window_seconds=60)
    secret_probe = "super-secret-attacker-probe-token-987654321"

    with caplog.at_level(logging.WARNING):
        async with TestClient(TestServer(_make_app(adapter))) as cli:
            # 4 bad attempts against alice from attacker IP -> 401 each
            for _ in range(4):
                resp = await cli.get(
                    "/p/alice/v1/capabilities",
                    headers=_auth_headers(secret_probe, forwarded_for="203.0.113.10"),
                )
                assert resp.status == 401
                assert resp.headers.get("Cache-Control") == "no-store"

            # 5th attempt from attacker IP against alice -> 429 with Retry-After + no-store
            blocked = await cli.get(
                "/p/alice/v1/capabilities",
                headers=_auth_headers(ALICE_KEY, forwarded_for="203.0.113.10"),
            )
            assert blocked.status == 429
            assert int(blocked.headers.get("Retry-After", "0")) >= 1
            assert blocked.headers.get("Cache-Control") == "no-store"
            body = await blocked.json()
            assert body["error"]["code"] == "auth_rate_limited"

            # Isolated by profile: bob on the SAME IP still succeeds
            bob_ok = await cli.get(
                "/p/bob/v1/capabilities",
                headers=_auth_headers(BOB_KEY, forwarded_for="203.0.113.10"),
            )
            assert bob_ok.status == 200

            # Isolated by IP: alice from a DIFFERENT IP still succeeds
            alice_other_ip = await cli.get(
                "/p/alice/v1/capabilities",
                headers=_auth_headers(ALICE_KEY, forwarded_for="203.0.113.11"),
            )
            assert alice_other_ip.status == 200

            # Valid auth before threshold clears accumulated failures on charlie
            for _ in range(3):
                r = await cli.get(
                    "/p/charlie/v1/capabilities",
                    headers=_auth_headers(secret_probe, forwarded_for="203.0.113.12"),
                )
                assert r.status == 401
            ok = await cli.get(
                "/p/charlie/v1/capabilities",
                headers=_auth_headers(CHARLIE_KEY, forwarded_for="203.0.113.12"),
            )
            assert ok.status == 200
            for _ in range(3):
                r = await cli.get(
                    "/p/charlie/v1/capabilities",
                    headers=_auth_headers(secret_probe, forwarded_for="203.0.113.12"),
                )
                assert r.status == 401

    # Never log any secret material
    all_logs = "\n".join(rec.getMessage() for rec in caplog.records)
    assert secret_probe not in all_logs
    assert ALICE_KEY not in all_logs
    assert BOB_KEY not in all_logs

    # Non-loopback remote peer cannot spoof X-Forwarded-For to escape its IP bucket
    direct_req = SimpleNamespace(
        remote="198.51.100.77",
        transport=None,
        headers={"X-Forwarded-For": "10.0.0.1", "X-Real-IP": "10.0.0.2"},
    )
    assert extract_client_ip(direct_req) == "198.51.100.77"


@pytest.mark.asyncio
async def test_15_concurrent_runs_across_3_profiles_true_parallel_per_user_and_zero_leakage(
    tmp_path, monkeypatch
):
    homes = _setup_multiplex_homes(tmp_path, monkeypatch)
    adapter = _make_adapter()
    # Confirm defaults comfortably allow 15 simultaneous runs (3 profiles x 5 runs)
    assert adapter._max_concurrent_runs >= 32
    assert adapter._max_concurrent_runs_per_profile >= 12

    runs_per_profile = 5
    per_profile_barriers = {
        name: threading.Barrier(runs_per_profile, timeout=10.0)
        for name in PROFILE_KEYS
    }
    overlap_lock = threading.Lock()
    active_by_profile: dict[str, int] = {name: 0 for name in PROFILE_KEYS}
    peak_by_profile: dict[str, int] = {name: 0 for name in PROFILE_KEYS}
    global_active = [0]
    global_peak = [0]
    worker_checks: list[tuple[str, str, Path, str]] = []

    class _ParallelFakeAgent:
        def __init__(self, *, stream_delta_callback=None, tool_progress_callback=None, session_id=None, **_kwargs):
            self._stream_cb = stream_delta_callback
            self._tool_cb = tool_progress_callback
            self.session_id = session_id or "sess"
            self.session_prompt_tokens = 11
            self.session_completion_tokens = 7
            self.session_total_tokens = 18
            self.provider = "mock"
            self.model = "mock-model"

        def run_conversation(self, *, user_message: str, **_kwargs):
            profile = _api_request_profile.get()
            home = get_hermes_home().resolve()
            active_key = secret_scope.get_secret("API_SERVER_KEY", "")
            assert profile in PROFILE_KEYS, f"Worker thread missing _api_request_profile: {profile!r}"
            assert home == homes[profile].resolve(), f"HERMES_HOME leaked: {home} != {homes[profile]}"
            assert active_key == PROFILE_KEYS[profile], f"Secret leaked for {profile}"

            with overlap_lock:
                active_by_profile[profile] += 1
                peak_by_profile[profile] = max(peak_by_profile[profile], active_by_profile[profile])
                global_active[0] += 1
                global_peak[0] = max(global_peak[0], global_active[0])
            try:
                # Barrier PROVES all 5 discussions for this user execute truly in parallel (never serialized)
                per_profile_barriers[profile].wait()

                if self._tool_cb:
                    self._tool_cb("tool.started", "mobile_notes", f"writing {user_message}")
                # Profile-scoped SQLite writes inside concurrent agent worker threads
                create_note(title=f"Note {user_message}", content=f"Private to {profile}: {user_message}")
                MobileDeliveryStore().deliver(
                    "notice",
                    f"Done {user_message}",
                    {"body": f"Delivered for {profile}"},
                    session_id=self.session_id,
                )
                if self._stream_cb:
                    self._stream_cb(f"chunk:{profile}:{user_message}")
                if self._tool_cb:
                    self._tool_cb("tool.completed", "mobile_notes", None, duration=0.01, is_error=False)
                with overlap_lock:
                    worker_checks.append((profile, user_message, home, active_key))
                return {"final_response": f"reply:{profile}:{user_message}"}
            finally:
                with overlap_lock:
                    active_by_profile[profile] -= 1
                    global_active[0] -= 1

    monkeypatch.setattr(adapter, "_create_agent", lambda **kw: _ParallelFakeAgent(**kw))

    async def _read_sse_events(cli: TestClient, profile: str, run_id: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        resp = await cli.get(
            f"/p/{profile}/v1/runs/{run_id}/events",
            headers=_auth_headers(PROFILE_KEYS[profile]),
        )
        assert resp.status == 200
        raw = await resp.text()
        for line in raw.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
        return events

    async with TestClient(TestServer(_make_app(adapter))) as cli:
        # Launch all 15 runs concurrently
        launch_specs = [
            (profile, f"{profile}-task-{idx}")
            for profile in PROFILE_KEYS
            for idx in range(runs_per_profile)
        ]

        async def _start_one(profile: str, msg: str):
            resp = await cli.post(
                f"/p/{profile}/v1/runs",
                headers=_auth_headers(PROFILE_KEYS[profile]),
                json={"input": msg, "session_id": f"sess-{msg}"},
            )
            assert resp.status == 202, await resp.text()
            data = await resp.json()
            return profile, msg, data["run_id"]

        launched = await asyncio.gather(*(_start_one(p, m) for p, m in launch_specs))
        assert len(launched) == 15

        # Stream SSE events for all 15 runs concurrently
        sse_results = await asyncio.gather(
            *(_read_sse_events(cli, profile, run_id) for profile, _msg, run_id in launched)
        )

        # Verify every run finished with exact isolated output & SSE completion
        for (profile, msg, run_id), events in zip(launched, sse_results):
            event_names = [ev.get("event") for ev in events]
            assert "run.completed" in event_names, f"Missing run.completed for {profile}/{run_id}: {event_names}"
            completed = [ev for ev in events if ev.get("event") == "run.completed"][-1]
            assert completed["output"] == f"reply:{profile}:{msg}"

            status_resp = await cli.get(
                f"/p/{profile}/v1/runs/{run_id}",
                headers=_auth_headers(PROFILE_KEYS[profile]),
            )
            assert status_resp.status == 200
            status_body = await status_resp.json()
            assert status_body["status"] == "completed"
            assert status_body["output"] == f"reply:{profile}:{msg}"

            # Cross-profile run status, SSE, and stop MUST fail closed (404)
            other_profile = next(p for p in PROFILE_KEYS if p != profile)
            other_headers = _auth_headers(PROFILE_KEYS[other_profile])
            assert (await cli.get(f"/p/{other_profile}/v1/runs/{run_id}", headers=other_headers)).status == 404
            assert (await cli.get(f"/p/{other_profile}/v1/runs/{run_id}/events", headers=other_headers)).status == 404
            assert (await cli.post(f"/p/{other_profile}/v1/runs/{run_id}/stop", headers=other_headers)).status == 404

        # Verify true parallelism per user & zero cross-profile state contamination
        for profile, key in PROFILE_KEYS.items():
            assert peak_by_profile[profile] == runs_per_profile
            notes_resp = await cli.get(f"/p/{profile}/api/notes", headers=_auth_headers(key))
            assert notes_resp.status == 200
            notes = (await notes_resp.json())["data"]
            assert len(notes) == runs_per_profile
            assert all(f"Private to {profile}:" in n["content"] for n in notes)

            deliveries_resp = await cli.get(f"/p/{profile}/api/mobile/deliveries", headers=_auth_headers(key))
            assert deliveries_resp.status == 200
            deliveries = (await deliveries_resp.json())["data"]
            assert len(deliveries) == runs_per_profile
            assert all(d["payload"]["body"] == f"Delivered for {profile}" for d in deliveries)

    assert len(worker_checks) == 15
    assert global_peak[0] >= runs_per_profile
    assert adapter.active_agent_work_count() == 0


@pytest.mark.asyncio
async def test_per_profile_fair_share_cap_prevents_noisy_neighbor_starvation(tmp_path, monkeypatch):
    _setup_multiplex_homes(tmp_path, monkeypatch)
    adapter = _make_adapter()
    adapter._max_concurrent_runs = 10
    adapter._max_concurrent_runs_per_profile = 3

    started = threading.Event()
    started_count = [0]
    started_lock = threading.Lock()
    release = threading.Event()

    class _BlockingAgent:
        def __init__(self, **_kwargs):
            self.session_id = "s"
            self.session_prompt_tokens = 1
            self.session_completion_tokens = 1
            self.session_total_tokens = 2

        def run_conversation(self, *, user_message: str, **_kwargs):
            with started_lock:
                started_count[0] += 1
                if started_count[0] >= 3:
                    started.set()
            assert release.wait(timeout=10.0)
            return {"final_response": f"ok:{user_message}"}

    monkeypatch.setattr(adapter, "_create_agent", lambda **kw: _BlockingAgent(**kw))

    async with TestClient(TestServer(_make_app(adapter))) as cli:
        # Alice fills all 3 of her per-profile slots
        alice_runs = []
        for idx in range(3):
            resp = await cli.post(
                "/p/alice/v1/runs",
                headers=_auth_headers(ALICE_KEY),
                json={"input": f"alice-{idx}"},
            )
            assert resp.status == 202
            alice_runs.append((await resp.json())["run_id"])

        await asyncio.to_thread(started.wait, 5.0)
        assert adapter.active_profile_agent_work_count("alice") == 3

        # Alice's 4th concurrent run is throttled (429) without touching Bob's capacity
        capped = await cli.post(
            "/p/alice/v1/runs",
            headers=_auth_headers(ALICE_KEY),
            json={"input": "alice-overflow"},
        )
        assert capped.status == 429
        assert capped.headers.get("Retry-After") == "1"
        assert capped.headers.get("Cache-Control") == "no-store"
        capped_body = await capped.json()
        assert capped_body["error"]["code"] == "rate_limit_exceeded"
        assert "alice" in capped_body["error"]["message"]

        # Bob is completely unaffected and launches 3 concurrent runs right away
        bob_runs = []
        for idx in range(3):
            resp = await cli.post(
                "/p/bob/v1/runs",
                headers=_auth_headers(BOB_KEY),
                json={"input": f"bob-{idx}"},
            )
            assert resp.status == 202
            bob_runs.append((await resp.json())["run_id"])

        assert adapter.active_profile_agent_work_count("bob") == 3
        assert adapter.active_agent_work_count() == 6

        # Release all runs and wait for completion
        release.set()
        for profile, run_ids in (("alice", alice_runs), ("bob", bob_runs)):
            for run_id in run_ids:
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    st = await (await cli.get(
                        f"/p/{profile}/v1/runs/{run_id}",
                        headers=_auth_headers(PROFILE_KEYS[profile]),
                    )).json()
                    if st["status"] == "completed":
                        break
                    await asyncio.sleep(0.02)
                assert st["status"] == "completed"

        # Let task retirement finish
        deadline = time.monotonic() + 2.0
        while adapter.active_agent_work_count() > 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        assert adapter.active_agent_work_count() == 0
        assert adapter.active_profile_agent_work_count("alice") == 0
        assert adapter.active_profile_agent_work_count("bob") == 0


def test_run_status_thread_safety_no_terminal_regression_and_waiting_for_device():
    adapter = _make_adapter()
    run_id = "run_regression_guard"
    adapter._set_run_status(run_id, "running")
    adapter._set_run_status(run_id, "waiting_for_device")
    assert adapter._active_structured_run_count() == 1

    # Cooperative "stopping" cannot be regressed back to "running" by late tool callback
    adapter._set_run_status(run_id, "stopping")
    after_tool = adapter._set_run_status(run_id, "running", last_event="tool.completed")
    assert after_tool["status"] == "stopping"

    # Terminal status cannot be regressed by concurrent worker callbacks
    adapter._set_run_status(run_id, "completed", output="final")
    assert adapter._active_structured_run_count() == 0

    errors: list[Exception] = []

    def _spam_non_terminal():
        try:
            for _ in range(100):
                st = adapter._set_run_status(run_id, "running", last_event="tool.completed")
                assert st["status"] == "completed"
        except Exception as exc:  # pragma: no cover - assertion failure collector
            errors.append(exc)

    threads = [threading.Thread(target=_spam_non_terminal) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
    assert not errors
    assert adapter._run_statuses[run_id]["status"] == "completed"


def test_last_resolved_model_isolated_across_profiles():
    adapter = _make_adapter()
    # Record a model for alice under a shared session key
    tok = _api_request_profile.set("alice")
    try:
        assert adapter._recover_or_record_model("openai/gpt-alice", {}, "shared-session") == "openai/gpt-alice"
        assert adapter._recover_or_record_model("", {}, "shared-session") == "openai/gpt-alice"
    finally:
        _api_request_profile.reset(tok)

    # Bob must NEVER recover Alice's model even with identical gateway_session_key
    tok = _api_request_profile.set("bob")
    try:
        assert adapter._recover_or_record_model("", {}, "shared-session") == ""
        assert adapter._recover_or_record_model("anthropic/claude-bob", {}, "shared-session") == "anthropic/claude-bob"
    finally:
        _api_request_profile.reset(tok)

    # Default profile must also remain completely isolated from named profiles
    assert adapter._recover_or_record_model("", {}, "shared-session") == ""
    assert adapter._recover_or_record_model("openai/gpt-default", {}, "shared-session") == "openai/gpt-default"
    assert adapter._last_resolved_model["shared-session"] == "openai/gpt-default"


def test_high_concurrency_sqlite_and_memory_stress_zero_busy_and_zero_lost_updates(tmp_path, monkeypatch):
    homes = _setup_multiplex_homes(tmp_path, monkeypatch)
    adapter = _make_adapter()
    errors: list[Exception] = []
    writers_per_profile = 10

    def _run_in_profile(profile: str, fn):
        tok = _api_request_profile.set(profile)
        try:
            with adapter._profile_scope(profile):
                return fn()
        finally:
            _api_request_profile.reset(tok)

    # Seed one note per profile to also stress concurrent cross-profile sharing + reads
    seed_notes = {
        profile: _run_in_profile(profile, lambda p=profile: create_note(title=f"Seed {p}", content=f"Content {p}"))["id"]
        for profile in PROFILE_KEYS
    }

    response_stores = {
        profile: ResponseStore(db_path=str(homes[profile] / "response_store.db"))
        for profile in PROFILE_KEYS
    }
    idem_stores = {
        profile: RunIdempotencyStore(db_path=str(homes[profile] / "runs_idempotency.db"))
        for profile in PROFILE_KEYS
    }

    def _worker(profile: str, idx: int) -> None:
        try:
            def _do():
                # 1. Notes write + concurrent read
                create_note(title=f"{profile}-note-{idx}", content=f"Body {profile} {idx}", tags=[profile, f"t{idx}"])
                assert len(list_notes(limit=50)) >= 1

                # 2. Deliveries write + concurrent read (shares mobile_state.db with UI store)
                delivery_store = MobileDeliveryStore()
                delivery_store.deliver("notice", f"Msg {idx}", {"body": f"{profile}-{idx}"})
                assert len(delivery_store.list_deliveries(limit=50)) >= 1

                # 3. UI revisions write + concurrent read (same mobile_state.db!)
                ui_store = MobileUiStore()
                ui_store.propose_or_apply(
                    [{"op": "clear_theme_override"}],
                    summary=f"Patch {profile} {idx}",
                    requires_apply=True,
                )
                assert "pending_revisions" in ui_store.get_state()

                # 4. Habits atomic counter increment + concurrent read
                record_habit_event("quick_action", f"act_{profile}")
                list_habit_suggestions()

                # 5. Memory + .pins.json concurrent RMW
                status, _payload = _sync_add_memory("memory", f"{profile} memory fact #{idx}", pin=(idx % 2 == 0))
                assert status == 201
                _sync_get_memory()

                # 6. Cross-profile share write + read
                target = "bob" if profile == "alice" else ("charlie" if profile == "bob" else "alice")
                grant_share("note", seed_notes[profile], target, "read", label=f"Share {idx}")
                list_outgoing_shares()
                list_incoming_shares()

                # 7. ResponseStore & RunIdempotencyStore concurrent writes
                response_stores[profile].put(f"resp_{idx}", {"profile": profile, "idx": idx})
                assert response_stores[profile].get(f"resp_{idx}") is not None
                outcome, _rec = idem_stores[profile].reserve(
                    f"scope_{profile}", f"key_{idx}", f"fp_{idx}", f"run_{profile}_{idx}", {"status": "queued"}
                )
                assert outcome == "created"

            _run_in_profile(profile, _do)
        except Exception as exc:  # pragma: no cover - captured for assertion
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=18) as pool:
        futures = [
            pool.submit(_worker, profile, idx)
            for profile in PROFILE_KEYS
            for idx in range(writers_per_profile)
        ]
        for fut in futures:
            fut.result(timeout=30.0)

    for store in response_stores.values():
        store.close()
    for store in idem_stores.values():
        store.close()

    assert not errors, f"Concurrency errors observed: {errors!r}"

    # Verify exact zero-lost-update counts and strict per-profile isolation
    for profile in PROFILE_KEYS:
        notes = _run_in_profile(profile, lambda: list_notes(limit=100))
        # 1 seed + writers_per_profile created notes
        assert len(notes) == 1 + writers_per_profile
        assert all(n["title"].startswith((f"{profile}-note-", f"Seed {profile}")) for n in notes)

        deliveries = _run_in_profile(profile, lambda: MobileDeliveryStore().list_deliveries(limit=100))
        assert len(deliveries) == writers_per_profile

        ui_history = _run_in_profile(profile, lambda: MobileUiStore().list_history(limit=100))
        assert len(ui_history) == writers_per_profile

        habits_payload = _run_in_profile(profile, lambda: list_habit_suggestions(limit=10))
        habits = habits_payload["data"]
        assert habits and habits[0]["action"] == f"act_{profile}"
        assert habits[0]["count"] == writers_per_profile

        mem = _run_in_profile(profile, _sync_get_memory)
        entries = mem["memory"]["entries"]
        assert len(entries) == writers_per_profile
        assert sum(1 for e in entries if e["pinned"]) == writers_per_profile // 2
        assert all(e["text"].startswith(f"{profile} memory fact #") for e in entries)
