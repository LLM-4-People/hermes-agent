"""Deterministic positive & negative tests for Fork PR #15:
Live UI Modding, Universal Phone Push (`mobile_deliver`), `run_progress` hooks,
and Approved Bidirectional Android Device Bridge (`request_device_action`).
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import copy_context
import json
from unittest.mock import MagicMock

from aiohttp.test_utils import TestClient, TestServer
import pytest

from gateway.platforms import api_server as api_server_mod
from gateway.platforms import api_server_device_bridge, api_server_ui
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tests.gateway.test_api_server_runs import _create_runs_app, _make_adapter
from tools.mobile_deliver import (
    mobile_deliver_tool,
    register_delivery_wakeup_hook,
    register_run_progress_hook,
    unregister_delivery_wakeup_hook,
    unregister_run_progress_hook,
)
from tools.mobile_device_bridge import (
    DeviceBridgeBroker,
    _device_bridge_available,
    current_device_capabilities,
    device_bridge_scope,
    request_device_action_tool,
    validate_device_capabilities,
)
from tools.mobile_ui_patch import (
    MobileUiStore,
    contrast_ratio,
    mobile_ui_patch_tool,
    validate_theme_override,
    validate_ui_patch_ops,
)


@contextmanager
def override_hermes_home(path):
    token = set_hermes_home_override(path)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _mount_full_mobile_app(adapter):
    app = _create_runs_app(adapter)
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    for method, path, handler in api_server_ui._http_routes(adapter):
        app.router.add_route(method, path, handler)
    for method, path, handler in api_server_device_bridge._http_routes(adapter):
        app.router.add_route(method, path, handler)
    return app


def _parse_sse(raw: bytes) -> list[dict]:
    events = []
    for block in raw.decode("utf-8", errors="replace").split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


# ── 1. Capabilities, Route Table & Auth Enforcement ──────────────────────────


@pytest.mark.asyncio
async def test_capabilities_routes_and_unauthenticated_rejection(tmp_path):
    adapter = _make_adapter()
    adapter._api_key = "secret-mobile-key"
    table_paths = {(m, p) for m, p, _ in adapter._http_route_table()}
    expected_routes = {
        ("GET", "/api/mobile/ui/state"),
        ("GET", "/api/mobile/ui/history"),
        ("POST", "/api/mobile/ui/revisions/{revision_id}/apply"),
        ("POST", "/api/mobile/ui/revisions/{revision_id}/revert"),
        ("GET", "/api/mobile/deliveries"),
        ("GET", "/api/mobile/deliveries/{delivery_id}"),
        ("POST", "/api/mobile/deliveries/{delivery_id}/ack"),
        ("POST", "/v1/runs/{run_id}/device-result"),
    }
    assert expected_routes <= table_paths

    app = _mount_full_mobile_app(adapter)
    headers = {"Authorization": "Bearer secret-mobile-key"}
    with override_hermes_home(tmp_path):
        async with TestClient(TestServer(app)) as client:
            caps_resp = await client.get("/v1/capabilities", headers=headers)
            assert caps_resp.status == 200
            caps = await caps_resp.json()
            for flag in ("live_ui_modding", "mobile_deliveries", "device_bridge", "run_progress_hooks"):
                assert caps["features"][flag] is True
            for ep in (
                "mobile_ui_state",
                "mobile_ui_history",
                "mobile_ui_apply",
                "mobile_ui_revert",
                "mobile_deliveries",
                "mobile_delivery",
                "mobile_delivery_ack",
                "run_device_result",
            ):
                assert ep in caps["endpoints"]

            # Every new endpoint must fail closed with 401 and Cache-Control: no-store.
            for method, path in (
                ("GET", "/api/mobile/ui/state"),
                ("GET", "/api/mobile/ui/history"),
                ("POST", "/api/mobile/ui/revisions/uirev_x/apply"),
                ("POST", "/api/mobile/ui/revisions/uirev_x/revert"),
                ("GET", "/api/mobile/deliveries"),
                ("GET", "/api/mobile/deliveries/dlv_x"),
                ("POST", "/api/mobile/deliveries/dlv_x/ack"),
                ("POST", "/v1/runs/run_x/device-result"),
            ):
                resp = await client.request(method, path)
                assert resp.status == 401, (method, path, resp.status)
                assert resp.headers.get("Cache-Control") == "no-store"


# ── 2. Live UI Modding: Validation, WCAG Guard, Apply/Revert & Cross-Profile ─


def test_ui_patch_wcag_and_malicious_payload_validation():
    # High-contrast valid palette passes.
    assert contrast_ratio("#1A1917", "#FAF8F5") >= 4.5
    cleaned, err = validate_theme_override({
        "accent_hex": "#0d6e6e",
        "accent_ink_hex": "#ffffff",
        "surface_hex": "#faf8f5",
        "ink_hex": "#1a1917",
        "density": "compact",
        "font_scale_step": 1,
        "surface_style": "paper",
    })
    assert err is None and cleaned["accent_hex"] == "#0D6E6E"

    # Low-contrast text on surface rejected.
    _, err_text = validate_theme_override({"surface_hex": "#FFFFFF", "ink_hex": "#CCCCCC"})
    assert err_text and "WCAG AA" in err_text

    # Low-contrast accent rejected.
    _, err_accent = validate_theme_override({"accent_hex": "#FFEE00", "accent_ink_hex": "#FFFFFF"})
    assert err_accent and "WCAG AA" in err_accent

    # Unknown theme keys or bad font scale rejected.
    assert validate_theme_override({"evil_key": "x"})[1] is not None
    assert validate_theme_override({"font_scale_step": 9})[1] is not None
    assert validate_theme_override({"font_scale_step": True})[1] is not None

    # Malicious / invalid ops rejected.
    bad_cases = [
        [],
        [{"op": "exec_js", "code": "alert(1)"}],
        [{"op": "upsert_surface", "surface_id": "../escape", "kind": "kpi_grid", "title": "T", "spec": {}}],
        [{"op": "upsert_surface", "surface_id": "ok_id", "kind": "webview", "title": "T", "spec": {}}],
        [{"op": "upsert_surface", "surface_id": "ok_id", "kind": "kpi_grid", "title": "<script>x</script>", "spec": {}}],
        [{"op": "upsert_surface", "surface_id": "ok_id", "kind": "markdown_card", "title": "Card", "spec": {"url": "javascript:alert(1)"}}],
        [{"op": "upsert_surface", "surface_id": "ok_id", "kind": "markdown_card", "title": "Card", "spec": {"uri": "intent://launch"}}],
        [{"op": "upsert_surface", "surface_id": "ok_id", "kind": "markdown_card", "title": "Card", "spec": {"p": "file:///etc/passwd"}}],
        [{"op": "upsert_surface", "surface_id": "ok_id", "kind": "markdown_card", "title": "\ud800", "spec": {}}],
        [{"op": "upsert_quick_action", "action_id": "qa1", "label": "Run", "action_kind": "open_destination", "destination": "secret"}],
    ]
    for bad in bad_cases:
        assert validate_ui_patch_ops(bad)[1] is not None, bad


def test_dynamic_widget_ops_and_widget_update_delivery(tmp_path):
    # Valid widget spec passes via both mobile_ui_patch and mobile_deliver(kind="widget_update").
    valid_widget = {
        "widget_id": "morning_kpi",
        "title": "Mon matin",
        "subtitle": "Résumé prêt",
        "template": "metric",
        "items": [{"label": "Tâches", "value": "3", "action_draft": "Ouvre ma liste du jour"}],
        "accent_hex": "#0D6E6E",
        "home_screen_safe": True,
    }
    with override_hermes_home(tmp_path):
        rev = json.loads(
            mobile_ui_patch_tool(
                ops=[{"op": "upsert_widget", **valid_widget}],
                summary="Add morning home-screen widget",
                requires_apply=False,
            )
        )
        assert rev["ok"] is True
        st = MobileUiStore(tmp_path).get_state()
        assert [w["widget_id"] for w in st["widgets"]] == ["morning_kpi"]
        assert st["widgets"][0]["home_screen_safe"] is True

        # Remove widget then revert removal -> widget restored cleanly!
        rm = json.loads(
            mobile_ui_patch_tool(
                ops=[{"op": "remove_widget", "widget_id": "morning_kpi"}],
                summary="Remove morning widget",
                requires_apply=False,
            )
        )
        assert rm["ok"] is True
        assert MobileUiStore(tmp_path).get_state()["widgets"] == []
        MobileUiStore(tmp_path).revert_revision(rm["revision"]["revision_id"])
        assert [w["widget_id"] for w in MobileUiStore(tmp_path).get_state()["widgets"]] == ["morning_kpi"]

        # Valid widget_update delivery passes.
        dlv = json.loads(mobile_deliver_tool(kind="widget_update", title="Widget update", payload=valid_widget))
        assert dlv["ok"] is True and dlv["delivery"]["kind"] == "widget_update"

        # Negative guards: missing/false home_screen_safe, low-contrast accent, bad template, secret token leak.
        for bad_override in (
            {"home_screen_safe": False},
            {"accent_hex": "#FFEE00"},
            {"template": "raw_html"},
            {"title": "Key sk-secret1234567890abcdef"},
            {"items": [{"label": "Token", "value": "Bearer supersecrettokenvalue1234"}]},
        ):
            bad_spec = {**valid_widget, **bad_override}
            assert json.loads(mobile_ui_patch_tool(ops=[{"op": "upsert_widget", **bad_spec}], summary="Bad"))["ok"] is False
            assert json.loads(mobile_deliver_tool(kind="widget_update", title="Bad", payload=bad_spec))["ok"] is False


@pytest.mark.asyncio
async def test_ui_patch_live_sse_apply_idempotent_revert_and_profile_isolation(tmp_path, monkeypatch):
    profile_a = tmp_path / "profile_a"
    profile_b = tmp_path / "profile_b"
    profile_a.mkdir()
    profile_b.mkdir()

    adapter = _make_adapter()
    adapter._api_key = "owner-key"
    agent = MagicMock()
    emitted_revisions = []

    def create(**kwargs):
        return agent

    def run(**kwargs):
        res1 = json.loads(
            mobile_ui_patch_tool(
                ops=[
                    {
                        "op": "upsert_surface",
                        "surface_id": "kpi_sprint",
                        "scope": "atelier",
                        "kind": "kpi_grid",
                        "title": "Sprint Metrics",
                        "tab": "metrics",
                        "spec": {"items": [{"label": "PRs", "value": "15"}]},
                    },
                    {
                        "op": "upsert_quick_action",
                        "action_id": "briefing",
                        "label": "Briefing",
                        "action_kind": "prompt",
                        "prompt": "Donne mon briefing du matin",
                    },
                    {
                        "op": "set_theme_override",
                        "theme": {"accent_hex": "#0D6E6E", "accent_ink_hex": "#FFFFFF", "density": "comfortable"},
                    },
                ],
                summary="Add Sprint Metrics dashboard and Briefing shortcut",
                scope="atelier",
                requires_apply=True,
                session_id="sess-ui-1",
            )
        )
        assert res1["ok"] is True
        emitted_revisions.append(res1["revision"]["revision_id"])
        return {"final_response": "Proposed live UI update."}

    agent.run_conversation.side_effect = run
    monkeypatch.setattr(adapter, "_create_agent", create)

    app = _mount_full_mobile_app(adapter)
    headers = {"Authorization": "Bearer owner-key"}

    with override_hermes_home(profile_a):
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/v1/runs", json={"input": "Build my sprint dashboard", "session_id": "sess-ui-1"}, headers=headers)
            assert resp.status == 202
            run_id = (await resp.json())["run_id"]
            async with client.get(f"/v1/runs/{run_id}/events", headers=headers) as stream:
                raw_sse = await stream.read()

            sse_events = _parse_sse(raw_sse)
            ui_events = [e for e in sse_events if e.get("event") == "ui.patch"]
            assert len(ui_events) == 1
            rev1_id = ui_events[0]["revision_id"]
            assert rev1_id == emitted_revisions[0]
            assert ui_events[0]["requires_apply"] is True and ui_events[0]["reversible"] is True

            # Before apply: active atelier_surfaces is empty; revision sits in pending_revisions.
            st0 = await (await client.get("/api/mobile/ui/state?session_id=sess-ui-1", headers=headers)).json()
            assert st0["atelier_surfaces"] == []
            assert any(r["revision_id"] == rev1_id for r in st0["pending_revisions"])

            # Apply rev1 (and verify idempotency on second call).
            ap1 = await client.post(f"/api/mobile/ui/revisions/{rev1_id}/apply", headers=headers)
            assert ap1.status == 200 and ap1.headers["Cache-Control"] == "no-store"
            assert (await ap1.json())["already_applied"] is False
            ap1_again = await (await client.post(f"/api/mobile/ui/revisions/{rev1_id}/apply", headers=headers)).json()
            assert ap1_again["already_applied"] is True

            st1 = await (await client.get("/api/mobile/ui/state", headers=headers)).json()
            assert [s["surface_id"] for s in st1["atelier_surfaces"]] == ["kpi_sprint"]
            assert [a["action_id"] for a in st1["quick_actions"]] == ["briefing"]
            assert st1["theme"]["accent_hex"] == "#0D6E6E"

            # Add rev2 directly (applied immediately), then revert rev2 -> restores rev1!
            rev2 = MobileUiStore(profile_a).propose_or_apply(
                ops=[{"op": "upsert_surface", "surface_id": "spark_tps", "scope": "atelier", "kind": "sparkline", "title": "TPS", "spec": {"points": [40, 82, 108]}}],
                summary="Add TPS sparkline",
                scope="atelier",
                requires_apply=False,
            )
            st2 = await (await client.get("/api/mobile/ui/state", headers=headers)).json()
            assert {s["surface_id"] for s in st2["atelier_surfaces"]} == {"kpi_sprint", "spark_tps"}

            # Guard: reverting a non-active applied revision (rev1 while rev2 is active) MUST return 409!
            conflict = await client.post(f"/api/mobile/ui/revisions/{rev1_id}/revert", headers=headers)
            assert conflict.status == 409 and conflict.headers["Cache-Control"] == "no-store"
            assert (await conflict.json())["error"]["code"] == "revision_not_active"
            # Active state stays synchronized at rev2.
            st_still_2 = await (await client.get("/api/mobile/ui/state", headers=headers)).json()
            assert st_still_2["active_revision_id"] == rev2["revision_id"]

            rv2 = await (await client.post(f"/api/mobile/ui/revisions/{rev2['revision_id']}/revert", headers=headers)).json()
            assert rv2["already_reverted"] is False and rv2["active_revision_id"] == rev1_id
            rv2_again = await (await client.post(f"/api/mobile/ui/revisions/{rev2['revision_id']}/revert", headers=headers)).json()
            assert rv2_again["already_reverted"] is True

            st_after_rv2 = await (await client.get("/api/mobile/ui/state", headers=headers)).json()
            assert [s["surface_id"] for s in st_after_rv2["atelier_surfaces"]] == ["kpi_sprint"]

            # Revert rev1 -> restores clean empty default state.
            rv1 = await (await client.post(f"/api/mobile/ui/revisions/{rev1_id}/revert", headers=headers)).json()
            assert rv1["active_revision_id"] is None
            st_empty = await (await client.get("/api/mobile/ui/state", headers=headers)).json()
            assert st_empty["atelier_surfaces"] == [] and st_empty["quick_actions"] == [] and st_empty["theme"] == {}

            # History & negative checks.
            hist = await (await client.get("/api/mobile/ui/history?limit=10", headers=headers)).json()
            assert len(hist["data"]) == 2
            assert (await client.get("/api/mobile/ui/history?limit=0", headers=headers)).status == 400
            assert (await client.post("/api/mobile/ui/revisions/uirev_missing/apply", headers=headers)).status == 404
            assert (await client.post("/api/mobile/ui/revisions/uirev_missing/revert", headers=headers)).status == 404

    # Cross-profile isolation: Profile B sees zero revisions from Profile A and cannot apply/revert them.
    with override_hermes_home(profile_b):
        async with TestClient(TestServer(app)) as client:
            st_b = await (await client.get("/api/mobile/ui/state", headers=headers)).json()
            assert st_b["atelier_surfaces"] == [] and st_b["pending_revisions"] == []
            assert (await client.post(f"/api/mobile/ui/revisions/{rev1_id}/apply", headers=headers)).status == 404
            assert (await client.post(f"/api/mobile/ui/revisions/{rev1_id}/revert", headers=headers)).status == 404


# ── 3. Universal Phone Push (`mobile_deliver`) & Zero-Leak Wakeup + Cron ────


@pytest.mark.asyncio
async def test_mobile_deliver_cron_zero_leak_wakeup_ack_and_profile_isolation(tmp_path, monkeypatch):
    profile_a = tmp_path / "profile_a"
    profile_b = tmp_path / "profile_b"
    profile_a.mkdir()
    profile_b.mkdir()

    wakeups: list[dict] = []
    register_delivery_wakeup_hook(wakeups.append)
    try:
        # Simulate a cron job delivering a private morning briefing with zero live adapter.
        monkeypatch.setenv("HERMES_CRON_JOB_ID", "cron_morning_brief")
        with override_hermes_home(profile_a):
            out = json.loads(
                mobile_deliver_tool(
                    kind="briefing",
                    title="PRIVATE_SECRET_TITLE_XYZ",
                    payload={"summary": "PRIVATE_BODY_NEVER_ON_PUSH", "items": ["PR #15 ready"]},
                    urgency="high",
                    session_id="sess-cron-1",
                )
            )
            assert out["ok"] is True
            dlv_id = out["delivery"]["delivery_id"]
            assert out["delivery"]["origin"] == "cron"

            # Reject malicious or invalid deliveries cleanly.
            bad_ui = json.loads(
                mobile_deliver_tool(kind="ui_patch", title="Bad patch", payload={"ops": [{"op": "exec_js"}]})
            )
            assert bad_ui["ok"] is False
            bad_kind = json.loads(mobile_deliver_tool(kind="shell_exec", title="Bad", payload={}))
            assert bad_kind["ok"] is False
            bad_xss = json.loads(mobile_deliver_tool(kind="notice", title="Ok", payload={"html": "<script>x</script>"}))
            assert bad_xss["ok"] is False
    finally:
        unregister_delivery_wakeup_hook(wakeups.append)

    # Zero-leak assertion: wakeup hook NEVER receives title or payload content.
    assert len(wakeups) == 1
    assert wakeups[0]["event_id"] == dlv_id
    assert wakeups[0]["kind"] == "briefing" and wakeups[0]["urgency"] == "high"
    serialized_wakeup = json.dumps(wakeups[0])
    assert "PRIVATE_SECRET_TITLE_XYZ" not in serialized_wakeup
    assert "PRIVATE_BODY_NEVER_ON_PUSH" not in serialized_wakeup

    adapter = _make_adapter()
    adapter._api_key = "owner-key"
    app = _mount_full_mobile_app(adapter)
    headers = {"Authorization": "Bearer owner-key"}

    with override_hermes_home(profile_a):
        async with TestClient(TestServer(app)) as client:
            listing = await client.get("/api/mobile/deliveries", headers=headers)
            assert listing.status == 200 and listing.headers["Cache-Control"] == "no-store"
            data = (await listing.json())["data"]
            assert len(data) == 1 and data[0]["delivery_id"] == dlv_id

            detail = await (await client.get(f"/api/mobile/deliveries/{dlv_id}", headers=headers)).json()
            assert detail["delivery"]["title"] == "PRIVATE_SECRET_TITLE_XYZ"

            # Ack is idempotent; unacked list drops it while include_acked=true retains it.
            ack1 = await (await client.post(f"/api/mobile/deliveries/{dlv_id}/ack", headers=headers)).json()
            assert ack1["acked"] is True and ack1["delivery"]["acked_at"] is not None
            ack2 = await (await client.post(f"/api/mobile/deliveries/{dlv_id}/ack", headers=headers)).json()
            assert ack2["delivery"]["acked_at"] == ack1["delivery"]["acked_at"]

            unacked = await (await client.get("/api/mobile/deliveries", headers=headers)).json()
            assert unacked["data"] == []
            all_items = await (await client.get("/api/mobile/deliveries?include_acked=true", headers=headers)).json()
            assert len(all_items["data"]) == 1

            # Negative query validation & 404.
            assert (await client.get("/api/mobile/deliveries?limit=0", headers=headers)).status == 400
            assert (await client.get("/api/mobile/deliveries?since=-5", headers=headers)).status == 400
            assert (await client.get("/api/mobile/deliveries?include_acked=maybe", headers=headers)).status == 400
            assert (await client.get("/api/mobile/deliveries/dlv_missing", headers=headers)).status == 404
            assert (await client.post("/api/mobile/deliveries/dlv_missing/ack", headers=headers)).status == 404

    # Cross-profile isolation: Profile B cannot see or ack Profile A's delivery.
    with override_hermes_home(profile_b):
        async with TestClient(TestServer(app)) as client:
            assert (await (await client.get("/api/mobile/deliveries?include_acked=true", headers=headers)).json())["data"] == []
            assert (await client.get(f"/api/mobile/deliveries/{dlv_id}", headers=headers)).status == 404
            assert (await client.post(f"/api/mobile/deliveries/{dlv_id}/ack", headers=headers)).status == 404


# ── 4. Approved Bidirectional Device Bridge + `run_progress` Hooks ───────────


def test_device_bridge_scope_expiry_and_capability_validation():
    assert validate_device_capabilities(["camera", "camera", "alarm_timer"]) == (["camera", "alarm_timer"], None)
    assert validate_device_capabilities("camera")[1] is not None
    assert validate_device_capabilities(["root_shell"])[1] is not None

    broker = DeviceBridgeBroker()
    assert _device_bridge_available() is False
    with device_bridge_scope("owner", capabilities=["camera"], run_id="r1", profile_key="default", broker=broker):
        assert _device_bridge_available() is True
        assert current_device_capabilities() == frozenset({"camera"})
        copied = copy_context()
        # Unadvertised capability fails closed immediately without blocking.
        denied = json.loads(request_device_action_tool("clipboard_read", "Read clipboard"))
        assert denied["ok"] is False and denied["error"] == "capability_not_advertised"
    # After scope exit, even the copied context sees zero capabilities.
    assert copied.run(current_device_capabilities) == frozenset()
    assert copied.run(_device_bridge_available) is False
    after_exit = json.loads(copied.run(request_device_action_tool, "camera", "Take photo"))
    assert after_exit["ok"] is False and after_exit["error"] == "device_bridge_unavailable"


@pytest.mark.asyncio
async def test_device_bridge_approve_decline_timeout_stop_and_progress_hooks(tmp_path, monkeypatch):
    adapter = _make_adapter()
    adapter._api_key = "owner-key"
    app = _mount_full_mobile_app(adapter)
    headers = {"Authorization": "Bearer owner-key"}

    progress_events: list[dict] = []
    register_run_progress_hook(progress_events.append)

    try:
        # 4A. Reject invalid device_capabilities with 400.
        async with TestClient(TestServer(app)) as client:
            bad = await client.post(
                "/v1/runs",
                json={"input": "hi", "device_capabilities": ["not_real_cap"]},
                headers=headers,
            )
            assert bad.status == 400

        # 4B. Full approve flow + progress hooks + negative device-result checks.
        agent = MagicMock()
        tool_outcomes = []

        def create_approve(**kwargs):
            return agent

        def run_approve(**kwargs):
            outcome = json.loads(
                request_device_action_tool("camera", "Capture receipt photo for expense report", {"quality": "high"})
            )
            tool_outcomes.append(outcome)
            return {"final_response": f"Got photo {outcome.get('data', {}).get('photo_id')}"}

        agent.run_conversation.side_effect = run_approve
        monkeypatch.setattr(adapter, "_create_agent", create_approve)

        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/v1/runs",
                json={"input": "Scan my receipt", "device_capabilities": ["camera"]},
                headers=headers,
            )
            assert resp.status == 202
            run_id = (await resp.json())["run_id"]

            # Wait for waiting_for_device status with pending_device_requests populated.
            req_id = None
            for _ in range(80):
                st = await (await client.get(f"/v1/runs/{run_id}", headers=headers)).json()
                if st.get("pending_device_requests"):
                    assert st["status"] == "waiting_for_device"
                    req_id = st["pending_device_requests"][0]["request_id"]
                    assert st["pending_device_requests"][0]["action_type"] == "camera"
                    break
                await asyncio.sleep(0.02)
            assert req_id is not None

            # Negative checks on /device-result before resolving.
            assert (await client.post(f"/v1/runs/{run_id}/device-result", json={"request_id": req_id, "status": "bogus"}, headers=headers)).status == 400
            assert (await client.post(f"/v1/runs/{run_id}/device-result", json={"request_id": req_id, "status": "completed", "extra": 1}, headers=headers)).status == 400
            assert (await client.post(f"/v1/runs/{run_id}/device-result", json={"request_id": req_id, "status": "completed", "data": {"x": "<script>1</script>"}}, headers=headers)).status == 400
            assert (await client.post(f"/v1/runs/{run_id}/device-result", json={"request_id": "dev_wrong", "status": "completed"}, headers=headers)).status == 404

            # Cross-profile isolation: resolve under a different profile context returns 404.
            tok = api_server_mod._api_request_profile.set("other_profile")
            try:
                cross = adapter._device_bridge_broker.resolve_request(
                    profile_key="other_profile", run_id=run_id, request_id=req_id, status="completed", data={}
                )
                assert cross[0] == "not_found"
            finally:
                api_server_mod._api_request_profile.reset(tok)

            # Resolve cleanly with approved data.
            ok_resp = await client.post(
                f"/v1/runs/{run_id}/device-result",
                json={"request_id": req_id, "status": "completed", "data": {"photo_id": "img_42"}},
                headers=headers,
            )
            assert ok_resp.status == 200 and ok_resp.headers["Cache-Control"] == "no-store"
            assert (await ok_resp.json())["accepted"] is True

            # Duplicate resolution rejected cleanly.
            dup = await client.post(
                f"/v1/runs/{run_id}/device-result",
                json={"request_id": req_id, "status": "completed"},
                headers=headers,
            )
            assert dup.status == 404

            async with client.get(f"/v1/runs/{run_id}/events", headers=headers) as stream:
                events = _parse_sse(await stream.read())
            names = [e.get("event") for e in events]
            assert "device.requested" in names and "device.settled" in names and "run.completed" in names
            assert tool_outcomes == [{
                "ok": True,
                "request_id": req_id,
                "action_type": "camera",
                "status": "completed",
                "data": {"photo_id": "img_42"},
            }]
            stages = [p["stage"] for p in progress_events if p["run_id"] == run_id]
            assert "run.started" in stages and "device.requested" in stages and "device.settled" in stages and "run.completed" in stages

        # 4C. Decline flow.
        decline_outcomes = []

        def run_decline(**kwargs):
            decline_outcomes.append(json.loads(request_device_action_tool("location_coarse", "Find nearby cafes")))
            return {"final_response": "Okay, continuing without location."}

        agent.run_conversation.side_effect = run_decline
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/v1/runs",
                json={"input": "Cafes?", "device_capabilities": ["location_coarse"]},
                headers=headers,
            )
            run_id = (await resp.json())["run_id"]
            req_id = None
            for _ in range(80):
                st = await (await client.get(f"/v1/runs/{run_id}", headers=headers)).json()
                if st.get("pending_device_requests"):
                    req_id = st["pending_device_requests"][0]["request_id"]
                    break
                await asyncio.sleep(0.02)
            assert req_id is not None
            r = await client.post(
                f"/v1/runs/{run_id}/device-result",
                json={"request_id": req_id, "status": "denied"},
                headers=headers,
            )
            assert r.status == 200
            async with client.get(f"/v1/runs/{run_id}/events", headers=headers) as stream:
                await stream.read()
            assert decline_outcomes[0]["ok"] is False and decline_outcomes[0]["status"] == "denied"

        # 4D. Timeout flow (1s bounded timeout).
        monkeypatch.setenv("HERMES_MOBILE_DEVICE_TIMEOUT_SECONDS", "1.0")
        timeout_outcomes = []

        def run_timeout(**kwargs):
            timeout_outcomes.append(json.loads(request_device_action_tool("alarm_timer", "Set 5 min timer")))
            return {"final_response": "Timer timed out."}

        agent.run_conversation.side_effect = run_timeout
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/v1/runs",
                json={"input": "Set timer", "device_capabilities": ["alarm_timer"]},
                headers=headers,
            )
            run_id = (await resp.json())["run_id"]
            async with client.get(f"/v1/runs/{run_id}/events", headers=headers) as stream:
                events = _parse_sse(await stream.read())
            settled = [e for e in events if e.get("event") == "device.settled"]
            assert settled and settled[0]["status"] == "timed_out"
            assert timeout_outcomes[0]["status"] == "timed_out"
            assert timeout_outcomes[0]["error"] == "device_request_timed_out"

        # 4E. Stop/cancel unblocks a waiting device request immediately.
        monkeypatch.setenv("HERMES_MOBILE_DEVICE_TIMEOUT_SECONDS", "60.0")
        stop_outcomes = []

        def run_stopped(**kwargs):
            stop_outcomes.append(json.loads(request_device_action_tool("share_sheet", "Share summary")))
            return {"final_response": "", "interrupted": True}

        agent.run_conversation.side_effect = run_stopped
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/v1/runs",
                json={"input": "Share this", "device_capabilities": ["share_sheet"]},
                headers=headers,
            )
            run_id = (await resp.json())["run_id"]
            for _ in range(80):
                st = await (await client.get(f"/v1/runs/{run_id}", headers=headers)).json()
                if st.get("pending_device_requests"):
                    break
                await asyncio.sleep(0.02)
            stop_resp = await client.post(f"/v1/runs/{run_id}/stop", headers=headers)
            assert stop_resp.status == 200
            async with client.get(f"/v1/runs/{run_id}/events", headers=headers) as stream:
                await stream.read()
            assert stop_outcomes and stop_outcomes[0]["status"] == "cancelled"
    finally:
        unregister_run_progress_hook(progress_events.append)


# ── 5. Mobile Companion Prompt Builder & Contextual `suggested_actions[]` ────


@pytest.mark.asyncio
async def test_mobile_companion_prompt_builder_and_suggested_actions(tmp_path, monkeypatch):
    from agent.mobile_companion_prompt import build_mobile_companion_instructions, combine_run_instructions
    from agent.run_instructions import effective_system_prompt
    from tools.mobile_ui_patch import validate_suggested_actions

    # Non-mobile runs get zero extra prompt overhead.
    assert build_mobile_companion_instructions(None) == ""
    assert build_mobile_companion_instructions([]) == ""
    assert combine_run_instructions("stay brief", None) == "stay brief"

    # Mobile prompt strictly reflects ONLY advertised capabilities.
    prompt = build_mobile_companion_instructions(["camera", "alarm_timer", "ui_surfaces_v2"])
    assert "camera" in prompt and "alarm_timer" in prompt and "upsert_widget" in prompt and "mobile_deliver" in prompt
    assert "clipboard_read" not in prompt and "dial_draft" not in prompt

    # Suggested actions validation (positive + negative bounds).
    valid_chips = [
        {"id": "pin_kpi", "label": "Épingler le widget", "kind": "pin_widget", "widget_id": "morning_kpi"},
        {"id": "save_mem", "label": "Retenir cette préférence", "kind": "save_memory", "prompt": "Retiens que je préfère le café noir"},
    ]
    cleaned, err = validate_suggested_actions(valid_chips)
    assert err is None and len(cleaned) == 2
    assert validate_suggested_actions([])[1] is not None
    assert validate_suggested_actions([{"id": f"c{i}", "label": "X", "kind": "prompt"} for i in range(4)])[1] is not None
    assert validate_suggested_actions([{"id": "a", "label": "X", "kind": "shell"}])[1] is not None
    assert validate_suggested_actions([{"id": "a", "label": "<script>x</script>", "kind": "prompt"}])[1] is not None

    # End-to-end run: mobile turn gets companion guidance + emits suggested_actions; next non-mobile turn sees neither.
    adapter = _make_adapter()
    adapter._api_key = "owner-key"
    agent = MagicMock()
    agent.ephemeral_system_prompt = None
    seen_prompts = []

    def create(**kwargs):
        return agent

    def run(**kwargs):
        seen_prompts.append(effective_system_prompt(agent, "BASE_SYSTEM"))
        if len(seen_prompts) == 1:
            res = json.loads(
                mobile_ui_patch_tool(
                    ops=[{
                        "op": "upsert_widget",
                        "widget_id": "morning_kpi",
                        "title": "Mon matin",
                        "template": "metric",
                        "items": [{"label": "Rappels", "value": "2"}],
                        "home_screen_safe": True,
                    }],
                    summary="Create morning widget",
                    suggested_actions=valid_chips,
                )
            )
            assert res["ok"] is True
        return {"final_response": "C'est prêt."}

    agent.run_conversation.side_effect = run
    monkeypatch.setattr(adapter, "_create_agent", create)
    app = _mount_full_mobile_app(adapter)
    headers = {"Authorization": "Bearer owner-key"}

    with override_hermes_home(tmp_path):
        async with TestClient(TestServer(app)) as client:
            r1 = await client.post(
                "/v1/runs",
                json={"input": "Aide-moi le matin", "device_capabilities": ["alarm_timer", "ui_surfaces_v2"]},
                headers=headers,
            )
            assert r1.status == 202
            run1_id = (await r1.json())["run_id"]
            async with client.get(f"/v1/runs/{run1_id}/events", headers=headers) as stream:
                events = _parse_sse(await stream.read())
            completed = next(e for e in events if e.get("event") == "run.completed")
            assert len(completed["suggested_actions"]) == 2
            st1 = await (await client.get(f"/v1/runs/{run1_id}", headers=headers)).json()
            assert [c["id"] for c in st1["suggested_actions"]] == ["pin_kpi", "save_mem"]

            # Subsequent ordinary (non-mobile) run must NOT inherit mobile companion prompt or suggestions.
            r2 = await client.post("/v1/runs", json={"input": "Hello"}, headers=headers)
            run2_id = (await r2.json())["run_id"]
            async with client.get(f"/v1/runs/{run2_id}/events", headers=headers) as stream:
                await stream.read()
            st2 = await (await client.get(f"/v1/runs/{run2_id}", headers=headers)).json()
            assert "suggested_actions" not in st2

            # Persisted suggested_actions survive reload in both /state (pending_revisions) and /history!
            ui_state = await (await client.get("/api/mobile/ui/state", headers=headers)).json()
            assert ui_state["pending_revisions"][0]["suggested_actions"][0]["id"] == "pin_kpi"
            ui_hist = await (await client.get("/api/mobile/ui/history", headers=headers)).json()
            assert ui_hist["data"][0]["suggested_actions"][1]["id"] == "save_mem"

    assert "[Hermes Mobile Companion]" in seen_prompts[0] and "alarm_timer" in seen_prompts[0]
    assert seen_prompts[1] == "BASE_SYSTEM"


# ── 6. Small-LLM Forgiving Normalization, Auto-Slugs & Self-Correcting Hints ─


def test_small_llm_forgiving_normalization_auto_slugs_and_error_hints(tmp_path):
    with override_hermes_home(tmp_path):
        # Single dict op (not wrapped in list), omitted summary, omitted IDs, accented French title,
        # named accent alias ("teal"), template alias ("tasks" -> "checklist"), omitted home_screen_safe,
        # string items, and suggested_actions without explicit ids.
        res = json.loads(
            mobile_ui_patch_tool(
                ops={
                    "op": "upsert_widget",
                    "title": "Mes tâches d’été",
                    "template": "tasks",
                    "accent": "teal",
                    "items": ["Acheter du café", {"label": "Budget", "value": 42}],
                },
                requires_apply=False,
                suggested_actions=[{"label": "Épingler ce widget", "kind": "pin_widget"}],
            )
        )
        assert res["ok"] is True
        widget_op = res["revision"]["ops"][0]
        assert widget_op["widget_id"] == "mes_taches_dete"
        assert widget_op["template"] == "checklist"
        assert widget_op["accent_hex"] == "#0D6E6E" and widget_op["accent_ink_hex"] == "#FFFFFF"
        assert widget_op["home_screen_safe"] is True
        assert widget_op["items"] == [{"label": "Acheter du café"}, {"label": "Budget", "value": "42"}]
        assert res["revision"]["suggested_actions"][0]["id"] == "epingler_ce_widget"

        # Surface alias + auto-slug + theme accent preset ("coral").
        res2 = json.loads(
            mobile_ui_patch_tool(
                ops=[
                    {"op": "upsert_surface", "title": "Vitesse API", "kind": "kpi"},
                    {"op": "upsert_quick_action", "label": "Résumé matin", "prompt": "Fais mon résumé"},
                    {"op": "set_theme_override", "theme": {"accent": "coral"}},
                ],
                requires_apply=False,
            )
        )
        assert res2["ok"] is True
        assert res2["revision"]["ops"][0]["surface_id"] == "vitesse_api"
        assert res2["revision"]["ops"][0]["kind"] == "kpi_grid"
        assert res2["revision"]["ops"][1]["action_id"] == "resume_matin"
        assert res2["revision"]["ops"][2]["theme"]["accent_hex"] == "#B83227"

        # Strict security preserved even when home_screen_safe is omitted: secrets & explicit False STILL fail with hint+example!
        secret_fail = json.loads(
            mobile_ui_patch_tool(ops={"op": "upsert_widget", "title": "Token sk-abcdef1234567890", "template": "kpi"})
        )
        assert secret_fail["ok"] is False and "hint" in secret_fail and "example" in secret_fail
        false_fail = json.loads(
            mobile_ui_patch_tool(
                ops={"op": "upsert_widget", "title": "Privé", "template": "kpi", "home_screen_safe": False}
            )
        )
        assert false_fail["ok"] is False and "hint" in false_fail

        # mobile_deliver forgiving aliases: kind="alert" -> "notice", urgency="low" -> "silent", plain string payload.
        dlv = json.loads(mobile_deliver_tool(kind="alert", title="Rappel doux", payload="Pense à boire de l'eau", urgency="low"))
        assert dlv["ok"] is True
        assert dlv["delivery"]["kind"] == "notice"
        assert dlv["delivery"]["urgency"] == "silent"
        assert dlv["delivery"]["payload"] == {"text": "Pense à boire de l'eau"}

        # mobile_deliver self-correcting error hint + example.
        bad_dlv = json.loads(mobile_deliver_tool(kind="invalid_kind", title="Bad", payload={}))
        assert bad_dlv["ok"] is False and "hint" in bad_dlv and "example" in bad_dlv

    # Device bridge alias normalization ("timer" -> "alarm_timer") still enforces advertised scope + self-correcting hints.
    broker = DeviceBridgeBroker()
    with device_bridge_scope("owner", capabilities=["alarm_timer"], run_id="r_alias", profile_key="default", broker=broker):
        bad_dev = json.loads(request_device_action_tool("photo", "Take photo"))
        assert bad_dev["ok"] is False and bad_dev["error"] == "capability_not_advertised"
        assert bad_dev["action_type"] == "camera" and "hint" in bad_dev
        bad_reason = json.loads(request_device_action_tool("timer", ""))
        assert bad_reason["ok"] is False and "hint" in bad_reason and "example" in bad_reason


# ── 7. Real Gateway-Authority TurnRunner End-to-End Integration ──────────────


def test_gateway_authority_turn_runner_wires_device_bridge_and_live_emitters(tmp_path):
    import threading
    from types import SimpleNamespace

    from gateway.platforms.api_server_device_bridge import ensure_device_bridge_broker
    from gateway.run_turn_runner import TurnRunner
    from gateway.session_api_turn import api_execution
    from gateway.turn_context import TurnContext

    # Build a minimal real-shaped authority + adapter fixture matching gateway/run.py.
    adapter = _make_adapter()
    ui_seen: list[dict] = []
    dlv_seen: list[dict] = []
    dev_events: list[tuple[str, dict]] = []

    live_session = SimpleNamespace(event_stream=SimpleNamespace(execution={"admission_id": "adm_auth_1"}))
    authority = SimpleNamespace(
        profile_id="default",
        logical_owner=lambda sid: sid,
        persistence=SimpleNamespace(sessions=SimpleNamespace(get=lambda sid: SimpleNamespace(runtime_generation=1))),
        sessions={"sess-auth-1": live_session},
        api_observers={
            "adm_auth_1": [{
                "run_id": "run_auth_1",
                "ui_patch_callback": ui_seen.append,
                "delivery_callback": dlv_seen.append,
            }]
        },
    )
    runner = SimpleNamespace(
        session_authority=authority,
        _consume_pending_native_image_paths=lambda key: [],
    )
    adapter.gateway_runner = runner
    broker = ensure_device_bridge_broker(adapter)
    broker.register_run_listener("default", "run_auth_1", lambda ev, p: dev_events.append((ev, p)))

    # Resolve pending device request from a helper thread once requested.
    def _approve_when_requested():
        for _ in range(100):
            pending = broker.list_pending("default", "run_auth_1")
            if pending:
                broker.resolve_request(
                    profile_key="default",
                    run_id="run_auth_1",
                    request_id=pending[0]["request_id"],
                    status="completed",
                    data={"ok_from_phone": True},
                )
                return
            import time
            time.sleep(0.01)

    approver = threading.Thread(target=_approve_when_requested, daemon=True)
    approver.start()

    prepared = {
        "adapter": adapter,
        "settings": {},
        "history": None,
        "content": "go",
        "turn_author": None,
        "relay_metadata": None,
        "run_instructions": None,
        "device_capabilities": ["alarm_timer", "ui_surfaces_v2"],
        "resume_unanswered": False,
    }
    token = api_execution.set(prepared)
    with override_hermes_home(tmp_path):
        try:
            turn = TurnRunner(
                runner,
                TurnContext(
                    message="go",
                    session_id="sess-auth-1",
                    session_key="api",
                    source=SimpleNamespace(user_id="owner", user_name="Owner", is_bot=False),
                ),
            )
            agent = SimpleNamespace(ephemeral_system_prompt=None, _cached_system_prompt="base")
            outcomes = {}

            def _run(message, **kwargs):
                outcomes["ui"] = json.loads(
                    mobile_ui_patch_tool(
                        ops={"op": "upsert_widget", "title": "Brief", "template": "metric", "accent": "teal"},
                        session_id="sess-auth-1",
                    )
                )
                outcomes["dlv"] = json.loads(
                    mobile_deliver_tool(kind="alert", title="Done", payload="All set", urgency="low", session_id="sess-auth-1")
                )
                outcomes["dev"] = json.loads(request_device_action_tool("timer", "Set 5m timer"))
                return {"final_response": "ok"}

            agent.run_conversation = _run
            turn._run_conversation_with_approval(agent, [], None, None, None)
        finally:
            api_execution.reset(token)
            approver.join(timeout=2.0)

    assert outcomes["ui"]["ok"] is True and len(ui_seen) == 1
    assert outcomes["dlv"]["ok"] is True and len(dlv_seen) == 1
    assert outcomes["dev"]["ok"] is True and outcomes["dev"]["data"] == {"ok_from_phone": True}
    assert [ev for ev, _ in dev_events] == ["device.requested", "device.settled"]
