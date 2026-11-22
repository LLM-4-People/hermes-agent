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
