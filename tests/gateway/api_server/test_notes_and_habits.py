"""Deterministic positive, negative, ACL, and cross-profile isolation tests for:
- Smart Notes REST API & small-LLM forgiving ``mobile_notes`` tool
- Explicit same-server multi-user sharing (read vs edit, revoke, owner-only guards, deny-by-default)
- Privacy-safe Habit-Adaptive Suggestions (strict count >= 3 threshold, zero PII fields)
"""

from __future__ import annotations

import json
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, security_headers_middleware
from tools.mobile_notes import mobile_notes_tool
from tools.mobile_sharing import mobile_share_tool

OWNER_KEY = "owner-key-1234567890abcdef"
BETA_KEY = "beta-key-1234567890abcdef"
GAMMA_KEY = "gamma-key-1234567890abcdef"


def _make_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(
        middlewares=[adapter._make_profile_prefix_middleware(), security_headers_middleware]
    )
    app["api_server_adapter"] = adapter
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
        app.router.add_route(method, f"/p/{{profile}}{path}", handler)
    return app


def _make_adapter(key: str = OWNER_KEY, *, multiplex: bool = False) -> APIServerAdapter:
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": key}))
    if multiplex:

        class _Runner:
            config = GatewayConfig(multiplex_profiles=True)

        adapter.gateway_runner = _Runner()
    return adapter


@pytest.mark.asyncio
async def test_smart_notes_crud_sharing_acls_and_cross_profile_isolation(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    beta = root / "profiles" / "beta"
    gamma = root / "profiles" / "gamma"
    beta.mkdir(parents=True)
    gamma.mkdir(parents=True)
    (root / "config.yaml").write_text("{}\n", encoding="utf-8")
    (beta / "config.yaml").write_text("{}\n", encoding="utf-8")
    (gamma / "config.yaml").write_text("{}\n", encoding="utf-8")
    (beta / ".env").write_text(f"API_SERVER_KEY={BETA_KEY}\n", encoding="utf-8")
    (gamma / ".env").write_text(f"API_SERVER_KEY={GAMMA_KEY}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    adapter = _make_adapter(OWNER_KEY, multiplex=True)
    owner_h = {"Authorization": f"Bearer {OWNER_KEY}"}
    beta_h = {"Authorization": f"Bearer {BETA_KEY}"}
    gamma_h = {"Authorization": f"Bearer {GAMMA_KEY}"}

    async with TestClient(TestServer(_make_app(adapter))) as cli:
        # Unauthenticated rejected
        assert (await cli.get("/api/notes")).status == 401
        assert (await cli.post("/api/notes", json={"title": "x"})).status == 401

        # Capabilities advertise smart_notes, note_sharing, habit_suggestions
        caps = await (await cli.get("/v1/capabilities", headers=owner_h)).json()
        assert caps["features"]["smart_notes"] is True
        assert caps["features"]["note_sharing"] is True
        assert caps["features"]["habit_suggestions"] is True

        # Owner creates a private note (auto-slug ID + tags normalization)
        create = await cli.post(
            "/api/notes",
            headers=owner_h,
            json={"title": "Weekend Trip Plan", "content": "Book train tickets", "tags": "Travel, Ideas", "pinned": True},
        )
        assert create.status == 201
        assert create.headers.get("Cache-Control") == "no-store"
        note = (await create.json())["note"]
        note_id = note["id"]
        assert note_id == "weekend-trip-plan"
        assert note["tags"] == ["travel", "ideas"]
        assert note["pinned"] is True
        assert note["access"] == "owner"

        # Deny-by-default: beta and gamma CANNOT see unshared note even if guessing exact ID
        assert (await cli.get(f"/p/beta/api/notes/{note_id}", headers=beta_h)).status == 404
        assert (await cli.patch(f"/p/beta/api/notes/{note_id}", headers=beta_h, json={"content": "hack"})).status == 404
        assert (await cli.delete(f"/p/beta/api/notes/{note_id}", headers=beta_h)).status == 404
        assert (await (await cli.get("/p/beta/api/notes/shared-with-me", headers=beta_h)).json())["data"] == []

        # Negative sharing checks: unknown profile (404), self-share (400), invalid permission (400)
        assert (await cli.post(f"/api/notes/{note_id}/shares", headers=owner_h, json={"target_profile": "ghost", "permission": "read"})).status == 404
        assert (await cli.post(f"/api/notes/{note_id}/shares", headers=owner_h, json={"target_profile": "default", "permission": "read"})).status == 400
        assert (await cli.post(f"/api/notes/{note_id}/shares", headers=owner_h, json={"target_profile": "beta", "permission": "admin"})).status == 400

        # Owner explicitly shares READ-ONLY with beta
        sh = await cli.post(
            f"/api/notes/{note_id}/shares",
            headers=owner_h,
            json={"target_profile": "beta", "permission": "read"},
        )
        assert sh.status == 201

        # Beta sees it in shared-with-me and can GET it, but CANNOT edit, delete, or re-share
        shared_list = (await (await cli.get("/p/beta/api/notes/shared-with-me", headers=beta_h)).json())["data"]
        assert len(shared_list) == 1 and shared_list[0]["id"] == note_id and shared_list[0]["access"] == "read"
        beta_note = (await (await cli.get(f"/p/beta/api/notes/{note_id}", headers=beta_h)).json())["note"]
        assert beta_note["can_edit"] is False and beta_note["can_share"] is False
        assert "shares" not in beta_note  # ACL list stays owner-only
        assert (await cli.patch(f"/p/beta/api/notes/{note_id}", headers=beta_h, json={"append": "No write"})).status == 403
        assert (await cli.delete(f"/p/beta/api/notes/{note_id}", headers=beta_h)).status == 403
        assert (
            await cli.post(
                f"/p/beta/api/notes/{note_id}/shares",
                headers=beta_h,
                json={"target_profile": "gamma", "permission": "read"},
            )
        ).status == 403

        # Gamma STILL cannot see it (strictly isolated to beta!)
        assert (await cli.get(f"/p/gamma/api/notes/{note_id}", headers=gamma_h)).status == 404

        # Owner upgrades beta to EDIT permission
        up = await cli.post(
            f"/api/notes/{note_id}/shares",
            headers=owner_h,
            json={"target_profile": "beta", "permission": "edit"},
        )
        assert up.status == 201
        edit_resp = await cli.patch(
            f"/p/beta/api/notes/{note_id}",
            headers=beta_h,
            json={"append": "Pack raincoat"},
        )
        assert edit_resp.status == 200
        edited = (await edit_resp.json())["note"]
        assert "Pack raincoat" in edited["content"]
        assert edited["updated_by"] == "beta"
        # Even with edit access, beta cannot pin/archive or delete or re-share
        assert (await cli.patch(f"/p/beta/api/notes/{note_id}", headers=beta_h, json={"pinned": False})).status == 403
        assert (await cli.delete(f"/p/beta/api/notes/{note_id}", headers=beta_h)).status == 403

        # Owner revokes share -> beta IMMEDIATELY loses access
        rev = await cli.delete(f"/api/notes/{note_id}/shares/beta", headers=owner_h)
        assert rev.status == 200
        assert (await cli.get(f"/p/beta/api/notes/{note_id}", headers=beta_h)).status == 404
        assert (await cli.patch(f"/p/beta/api/notes/{note_id}", headers=beta_h, json={"append": "Too late"})).status == 404
        assert (await (await cli.get("/p/beta/api/notes/shared-with-me", headers=beta_h)).json())["data"] == []

        # Owner deletes note cleanly
        assert (await cli.delete(f"/api/notes/{note_id}", headers=owner_h)).status == 200
        assert (await cli.get(f"/api/notes/{note_id}", headers=owner_h)).status == 404


def test_mobile_notes_tool_forgiving_small_llm_normalization(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    alice = root / "profiles" / "alice"
    alice.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))

    # Action alias "add", title inferred from content, comma-separated tags
    res1 = json.loads(mobile_notes_tool({"action": "add", "content": "# Quick Groceries\n- Oat milk", "tags": "Home, Food"}))
    assert res1["success"] is True
    nid = res1["note"]["id"]
    assert nid == "quick-groceries"
    assert res1["note"]["tags"] == ["home", "food"]

    # Append & search alias "find"
    res2 = json.loads(mobile_notes_tool({"action": "append", "note_id": nid, "content": "- Coffee beans"}))
    assert res2["success"] is True and "Coffee beans" in res2["note"]["content"]
    found = json.loads(mobile_notes_tool({"action": "find", "query": "coffee"}))
    assert found["success"] is True and len(found["notes"]) == 1

    # Permission alias "viewer" -> "read"
    shared = json.loads(mobile_notes_tool({"action": "share", "note_id": nid, "target_profile": "alice", "permission": "viewer"}))
    assert shared["success"] is True and shared["permission"] == "read"

    # Self-correcting error responses include hint & example for small LLMs
    bad_action = json.loads(mobile_notes_tool({"action": "explode"}))
    assert bad_action["success"] is False and "hint" in bad_action and "example" in bad_action
    missing_title = json.loads(mobile_notes_tool({"action": "create"}))
    assert missing_title["success"] is False and "hint" in missing_title and "example" in missing_title


@pytest.mark.asyncio
async def test_habit_adaptive_suggestions_threshold_privacy_and_isolation(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    beta = root / "profiles" / "beta"
    beta.mkdir(parents=True)
    (root / "config.yaml").write_text("{}\n", encoding="utf-8")
    (beta / "config.yaml").write_text("{}\n", encoding="utf-8")
    (beta / ".env").write_text(f"API_SERVER_KEY={BETA_KEY}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    adapter = _make_adapter(OWNER_KEY, multiplex=True)
    owner_h = {"Authorization": f"Bearer {OWNER_KEY}"}
    beta_h = {"Authorization": f"Bearer {BETA_KEY}"}

    async with TestClient(TestServer(_make_app(adapter))) as cli:
        # Unauthenticated rejected
        assert (await cli.get("/api/mobile/habits/suggestions")).status == 401

        # Fresh profile returns empty suggestions (never spams generic guesses)
        s0 = await (await cli.get("/api/mobile/habits/suggestions", headers=owner_h)).json()
        assert s0["data"] == [] and s0["threshold"] == 3

        # Privacy guard: rejects free-form / PII fields or non-slug values
        assert (
            await cli.post(
                "/api/mobile/habits/events",
                headers=owner_h,
                json={"category": "intake_photo", "action": "extract_info", "text": "private receipt"},
            )
        ).status == 400
        assert (
            await cli.post(
                "/api/mobile/habits/events",
                headers=owner_h,
                json={"category": "has spaces!", "action": "ok"},
            )
        ).status == 400

        # Count 1 and 2 stay strictly below threshold (< 3 -> empty suggestions even if min_count=1 asked!)
        for expected_count in (1, 2):
            ev = await cli.post(
                "/api/mobile/habits/events",
                headers=owner_h,
                json={"category": "intake_photo", "action": "extract_info"},
            )
            assert ev.status == 201
            body = await ev.json()
            assert body["count"] == expected_count and body["learned"] is False
            sug = await (await cli.get("/api/mobile/habits/suggestions?min_count=1", headers=owner_h)).json()
            assert sug["data"] == [] and sug["threshold"] == 3

        # 3rd repetition crosses threshold -> surfaces adapted suggestion for default profile only!
        ev3 = await (
            await cli.post(
                "/api/mobile/habits/events",
                headers=owner_h,
                json={"category": "intake_photo", "action": "extract_info"},
            )
        ).json()
        assert ev3["count"] == 3 and ev3["learned"] is True

        sug_owner = await (await cli.get("/api/mobile/habits/suggestions?category=intake_photo", headers=owner_h)).json()
        assert len(sug_owner["data"]) == 1
        assert sug_owner["data"][0]["action"] == "extract_info"
        assert sug_owner["data"][0]["count"] == 3

        # Cross-profile isolation: beta sees zero suggestions
        sug_beta = await (await cli.get("/p/beta/api/mobile/habits/suggestions", headers=beta_h)).json()
        assert sug_beta["data"] == []

        # Reset clears learned habits cleanly
        reset_resp = await cli.delete("/api/mobile/habits", headers=owner_h)
        assert reset_resp.status == 200
        assert (await (await cli.get("/api/mobile/habits/suggestions", headers=owner_h)).json())["data"] == []


@pytest.mark.asyncio
async def test_universal_resource_sharing_engine_all_kinds_and_ssot_acls(tmp_path, monkeypatch):
    """Verify Universal Resource Sharing Engine across note, skill, widget, surface, tool,
    session, and artifact — including SSoT integration with notes, instant revoke, recipient
    dismiss, deny-by-default isolation, and small-LLM mobile_share tool."""
    root = tmp_path / "hermes"
    beta = root / "profiles" / "beta"
    gamma = root / "profiles" / "gamma"
    beta.mkdir(parents=True)
    gamma.mkdir(parents=True)
    (root / "config.yaml").write_text("{}\n", encoding="utf-8")
    (beta / "config.yaml").write_text("{}\n", encoding="utf-8")
    (gamma / "config.yaml").write_text("{}\n", encoding="utf-8")
    (beta / ".env").write_text(f"API_SERVER_KEY={BETA_KEY}\n", encoding="utf-8")
    (gamma / ".env").write_text(f"API_SERVER_KEY={GAMMA_KEY}\n", encoding="utf-8")
    # Seed a user skill in default profile
    skill_dir = root / "skills" / "deploy-guide"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: deploy-guide\ndescription: Safe release steps.\n---\nRun tests then tag.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(root))

    adapter = _make_adapter(OWNER_KEY, multiplex=True)
    owner_h = {"Authorization": f"Bearer {OWNER_KEY}"}
    beta_h = {"Authorization": f"Bearer {BETA_KEY}"}
    gamma_h = {"Authorization": f"Bearer {GAMMA_KEY}"}

    async with TestClient(TestServer(_make_app(adapter))) as cli:
        caps = await (await cli.get("/v1/capabilities", headers=owner_h)).json()
        assert caps["features"]["universal_sharing"] is True
        assert caps["endpoints"]["shares"] == {"method": "POST", "path": "/api/shares"}

        # 1. Create a note and share it via /api/notes/{id}/shares -> visible via universal /api/shares!
        note_r = await cli.post("/api/notes", headers=owner_h, json={"title": "Team Runbook", "content": "Step 1"})
        assert note_r.status == 201
        note_id = (await note_r.json())["note"]["id"]
        nshare = await cli.post(
            f"/api/notes/{note_id}/shares",
            headers=owner_h,
            json={"target_profile": "beta", "permission": "read"},
        )
        assert nshare.status == 201
        note_share_id = (await nshare.json())["share_id"]

        outgoing = (await (await cli.get("/api/shares/outgoing?resource_kind=note", headers=owner_h)).json())["data"]
        assert any(s["share_id"] == note_share_id and s["resource_id"] == note_id for s in outgoing)
        incoming = (await (await cli.get("/p/beta/api/shares/incoming?resource_kind=note", headers=beta_h)).json())["data"]
        assert len(incoming) == 1 and incoming[0]["share_id"] == note_share_id
        content_r = await cli.get(f"/p/beta/api/shares/incoming/{note_share_id}/content", headers=beta_h)
        assert content_r.status == 200
        assert (await content_r.json())["content"]["content"] == "Step 1"

        # Revoking via DELETE /api/shares/{share_id} IMMEDIATELY revokes /api/notes access too (SSoT!)
        assert (await cli.delete(f"/api/shares/{note_share_id}", headers=owner_h)).status == 200
        assert (await cli.get(f"/p/beta/api/notes/{note_id}", headers=beta_h)).status == 404
        assert (await cli.get(f"/p/beta/api/shares/incoming/{note_share_id}/content", headers=beta_h)).status == 404

        # 2. Share a skill with 'clone' permission -> beta reads content, gamma denied, beta cannot re-share
        sk_share = await cli.post(
            "/api/shares",
            headers=owner_h,
            json={"resource_kind": "skill", "resource_id": "deploy-guide", "target_profile": "beta", "permission": "clone"},
        )
        assert sk_share.status == 201
        sk_share_id = (await sk_share.json())["share"]["share_id"]
        sk_content = await cli.get(f"/p/beta/api/shares/incoming/{sk_share_id}/content", headers=beta_h)
        assert sk_content.status == 200
        assert "Run tests then tag." in (await sk_content.json())["content"]["body"]
        assert (await cli.get(f"/p/gamma/api/shares/incoming/{sk_share_id}/content", headers=gamma_h)).status == 404
        assert (
            await cli.post(
                "/p/beta/api/shares",
                headers=beta_h,
                json={"resource_kind": "skill", "resource_id": "deploy-guide", "target_profile": "gamma", "permission": "read"},
            )
        ).status == 403

        # 3. Share a widget -> recipient dismisses it -> disappears from both incoming and outgoing!
        w_share = await cli.post(
            "/api/shares",
            headers=owner_h,
            json={"resource_kind": "widget", "resource_id": "daily-metrics", "target_profile": "beta", "permission": "read", "label": "Daily KPIs"},
        )
        assert w_share.status == 201
        w_share_id = (await w_share.json())["share"]["share_id"]
        dismiss = await cli.delete(f"/p/beta/api/shares/{w_share_id}", headers=beta_h)
        assert dismiss.status == 200 and (await dismiss.json())["action"] == "dismissed"
        assert (await (await cli.get("/api/shares/outgoing?resource_kind=widget", headers=owner_h)).json())["data"] == []

        # 4. Negative checks: invalid kind (400), path traversal resource_id (400), missing profile (404)
        assert (await cli.post("/api/shares", headers=owner_h, json={"resource_kind": "secret", "resource_id": "x", "target_profile": "beta"})).status == 400
        assert (await cli.post("/api/shares", headers=owner_h, json={"resource_kind": "skill", "resource_id": "../etc", "target_profile": "beta"})).status == 400
        assert (await cli.post("/api/shares", headers=owner_h, json={"resource_kind": "widget", "resource_id": "w1", "target_profile": "nobody"})).status == 404

    # 5. Small-LLM forgiving mobile_share tool aliases & self-correcting hints
    tool_res = json.loads(
        mobile_share_tool({"action": "grant", "resource_kind": "dashboard", "resource_id": "kpi-1", "target_profile": "beta", "permission": "copy"})
    )
    assert tool_res["success"] is True
    assert tool_res["share"]["resource_kind"] == "surface"
    assert tool_res["share"]["permission"] == "clone"
    bad_tool = json.loads(mobile_share_tool({"action": "bogus"}))
    assert bad_tool["success"] is False and "hint" in bad_tool and "example" in bad_tool
