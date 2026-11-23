"""Comprehensive tests for Track 19A: ephemeral pairing tickets, multi-profile management,
first-profile admin & admin settings API, localized onboarding, LLM tool, and CLI wizard.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer
import pytest

from gateway.config import Platform, PlatformConfig
from gateway.i18n import user_lang_scope
from gateway.mobile_pairing import PairingError, PairingStore, build_pairing_deep_link
from gateway.mobile_profiles import (
    ProfileManagementError,
    create_managed_profile,
    delete_managed_profile,
    get_admin_settings,
    is_admin_profile,
    list_managed_profiles,
    read_profile_api_key,
    rotate_managed_profile_key,
    update_admin_settings,
    update_managed_profile,
    write_profile_api_key,
)
from gateway.platforms.api_server import APIServerAdapter
from scripts import mobile_setup
from tools.mobile_profile_manager import manage_profiles_tool
from tools.mobile_ui_patch import validate_interactive_cards


def _make_adapter(monkeypatch, tmp_path: Path, *, default_key: str = "admin-super-secret-key-123456"):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("API_SERVER_KEY", default_key)
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: tmp_path)
    monkeypatch.setattr("gateway.mobile_pairing._default_root_home", lambda: tmp_path)
    monkeypatch.setattr("gateway.mobile_profiles.default_root_home", lambda: tmp_path)
    monkeypatch.setattr("gateway.platforms.api_server_profiles.default_root_home", lambda: tmp_path)
    monkeypatch.setattr("tools.mobile_profile_manager.default_root_home", lambda: tmp_path)
    write_profile_api_key("default", default_key, tmp_path)

    adapter = APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "host": "127.0.0.1",
                "port": 0,
                "api_key": default_key,
                "auth_failure_max_attempts": 4,
                "auth_failure_window_seconds": 60,
            },
        )
    )
    adapter._api_key = default_key
    adapter._model_name = "test-model-v1"
    from gateway.platforms.api_server_auth_guard import AuthFailureLimiter

    adapter._auth_limiter = AuthFailureLimiter(max_attempts=4, window_seconds=60)

    class _FakeCfg:
        multiplex_profiles = True

    class _FakeRunner:
        config = _FakeCfg()

    adapter.gateway_runner = _FakeRunner()
    return adapter


async def _start_test_client(adapter: APIServerAdapter):
    from aiohttp import web
    from gateway.platforms import api_server_runs as _api_runs

    app = web.Application(
        middlewares=[
            adapter._make_profile_prefix_middleware(),
            _api_runs._LANGUAGE_SCOPE_MIDDLEWARE,
        ]
    )
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
        app.router.add_route(method, f"/p/{{profile}}{path}", handler)
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    return client


# ── 1. Ephemeral Pairing Store: Atomic Race, Expiry, Metadata Redaction ──────


def test_pairing_store_atomic_single_use_race(tmp_path: Path):
    store = PairingStore(tmp_path)
    ticket = store.create_ticket(
        profile_slug="default",
        display_name="Primary",
        mode="device",
        created_by="default",
        ttl_seconds=300,
        host="https://hermes.example.com",
    )
    assert ticket["deep_link"].startswith("hermes://pair?host=https%3A%2F%2Fhermes.example.com&code=")
    # Active list NEVER returns the plaintext code or hash.
    active = store.list_active_tickets(profile_slug="default")
    assert len(active) == 1
    assert "code" not in active[0]
    assert "code_hash" not in active[0]
    assert active[0]["ticket_id"] == ticket["ticket_id"]

    # 16 concurrent threads race to claim the exact same code: strictly 1 succeeds!
    def _attempt():
        try:
            return ("ok", store.claim_ticket(ticket["code"]))
        except PairingError as exc:
            return (exc.code, exc.status_code)

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda _: _attempt(), range(16)))

    winners = [r for r in results if r[0] == "ok"]
    losers = [r for r in results if r[0] != "ok"]
    assert len(winners) == 1
    assert len(losers) == 15
    assert all(code == "pairing_ticket_consumed" and status == 410 for code, status in losers)


def test_pairing_store_expiry_and_revoke(tmp_path: Path):
    now = [1000.0]
    store = PairingStore(tmp_path, clock=lambda: now[0])
    t1 = store.create_ticket(profile_slug="alice", display_name="Alice", ttl_seconds=60)
    t2 = store.create_ticket(profile_slug="alice", display_name="Alice", ttl_seconds=300)

    # Non-admin bob cannot revoke alice's ticket.
    with pytest.raises(PairingError) as denied:
        store.revoke_ticket(t2["ticket_id"], allowed_profile="bob", is_admin=False)
    assert denied.value.status_code == 403

    # Alice revokes t2.
    rev = store.revoke_ticket(t2["ticket_id"], allowed_profile="alice", is_admin=False)
    assert rev["revoked"] is True
    with pytest.raises(PairingError) as revoked_claim:
        store.claim_ticket(t2["code"])
    assert revoked_claim.value.status_code == 410
    assert revoked_claim.value.code == "pairing_ticket_revoked"

    # Advance clock past t1 expiry.
    now[0] += 61.0
    with pytest.raises(PairingError) as expired_claim:
        store.claim_ticket(t1["code"])
    assert expired_claim.value.status_code == 410
    assert expired_claim.value.code == "pairing_ticket_expired"
    assert store.list_active_tickets(profile_slug="alice") == []


# ── 2. End-to-End HTTP Routes: Pairing, Profiles, Admin Settings, Isolation ──


@pytest.mark.asyncio
async def test_http_pairing_claim_rate_limit_and_capabilities(monkeypatch, tmp_path: Path):
    default_key = "admin-super-secret-key-123456"
    adapter = _make_adapter(monkeypatch, tmp_path, default_key=default_key)
    client = await _start_test_client(adapter)
    try:
        # Capabilities expose frictionless_onboarding, profile_management, and admin_settings.
        cap_resp = await client.get("/v1/capabilities", headers={"Authorization": f"Bearer {default_key}"})
        assert cap_resp.status == 200
        caps = await cap_resp.json()
        assert caps["features"]["frictionless_onboarding"] is True
        assert caps["features"]["profile_management"] is True
        assert caps["features"]["admin_settings"] is True
        assert "mobile_pairing_claim" in caps["endpoints"]
        assert "mobile_admin_settings_get" in caps["endpoints"]

        # Create a device pairing ticket for default (admin) profile.
        create_resp = await client.post(
            "/api/mobile/pairing/create",
            headers={"Authorization": f"Bearer {default_key}"},
            json={"host": "https://hermes.example.com", "ttl_seconds": 300},
        )
        assert create_resp.status == 201
        assert create_resp.headers.get("Cache-Control") == "no-store"
        ticket = await create_resp.json()
        assert ticket["profile_slug"] == "default"
        assert ticket["code"]

        # Claim unauthenticated with French Accept-Language -> 200 + Cache-Control: no-store.
        claim_resp = await client.post(
            "/api/mobile/pairing/claim",
            headers={"Accept-Language": "fr-CA,fr;q=0.9"},
            json={"code": ticket["code"], "host": "https://hermes.example.com"},
        )
        assert claim_resp.status == 200
        assert claim_resp.headers.get("Cache-Control") == "no-store"
        claimed = await claim_resp.json()
        assert claimed["api_key"] == default_key
        assert claimed["profile_slug"] == "default"
        assert claimed["is_admin"] is True
        assert claimed["role"] == "admin"
        assert claimed["onboarding"]["language"] == "fr"
        assert claimed["onboarding"]["starters"]

        # Second claim attempt MUST fail with 410.
        replay = await client.post("/api/mobile/pairing/claim", json={"code": ticket["code"]})
        assert replay.status == 410

        # Repeated invalid claim attempts trigger AuthFailureLimiter -> 429!
        statuses = []
        for idx in range(5):
            r = await client.post(
                "/api/mobile/pairing/claim",
                json={"code": f"bogus-pairing-code-attempt-{idx}"},
            )
            statuses.append(r.status)
        assert 429 in statuses
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_http_profiles_admin_isolation_last_admin_guard_and_key_rotation(monkeypatch, tmp_path: Path):
    default_key = "admin-super-secret-key-123456"
    adapter = _make_adapter(monkeypatch, tmp_path, default_key=default_key)
    client = await _start_test_client(adapter)
    try:
        admin_hdr = {"Authorization": f"Bearer {default_key}"}

        # 1. First profile (default) is automatically admin.
        assert is_admin_profile("default", tmp_path) is True
        onboard_en = await client.get("/api/mobile/onboarding", headers={**admin_hdr, "X-Hermes-Lang": "en"})
        assert onboard_en.status == 200
        ob_en = await onboard_en.json()
        assert ob_en["is_admin"] is True
        assert ob_en["profiles_manageable"] is True
        assert ob_en["language"] == "en"

        onboard_fr = await client.get("/api/mobile/onboarding", headers={**admin_hdr, "X-Hermes-Lang": "fr"})
        assert onboard_fr.status == 200
        ob_fr = await onboard_fr.json()
        assert ob_fr["language"] == "fr"
        assert ob_fr["starters"][0]["title"] != ob_en["starters"][0]["title"]

        # 2. Admin creates an isolated French member profile 'alex' with invite ticket.
        create_alex = await client.post(
            "/api/mobile/profiles",
            headers=admin_hdr,
            json={
                "slug": "alex",
                "display_name": "Alex",
                "preferred_language": "fr",
                "badge_color": "indigo",
                "create_invite": True,
                "host": "https://hermes.example.com",
            },
        )
        assert create_alex.status == 201
        alex_data = await create_alex.json()
        alex_key = alex_data["api_key"]
        assert alex_data["profile"]["slug"] == "alex"
        assert alex_data["profile"]["is_admin"] is False
        assert alex_data["profile"]["role"] == "member"
        assert alex_data["pairing_ticket"]["mode"] == "invite"

        # Fail-closed default: new non-admin members do NOT get host terminal/code_execution by default.
        alex_hdr = {"Authorization": f"Bearer {alex_key}"}
        alex_prof = await (await client.get("/p/alex/api/profile", headers=alex_hdr)).json()
        assert "terminal" not in alex_prof["enabled_toolsets"]
        assert "code_execution" not in alex_prof["enabled_toolsets"]
        assert "web" in alex_prof["enabled_toolsets"]

        # New admin profile keeps full default toolsets (including terminal).
        create_bob = await client.post(
            "/api/mobile/profiles",
            headers=admin_hdr,
            json={"slug": "bob", "display_name": "Bob", "is_admin": True},
        )
        assert create_bob.status == 201
        bob_key = (await create_bob.json())["api_key"]
        bob_prof = await (
            await client.get("/p/bob/api/profile", headers={"Authorization": f"Bearer {bob_key}"})
        ).json()
        assert "terminal" in bob_prof["enabled_toolsets"]
        assert (
            await client.delete("/api/mobile/profiles/bob", headers=admin_hdr, json={"confirm_slug": "bob"})
        ).status == 200

        # Claim Alex's invite ticket unauthenticated.
        claim_alex = await client.post(
            "/api/mobile/pairing/claim",
            json={"code": alex_data["pairing_ticket"]["code"], "host": "https://hermes.example.com"},
        )
        assert claim_alex.status == 200
        alex_claimed = await claim_alex.json()
        assert alex_claimed["api_key"] == alex_key
        assert alex_claimed["profile_slug"] == "alex"
        assert alex_claimed["server_url"] == "https://hermes.example.com/p/alex"
        assert alex_claimed["is_admin"] is False

        # 3. Cross-profile privilege escalation checks: member 'alex' CANNOT access admin routes or other profiles!
        # List profiles as alex -> sees ONLY alex (never default!).
        alex_list = await client.get("/p/alex/api/mobile/profiles", headers=alex_hdr)
        assert alex_list.status == 200
        alex_list_body = await alex_list.json()
        assert alex_list_body["is_admin"] is False
        assert [p["slug"] for p in alex_list_body["data"]] == ["alex"]

        # Member alex gets 403 on create profile, edit default, invite default, delete default, admin settings!
        assert (await client.post("/p/alex/api/mobile/profiles", headers=alex_hdr, json={"slug": "eve"})).status == 403
        assert (
            await client.patch("/p/alex/api/mobile/profiles/default", headers=alex_hdr, json={"display_name": "Pwned"})
        ).status == 403
        assert (
            await client.patch("/p/alex/api/mobile/profiles/alex", headers=alex_hdr, json={"is_admin": True})
        ).status == 403
        assert (await client.post("/p/alex/api/mobile/profiles/default/invite", headers=alex_hdr, json={})).status == 403
        assert (
            await client.delete("/p/alex/api/mobile/profiles/default", headers=alex_hdr, json={"confirm_slug": "default"})
        ).status == 403
        assert (await client.get("/p/alex/api/mobile/admin/settings", headers=alex_hdr)).status == 403
        assert (
            await client.patch("/p/alex/api/mobile/admin/settings", headers=alex_hdr, json={"pairing_ttl_seconds": 600})
        ).status == 403

        # 4. Last-admin guard: default cannot demote itself when it is the only admin (409), nor delete default (409).
        demote_last = await client.patch(
            "/api/mobile/profiles/default", headers=admin_hdr, json={"is_admin": False}
        )
        assert demote_last.status == 409
        del_default = await client.delete(
            "/api/mobile/profiles/default", headers=admin_hdr, json={"confirm_slug": "default"}
        )
        assert del_default.status == 409

        # Promote alex -> now 2 admins; default can be demoted? Still safe, then re-promote default & demote alex.
        assert (
            await client.patch("/api/mobile/profiles/alex", headers=admin_hdr, json={"is_admin": True})
        ).status == 200
        assert (
            await client.patch("/api/mobile/profiles/alex", headers=admin_hdr, json={"is_admin": False})
        ).status == 200

        # 5. Admin settings GET + PATCH persist cleanly and enforce allow_self_device_pairing=False.
        adm_get = await client.get("/api/mobile/admin/settings", headers=admin_hdr)
        assert adm_get.status == 200
        adm_body = await adm_get.json()
        assert adm_body["summary"]["profile_count"] == 2

        adm_patch = await client.patch(
            "/api/mobile/admin/settings",
            headers=admin_hdr,
            json={
                "default_language": "fr",
                "pairing_ttl_seconds": 450,
                "max_concurrent_runs": 24,
                "per_profile_concurrent_limit": 8,
                "allow_self_device_pairing": False,
            },
        )
        assert adm_patch.status == 200
        assert get_admin_settings(tmp_path)["pairing_ttl_seconds"] == 450
        assert adapter._max_concurrent_runs == 24
        assert adapter._max_concurrent_runs_per_profile == 8

        # With allow_self_device_pairing=False, member alex gets 403 when trying to pair another device.
        pair_blocked = await client.post("/p/alex/api/mobile/pairing/create", headers=alex_hdr, json={})
        assert pair_blocked.status == 403

        # 6. Rotate Alex's key -> old key IMMEDIATELY rejected (401), new key works (200).
        rot_resp = await client.post(
            "/p/alex/api/mobile/profiles/alex/rotate-key",
            headers=alex_hdr,
            json={"create_invite": False},
        )
        assert rot_resp.status == 200
        new_alex_key = (await rot_resp.json())["api_key"]
        assert new_alex_key != alex_key
        assert (await client.get("/p/alex/api/mobile/profiles", headers=alex_hdr)).status == 401
        assert (
            await client.get("/p/alex/api/mobile/profiles", headers={"Authorization": f"Bearer {new_alex_key}"})
        ).status == 200

        # 7. Safe offboarding requires confirm_slug == slug.
        bad_del = await client.delete(
            "/api/mobile/profiles/alex", headers=admin_hdr, json={"confirm_slug": "wrong"}
        )
        assert bad_del.status == 400
        good_del = await client.delete(
            "/api/mobile/profiles/alex", headers=admin_hdr, json={"confirm_slug": "alex"}
        )
        assert good_del.status == 200
        assert (await good_del.json())["deleted"] is True
    finally:
        await client.close()


# ── 3. LLM Tool Isolation, Explicit Approval Guard & Pairing Interactive Card ──


def test_llm_manage_profiles_tool_isolation_and_approval_gate(monkeypatch, tmp_path: Path):
    _make_adapter(monkeypatch, tmp_path)
    create_managed_profile(caller_slug="default", slug="sam", display_name="Sam", root_home=tmp_path)

    # Member sees only self and gets 403-equivalent error trying to create user.
    sam_list = json.loads(manage_profiles_tool("list_profiles", caller_profile="sam", root_home=tmp_path))
    assert sam_list["ok"] is True
    assert sam_list["is_admin"] is False
    assert [p["slug"] for p in sam_list["profiles"]] == ["sam"]

    sam_escalate = json.loads(
        manage_profiles_tool(
            "create_user_profile_and_invite",
            caller_profile="sam",
            root_home=tmp_path,
            slug="mallory",
            confirmed=True,
        )
    )
    assert sam_escalate["ok"] is False
    assert sam_escalate["code"] == "admin_required"

    # Admin creating user WITHOUT confirmed=True -> approval gate, zero mutation!
    with user_lang_scope("fr"):
        pending = json.loads(
            manage_profiles_tool(
                "create_user_profile_and_invite",
                caller_profile="default",
                root_home=tmp_path,
                slug="claire",
                display_name="Claire",
                preferred_language="fr",
                host="https://hermes.example.com",
            )
        )
    assert pending["ok"] is False
    assert pending["requires_approval"] is True
    assert "claire" in pending["approval_prompt"].lower()
    assert not (tmp_path / "profiles" / "claire").exists()

    # Admin with confirmed=True -> creates isolated profile + valid pairing_invite interactive card!
    with user_lang_scope("fr"):
        approved = json.loads(
            manage_profiles_tool(
                "create_user_profile_and_invite",
                caller_profile="default",
                root_home=tmp_path,
                slug="claire",
                display_name="Claire",
                preferred_language="fr",
                host="https://hermes.example.com",
                confirmed=True,
            )
        )
    assert approved["ok"] is True
    assert approved["profile"]["slug"] == "claire"
    card = approved["interactive_card"]
    assert card["kind"] == "pairing_invite"
    validated_cards, err = validate_interactive_cards([card])
    assert err is None
    assert validated_cards and validated_cards[0]["kind"] == "pairing_invite"

    # Rotate key requires explicit approval too.
    rot_pending = json.loads(
        manage_profiles_tool("rotate_profile_key", caller_profile="sam", root_home=tmp_path)
    )
    assert rot_pending["requires_approval"] is True


# ── 4. CLI Setup Wizard & Pure-Python QR Renderer ────────────────────────────


def test_mobile_setup_cli_invite_pair_and_qr(monkeypatch, tmp_path: Path, capsys):
    _make_adapter(monkeypatch, tmp_path)
    rc = mobile_setup.main([
        "--invite", "nina",
        "--display-name", "Nina",
        "--lang", "fr",
        "--public-url", "https://hermes.example.com",
        "--hermes-home", str(tmp_path),
        "--qr",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "hermes://pair?" in out
    assert "█" in out or "▀" in out or "▄" in out
    assert read_profile_api_key("nina", tmp_path)

    # --pair --json returns structured payload with valid deep_link.
    rc2 = mobile_setup.main([
        "--pair",
        "--profile", "nina",
        "--public-url", "https://hermes.example.com",
        "--hermes-home", str(tmp_path),
        "--json",
    ])
    assert rc2 == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["profile"]["slug"] == "nina"
    assert parsed["pairing_ticket"]["deep_link"].startswith("hermes://pair?")

    # QR encoder produces square matrix with standard 7x7 finder patterns.
    matrix = mobile_setup.encode_qr_matrix(build_pairing_deep_link("https://hermes.example.com", "abc1234567890"))
    n = len(matrix)
    assert n >= 21 and all(len(row) == n for row in matrix)
    assert all(matrix[0][c] for c in range(7))
    assert all(matrix[0][n - 7 + c] for c in range(7))
    assert all(matrix[n - 7][c] for c in range(7))
