"""Comprehensive tests for Track 19A: ephemeral pairing tickets, multi-profile management,
first-profile admin & admin settings API, localized onboarding, LLM tool, and CLI wizard.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import stat
import struct
from unittest.mock import patch
import zlib

from aiohttp.test_utils import TestClient, TestServer
import pytest

from gateway.config import Platform, PlatformConfig
from gateway.i18n import user_lang_scope
from gateway.mobile_pairing import (
    PairingError,
    PairingStore,
    build_pairing_deep_link,
    normalize_public_host,
    resolve_public_host,
)
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
        assert "mobile_pairing_inspect" in caps["endpoints"]
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


def _extract_qr_format_copies(matrix: list[list[bool]]) -> tuple[int, int]:
    """Read both ISO/IEC 18004 Figure 25 format-info copies (bit 0 = LSB, bit 14 = MSB)."""
    size = len(matrix)
    copy1 = 0
    for i in range(6):
        copy1 |= (1 if matrix[i][8] else 0) << i
    copy1 |= (1 if matrix[7][8] else 0) << 6
    copy1 |= (1 if matrix[8][8] else 0) << 7
    copy1 |= (1 if matrix[8][7] else 0) << 8
    for i in range(9, 15):
        copy1 |= (1 if matrix[8][14 - i] else 0) << i

    copy2 = 0
    for i in range(8):
        copy2 |= (1 if matrix[8][size - 1 - i] else 0) << i
    for i in range(8, 15):
        copy2 |= (1 if matrix[size - 15 + i][8] else 0) << i
    return copy1, copy2


def _extract_qr_version_copies(matrix: list[list[bool]]) -> tuple[int, int]:
    """Read both ISO/IEC 18004 Figure 26 18-bit version-info blocks (Version 7+)."""
    size = len(matrix)
    top_right = 0
    bottom_left = 0
    for i in range(18):
        a, b = size - 11 + (i % 3), i // 3
        top_right |= (1 if matrix[b][a] else 0) << i
        bottom_left |= (1 if matrix[a][b] else 0) << i
    return top_right, bottom_left


def _qr_function_mask(version: int, size: int) -> list[list[bool]]:
    """Return the ISO/IEC 18004 function-module mask for *version* and *size*."""
    func = [[False] * size for _ in range(size)]
    for r0, c0 in ((0, 0), (0, size - 7), (size - 7, 0)):
        for dr in range(-1, 8):
            for dc in range(-1, 8):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < size and 0 <= c < size:
                    func[r][c] = True
    for i in range(8, size - 8):
        func[6][i] = func[i][6] = True
    centers = mobile_setup._QR_VERSIONS[version][3]
    last_center = len(centers) - 1
    for ri, r in enumerate(centers):
        for ci, c in enumerate(centers):
            if (ri == 0 and ci == 0) or (ri == 0 and ci == last_center) or (ri == last_center and ci == 0):
                continue
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    func[r + dr][c + dc] = True
    for i in range(9):
        func[i][8] = func[8][i] = True
    for i in range(8):
        func[8][size - 1 - i] = func[size - 1 - i][8] = True
    if version >= 7:
        for i in range(18):
            a, b = size - 11 + (i % 3), i // 3
            func[b][a] = func[a][b] = True
    return func


def _read_qr_data_bits(matrix: list[list[bool]], func: list[list[bool]], mask: int) -> list[int]:
    """Read and unmask data bits in standard ISO/IEC 18004 two-column zigzag order."""
    size = len(matrix)
    bits: list[int] = []
    col = size - 1
    upward = True
    while col > 0:
        if col == 6:
            col -= 1
        row_iter = range(size - 1, -1, -1) if upward else range(size)
        for r in row_iter:
            for dc in (0, -1):
                c = col + dc
                if func[r][c]:
                    continue
                val = 1 if matrix[r][c] else 0
                if mobile_setup._mask_bit(mask, r, c):
                    val ^= 1
                bits.append(val)
        col -= 2
        upward = not upward
    return bits


def _decode_qr_byte_payload_from_matrix(matrix: list[list[bool]]) -> str:
    """Pure-Python ISO/IEC 18004 byte-mode decoder verifying format, mask, zigzag, and deinterleaving."""
    size = len(matrix)
    version = (size - 17) // 4
    copy1, copy2 = _extract_qr_format_copies(matrix)
    assert copy1 == copy2
    unmasked_fmt = copy1 ^ 0x5412
    fmt_data = unmasked_fmt >> 10
    ec_bits = (fmt_data >> 3) & 0x3
    mask = fmt_data & 0x7
    assert ec_bits == 0b01
    assert 0 <= mask <= 7
    assert copy1 == mobile_setup._format_bits(0b01, mask)

    func = _qr_function_mask(version, size)
    bits = _read_qr_data_bits(matrix, func, mask)

    cap, _, blocks, _ = mobile_setup._QR_VERSIONS[version]
    codewords = [
        sum(bits[i + k] << (7 - k) for k in range(8))
        for i in range(0, cap * 8, 8)
    ]
    short_len = cap // blocks
    num_long = cap % blocks
    num_short = blocks - num_long
    block_lens = [short_len + (1 if idx >= num_short else 0) for idx in range(blocks)]
    reconstructed: list[list[int]] = [[] for _ in range(blocks)]
    pos = 0
    for col_idx in range(short_len + (1 if num_long else 0)):
        for blk_idx in range(blocks):
            if col_idx < block_lens[blk_idx]:
                reconstructed[blk_idx].append(codewords[pos])
                pos += 1
    data_bytes = [b for blk in reconstructed for b in blk]
    data_bits = [(b >> s) & 1 for b in data_bytes for s in range(7, -1, -1)]
    mode = sum(data_bits[k] << (3 - k) for k in range(4))
    assert mode == 0b0100
    len_width = 8 if version <= 9 else 16
    length = sum(data_bits[4 + k] << (len_width - 1 - k) for k in range(len_width))
    start = 4 + len_width
    raw = bytes(
        sum(data_bits[start + i * 8 + k] << (7 - k) for k in range(8))
        for i in range(length)
    )
    return raw.decode("utf-8")


def _decode_qr_matrix_from_unicode(rendered: str, *, border: int, invert: bool) -> list[list[bool]]:
    rows: list[list[bool]] = []
    for line in rendered.splitlines():
        top: list[bool] = []
        bottom: list[bool] = []
        for ch in line:
            t_fg = ch in {"█", "▀"}
            b_fg = ch in {"█", "▄"}
            top.append((not t_fg) if invert else t_fg)
            bottom.append((not b_fg) if invert else b_fg)
        rows.append(top)
        rows.append(bottom)
    padded_width = len(rows[0])
    core_size = padded_width - border * 2
    return [rows[border + r][border : border + core_size] for r in range(core_size)]


def _decode_qr_matrix_from_png(png_bytes: bytes, *, scale: int = 10, border: int = 4) -> list[list[bool]]:
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    pos = 8
    width = height = 0
    idat_parts: list[bytes] = []
    while pos < len(png_bytes):
        length = struct.unpack(">I", png_bytes[pos : pos + 4])[0]
        ctype = png_bytes[pos + 4 : pos + 8]
        cdata = png_bytes[pos + 8 : pos + 8 + length]
        crc_actual = struct.unpack(">I", png_bytes[pos + 8 + length : pos + 12 + length])[0]
        crc_expected = zlib.crc32(cdata, zlib.crc32(ctype)) & 0xFFFFFFFF
        assert crc_actual == crc_expected
        if ctype == b"IHDR":
            width, height, bit_depth, color_type, *_ = struct.unpack(">IIBBBBB", cdata)
            assert width == height and bit_depth == 8 and color_type == 0
        elif ctype == b"IDAT":
            idat_parts.append(cdata)
        elif ctype == b"IEND":
            break
        pos += 12 + length
    raw = zlib.decompress(b"".join(idat_parts))
    stride = 1 + width
    assert len(raw) == height * stride
    padded_modules = width // scale
    core_size = padded_modules - border * 2
    matrix: list[list[bool]] = []
    for r in range(core_size):
        py = (border + r) * scale + scale // 2
        row_start = py * stride
        assert raw[row_start] == 0
        row: list[bool] = []
        for c in range(core_size):
            px = (border + c) * scale + scale // 2
            pixel = raw[row_start + 1 + px]
            assert pixel in (0, 255)
            row.append(pixel == 0)
        matrix.append(row)
    return matrix


def test_qr_format_and_version_bits_iso18004_versions_1_to_10():
    # Payloads chosen to exercise every single QR Version 1 through 10.
    version_lengths = {1: 12, 2: 24, 3: 40, 4: 64, 5: 90, 6: 120, 7: 144, 8: 175, 9: 215, 10: 250}
    assert mobile_setup._format_bits(0b01, 0) == 0x77C4  # ISO/IEC 18004 Table C.1 EC-L mask 0

    try:
        import cv2
        import numpy as np
    except ImportError:
        cv2 = None
        np = None

    for version, length in version_lengths.items():
        payload = f"https://chat.hermes.example.com/mobile?v={version}&p=" + ("a" * length)
        payload = payload[:length]
        assert mobile_setup._choose_qr_version(len(payload.encode("utf-8"))) == version
        matrix = mobile_setup.encode_qr_matrix(payload)
        size = 17 + 4 * version
        assert len(matrix) == size

        copy1, copy2 = _extract_qr_format_copies(matrix)
        assert copy1 == copy2, f"V{version} Copy 1 vs Copy 2 mismatch"
        mask = ((copy1 ^ 0x5412) >> 10) & 0x7
        expected_fmt = mobile_setup._format_bits(0b01, mask)
        assert copy1 == expected_fmt, f"V{version} Copy 1 mismatch"
        assert copy2 == expected_fmt, f"V{version} Copy 2 mismatch"
        assert matrix[size - 8][8] is True

        if version >= 7:
            expected_vbits = mobile_setup._version_bits(version)
            tr, bl = _extract_qr_version_copies(matrix)
            assert tr == expected_vbits, f"V{version} top-right version block mismatch"
            assert bl == expected_vbits, f"V{version} bottom-left version block mismatch"
            # Timing-row alignment center must be present (not skipped by timing pattern).
            mid_center = mobile_setup._QR_VERSIONS[version][3][1]
            for cr, cc in ((6, mid_center), (mid_center, 6)):
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        assert matrix[cr + dr][cc + dc] is (max(abs(dr), abs(dc)) != 1)

        assert _decode_qr_byte_payload_from_matrix(matrix) == payload

        if cv2 is not None and np is not None:
            png_bytes = mobile_setup.encode_qr_png(matrix, scale=8, border=4)
            img = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
            assert decoded == payload, f"OpenCV failed to decode V{version}"


def test_qr_unicode_border_invert_and_png_roundtrip(monkeypatch, tmp_path: Path, capsys):
    _make_adapter(monkeypatch, tmp_path)
    target_url = "https://chat.hermes.example.com/mobile"
    deep_link = build_pairing_deep_link(target_url, "single_use_code_for_roundtrip_123456")

    # 1. Default Unicode border is 4 (ISO/IEC 18004 quiet zone) and round-trips cleanly.
    rendered = mobile_setup.render_qr_unicode(deep_link)
    lines = rendered.splitlines()
    assert lines[0] == " " * len(lines[0])
    assert lines[1] == " " * len(lines[1])
    assert all(line[:4] == "    " and line[-4:] == "    " for line in lines)
    recovered = _decode_qr_matrix_from_unicode(rendered, border=4, invert=False)
    assert _decode_qr_byte_payload_from_matrix(recovered) == deep_link

    # 2. Inverted Unicode paints the quiet zone white (█) for dark-mode terminals and round-trips identically.
    inverted = mobile_setup.render_qr_unicode(deep_link, invert=True)
    inv_lines = inverted.splitlines()
    assert inv_lines[0] == "█" * len(inv_lines[0])
    assert inv_lines[1] == "█" * len(inv_lines[1])
    recovered_inv = _decode_qr_matrix_from_unicode(inverted, border=4, invert=True)
    assert recovered_inv == recovered
    assert _decode_qr_byte_payload_from_matrix(recovered_inv) == deep_link

    # 3. CLI --qr --invert --png writes a crisp 0600 PNG that decodes back to the exact deep link.
    png_out = tmp_path / "pair_qr.png"
    rc = mobile_setup.main([
        "--pair",
        "--public-url", target_url,
        "--hermes-home", str(tmp_path),
        "--qr",
        "--invert",
        "--png", str(png_out),
    ])
    assert rc == 0
    stdout = capsys.readouterr().out
    link_line = next(line for line in stdout.splitlines() if line.startswith("Pairing link: "))
    emitted_link = link_line.split("Pairing link: ", 1)[1].strip()
    assert emitted_link.startswith("hermes://pair?host=https%3A%2F%2Fchat.hermes.example.com%2Fmobile&code=")
    assert f"QR PNG (0600): {png_out.resolve()}" in stdout
    assert png_out.is_file()
    assert stat.S_IMODE(png_out.stat().st_mode) == 0o600

    png_bytes = png_out.read_bytes()
    png_matrix = _decode_qr_matrix_from_png(png_bytes, scale=10, border=4)
    assert _decode_qr_byte_payload_from_matrix(png_matrix) == emitted_link

    try:
        import cv2
        import numpy as np
    except ImportError:
        cv2 = None
        np = None
    if cv2 is not None and np is not None:
        img = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
        assert decoded == emitted_link


def test_resolve_public_host_precedence_and_fail_closed(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("HERMES_PUBLIC_URL", raising=False)
    monkeypatch.delenv("API_SERVER_PUBLIC_URL", raising=False)

    # Tier 4a: fallback to request_scheme://request_host.
    assert (
        resolve_public_host(root_home=tmp_path, request_scheme="http", request_host="127.0.0.1:8642")
        == "http://127.0.0.1:8642"
    )
    # Tier 4b: reverse-proxy headers including X-Forwarded-Prefix preserve /mobile subpath.
    proxy_headers = {
        "X-Forwarded-Proto": "https, http",
        "X-Forwarded-Host": "chat.hermes.example.com, internal:8642",
        "X-Forwarded-Prefix": "/mobile/",
    }
    assert (
        resolve_public_host(
            root_home=tmp_path,
            request_headers=proxy_headers,
            request_scheme="http",
            request_host="127.0.0.1:8642",
        )
        == "https://chat.hermes.example.com/mobile"
    )

    # Tier 3a: config.yaml gateway.api_server.public_url beats reverse-proxy headers.
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("gateway:\n  api_server:\n    public_url: https://gw.example.com/mobile/\n", encoding="utf-8")
    assert (
        resolve_public_host(root_home=tmp_path, request_headers=proxy_headers)
        == "https://gw.example.com/mobile"
    )

    # Tier 3b: config.yaml api_server.public_url beats gateway.api_server.public_url.
    cfg_path.write_text(
        "api_server:\n  public_url: https://top.example.com/mobile/\n"
        "gateway:\n  api_server:\n    public_url: https://gw.example.com/mobile/\n",
        encoding="utf-8",
    )
    assert (
        resolve_public_host(root_home=tmp_path, request_headers=proxy_headers)
        == "https://top.example.com/mobile"
    )

    # Tier 2a: .env API_SERVER_PUBLIC_URL beats config.yaml and does NOT mutate os.environ.
    env_path = tmp_path / ".env"
    env_path.write_text('API_SERVER_PUBLIC_URL="https://dotenv-api.example.com/mobile/"\n', encoding="utf-8")
    assert resolve_public_host(root_home=tmp_path, request_headers=proxy_headers) == "https://dotenv-api.example.com/mobile"
    assert "API_SERVER_PUBLIC_URL" not in os.environ

    # Tier 2b: .env HERMES_PUBLIC_URL beats .env API_SERVER_PUBLIC_URL (supports inline comment).
    env_path.write_text(
        'API_SERVER_PUBLIC_URL="https://dotenv-api.example.com/mobile"\n'
        "export HERMES_PUBLIC_URL=https://dotenv-hermes.example.com/mobile/  # prod\n",
        encoding="utf-8",
    )
    assert (
        resolve_public_host(root_home=tmp_path, request_headers=proxy_headers)
        == "https://dotenv-hermes.example.com/mobile"
    )
    assert "HERMES_PUBLIC_URL" not in os.environ

    # Tier 2c: os.environ beats .env.
    monkeypatch.setenv("API_SERVER_PUBLIC_URL", "https://env-api.example.com/mobile/")
    assert resolve_public_host(root_home=tmp_path) == "https://env-api.example.com/mobile"
    monkeypatch.setenv("HERMES_PUBLIC_URL", "https://env-hermes.example.com/mobile/")
    assert resolve_public_host(root_home=tmp_path) == "https://env-hermes.example.com/mobile"

    # Tier 1: explicit raw_host overrides everything and preserves subpath.
    assert (
        resolve_public_host("https://chat.hermes.example.com/mobile/", root_home=tmp_path)
        == "https://chat.hermes.example.com/mobile"
    )

    # Fail-closed validation rejects embedded credentials, control chars, whitespace, query, fragment, bad schemes/ports.
    for bad in (
        "https://user:pass@chat.hermes.example.com/mobile",
        "https://:@chat.hermes.example.com/mobile",
        "https://chat.hermes.example.com/mobile?foo=bar",
        "https://chat.hermes.example.com/mobile#frag",
        "https://chat.hermes.example.com/\nmobile",
        "https://chat.hermes.example.com /mobile",
        "javascript:alert(1)",
        "ftp://chat.hermes.example.com",
        "https://chat.hermes.example.com:99999/mobile",
    ):
        with pytest.raises(PairingError) as exc:
            normalize_public_host(bad)
        assert exc.value.code == "invalid_host"

    monkeypatch.delenv("HERMES_PUBLIC_URL", raising=False)
    monkeypatch.delenv("API_SERVER_PUBLIC_URL", raising=False)
    env_path.unlink()
    cfg_path.unlink()
    for bad_prefix in ("//evil.com", "/mobile/../secret", "/mobile?x=1", "@evil.com", "/mobile\r\n"):
        with pytest.raises(PairingError) as exc:
            resolve_public_host(
                root_home=tmp_path,
                request_headers={
                    "X-Forwarded-Proto": "https",
                    "X-Forwarded-Host": "chat.hermes.example.com",
                    "X-Forwarded-Prefix": bad_prefix,
                },
            )
        assert exc.value.code == "invalid_host"


@pytest.mark.asyncio
async def test_public_host_ssot_across_api_cli_and_tool(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.delenv("HERMES_PUBLIC_URL", raising=False)
    monkeypatch.delenv("API_SERVER_PUBLIC_URL", raising=False)
    default_key = "admin-super-secret-key-123456"
    adapter = _make_adapter(monkeypatch, tmp_path, default_key=default_key)

    # Configure HERMES_PUBLIC_URL in <root_home>/.env (no env var, no explicit host argument).
    env_path = tmp_path / ".env"
    existing = env_path.read_text(encoding="utf-8-sig")
    env_path.write_text(existing + "HERMES_PUBLIC_URL=https://chat.hermes.example.com/mobile/\n", encoding="utf-8")

    # 1. CLI mobile_setup.py picks up .env HERMES_PUBLIC_URL automatically.
    rc = mobile_setup.main(["--pair", "--hermes-home", str(tmp_path), "--json"])
    assert rc == 0
    cli_out = json.loads(capsys.readouterr().out)
    assert cli_out["pairing_ticket"]["deep_link"].startswith(
        "hermes://pair?host=https%3A%2F%2Fchat.hermes.example.com%2Fmobile&code="
    )

    # 2. LLM tool manage_profiles picks up .env HERMES_PUBLIC_URL automatically.
    tool_out = json.loads(
        manage_profiles_tool("create_device_pairing_link", caller_profile="default", root_home=tmp_path)
    )
    assert tool_out["ok"] is True
    assert tool_out["pairing_ticket"]["deep_link"].startswith(
        "hermes://pair?host=https%3A%2F%2Fchat.hermes.example.com%2Fmobile&code="
    )

    # 3. API server routes pick up .env HERMES_PUBLIC_URL (and fall back to X-Forwarded-Prefix when absent).
    client = await _start_test_client(adapter)
    try:
        resp = await client.post(
            "/api/mobile/pairing/create",
            headers={"Authorization": f"Bearer {default_key}"},
            json={},
        )
        assert resp.status == 201
        ticket = await resp.json()
        assert ticket["deep_link"].startswith(
            "hermes://pair?host=https%3A%2F%2Fchat.hermes.example.com%2Fmobile&code="
        )

        # Claiming returns base_host + server_url with /mobile preserved.
        claim_resp = await client.post("/api/mobile/pairing/claim", json={"code": ticket["code"]})
        assert claim_resp.status == 200
        claimed = await claim_resp.json()
        assert claimed["base_host"] == "https://chat.hermes.example.com/mobile"
        assert claimed["server_url"] == "https://chat.hermes.example.com/mobile"

        # Without .env HERMES_PUBLIC_URL, X-Forwarded-Prefix is honored cleanly.
        env_path.write_text(existing, encoding="utf-8")
        resp2 = await client.post(
            "/api/mobile/pairing/create",
            headers={
                "Authorization": f"Bearer {default_key}",
                "X-Forwarded-Proto": "https",
                "X-Forwarded-Host": "chat.hermes.example.com",
                "X-Forwarded-Prefix": "/mobile",
            },
            json={},
        )
        assert resp2.status == 201
        ticket2 = await resp2.json()
        assert ticket2["deep_link"].startswith(
            "hermes://pair?host=https%3A%2F%2Fchat.hermes.example.com%2Fmobile&code="
        )
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_pairing_inspect_non_consuming_and_lifecycle(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("HERMES_PUBLIC_URL", raising=False)
    monkeypatch.delenv("API_SERVER_PUBLIC_URL", raising=False)
    default_key = "admin-super-secret-key-123456"
    adapter = _make_adapter(monkeypatch, tmp_path, default_key=default_key)
    client = await _start_test_client(adapter)
    try:
        admin_hdr = {"Authorization": f"Bearer {default_key}"}
        # Create a pairing ticket for default with a friendly display name.
        create_resp = await client.post(
            "/api/mobile/pairing/create",
            headers=admin_hdr,
            json={"host": "https://chat.hermes.example.com/mobile", "display_name": "Jacob", "ttl_seconds": 300},
        )
        assert create_resp.status == 201
        ticket = await create_resp.json()
        assert "&name=Jacob" in ticket["deep_link"]

        # Inspecting repeatedly NEVER consumes the ticket and NEVER returns api_key.
        for _ in range(3):
            insp = await client.post(
                "/api/mobile/pairing/inspect",
                json={"code": ticket["code"], "host": "https://chat.hermes.example.com/mobile"},
            )
            assert insp.status == 200
            assert insp.headers.get("Cache-Control") == "no-store"
            body = await insp.json()
            assert body["object"] == "hermes.mobile.pairing.inspect"
            assert body["valid"] is True
            assert body["server_url"] == "https://chat.hermes.example.com/mobile"
            assert body["base_host"] == "https://chat.hermes.example.com/mobile"
            assert body["profile_slug"] == "default"
            assert body["display_name"] == "Jacob"
            assert body["mode"] == "device"
            assert body["ttl_remaining_seconds"] > 0
            assert "api_key" not in body
            assert "code" not in body

        # Named profile inspect returns /p/<slug> server_url and never leaks api_key.
        create_alex = await client.post(
            "/api/mobile/profiles",
            headers=admin_hdr,
            json={
                "slug": "alex",
                "display_name": "Alex",
                "create_invite": True,
                "host": "https://chat.hermes.example.com/mobile",
            },
        )
        assert create_alex.status == 201
        alex_ticket = (await create_alex.json())["pairing_ticket"]
        assert "&profile=alex&name=Alex" in alex_ticket["deep_link"]
        alex_insp = await client.post(
            "/api/mobile/pairing/inspect",
            json={"code": alex_ticket["code"], "host": "https://chat.hermes.example.com/mobile"},
        )
        assert alex_insp.status == 200
        alex_body = await alex_insp.json()
        assert alex_body["server_url"] == "https://chat.hermes.example.com/mobile/p/alex"
        assert alex_body["profile_slug"] == "alex"
        assert alex_body["display_name"] == "Alex"
        assert "api_key" not in alex_body

        # Claiming the ticket succeeds after repeated inspects; then BOTH inspect and claim return 410.
        claim_resp = await client.post(
            "/api/mobile/pairing/claim",
            json={"code": ticket["code"], "host": "https://chat.hermes.example.com/mobile"},
        )
        assert claim_resp.status == 200
        assert (await claim_resp.json())["api_key"] == default_key

        after_inspect = await client.post("/api/mobile/pairing/inspect", json={"code": ticket["code"]})
        assert after_inspect.status == 410
        after_claim = await client.post("/api/mobile/pairing/claim", json={"code": ticket["code"]})
        assert after_claim.status == 410

        # Revoked ticket returns 410 on inspect.
        rev_resp = await client.delete(f"/api/mobile/pairing/{alex_ticket['ticket_id']}", headers=admin_hdr)
        assert rev_resp.status == 200
        rev_insp = await client.post("/api/mobile/pairing/inspect", json={"code": alex_ticket["code"]})
        assert rev_insp.status == 410

        # Unknown code returns 404, and repeated invalid inspects hit the shared rate limiter (429).
        statuses = []
        for idx in range(5):
            r = await client.post("/api/mobile/pairing/inspect", json={"code": f"unknown-inspect-code-{idx:04d}"})
            statuses.append(r.status)
        assert 404 in statuses
        assert 429 in statuses
    finally:
        await client.close()


def test_pairing_store_inspect_expired_and_deep_link_name_qr(tmp_path: Path):
    now = [2000.0]
    store = PairingStore(tmp_path, clock=lambda: now[0])

    # Missing DB fails closed with 404 without creating a database file.
    with pytest.raises(PairingError) as missing_exc:
        store.inspect_ticket("nonexistent_code_123456")
    assert missing_exc.value.status_code == 404
    assert not store.db_path.exists()

    ticket = store.create_ticket(
        profile_slug="default",
        display_name="Jacob",
        mode="device",
        created_by="default",
        ttl_seconds=60,
        host="https://chat.hermes.example.com/mobile",
    )
    deep_link = ticket["deep_link"]
    assert deep_link.startswith("hermes://pair?host=https%3A%2F%2Fchat.hermes.example.com%2Fmobile&code=")
    assert deep_link.endswith("&name=Jacob")
    assert "profile=" not in deep_link

    # Direct helper omits name when empty and rejects control characters fail-closed.
    plain_link = build_pairing_deep_link("https://chat.hermes.example.com/mobile", "code_1234567890")
    assert "name=" not in plain_link
    with pytest.raises(PairingError) as bad_name:
        build_pairing_deep_link("https://chat.hermes.example.com/mobile", "code_1234567890", display_name="Bad\nName")
    assert bad_name.value.code == "invalid_display_name"

    # Deep link with &name=Jacob round-trips cleanly through QR matrix, PNG, and OpenCV when available.
    matrix = mobile_setup.encode_qr_matrix(deep_link)
    assert _decode_qr_byte_payload_from_matrix(matrix) == deep_link
    png_bytes = mobile_setup.encode_qr_png(matrix, scale=10, border=4)
    assert _decode_qr_byte_payload_from_matrix(_decode_qr_matrix_from_png(png_bytes, scale=10, border=4)) == deep_link
    try:
        import cv2
        import numpy as np
    except ImportError:
        cv2 = None
        np = None
    if cv2 is not None and np is not None:
        img = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
        assert decoded == deep_link

    # Inspect succeeds while active and returns 410 once expired.
    info = store.inspect_ticket(ticket["code"])
    assert info["display_name"] == "Jacob"
    assert info["ttl_remaining_seconds"] == 60
    now[0] += 61.0
    with pytest.raises(PairingError) as exp_exc:
        store.inspect_ticket(ticket["code"])
    assert exp_exc.value.status_code == 410
    assert exp_exc.value.code == "pairing_ticket_expired"
