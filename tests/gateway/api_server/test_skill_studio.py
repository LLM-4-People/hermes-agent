"""Deterministic positive, negative, and cross-profile tests for Skill Studio CRUD API."""

from __future__ import annotations

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, security_headers_middleware

OWNER_KEY = "owner-key-1234567890abcdef"
BETA_KEY = "beta-key-1234567890abcdef"


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

    class _Runner:
        config = GatewayConfig(multiplex_profiles=multiplex)

    adapter.gateway_runner = _Runner()
    return adapter


_VALID_SKILL_MD = (
    "---\n"
    "name: release-check\n"
    "description: Verify build artifacts before tagging a release.\n"
    "tags: [release, ci]\n"
    "---\n\n"
    "# Release Check\n\n"
    "1. Run unit tests.\n"
    "2. Check changelog.\n"
)


@pytest.mark.asyncio
async def test_skill_studio_full_crud_toggle_and_guards(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("# Operator comment preserved\nskills:\n  disabled: []\n", encoding="utf-8")

    # Seed a bundled (builtin) skill and an essential skill
    bundled_dir = tmp_path / "skills" / "builtin-demo"
    bundled_dir.mkdir(parents=True)
    (bundled_dir / "SKILL.md").write_text(
        "---\nname: builtin-demo\ndescription: Bundled skill.\n---\nBundled instructions.\n",
        encoding="utf-8",
    )
    (tmp_path / "skills" / ".bundled_manifest").write_text(
        "builtin-demo:deadbeef\n", encoding="utf-8"
    )
    essential_dir = tmp_path / "skills" / "hermes-agent"
    essential_dir.mkdir(parents=True)
    (essential_dir / "SKILL.md").write_text(
        "---\nname: hermes-agent\ndescription: Essential agent skill.\n---\nCore instructions.\n",
        encoding="utf-8",
    )

    adapter = _make_adapter()
    headers = {"Authorization": f"Bearer {OWNER_KEY}"}
    async with TestClient(TestServer(_make_app(adapter))) as cli:
        # Unauthenticated rejected
        assert (await cli.post("/v1/skills", json={"name": "x", "content": _VALID_SKILL_MD})).status == 401

        # Create user skill
        create = await cli.post(
            "/v1/skills",
            headers=headers,
            json={"name": "release-check", "category": "devops", "content": _VALID_SKILL_MD},
        )
        assert create.status == 201
        created = await create.json()
        assert created["skill"]["name"] == "release-check"
        assert created["skill"]["source"] == "user"
        assert created["skill"]["writable"] is True
        assert created["skill"]["tags"] == ["release", "ci"]

        # Duplicate create -> 409
        dup = await cli.post(
            "/v1/skills",
            headers=headers,
            json={"name": "release-check", "category": "devops", "content": _VALID_SKILL_MD},
        )
        assert dup.status == 409

        # Negative: malformed names / traversal / bad frontmatter
        for bad_name in ("../escape", "Bad Uppercase", "a/b", ""):
            bad = await cli.post("/v1/skills", headers=headers, json={"name": bad_name, "content": _VALID_SKILL_MD})
            assert bad.status == 400
        assert (await cli.post("/v1/skills", headers=headers, json={"name": "nofm", "content": "plain text"})).status == 400
        assert (await cli.get("/v1/skills/..%2F..%2Fetc%2Fpasswd", headers=headers)).status == 400

        # Read detail
        detail_resp = await cli.get("/v1/skills/release-check", headers=headers)
        assert detail_resp.status == 200
        detail = await detail_resp.json()
        assert detail["frontmatter"]["name"] == "release-check"
        assert "Run unit tests." in detail["body"]

        # Patch skill
        patch_resp = await cli.put(
            "/v1/skills/release-check",
            headers=headers,
            json={"old_string": "Check changelog.", "new_string": "Verify release notes."},
        )
        assert patch_resp.status == 200
        assert "Verify release notes." in (await patch_resp.json())["skill"]["body"]

        # Full rewrite rejected when both content and patch provided
        assert (
            await cli.put(
                "/v1/skills/release-check",
                headers=headers,
                json={"content": _VALID_SKILL_MD, "old_string": "a", "new_string": "b"},
            )
        ).status == 400

        # Toggle disabled & verify include_disabled filter + YAML comment preservation
        off = await cli.put("/v1/skills/release-check/enabled", headers=headers, json={"enabled": False})
        assert off.status == 200
        assert (await off.json())["enabled"] is False
        assert "# Operator comment preserved" in (tmp_path / "config.yaml").read_text(encoding="utf-8")

        enabled_only = (await (await cli.get("/v1/skills", headers=headers)).json())["data"]
        assert all(row["name"] != "release-check" for row in enabled_only)

        with_disabled = (await (await cli.get("/v1/skills?include_disabled=true", headers=headers)).json())["data"]
        row = next(r for r in with_disabled if r["name"] == "release-check")
        assert row["enabled"] is False
        assert row["tags"] == ["release", "ci"]

        # Essential skill cannot be disabled -> 403
        ess = await cli.put("/v1/skills/hermes-agent/enabled", headers=headers, json={"enabled": False})
        assert ess.status == 403

        # Builtin skill cannot be modified or deleted -> 403
        assert (await cli.delete("/v1/skills/builtin-demo", headers=headers)).status == 403
        assert (await cli.put("/v1/skills/builtin-demo", headers=headers, json={"content": _VALID_SKILL_MD})).status == 403

        # User skill can be deleted -> 200, then 404
        deleted = await cli.delete("/v1/skills/release-check", headers=headers)
        assert deleted.status == 200
        assert (await deleted.json())["deleted"] is True
        assert (await cli.get("/v1/skills/release-check", headers=headers)).status == 404


@pytest.mark.asyncio
async def test_skill_studio_cross_profile_isolation(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    beta = root / "profiles" / "beta"
    beta.mkdir(parents=True)
    (root / "config.yaml").write_text("{}\n", encoding="utf-8")
    (beta / "config.yaml").write_text("{}\n", encoding="utf-8")
    (beta / ".env").write_text(f"API_SERVER_KEY={BETA_KEY}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.chdir(tmp_path)

    adapter = _make_adapter(OWNER_KEY, multiplex=True)
    owner_h = {"Authorization": f"Bearer {OWNER_KEY}"}
    beta_h = {"Authorization": f"Bearer {BETA_KEY}"}

    async with TestClient(TestServer(_make_app(adapter))) as cli:
        r = await cli.post("/p/beta/v1/skills", headers=beta_h, json={"name": "beta-only", "content": _VALID_SKILL_MD.replace("release-check", "beta-only")})
        assert r.status == 201
        assert (await cli.get("/v1/skills/beta-only", headers=owner_h)).status == 404
        assert (await cli.delete("/v1/skills/beta-only", headers=owner_h)).status == 404
        assert (await cli.get("/p/beta/v1/skills/beta-only", headers=beta_h)).status == 200
