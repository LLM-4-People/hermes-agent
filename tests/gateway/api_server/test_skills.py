"""Authenticated discovery of the profile's usable skill catalog."""

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_server import _create_app, _make_adapter


@pytest.mark.asyncio
async def test_skills_returns_enabled_profile_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_PLATFORM", "telegram")
    (tmp_path / "config.yaml").write_text(
        "skills:\n  disabled: [disabled-skill]\n  platform_disabled:\n    api_server: [cli-only]\n    telegram: [notes]\n",
        encoding="utf-8",
    )
    for path, name in (
        ("uncategorized", "first"),
        ("writing/notes", "notes"),
        ("editing/shared", "shared"),
        ("writing/shared", "shared"),
        ("disabled-skill", "disabled-skill"),
        ("cli-only", "cli-only"),
    ):
        directory = tmp_path / "skills" / path
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Description for {name}\n---\nPrivate instructions.",
            encoding="utf-8",
        )
    async with TestClient(TestServer(_create_app(_make_adapter(api_key="sk-secret")))) as cli:
        for headers in ({}, {"Authorization": "Bearer invalid"}):
            assert (await cli.get("/v1/skills", headers=headers)).status == 401
        resp = await cli.get("/v1/skills", headers={"Authorization": "Bearer sk-secret"})
        assert resp.status == 200
        assert await resp.json() == {"object": "list", "data": [
            {"name": "first", "description": "Description for first", "category": None},
            {"name": "editing/shared", "description": "Description for shared", "category": "editing"},
            {"name": "notes", "description": "Description for notes", "category": "writing"},
            {"name": "writing/shared", "description": "Description for shared", "category": "writing"},
        ]}
