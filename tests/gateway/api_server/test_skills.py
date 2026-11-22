"""Authenticated discovery of the profile's usable skill catalog."""

import json

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


@pytest.mark.asyncio
async def test_bad_frontmatter_does_not_hide_healthy_skills(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    for folder, metadata in {
        "good": "description: Text description",
        "list": "description: [one, two]",
        "map": "description: {one: two}",
        "set": "description: !!set {one: null}",
        "invalid-name": "name: [one, two]",
    }.items():
        skill = tmp_path / "skills" / folder
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(f"---\n{metadata}\n---\nFallback description.", encoding="utf-8")

    async with TestClient(TestServer(_create_app(_make_adapter(api_key="sk-secret")))) as cli:
        resp = await cli.get("/v1/skills", headers={"Authorization": "Bearer sk-secret"})
        assert resp.status == 200
        entries = (await resp.json())["data"]
    assert {row["name"]: row["description"] for row in entries} == {
        "good": "Text description", "list": "Fallback description.",
        "map": "Fallback description.", "set": "Fallback description.",
    }


@pytest.mark.asyncio
async def test_advertised_locators_load_the_intended_package(tmp_path, monkeypatch):
    from tools.skills_tool import skill_view

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "skills"
    long_name = "long-" + "a" * 64
    for path, declared in (("writing/brief", "brief"), ("writing/long-folder", long_name)):
        skill = root / path
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            f"---\nname: {declared}\ndescription: An installed skill\n---\nInstructions for {path}.",
            encoding="utf-8",
        )
    archive = root / "archive"
    archive.mkdir()
    (archive / "brief.md").write_text("A separate legacy skill.", encoding="utf-8")
    async with TestClient(TestServer(_create_app(_make_adapter(api_key="sk-secret")))) as cli:
        resp = await cli.get("/v1/skills", headers={"Authorization": "Bearer sk-secret"})
        assert resp.status == 200
        locators = [row["name"] for row in (await resp.json())["data"]]
    assert set(locators) == {"writing/brief", long_name}
    for name, path in (("writing/brief", "writing/brief"), (long_name, "writing/long-folder")):
        assert f"Instructions for {path}." in json.loads(skill_view(name, preprocess=False))["content"]
    assert not json.loads(skill_view("brief", preprocess=False))["success"]
