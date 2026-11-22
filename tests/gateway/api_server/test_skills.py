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
    from tools.skills_tool import skill_view

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    for folder, metadata in {
        "good": "description: Text description",
        "list": "description: [one, two]",
        "map": "description: {one: two}",
        "set": "description: !!set {one: null}",
        "dated": "description: 2025-11-14",
        "nested": "description: {updated: 2025-11-14}",
        "optional": (
            "description: Typed fields\ncompatibility: 2025-11-14\n"
            "metadata: {last_updated: 2025-11-14, hermes: [wrong-type], channel: {status: ready}}"
        ),
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
        "map": "Fallback description.", "set": "Fallback description.", "dated": "Fallback description.",
        "nested": "Fallback description.", "optional": "Typed fields",
    }
    for row in entries:
        view = json.loads(skill_view(row["name"], preprocess=False))
        assert view["success"] and view["description"] == row["description"]
        if row["name"] == "optional":
            assert view["metadata"] == {"channel": {"status": "ready"}}
            assert "compatibility" not in view


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


@pytest.mark.asyncio
async def test_legacy_lookup_treats_locators_as_literal_names(tmp_path, monkeypatch):
    from tools.skills_tool import skill_view

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "skills"
    cases = {"plan.v2": "plan.md", "note[1]": "note1.md", "guide.md": "guide.md"}
    for name, legacy in cases.items():
        directory = root / "writing" / name
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(f"---\nname: {name}\n---\nPackage {name}.", encoding="utf-8")
        (root / legacy).write_text(f"Legacy {legacy}.", encoding="utf-8")
    async with TestClient(TestServer(_create_app(_make_adapter(api_key="sk-secret")))) as cli:
        resp = await cli.get("/v1/skills", headers={"Authorization": "Bearer sk-secret"})
        assert resp.status == 200
        locators = {row["name"] for row in (await resp.json())["data"]}
    assert locators == {"plan.v2", "note[1]", "writing/guide.md"}
    for name in locators:
        assert f"Package {name.split('/')[-1]}." in json.loads(skill_view(name, preprocess=False))["content"]
    for name in ("plan", "plan.md", "note1", "note1.md"):
        assert "Legacy " in json.loads(skill_view(name, preprocess=False))["content"]


@pytest.mark.asyncio
async def test_catalog_reloads_when_root_roles_change(tmp_path, monkeypatch):
    from tools.skills_tool import skill_view

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    for directory in ("first/north/one", "second/south/two"):
        skill = tmp_path / directory
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(f"---\nname: shared\n---\nSource {directory}", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("skills:\n  external_dirs: [first, second]\n", encoding="utf-8")
    async with TestClient(TestServer(_create_app(_make_adapter(api_key="sk-secret")))) as cli:
        async def names():
            resp = await cli.get("/v1/skills", headers={"Authorization": "Bearer sk-secret"})
            assert resp.status == 200
            return {row["name"] for row in (await resp.json())["data"]}

        assert await names() == {"north/one", "south/two"}
        config.write_text("skills:\n  create_dir: first\n  external_dirs: [second]\n", encoding="utf-8")
        assert await names() == {"shared"}
        assert "Source first/north/one" in json.loads(skill_view("shared", preprocess=False))["content"]


@pytest.mark.asyncio
async def test_markdown_directories_and_folded_names_load_as_packages(tmp_path, monkeypatch):
    from tools.skills_tool import skill_view

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    for folder, definition in (("notes/release", "release"), ("notes/release.md", "release-note"),
                               ("routine", ">\n  review")):
        skill = tmp_path / "skills" / folder
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(f"---\nname: {definition}\n---\nPackage {folder}.", encoding="utf-8")
    async with TestClient(TestServer(_create_app(_make_adapter(api_key="sk-secret")))) as cli:
        resp = await cli.get("/v1/skills", headers={"Authorization": "Bearer sk-secret"})
        assert resp.status == 200
        locators = {row["name"] for row in (await resp.json())["data"]}
    expected = {"release": "notes/release", "release-note": "notes/release.md", "review": "routine"}
    assert locators == expected.keys()
    for name, folder in expected.items():
        assert f"Package {folder}." in json.loads(skill_view(name, preprocess=False))["content"]
    assert "Package routine." in json.loads(skill_view("review\n", preprocess=False))["content"]


@pytest.mark.asyncio
async def test_catalog_reads_the_complete_frontmatter(tmp_path, monkeypatch):
    from tools.skills_tool import skill_view

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("skills:\n  disabled: [private-journal]\n", encoding="utf-8")
    credits = "Shared with permission. " * 200
    for folder, declared in (("routine", "follow-up"), ("private", "private-journal")):
        skill = tmp_path / "skills" / folder
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            f"---\nname: {declared}\ndescription: A practice journal\nmetadata:\n  credits: {credits}\n---\nNotes.",
            encoding="utf-8",
        )
    async with TestClient(TestServer(_create_app(_make_adapter(api_key="sk-secret")))) as cli:
        resp = await cli.get("/v1/skills", headers={"Authorization": "Bearer sk-secret"})
        assert resp.status == 200
        assert (await resp.json())["data"] == [{"name": "follow-up", "description": "A practice journal", "category": None}]
    view = json.loads(skill_view("follow-up", preprocess=False))
    assert view["success"] and view["description"] == "A practice journal"


@pytest.mark.asyncio
async def test_catalog_publishes_only_single_line_loadable_identifiers(tmp_path, monkeypatch):
    from tools.skills_tool import skill_view

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    cases = {
        "notes/multiline": "|\n  review\n  notes",
        "notes/control": r'"draft\x7F\x80end"',
        "notes/line": r'"draft\u2028line"',
        "notes/paragraph": r'"draft\u2029paragraph"',
        "notes/reports": '"R: Report"',
        "notes/research": '"R&D: overview"',
        "notes/scoped": '"research:report"',
        "notes/generated": json.dumps("R📌"),
        "notes/incomplete": r'"draft\uD83D"',
        "notes/hide\nthis": r'"bad\x01name"',
        "notes/保留": "Résumé 📌",
    }
    for folder, definition in cases.items():
        skill = tmp_path / "skills" / folder
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(f"---\nname: {definition}\n---\nPackage {folder}.", encoding="utf-8")
    async with TestClient(TestServer(_create_app(_make_adapter(api_key="sk-secret")))) as cli:
        resp = await cli.get("/v1/skills", headers={"Authorization": "Bearer sk-secret"})
        assert resp.status == 200
        locators = {row["name"] for row in (await resp.json())["data"]}
    expected = {folder: folder for folder in cases if folder not in {"notes/hide\nthis", "notes/保留"}}
    expected["Résumé 📌"] = "notes/保留"
    assert locators == expected.keys()
    for name, folder in expected.items():
        assert f"Package {folder}." in json.loads(skill_view(name, preprocess=False))["content"]
