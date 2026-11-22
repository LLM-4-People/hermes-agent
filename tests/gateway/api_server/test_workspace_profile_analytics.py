"""Deterministic positive, negative, and cross-profile isolation tests for:
Workspace/Git Explorer, Profile Defaults & SOUL.md, Usage Analytics, Cron Runs,
Session Multi-File Uploads + Run Attachments, and SSoT Capabilities Advertisement.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import types
import pytest
from aiohttp import FormData, web
from ruamel.yaml import YAML
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    body_limit_middleware,
    security_headers_middleware,
)
from gateway.platforms.api_server_workspace import apply_run_attachments, resolve_session_attachments
from gateway.session_authorities import SessionAuthorities
from hermes_constants import WEBAPP_ATTACHMENT_MAX_BYTES
from persistence.conversation.sessions import NewSession
from persistence.conversation.usage import TokenCounts, UsageDelta
from persistence_assembly import open_persistence, profile_scope

OWNER_KEY = "owner-key-1234567890abcdef"
BETA_KEY = "beta-key-1234567890abcdef"


def _make_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(
        middlewares=[
            adapter._make_profile_prefix_middleware(),
            body_limit_middleware,
            security_headers_middleware,
        ]
    )
    app["api_server_adapter"] = adapter
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
        app.router.add_route(method, f"/p/{{profile}}{path}", handler)
    return app


def _make_adapter(
    key: str = OWNER_KEY,
    *,
    multiplex: bool = False,
    profile_homes: tuple[Path, ...] = (),
) -> APIServerAdapter:
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": key}))
    if multiplex:
        assert profile_homes
        authorities = SessionAuthorities(profile_homes[0], multiplexed=True)
        for home in profile_homes:
            authorities.add(home, types.SimpleNamespace(persistence=open_persistence(profile_scope(home=home))))

        class _Runner:
            config = GatewayConfig(multiplex_profiles=True)
            session_authorities = authorities

        adapter.gateway_runner = _Runner()
    return adapter


def _init_git_repo(repo: Path) -> None:
    def _git(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=str(repo),
            check=True,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=15,
        )

    _git("init")
    _git("config", "user.email", "test@example.com")
    _git("config", "user.name", "Test User")
    (repo / "src").mkdir(parents=True, exist_ok=True)
    (repo / "src" / "main.py").write_text("print('v1')\n", encoding="utf-8")
    (repo / ".env").write_text("API_KEY=super-secret-value\n", encoding="utf-8")
    _git("add", "src/main.py")
    _git("commit", "-m", "initial commit")
    # Modify tracked file and also add secret file change
    (repo / "src" / "main.py").write_text("print('v1')\nprint('v2')\n", encoding="utf-8")


@pytest.mark.asyncio
async def test_workspace_explorer_jail_symlinks_secrets_and_git_diff(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    outside = tmp_path / "outside_secret.txt"
    outside.write_text("OUTSIDE_HOST_DATA\n", encoding="utf-8")

    workspace = tmp_path / "project"
    workspace.mkdir()
    _init_git_repo(workspace)
    (workspace / "binary.bin").write_bytes(b"\x00\x01\x02\xffbinary")
    (workspace / "escape_link.txt").symlink_to(outside)
    (workspace / "env_alias.txt").symlink_to(workspace / ".env")
    (workspace / "git_cfg_link.txt").symlink_to(workspace / ".git" / "config")

    adapter = _make_adapter()
    store = adapter._store_beside(adapter._ensure_store())
    store.sessions.create(NewSession(session_id="ws-sess", source="api_server", model="gpt-5", cwd=str(workspace)))
    store.sessions.create(NewSession(session_id="no-cwd-sess", source="api_server", model="gpt-5"))

    headers = {"Authorization": f"Bearer {OWNER_KEY}"}
    async with TestClient(TestServer(_make_app(adapter))) as cli:
        # Tree lists safe files, hides .git, .env, escaping symlink, and secret symlink
        tree_resp = await cli.get("/api/sessions/ws-sess/workspace/tree?depth=2", headers=headers)
        assert tree_resp.status == 200
        assert tree_resp.headers.get("Cache-Control") == "no-store"
        tree = await tree_resp.json()
        paths = {e["path"] for e in tree["entries"]}
        assert "src/main.py" in paths
        assert ".env" not in paths
        assert ".git" not in paths
        assert "escape_link.txt" not in paths
        assert "env_alias.txt" not in paths
        assert "git_cfg_link.txt" not in paths

        # Safe text file read with line slicing
        file_resp = await cli.get("/api/sessions/ws-sess/workspace/file?path=src/main.py&start_line=2&end_line=2", headers=headers)
        assert file_resp.status == 200
        fdata = await file_resp.json()
        assert fdata["language"] == "python"
        assert fdata["lines"] == ["print('v2')"]
        assert len(fdata["sha256"]) == 64

        # Binary rejected with 415
        assert (await cli.get("/api/sessions/ws-sess/workspace/file?path=binary.bin", headers=headers)).status == 415

        # Security negatives: traversal (400), absolute path (400), symlink escape (403), secret paths (403)
        assert (await cli.get("/api/sessions/ws-sess/workspace/file?path=../outside_secret.txt", headers=headers)).status == 400
        assert (await cli.get(f"/api/sessions/ws-sess/workspace/file?path={outside}", headers=headers)).status == 400
        assert (await cli.get("/api/sessions/ws-sess/workspace/file?path=escape_link.txt", headers=headers)).status == 403
        assert (await cli.get("/api/sessions/ws-sess/workspace/file?path=.env", headers=headers)).status == 403
        assert (await cli.get("/api/sessions/ws-sess/workspace/file?path=.git/config", headers=headers)).status == 403
        assert (await cli.get("/api/sessions/ws-sess/workspace/file?path=env_alias.txt", headers=headers)).status == 403
        assert (await cli.get("/api/sessions/ws-sess/workspace/file?path=git_cfg_link.txt", headers=headers)).status == 403

        # Session without CWD fails closed (404)
        assert (await cli.get("/api/sessions/no-cwd-sess/workspace/tree", headers=headers)).status == 404

        # Git diff exposes safe tracked changes and never leaks .env
        diff_resp = await cli.get("/api/sessions/ws-sess/workspace/diff", headers=headers)
        assert diff_resp.status == 200
        diff_data = await diff_resp.json()
        assert diff_data["is_git_repo"] is True
        assert any(f["path"] == "src/main.py" for f in diff_data["files"])
        assert all(f["path"] != ".env" for f in diff_data["files"])
        assert "print('v2')" in diff_data["diff"]
        assert "super-secret-value" not in diff_data["diff"]
        # Direct diff on .env blocked
        assert (await cli.get("/api/sessions/ws-sess/workspace/diff?path=.env", headers=headers)).status == 403


@pytest.mark.asyncio
async def test_profile_defaults_soul_capabilities_and_secret_safety(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "# Keep this comment\nmodel:\n  default: gpt-5\n  provider: openai\n",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-never-leak-this\n", encoding="utf-8")
    adapter = _make_adapter()
    headers = {"Authorization": f"Bearer {OWNER_KEY}"}

    async with TestClient(TestServer(_make_app(adapter))) as cli:
        # Capabilities advertise all new features, endpoints, and SSoT max_artifact_bytes
        caps = await (await cli.get("/v1/capabilities", headers=headers)).json()
        assert caps["max_artifact_bytes"] == WEBAPP_ATTACHMENT_MAX_BYTES
        assert caps["limits"]["max_artifact_bytes"] == WEBAPP_ATTACHMENT_MAX_BYTES
        for flag in (
            "memory_api", "memory_write_api", "session_search", "session_fork_at_turn",
            "session_export", "session_uploads", "skill_studio", "workspace_browser",
            "usage_analytics", "profile_defaults", "cron_run_history",
        ):
            assert caps["features"][flag] is True, flag

        # GET /api/profile returns summary + max_artifact_bytes and zero secrets
        prof_resp = await cli.get("/api/profile", headers=headers)
        assert prof_resp.status == 200
        prof_text = await prof_resp.text()
        assert "sk-never-leak-this" not in prof_text
        prof = await prof_resp.json()
        assert prof["max_artifact_bytes"] == WEBAPP_ATTACHMENT_MAX_BYTES
        assert prof["default_model"] == "gpt-5"

        # PATCH /api/profile/defaults allowlisted update preserves comments
        patch = await cli.patch(
            "/api/profile/defaults",
            headers=headers,
            json={
                "model": "claude-sonnet-4",
                "provider": "anthropic",
                "reasoning_effort": "high",
                "enabled_toolsets": ["web", "file"],
                "preferred_language": "fr",
            },
        )
        assert patch.status == 200
        updated = await patch.json()
        assert updated["default_model"] == "claude-sonnet-4"
        assert updated["default_provider"] == "anthropic"
        assert updated["reasoning_effort"] == "high"
        assert updated["preferred_language"] == "fr"
        assert updated["enabled_toolsets"] == ["file", "web"]
        safe_yaml = YAML(typ="safe")
        cfg_fr = safe_yaml.load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
        assert cfg_fr["display"]["language"] == "fr"
        assert cfg_fr["display"]["preferred_language"] == "fr"
        assert "# Keep this comment" in (tmp_path / "config.yaml").read_text(encoding="utf-8")

        # Reset preferred_language back to auto -> clears stale display.language on disk & keeps comment
        auto_patch = await cli.patch("/api/profile/defaults", headers=headers, json={"preferred_language": "auto"})
        assert auto_patch.status == 200 and (await auto_patch.json())["preferred_language"] == "auto"
        cfg_auto = safe_yaml.load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
        assert cfg_auto["display"]["preferred_language"] == "auto"
        assert "language" not in cfg_auto["display"]
        assert "# Keep this comment" in (tmp_path / "config.yaml").read_text(encoding="utf-8")
        assert (await cli.patch("/api/profile/defaults", headers=headers, json={"preferred_language": "de"})).status == 400

        # Negative: disallowed keys / invalid effort / unknown toolset
        assert (await cli.patch("/api/profile/defaults", headers=headers, json={"api_key": "sk-x"})).status == 400
        assert (await cli.patch("/api/profile/defaults", headers=headers, json={"env": {"A": "B"}})).status == 400
        assert (await cli.patch("/api/profile/defaults", headers=headers, json={"reasoning_effort": "turbo"})).status == 400
        assert (await cli.patch("/api/profile/defaults", headers=headers, json={"enabled_toolsets": ["nonexistent_ts"]})).status == 400

        # SOUL.md GET/PUT + size cap & symlink guard
        soul_file = tmp_path / "SOUL.md"
        soul_file.unlink(missing_ok=True)
        s0 = await (await cli.get("/api/profile/soul", headers=headers)).json()
        assert s0["exists"] is False
        put_ok = await cli.put("/api/profile/soul", headers=headers, json={"content": "You are calm and concise."})
        assert put_ok.status == 200
        s1 = await (await cli.get("/api/profile/soul", headers=headers)).json()
        assert s1["exists"] is True and s1["content"] == "You are calm and concise."
        assert (await cli.put("/api/profile/soul", headers=headers, json={"content": "a" * 70_000})).status == 413

        # Escaping symlink on SOUL.md rejected (403)
        outside_soul = tmp_path.parent / "outside_soul.md"
        outside_soul.write_text("outside", encoding="utf-8")
        soul_file.unlink()
        soul_file.symlink_to(outside_soul)
        assert (await cli.get("/api/profile/soul", headers=headers)).status == 403
        assert (await cli.put("/api/profile/soul", headers=headers, json={"content": "overwrite"})).status == 403


@pytest.mark.asyncio
async def test_analytics_cron_runs_and_session_uploads_isolation(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    beta = root / "profiles" / "beta"
    beta.mkdir(parents=True)
    (root / "config.yaml").write_text("{}\n", encoding="utf-8")
    (beta / "config.yaml").write_text("{}\n", encoding="utf-8")
    (beta / ".env").write_text(f"API_SERVER_KEY={BETA_KEY}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    adapter = _make_adapter(OWNER_KEY, multiplex=True, profile_homes=(root, beta))
    owner_h = {"Authorization": f"Bearer {OWNER_KEY}"}
    beta_h = {"Authorization": f"Bearer {BETA_KEY}"}

    # Seed session + usage in default profile
    with adapter._profile_scope(None):
        store = adapter._store_beside(adapter._ensure_store())
        store.sessions.create(NewSession(session_id="u-sess", source="api_server", model="gpt-5"))
        store.sessions.create(NewSession(session_id="other-sess", source="api_server", model="gpt-5"))
        store.usage.queue_usage(
            "u-sess",
            UsageDelta(tokens=TokenCounts(input_tokens=1000, output_tokens=250)),
        )
        store.usage.flush_usage()

        from cron.jobs import create_job
        job = create_job(prompt="ping", schedule="every 1h", name="hourly-ping")
        out_dir = root / "cron" / "output" / job["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "2026-11-22_12-00-00.md").write_text("Hourly ping succeeded cleanly.\n", encoding="utf-8")

    async with TestClient(TestServer(_make_app(adapter))) as cli:
        # Analytics usage (default profile sees tokens; beta sees 0)
        an = await cli.get("/api/analytics/usage?days=7", headers=owner_h)
        assert an.status == 200
        an_data = await an.json()
        assert an_data["totals"]["total_input"] >= 1000
        assert (await cli.get("/api/analytics/usage?days=0", headers=owner_h)).status == 400
        assert (await cli.get("/api/analytics/usage?session_id=missing", headers=owner_h)).status == 404

        beta_an = await (await cli.get("/p/beta/api/analytics/usage?days=7", headers=beta_h)).json()
        assert (beta_an["totals"].get("total_input") or 0) == 0

        # Cron runs: default sees output run; beta gets 404 (isolated!)
        runs_resp = await cli.get(f"/api/jobs/{job['id']}/runs", headers=owner_h)
        assert runs_resp.status == 200
        runs_data = await runs_resp.json()
        assert len(runs_data["runs"]) == 1
        assert "Hourly ping succeeded" in (runs_data["runs"][0]["preview"] or "")
        assert (await cli.get(f"/p/beta/api/jobs/{job['id']}/runs", headers=beta_h)).status == 404

        # Multi-file session upload (PDF magic bytes)
        form = FormData()
        form.add_field("file", b"%PDF-1.7\n%mock pdf bytes", filename="report.pdf", content_type="application/octet-stream")
        up_resp = await cli.post("/api/sessions/u-sess/uploads", headers=owner_h, data=form)
        assert up_resp.status == 201
        up = await up_resp.json()
        assert up["mime_type"] == "application/pdf"
        assert up["max_artifact_bytes"] == WEBAPP_ATTACHMENT_MAX_BYTES
        upload_id = up["upload_id"]

        # Attachment resolution succeeds for owning session, fails for another session
        with adapter._profile_scope(None):
            resolved, err = resolve_session_attachments("u-sess", [upload_id])
            assert err is None and len(resolved) == 1 and resolved[0]["mime_type"] == "application/pdf"
            _, wrong_err = resolve_session_attachments("other-sess", [upload_id])
            assert wrong_err is not None and wrong_err[0] in (403, 404)
            _, bad_id_err = resolve_session_attachments("u-sess", ["../etc/passwd"])
            assert bad_id_err is not None and bad_id_err[0] == 400

        # Empty upload rejected (400), non-multipart rejected (400)
        empty_form = FormData()
        empty_form.add_field("file", b"", filename="empty.txt")
        assert (await cli.post("/api/sessions/u-sess/uploads", headers=owner_h, data=empty_form)).status == 400
        assert (await cli.post("/api/sessions/u-sess/uploads", headers=owner_h, json={"file": "x"})).status == 400

        # POST /v1/runs rejects unknown or foreign attachment IDs cleanly before admitting run
        bad_run = await cli.post(
            "/v1/runs",
            headers=owner_h,
            json={"input": "read this", "session_id": "other-sess", "attachments": [upload_id]},
        )
        assert bad_run.status in (403, 404)


@pytest.mark.asyncio
async def test_upload_compaction_lineage_multimodal_and_abort_cleanup(tmp_path, monkeypatch):
    """Regression suite for #91 findings: multimodal attachment preservation,
    compacted-session root/tip lineage alignment, stream abort cleanup, and sanitized 500s."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    store = adapter._store_beside(adapter._ensure_store())
    # Create root session and a compression continuation child (tip)
    store.sessions.create(NewSession(session_id="thread-root", source="api_server", model="gpt-5"))
    store.sessions.end("thread-root", "compression")
    store.sessions.create(
        NewSession(
            session_id="thread-tip",
            source="api_server",
            model="gpt-5",
            parent_session_id="thread-root",
        )
    )
    headers = {"Authorization": f"Bearer {OWNER_KEY}"}
    async with TestClient(TestServer(_make_app(adapter))) as cli:
        # Upload addressed to root_id; run addressed to compacted tip_id MUST resolve seamlessly
        form = FormData()
        form.add_field("file", b"def hello():\n    return 42\n", filename="app.py")
        up_resp = await cli.post("/api/sessions/thread-root/uploads", headers=headers, data=form)
        assert up_resp.status == 201
        upload = await up_resp.json()
        assert upload["thread_id"] == "thread-root"
        upload_id = upload["upload_id"]

        # Multimodal list input preserves image_url block + appends structured text block
        multimodal_input = [
            {"type": "text", "text": "Inspect the attached code and image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
        ]
        updated_msg, att_err = await apply_run_attachments(
            adapter,
            "thread-tip",
            {"attachments": [upload_id]},
            multimodal_input,
            lambda m, code=None: {"error": {"message": m, "code": code}},
        )
        assert att_err is None
        assert isinstance(updated_msg, list) and len(updated_msg) == 3
        assert updated_msg[0] == multimodal_input[0]
        assert updated_msg[1] == multimodal_input[1]
        assert updated_msg[2]["type"] == "text" and upload_id in updated_msg[2]["text"]

        # Stream abort (non-OSError) leaves zero leaked partial files in uploads/
        uploads_dir = tmp_path / "uploads"
        before_files = set(uploads_dir.rglob("*"))

        class _ExplodingPart:
            filename = "broken.bin"
            name = "file"
            headers = {}

            async def read_chunk(self, _n):
                raise ValueError("simulated client disconnect mid-stream")

        class _FakeReader:
            def __init__(self):
                self._yielded = False

            async def next(self):
                if not self._yielded:
                    self._yielded = True
                    return _ExplodingPart()
                return None

        from gateway.platforms.api_server_workspace import handle_session_upload

        class _FakeReq:
            match_info = {"session_id": "thread-root"}
            content_type = "multipart/form-data"
            headers = {"Authorization": f"Bearer {OWNER_KEY}"}

            async def multipart(self):
                return _FakeReader()

        with pytest.raises(ValueError, match="simulated client disconnect"):
            await handle_session_upload(adapter, _FakeReq())
        assert set(uploads_dir.rglob("*")) == before_files
