"""Regression tests for Round 1 Consolidated PR: app_presence wiring, i18n middleware,
capabilities flags, run-state concurrency/leak fixes, lineage total visibility, and fork guard SSoT."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import threading
import types

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from agent.mobile_companion_prompt import combine_run_instructions, normalize_user_language
from gateway.platforms import api_server_runs as _api_runs
from gateway.platforms.api_server_ui import (
    MOBILE_FEATURE_FLAGS,
    language_scope_middleware,
    optional_user_lang_scope,
)
from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import NewMessage, TranscriptRead, TranscriptView
from persistence.core.identity import SessionId
from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.transcript_pages import get_transcript_page_cursor
from persistence_assembly import store_location
from scripts.check_fork_preservation import check_preservation
from tests.fixtures.storage_seed import seed_session
from tools.mobile_app_presence import ALLOWED_SCREENS, sanitize_app_presence
from tools.mobile_device_bridge import DEEP_ANDROID_ACTIONS


def test_allowed_screens_include_android_workspace_routes_and_prompt_wiring():
    for route in (
        "discussion",
        "home",
        "brain",
        "activity",
        "presence_control",
        "workflow_templates",
        "connection_doctor",
    ):
        assert route in ALLOWED_SCREENS
        clean = sanitize_app_presence({"current_screen": route, "active_run_count": 2})
        assert clean is not None and clean["current_screen"] == route

    # Deep Android 16 capabilities and app_presence flow into combine_run_instructions.
    combined = combine_run_instructions(
        "Keep answers concise.",
        list(DEEP_ANDROID_ACTIONS),
        user_language="fr",
        app_presence={
            "current_screen": "presence_control",
            "active_run_count": 1,
            "continuous_location": {"opted_in": False, "lat": 45.5, "lon": -73.5},
        },
    )
    assert combined is not None
    assert "<mobile_app_presence>" in combined
    assert "screen=presence_control" in combined
    assert "45.5" not in combined  # fail-closed when opted_in is False
    for action in DEEP_ANDROID_ACTIONS:
        assert action in combined

    # X-Hermes-Lang fallback works cleanly when body user_language is omitted.
    lang, err = normalize_user_language(None, accept_language="en-US", x_hermes_lang="fr-CA")
    assert err is None and lang == "fr"


def test_mobile_feature_flags_advertise_all_phase2_capabilities():
    for flag in (
        "app_presence_v2",
        "interactive_cards",
        "deep_android_bridge",
        "cursor_pagination",
        "bilingual_i18n",
    ):
        assert MOBILE_FEATURE_FLAGS.get(flag) is True


def test_run_user_message_validates_app_presence_fail_closed():
    def _err(msg, code="invalid_request_error", **_kw):
        return {"error": {"message": msg, "code": code}}

    # Valid payload accepted
    msg, err_resp = _api_runs._run_user_message(
        {"input": "Bonjour", "app_presence": {"current_screen": "discussion", "draft_chars": 12}},
        _err,
    )
    assert msg == "Bonjour"
    assert err_resp is None

    # Invalid payload type rejected with 400 invalid_app_presence
    msg_bad, err_bad = _api_runs._run_user_message(
        {"input": "Bonjour", "app_presence": ["not-an-object"]},
        _err,
    )
    assert msg_bad is None
    assert err_bad is not None and err_bad.status == 400
    assert b"invalid_app_presence" in err_bad.body


@pytest.mark.asyncio
async def test_language_scope_middleware_binds_gateway_i18n_when_present(monkeypatch):
    observed: list[str] = []
    fake_i18n = types.ModuleType("gateway.i18n")

    def _resolve_request_lang(headers, *, explicit_lang=None):
        if explicit_lang:
            return explicit_lang
        if headers and str(headers.get("X-Hermes-Lang", "")).lower().startswith("fr"):
            return "fr"
        return "en"

    @contextmanager
    def _user_lang_scope(lang):
        observed.append(lang)
        yield lang

    fake_i18n.resolve_request_lang = _resolve_request_lang  # type: ignore[attr-defined]
    fake_i18n.user_lang_scope = _user_lang_scope  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gateway.i18n", fake_i18n)

    with optional_user_lang_scope({"X-Hermes-Lang": "fr"}) as active:
        assert active == "fr"

    app = web.Application(middlewares=[language_scope_middleware])

    async def _handler(_request):
        return web.json_response({"ok": True})

    app.router.add_get("/ping", _handler)
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/ping", headers={"X-Hermes-Lang": "fr-CA"})
        assert resp.status == 200
    assert observed == ["fr", "fr"]


def test_make_run_event_callback_holds_run_state_lock_across_status_update():
    """Auditor 2 #2: _push() must hold _run_state_lock continuously across status read and _set_run_status."""

    class _TrackingRLock:
        def __init__(self) -> None:
            self._inner = threading.RLock()
            self.depth = 0

        def __enter__(self):
            self._inner.acquire()
            self.depth += 1
            return self

        def __exit__(self, *args):
            self.depth -= 1
            self._inner.release()

    lock = _TrackingRLock()
    depth_during_set: list[int] = []

    class _FakeAdapter:
        def __init__(self) -> None:
            self._run_state_lock = lock
            self._run_statuses = {"run_1": {"status": "running"}}
            self._run_streams = {}

        def _set_run_status(self, run_id: str, status: str, **fields):
            depth_during_set.append(lock.depth)
            self._run_statuses[run_id] = {"status": status, **fields}
            return self._run_statuses[run_id]

    fake_api_mod = types.SimpleNamespace(
        redact_sensitive_text=lambda s, **_kw: s,
        _api_request_profile=types.SimpleNamespace(get=lambda: "default"),
    )
    adapter = _FakeAdapter()
    cb = _api_runs._make_run_event_callback(adapter, "run_1", asyncio.new_event_loop(), _api_server=fake_api_mod)
    cb("tool.started", "terminal", "ls", {"command": "ls"})
    assert depth_during_set == [1]


@pytest.mark.asyncio
async def test_handle_runs_does_not_leak_owner_or_profile_on_rejected_attachment(api):
    """Auditor 2 #3: rejected attachments must not leave orphaned entries in _run_owners or _run_profiles."""
    api._api_key = "leak-test-key"
    api._run_owners.clear()
    api._run_profiles.clear()
    app = web.Application()
    app.router.add_post("/v1/runs", api._handle_runs)

    async with TestClient(TestServer(app)) as client:
        resp = await client.post(
            "/v1/runs",
            headers={"Authorization": "Bearer leak-test-key"},
            json={
                "input": "hello",
                "attachments": [{"artifact_id": "../escape-attempt"}],
            },
        )
        assert resp.status in (400, 404)
    assert api._run_owners == {}
    assert api._run_profiles == {}


def test_lineage_fast_keyset_total_matches_winning_copy_visibility(owner):
    """Auditor 3 #6: if a newer winning copy is display_kind='hidden', total must not over-count."""
    store = owner.persistence
    seg0 = seed_session(store, "vis-seg-0", "api_server")
    store.messages.append_batch(
        seg0,
        [
            NewMessage(role="user", content="u0", timestamp=1_700_000_001.0),
            NewMessage(role="assistant", content="a0", timestamp=1_700_000_002.0),
            NewMessage(role="user", content="carried-hidden", timestamp=1_700_000_003.0),
        ],
    )
    seg1 = "vis-seg-1"
    store.lineage.publish_compression_child(
        SessionId(seg0),
        CompressionChild(session_id=SessionId(seg1), source="api_server"),
        [
            NewMessage(role="assistant", content="[CONTEXT COMPACTION SUMMARY]", compressed_summary=True),
            NewMessage(role="user", content="carried-hidden", display_kind="hidden"),
        ],
        require_compression_lease=False,
    )
    store.messages.append_batch(
        seg1,
        [NewMessage(role="assistant", content="a1", timestamp=1_700_000_005.0)],
    )

    req = TranscriptRead(view=TranscriptView.DISPLAY, limit=20, offset=0, latest=True, include_lineage=True)
    with StoreHolder(store_location(store.scope), read_only=True) as reader:
        rows, total, _root, has_more = get_transcript_page_cursor(reader, seg1, req, True)
    assert has_more is False
    visible_contents = [r["content"] for r in rows]
    assert "carried-hidden" not in visible_contents
    assert total == len(rows) == 3  # u0, a0, a1 (hidden winning copy excluded from both rows and total)


def test_fork_preservation_dynamically_protects_workflows_from_manifest(tmp_path: Path):
    """Auditor 7 #8: scripts/fork-owned-workflows.txt is the SSoT for protected workflows + check_writing_quality.py."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def _git(*args: str) -> str:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
        return proc.stdout.strip()

    _git("init", "-b", "main")
    _git("config", "user.email", "test@example.com")
    _git("config", "user.name", "Test")
    (repo / "scripts").mkdir()
    (repo / ".github" / "workflows").mkdir(parents=True)
    (repo / "scripts" / "fork-owned-workflows.txt").write_text("auto-release.yml\npr-labeler.yml\n", encoding="utf-8")
    (repo / "scripts" / "check_writing_quality.py").write_text("# v1\n", encoding="utf-8")
    (repo / ".github" / "workflows" / "auto-release.yml").write_text("name: release\n", encoding="utf-8")
    (repo / "upstream.txt").write_text("base\n", encoding="utf-8")
    _git("add", ".")
    _git("commit", "-m", "base")
    _git("branch", "upstream-main")

    # Upstream alters auto-release.yml (present at merge-base, declared only in fork-owned-workflows.txt)
    _git("checkout", "upstream-main")
    (repo / ".github" / "workflows" / "auto-release.yml").write_text("name: clobbered\n", encoding="utf-8")
    (repo / "scripts" / "check_writing_quality.py").write_text("# clobbered\n", encoding="utf-8")
    _git("add", ".")
    _git("commit", "-m", "upstream clobber")

    # Merge into main (clean fast-forward or merge without conflicts)
    _git("checkout", "main")
    _git("merge", "--no-ff", "upstream-main", "-m", "merge upstream")
    # Rewind main ref to pre-merge commit so main..HEAD reflects the candidate merge.
    base_sha = _git("rev-parse", "HEAD~1")
    result = check_preservation(repo=repo, base_ref=base_sha, upstream_ref="upstream-main", merged_ref="HEAD")
    assert result.ok is False
    assert ".github/workflows/auto-release.yml" in result.protected_violations
    assert "scripts/check_writing_quality.py" in result.protected_violations
