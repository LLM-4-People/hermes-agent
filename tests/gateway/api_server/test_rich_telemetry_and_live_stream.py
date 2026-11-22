"""End-to-end deterministic positive & negative tests for Fork PR #12:
1. Rich Run Telemetry (`usage` fields + `turn.metrics` SSE + explicit null for unknown).
2. Structured Visual Approval Diffs (`approval.request` SSE + `pending_controls[]`, redacted/bounded).
3. Live Coding Stream (opt-in `?detail=full` -> `reasoning.delta`, `subagent.tool.{started,completed}`,
   `turn.metrics`, lean default stream, 400 invalid params, 401 unauth, cross-profile isolation).
4. Enriched `/api/model/options` capabilities & pricing + `/v1/capabilities` feature flags.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

import hermes_cli.inventory as inv
from agent.turn_usage import record_response_usage, reset_turn_telemetry
from gateway.platforms import api_server, api_server_runs
from gateway.session_events import SessionEvents
from gateway.session_pending_controls import PendingControls
from tests.gateway.test_api_server_runs import _claim_run, _create_runs_app, _make_adapter
from tools import approval
from tools.approval_gateway_wait import _await_gateway_decision


def _parse_sse(body: str) -> list[dict]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


@pytest.mark.asyncio
async def test_detail_full_emits_reasoning_subagent_tools_and_turn_metrics_while_default_stays_lean():
    adapter = _make_adapter(api_key="test-key")
    headers = {"Authorization": "Bearer test-key"}
    secret = "sk-" + "Z9y8X7w6V5u4T3s2R1q0P9o8N7m6L5k4J3i2H1g0F9e8"

    def create_agent(**kwargs):
        stream_cb = kwargs["stream_delta_callback"]
        reasoning_cb = kwargs["reasoning_callback"]
        progress_cb = kwargs["tool_progress_callback"]

        agent = SimpleNamespace(
            model="claude-sonnet-4-6",
            provider="anthropic",
            api_mode="anthropic_messages",
            base_url="",
            api_key="",
            platform="api_server",
            session_id="sess-live-1",
            _session_db=None,
            _session_db_created=False,
            context_compressor=SimpleNamespace(
                threshold_tokens=100_000, context_length=200_000,
                awaiting_real_usage_after_compression=False,
                _verify_compaction_cleared_threshold=False,
                _context_probed=False, _context_probe_persistable=False,
                update_from_response=lambda _u: None, note_usage_less_response=lambda: None,
            ),
            session_prompt_tokens=0, session_completion_tokens=0, session_total_tokens=0,
            session_input_tokens=0, session_output_tokens=0, session_cache_read_tokens=0,
            session_cache_write_tokens=0, session_reasoning_tokens=0, session_api_calls=0,
            session_estimated_cost_usd=0.0, session_cost_status="unknown", session_cost_source="none",
            _api_latency_history=deque(maxlen=10), _api_output_history=deque(maxlen=10),
            verbose_logging=False, quiet_mode=True, log_prefix="",
            _safe_print=lambda *_a, **_kw: None, _vprint=lambda *_a, **_kw: None,
            _last_api_first_chunk_at=None, client=None,
        )

        def run_conversation(**_kw):
            reset_turn_telemetry(agent)
            reasoning_cb(f"Thinking with leaked key {secret}")
            progress_cb(
                "tool.started", "terminal", f"run {secret}", {"command": f"echo {secret}"},
                tool_call_id="call_1",
            )
            progress_cb(
                "subagent.tool", "read_file", f"read {secret}", {"path": "app.py", "token": secret},
                subagent_id="sub_1", delegation_id="del_1", tool_call_id="sub_call_1",
            )
            progress_cb(
                "subagent.tool.completed", "read_file", f"done {secret}", None,
                subagent_id="sub_1", delegation_id="del_1", tool_call_id="sub_call_1",
                duration=0.25, is_error=False, result={"ok": True, "secret": secret},
            )
            progress_cb(
                "tool.completed", "terminal", None, None,
                tool_call_id="call_1", duration=0.42, is_error=False, result="ok",
            )
            stream_cb("Final answer")
            agent._last_api_first_chunk_at = 5000.3
            record_response_usage(
                agent,
                SimpleNamespace(
                    usage={
                        "input_tokens": 1000,
                        "output_tokens": 150,
                        "cache_read_input_tokens": 4000,
                        "cache_creation_input_tokens": 500,
                        "output_tokens_details": {"thinking_tokens": 60},
                    },
                    id="msg_live",
                ),
                messages=[], api_call_count=1, api_duration=1.8,
                compression_attempts=0, max_compression_attempts=3, api_start_time=5000.0,
            )
            return {"final_response": "Final answer"}

        agent.run_conversation = run_conversation
        return agent

    app = _create_runs_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_create_agent", side_effect=create_agent):
            resp = await cli.post("/v1/runs", json={"input": "debug this"}, headers=headers)
            assert resp.status == 202
            run_id = (await resp.json())["run_id"]

            # 1. Default stream stays lean: no reasoning.delta, subagent.tool.*, or turn.metrics.
            default_body = await (await cli.get(f"/v1/runs/{run_id}/events", headers=headers)).text()
            default_events = _parse_sse(default_body)
            default_names = [e["event"] for e in default_events]
            assert "reasoning.delta" not in default_names
            assert "subagent.tool.started" not in default_names
            assert "subagent.tool.completed" not in default_names
            assert "turn.metrics" not in default_names
            assert "tool.started" in default_names
            assert "tool.completed" in default_names
            assert default_names[-1] == "run.completed"

            # tool.started carries redacted args_preview; tool.completed carries duration_ms & is_error.
            t_start = next(e for e in default_events if e["event"] == "tool.started")
            assert t_start["tool_call_id"] == "call_1"
            assert "args_preview" in t_start
            assert secret not in t_start["args_preview"]
            assert secret not in t_start["preview"]
            t_done = next(e for e in default_events if e["event"] == "tool.completed")
            assert t_done["duration_ms"] == 420
            assert t_done["is_error"] is False

            # 2. Full stream (`?detail=full`) emits reasoning.delta, subagent.tool.*, and turn.metrics!
            full_body = await (await cli.get(f"/v1/runs/{run_id}/events?detail=full", headers=headers)).text()
            full_events = _parse_sse(full_body)
            full_names = [e["event"] for e in full_events]
            assert "reasoning.delta" in full_names
            assert "subagent.tool.started" in full_names
            assert "subagent.tool.completed" in full_names
            assert "turn.metrics" in full_names
            assert full_names.index("turn.metrics") < full_names.index("run.completed")
            assert secret not in full_body

            sub_started = next(e for e in full_events if e["event"] == "subagent.tool.started")
            assert sub_started["subagent_id"] == "sub_1"
            assert sub_started["delegation_id"] == "del_1"
            assert sub_started["tool"] == "read_file"
            assert "args_preview" in sub_started

            sub_done = next(e for e in full_events if e["event"] == "subagent.tool.completed")
            assert sub_done["subagent_id"] == "sub_1"
            assert sub_done["duration_ms"] == 250
            assert sub_done["is_error"] is False

            metrics_evt = next(e for e in full_events if e["event"] == "turn.metrics")
            completed_evt = next(e for e in full_events if e["event"] == "run.completed")
            status_json = await (await cli.get(f"/v1/runs/{run_id}", headers=headers)).json()

            for payload in (metrics_evt["usage"], completed_evt["usage"], status_json["usage"]):
                assert payload["scope"] == "run"
                assert payload["input_tokens"] == 5500
                assert payload["output_tokens"] == 150
                assert payload["cache_read_tokens"] == 4000
                assert payload["cache_write_tokens"] == 500
                assert payload["cache_hit_ratio"] == pytest.approx(4000 / 5500, abs=1e-4)
                assert payload["reasoning_tokens"] == 60
                assert payload["cost_usd"] is not None and payload["cost_usd"] > 0
                assert payload["cost_source"] == "official_docs_snapshot"
                assert payload["ttft_ms"] == 300
                assert payload["duration_ms"] is not None and payload["duration_ms"] >= 0
                assert payload["tokens_per_sec"] == 100.0
                assert payload["api_calls"] == 1


@pytest.mark.asyncio
async def test_detail_stream_negative_validation_auth_and_cross_profile_isolation():
    adapter = _make_adapter(api_key="profile-a-key")
    headers = {"Authorization": "Bearer profile-a-key"}
    run_id = "run-profile-a"
    foreign_id = "run-profile-b"
    _claim_run(adapter, run_id)
    adapter._run_streams[run_id] = api_server_runs._RunStream()
    adapter._run_streams[run_id].put_nowait(
        api_server_runs._run_event(run_id, "reasoning.delta", text="private-a"), channel="reasoning"
    )
    adapter._run_streams[run_id].put_nowait(None)

    adapter._run_owners[foreign_id] = "foreign-profile-scope"
    adapter._run_streams[foreign_id] = api_server_runs._RunStream()
    adapter._run_streams[foreign_id].put_nowait(
        api_server_runs._run_event(foreign_id, "reasoning.delta", text="secret-profile-b"), channel="reasoning"
    )
    adapter._run_streams[foreign_id].put_nowait(None)

    app = _create_runs_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        # Unauthenticated rejected with 401.
        assert (await cli.get(f"/v1/runs/{run_id}/events?detail=full")).status == 401
        # Cross-profile run rejected with 404 (no leak of profile B reasoning).
        foreign_resp = await cli.get(f"/v1/runs/{foreign_id}/events?detail=full", headers=headers)
        assert foreign_resp.status == 404
        assert "secret-profile-b" not in await foreign_resp.text()
        # Invalid query detail / boolean flags rejected with 400.
        assert (await cli.get(f"/v1/runs/{run_id}/events?detail=bogus", headers=headers)).status == 400
        assert (await cli.get(f"/v1/runs/{run_id}/events?include_reasoning=maybe", headers=headers)).status == 400
        # Invalid body detail / flags on POST /v1/runs rejected with 400.
        for bad_body in (
            {"input": "hi", "detail": "verbose"},
            {"input": "hi", "detail": 123},
            {"input": "hi", "include_reasoning": "yes"},
            {"input": "hi", "include_subagent_tools": 1},
        ):
            r = await cli.post("/v1/runs", json=bad_body, headers=headers)
            assert r.status == 400


@pytest.mark.asyncio
async def test_approval_sse_and_pending_controls_carry_redacted_visual_diff():
    adapter = _make_adapter(api_key="test-key")
    headers = {"Authorization": "Bearer test-key"}
    run_id = "run-visual-diff"
    secret = "sk-" + "Q1w2E3r4T5y6U7i8O9p0A1s2D3f4G5h6J7k8L9z0X1c2"
    _claim_run(adapter, run_id)
    adapter._run_approval_sessions[run_id] = run_id
    adapter._set_run_status(run_id, "running")
    stream = adapter._run_streams[run_id] = api_server_runs._RunStream()
    notify = api_server_runs._make_approval_notify(
        adapter, SimpleNamespace(run_id=run_id, queue=stream), _api_server=api_server
    )
    approval_payload = {
        "command": f"Edit app.py (token {secret})",
        "description": "Approve file patch",
        "pattern_key": "editor-1",
        "allow_session": False,
        "allow_permanent": False,
        "private_context": "must-not-leak",
        "edit": {
            "tool_name": "patch",
            "path": "src/service.py",
            "old_text": "def f():\n    return 1\n",
            "new_text": f"def f():\n    key = '{secret}'\n    return 2\n",
        },
    }
    worker = asyncio.create_task(
        asyncio.to_thread(_await_gateway_decision, run_id, notify, approval_payload)
    )
    try:
        # Wait until the approval event lands in the SSE stream backlog.
        for _ in range(100):
            if stream.backlog:
                break
            await asyncio.sleep(0.02)
        assert stream.backlog, "approval.request event never landed"
        sse_event = stream.backlog[-1][1]
        assert sse_event["event"] == "approval.request"
        assert "edit" not in sse_event
        assert secret not in json.dumps(sse_event)
        assert sse_event["preview"]["kind"] == "file_patch"
        assert sse_event["preview"]["diff"]["path"] == "src/service.py"
        assert sse_event["preview"]["diff"]["language"] == "python"
        assert sse_event["preview"]["diff"]["additions"] == 2
        assert sse_event["preview"]["diff"]["deletions"] == 1

        # Poll GET /v1/runs/{run_id} and confirm pending_controls[] carries the same visual diff.
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            status_resp = await cli.get(f"/v1/runs/{run_id}", headers=headers)
            assert status_resp.status == 200
            snapshot = await status_resp.json()
            assert secret not in json.dumps(snapshot)
            assert "must-not-leak" not in json.dumps(snapshot)
            prompt = snapshot["pending_controls"][0]
            assert "edit" not in prompt
            assert prompt["preview"]["diff"]["path"] == "src/service.py"
            assert prompt["preview"]["diff"]["language"] == "python"
            assert prompt["preview"]["diff"]["additions"] == 2

        # Canonical PendingControls.register also attaches preview.diff cleanly.
        controls = PendingControls(SessionEvents())
        controls.register("sess-1", run_id, 1, {**approval_payload, "request_id": "req-canon"})
        canon_prompt = controls.pending["req-canon"][1]
        assert canon_prompt["preview"]["kind"] == "file_patch"
        assert canon_prompt["preview"]["diff"]["language"] == "python"
        assert secret not in json.dumps(canon_prompt["preview"])
    finally:
        approval.unregister_gateway_notify(run_id)
        await worker


@pytest.mark.asyncio
async def test_capabilities_and_model_options_expose_normalized_metadata(monkeypatch):
    from agent import models_dev
    from hermes_cli import models_pricing

    adapter = _make_adapter(api_key="test-key")
    headers = {"Authorization": "Bearer test-key"}

    app = _create_runs_app(adapter)
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_get("/api/model/options", adapter._handle_model_options)

    registry = {
        "anthropic": {
            "models": {
                "claude-sonnet-4-6": {
                    "id": "claude-sonnet-4-6",
                    "tool_call": True,
                    "reasoning": True,
                    "structured_output": True,
                    "modalities": {"input": ["text", "image"]},
                    "limit": {"context": 200000, "output": 64000},
                }
            }
        }
    }
    monkeypatch.setattr(models_dev, "_models_dev_cache", registry)
    monkeypatch.setattr(models_dev, "_models_dev_cache_time", float("inf"))
    monkeypatch.setattr(models_pricing, "get_pricing_for_provider", lambda _slug, **_kw: {})
    monkeypatch.setattr(inv, "_moa_provider_row", lambda *_a, **_kw: None)
    monkeypatch.setattr(inv, "_local_runtime_row", lambda *_a, **_kw: None)
    monkeypatch.setattr(inv, "_strip_aggregator_overlaps", lambda _rows: None)
    monkeypatch.setattr(inv, "_prewarm_pricing_async", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers",
        lambda **_kw: [
            {
                "slug": "anthropic",
                "name": "Anthropic",
                "models": ["claude-sonnet-4-6", "mystery-unknown-model"],
                "total_models": 2,
                "is_current": True,
                "is_user_defined": False,
                "source": "built-in",
            }
        ],
    )

    async with TestClient(TestServer(app)) as cli:
        caps = await (await cli.get("/v1/capabilities", headers=headers)).json()
        for flag in (
            "run_metrics_v2",
            "structured_approval_diffs",
            "live_coding_stream",
            "model_capability_metadata",
        ):
            assert caps["features"][flag] is True

        opts_resp = await cli.get("/api/model/options?include_unconfigured=false", headers=headers)
        assert opts_resp.status == 200
        opts = await opts_resp.json()
        anth = next(row for row in opts["providers"] if row["slug"] == "anthropic")

        sonnet_caps = anth["capabilities"]["claude-sonnet-4-6"]
        assert sonnet_caps["context_length"] == 200000
        assert sonnet_caps["max_output_tokens"] == 64000
        assert sonnet_caps["vision"] is True
        assert sonnet_caps["tools"] is True
        assert sonnet_caps["reasoning"] is True
        assert sonnet_caps["structured_output"] is True

        # Unknown model keeps explicit None for unknown metadata, never fabricated zeros.
        unknown_caps = anth["capabilities"]["mystery-unknown-model"]
        assert unknown_caps["context_length"] is None
        assert unknown_caps["max_output_tokens"] is None
        assert unknown_caps["vision"] is None
        assert unknown_caps["tools"] is None
        assert unknown_caps["structured_output"] is None

        # Pricing includes official docs snapshot numeric per-1M fields for known model,
        # and does NOT fabricate pricing for the unknown model.
        sonnet_price = anth["pricing"]["claude-sonnet-4-6"]
        assert sonnet_price["input_per_1m"] == 3.0
        assert sonnet_price["output_per_1m"] == 15.0
        assert sonnet_price["cache_read_per_1m"] == 0.3
        assert sonnet_price["cache_write_per_1m"] == 3.75
        assert sonnet_price["input"] == "$3.00"
        assert "mystery-unknown-model" not in anth["pricing"]
