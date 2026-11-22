"""Tool availability follows the submitted run across transport and execution boundaries."""
import json
from unittest.mock import MagicMock, patch

from aiohttp.test_utils import TestClient, TestServer
import pytest

from tests.gateway.test_api_server_runs import _create_runs_app, _make_adapter
from tools.registry import registry


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", [None, "file", [""], [False], ["file", "file"], ["not-an-enabled-toolset"]])
async def test_invalid_selection_never_starts(selected):
    adapter = _make_adapter()
    async with TestClient(TestServer(_create_runs_app(adapter))) as client:
        with patch.object(adapter, "_create_agent") as create:
            response = await client.post("/v1/runs", json={"input": "hello", "allowed_toolsets": selected})
            assert response.status == 400
        create.assert_not_called()


@pytest.mark.asyncio
async def test_empty_selection_survives_worker_hop_and_resets():
    adapter = _make_adapter()
    def run(**kwargs):
        from model_tools import get_tool_definitions
        assert get_tool_definitions(enabled_toolsets=["file"], quiet_mode=True) == []
        assert "run" in json.loads(registry.dispatch("unknown_control_probe", {}))["error"]
        return {"final_response": "done"}

    agent = MagicMock()
    agent.run_conversation.side_effect = run
    async with TestClient(TestServer(_create_runs_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", return_value=agent):
            response = await client.post("/v1/runs", json={"input": "hello", "allowed_toolsets": []})
            assert response.status == 202
            run_id = (await response.json())["run_id"]
            async with client.get(f"/v1/runs/{run_id}/events") as events:
                await events.read()
            status = await (await client.get(f"/v1/runs/{run_id}")).json()
            assert status["status"] == "completed", status
        assert "Unknown tool" in json.loads(registry.dispatch("unknown_control_probe", {}))["error"]


def test_recovered_api_admission_restricts_its_execution(api, owner):
    from gateway.session_api_turn import admit_api_turn, api_execution, api_policy_scope, prepare_api_execution
    from model_tools import get_tool_definitions
    _, ref, row = admit_api_turn(api, session_id="restricted", request_id="queued-turn",
                                 user_message="hello", conversation_history=[], allowed_toolsets=[])
    assert "allowed_toolsets" not in prepare_api_execution(owner, ref, {"text": "unrelated follow-up"})["settings"]
    context = api_execution.set(prepare_api_execution(owner, ref, row["payload"]))
    try:
        with api_policy_scope():
            assert get_tool_definitions(enabled_toolsets=["file"], quiet_mode=True) == []
    finally:
        api_execution.reset(context)


def test_individual_exclusions_follow_nested_hermes_dispatch_and_do_not_affect_other_runs(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from gateway.platforms.api_server_request_options import run_tool_scope
    from model_tools import get_tool_definitions
    from tools.thread_context import propagate_context_to_thread

    def visible():
        return {entry["function"]["name"] for entry in get_tool_definitions(["file"], quiet_mode=True)}

    def worker():
        assert "read_file" in visible()
        assert "write_file" not in visible()
        error = json.loads(registry.dispatch("write_file", {"path": str(tmp_path / "note"), "content": "test"}))
        assert "run" in error["error"]

    assert "write_file" in visible()
    with run_tool_scope({"allowed_toolsets": ["file"], "excluded_tools": ["write_file"]}):
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(propagate_context_to_thread(worker)).result()
        assert not (tmp_path / "note").exists()
    assert "write_file" in visible()


@pytest.mark.asyncio
@pytest.mark.parametrize("exclusions", [None, ["not-a-tool"], [False], ["write_file", "write_file"]])
async def test_invalid_tool_exclusions_never_start(exclusions):
    adapter = _make_adapter()
    async with TestClient(TestServer(_create_runs_app(adapter))) as client:
        with patch.object(adapter, "_create_agent") as create:
            response = await client.post("/v1/runs", json={"input": "hello", "excluded_tools": exclusions})
            assert response.status == 400
        create.assert_not_called()


def test_discovery_and_dispatch_share_provider_tool_membership(monkeypatch):
    from types import SimpleNamespace

    from agent.memory_manager_tools import inject_memory_provider_tools
    from gateway.platforms.api_server_request_options import RunToolInventory, run_tool_scope
    from tools.hosted_tool_policy import scoped_tool_schemas, tool_execution_refusal

    config = {"platform_toolsets": {"api_server": ["memory"]}, "memory": {"provider": "example"},
              "mcp_servers": {"example-mcp": {"url": "https://example.invalid"}}}
    monkeypatch.setattr("hermes_cli.config_effective.load_user_config_effective", lambda *args, **kwargs: config)
    schema = {"name": "lookup_note", "description": "Look up a note", "parameters": {"type": "object"}}
    monkeypatch.setattr("plugins.plugin_loader.inspect_provider_tool_names",
                        lambda cfg, name: {"lookup_note"} if name == "memory" else set())
    inventory = RunToolInventory(config)
    rows = {row["name"]: row for row in inventory.rows()}
    assert rows["example-mcp"]["enabled"]
    assert "lookup_note" in rows["memory"]["tools"]

    manager = SimpleNamespace(get_all_tool_schemas=lambda: [schema], has_tool=lambda name: name == "lookup_note")
    agent = SimpleNamespace(enabled_toolsets=["memory"], _memory_manager=manager, tools=[], valid_tool_names=set())
    inject_memory_provider_tools(agent)
    with run_tool_scope({"allowed_toolsets": ["memory"]}):
        assert scoped_tool_schemas(agent, agent.tools) == agent.tools
        assert tool_execution_refusal("lookup_note", agent) is None
        assert tool_execution_refusal("lookup_note") is not None
    with run_tool_scope({"excluded_tools": ["lookup_note"]}):
        assert scoped_tool_schemas(agent, agent.tools) == []
        assert tool_execution_refusal("lookup_note", agent) is not None


@pytest.mark.asyncio
async def test_unreadable_operator_policy_refuses_restricted_run(tmp_path, monkeypatch, caplog):
    (tmp_path / "config.yaml").write_text("secret-key: [private-test-sentinel", encoding="utf-8")
    monkeypatch.setattr("gateway.run._gateway_config_home", lambda: tmp_path)
    adapter = _make_adapter()
    app = _create_runs_app(adapter)
    app.router.add_get("/v1/toolsets", adapter._handle_toolsets)
    async with TestClient(TestServer(app)) as client:
        catalog = await client.get("/v1/toolsets")
        assert catalog.status == 503
        assert "private-test-sentinel" not in await catalog.text()
        with patch.object(adapter, "_create_agent") as create:
            response = await client.post("/v1/runs", json={"input": "hello", "allowed_toolsets": ["file"]})
            assert response.status == 503
            assert "private-test-sentinel" not in await response.text()
        create.assert_not_called()
    assert "private-test-sentinel" not in caplog.text


def test_execution_policy_backend_outage_is_safe_for_detached_worker(monkeypatch, caplog):
    from gateway.platforms.api_server_request_options import RunToolPolicyUnavailable, run_tool_scope
    from hermes_cli.config_backend import ConfigBackendUnavailable

    def unavailable(*args, **kwargs):
        raise ConfigBackendUnavailable("private-backend-sentinel")

    monkeypatch.setattr("hermes_cli.config_effective.load_user_config_effective", unavailable)
    with pytest.raises(RunToolPolicyUnavailable, match="policy is unavailable") as failure:
        run_tool_scope({"allowed_toolsets": []})
    assert failure.value.__suppress_context__
    assert "private-backend-sentinel" not in caplog.text


def test_canonical_run_usage_requires_a_per_turn_token_count():
    from gateway.platforms.api_server_authority_runs import canonical_run_usage

    assert canonical_run_usage({"output_tokens": 0}) == {"output_tokens": 0, "scope": "run"}
    assert canonical_run_usage({"output_tokens": 9}) == {"output_tokens": 9, "scope": "run"}
    for usage in ({}, {"output_tokens": -1}, {"output_tokens": True}, None):
        assert canonical_run_usage(usage) == usage


def test_registered_exclusions_skip_unrelated_provider_host():
    from gateway.platforms.api_server_request_options import RunToolInventory

    inventory = RunToolInventory({"platform_toolsets": {"api_server": ["file", "memory"]}})
    with patch.object(inventory, "names", side_effect=AssertionError("unnecessary provider inspection")):
        assert inventory.validate({"excluded_tools": ["write_file"]}) == {"excluded_tools": ["write_file"]}


@pytest.mark.asyncio
async def test_capabilities_advertise_both_supported_controls():
    from tests.gateway.test_api_server import _create_app

    async with TestClient(TestServer(_create_app(_make_adapter()))) as client:
        response = await client.get("/v1/capabilities")
        assert response.status == 200
        features = (await response.json())["features"]
        assert features["run_toolset_selection"] is True
        assert features["run_tool_exclusions"] is True


def test_run_selection_keeps_the_shared_session_tool_pin_intact():
    from types import SimpleNamespace
    from tools.hosted_tool_policy import restrict_tools
    from tools.mcp_tool_agent import persist_agent_tool_names, restore_agent_tool_prefix

    db = MagicMock()
    fresh = [{"type": "function", "function": {"name": name}} for name in ("read_file", "write_file")]
    agent = SimpleNamespace(_session_db=db, session_id="same-session", tools=fresh)
    with restrict_tools(["file"], ["read_file", "write_file"], []):
        assert not restore_agent_tool_prefix(agent, {"tools": fresh[:1]})
        assert agent.tools == fresh
        persist_agent_tool_names(agent)
    db.update_session_tool_names.assert_not_called()
