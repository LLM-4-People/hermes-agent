"""A real API run can ask a disconnected client several questions in one model tool call."""
import json
import sqlite3
from pathlib import Path

import requests

from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall
from tests.gateway.fixtures.local_recovery_probe import child_env, daemon
from tests.waiting import wait_until


QUESTIONS = [
    {"question": "Which city?"},
    {"question": "Pick two colors", "choices": ["Red", "Blue", "Green"], "multi_select": True},
    {"question": "Any further details?"},
]


def _status(url, headers):
    response = requests.get(url, headers=headers, timeout=5)
    response.raise_for_status()
    return response.json()


def _ledger(home, run_id):
    with sqlite3.connect(f"file:{home / 'runs_idempotency.db'}?mode=ro", uri=True) as db:
        row = db.execute("SELECT status_json FROM run_idempotency WHERE run_id=?", (run_id,)).fetchone()
    return json.loads(row[0])


def _answer_prompts(url, headers, home, run_id):
    last_id = None
    prompts = []
    for question, answer in zip(QUESTIONS, ["Montréal", json.dumps(["Red (Recommended)", "Green"]), ""]):
        def next_prompt():
            snapshot = _status(url, headers)
            assert snapshot["status"] not in {"completed", "failed", "cancelled"}, snapshot
            return next((p for p in snapshot.get("pending_controls", []) if p["prompt_id"] != last_id), None)
        prompt = wait_until(next_prompt, timeout=30, interval=.1, what="the next canonical question")
        assert prompt["question"] == question["question"]
        assert prompt["kind"] == "clarify" and prompt["multi_select"] == question.get("multi_select", False)
        assert prompt["choices"] == (["Red (Recommended)", "Blue", "Green"] if question.get("choices") else [])
        last_id = prompt["prompt_id"]
        def waiting_receipt():
            receipt = _ledger(home, run_id)
            return receipt if list(receipt.get("pending_questions", {})) == [last_id] else None
        receipt = wait_until(waiting_receipt, what="the durable pending question identity")
        assert receipt["status"] == _status(url, headers)["status"] == "waiting_for_input"
        assert receipt["pending_questions"][last_id] > 0
        assert all(secret not in json.dumps(receipt, ensure_ascii=False) for secret in [question["question"], "Recommended", "Montréal"])
        assert requests.post(url + "/steer", headers=headers, json={"input": "Later"}, timeout=5).status_code == 409
        if not prompts:
            # Disconnect a real SSE subscription, then answer from a fresh connection.
            with requests.get(url + "/events", headers=headers, stream=True, timeout=10) as events:
                assert next(event for event in _events(events) if event["event"] == "clarify.request")["prompt_id"] == last_id
        prompts.append(last_id)
        payload = {"request_id": last_id, "execution_generation": prompt["execution_generation"], "answer": answer}
        assert requests.post(url + "/clarify", json=payload, timeout=5).status_code == 401
        for mismatch in [{"request_id": "unknown-id"}, {"execution_generation": -1}]:
            assert requests.post(url + "/clarify", headers=headers, json={**payload, **mismatch}, timeout=5).status_code == 409
        response = requests.post(url + "/clarify", headers=headers, json=payload, timeout=5)
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "resolved"
        assert requests.post(url + "/clarify", headers=headers, json=payload, timeout=5).status_code == 409


    return prompts


def _events(stream):
    return (json.loads(line[6:]) for line in stream.iter_lines(chunk_size=1) if line.startswith(b"data: "))


def _stop_open_question(base, headers, home):
    accepted = requests.post(base + "/v1/runs", headers={**headers, "Idempotency-Key": "cancel-question"},
                             json={"input": "One more", "session_id": "question-stop", "allowed_toolsets": ["clarify"]},
                             timeout=10)
    assert accepted.status_code == 202, accepted.text
    run_id = accepted.json()["run_id"]
    url = base + "/v1/runs/" + run_id
    prompt = wait_until(lambda: _status(url, headers).get("pending_controls"), timeout=30, what="a stoppable prompt")[0]
    assert requests.post(url + "/stop", headers=headers, json={}, timeout=5).json()["status"] == "stopping"
    assert _status(url, headers)["pending_controls"] == []
    assert "pending_questions" not in _ledger(home, run_id)
    late = {"request_id": prompt["prompt_id"], "execution_generation": prompt["execution_generation"], "answer": "late"}
    assert requests.post(url + "/clarify", headers=headers, json=late, timeout=5).status_code == 409
    wait_until(lambda: _status(url, headers)["status"] == "cancelled", what="the stopped question to retire")


def test_ordinary_daemon_run_clarifications_reconnect(tmp_path, unused_tcp_port):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / "state", tmp_path / "user"
    home.mkdir(mode=0o700)
    user.mkdir()
    headers = {"Authorization": "Bearer loopback-only-api-key"}
    base = f"http://127.0.0.1:{unused_tcp_port}"
    with FakeLLMServer([ToolCall("clarify", {"questions": QUESTIONS}), Text("Questions complete"),
                        ToolCall("clarify", {"questions": QUESTIONS[:1]})]) as model:
        config = {"model": {"provider": "custom", "default": "question-model", "base_url": model.base_url},
                  "gateway": {"multiplex_profiles": False, "platforms": {"api_server": {
                      "enabled": True, "port": unused_tcp_port, "host": "127.0.0.1"}}},
                  "auxiliary": {"title_generation": {"enabled": False}},
                  "platform_toolsets": {"api_server": ["hermes-api-server", "clarify"]}}
        (home / "config.yaml").write_text(json.dumps(config))
        env = dict(child_env(), HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
                   OPENAI_API_KEY="loopback-only-model", OPENAI_BASE_URL=model.base_url,
                   API_SERVER_KEY="loopback-only-api-key", API_SERVER_ENABLED="true", API_SERVER_PORT=str(unused_tcp_port))
        with daemon(root, home, env, barrier=False):
            accepted = requests.post(base + "/v1/runs", headers={**headers, "Idempotency-Key": "questions"},
                                     json={"input": "Ask questions", "session_id": "question-test",
                                           "allowed_toolsets": ["clarify"]}, timeout=15)
            assert accepted.status_code == 202, accepted.text
            run_id = accepted.json()["run_id"]
            url = base + "/v1/runs/" + run_id
            prompt_ids = _answer_prompts(url, headers, home, run_id)
            wait_until(lambda: _status(url, headers)["status"] == "completed", what="the resumed inference result")
            assert _status(url, headers)["output"] == "Questions complete"
            wait_until(lambda: _ledger(home, run_id)["status"] == "completed", what="the durable terminal receipt")
            assert "pending_questions" not in _ledger(home, run_id)
            with requests.get(url + "/events", headers=headers, stream=True, timeout=10) as stream:
                events = list(_events(stream))
            for kind in ("clarify.request", "clarify.settled"):
                assert [e["prompt_id"] for e in events if e["event"] == kind] == prompt_ids
            capabilities = _status(base + "/v1/capabilities", headers)
            assert capabilities["features"]["run_clarify_response"]
            assert capabilities["endpoints"]["run_clarify"] == {"method": "POST", "path": "/v1/runs/{run_id}/clarify"}
            history = _status(base + "/api/sessions/question-test/messages?include_compacted=true", headers)["data"]
            clarify_call = next(c for m in history for c in (m.get("tool_calls") or []) if c["function"]["name"] == "clarify")
            result_row = next(m for m in history if m["role"] == "tool" and m["tool_call_id"] == clarify_call["id"])
            _stop_open_question(base, headers, home)
        first, resumed, _stopped = model.main_requests()
        assert [t["function"]["name"] for t in first["tools"]] == ["clarify"]
        tool_result = json.loads(next(m["content"] for m in resumed["messages"] if m["role"] == "tool"))
        assert json.loads(result_row["content"]) == tool_result
        assert tool_result["outcome"] == "submitted"
        assert [(r["status"], r["user_response"]) for r in tool_result["responses"]] == [
            ("answered", "Montréal"), ("answered", ["Red", "Green"]), ("skipped", None)]


def test_overlapping_pending_control_receipts_keep_individual_identities():
    from asyncio import Queue
    from gateway.platforms.api_server_runs import _RunLaunch
    from tests.gateway.test_api_server import _make_adapter
    adapter, queue = _make_adapter(), Queue()
    run = _RunLaunch(owner=adapter, run_id="controlled", queue=queue, session_id="session",
                     gateway_session_key=None, declared_selected=False, user_message="", conversation_history=[],
                     session_history_delivery=False, agent_kwargs={}, request_profile=None)
    adapter._run_streams[run.run_id] = queue
    state = adapter._set_run_status(run.run_id, "running")

    def event(family, action, prompt, timestamp):
        run.put_control({"event": family + "." + action, "timestamp": timestamp, "prompt_id": prompt,
                         "question": "confidential question", "choices": ["private"]})
    event("clarify", "request", "q1", 10)
    event("clarify", "request", "q1", 11)
    event("clarify", "request", "q2", 12)
    event("approval", "request", "a1", 13)
    assert state["pending_questions"] == {"q1": 10, "q2": 12}
    assert state["pending_approvals"] == {"a1": 13}
    assert state["status"] == "waiting_for_approval"
    event("clarify", "settled", "q2", 14)
    assert state["status"] == "waiting_for_approval" and state["pending_questions"] == {"q1": 10}
    event("approval", "settled", "a1", 15)
    assert state["status"] == "waiting_for_input" and "pending_approvals" not in state
    assert adapter._active_structured_run_count() == 1
    adapter._set_run_status(run.run_id, "cancelled")
    event("clarify", "request", "late", 16)
    assert state["status"] == "cancelled" and "pending_questions" not in state
    assert "confidential" not in json.dumps(state) and "private" not in json.dumps(state)



def test_standalone_adapter_does_not_advertise_canonical_clarification():
    import asyncio
    from aiohttp.test_utils import TestClient, TestServer
    from tests.gateway.test_api_server import _create_app, _make_adapter
    async def check():
        async with TestClient(TestServer(_create_app(_make_adapter()))) as client:
            response = await client.get("/v1/capabilities")
            info = await response.json()
            assert info["features"]["run_clarify_response"] is False and "run_clarify" not in info["endpoints"]
    asyncio.run(check())
