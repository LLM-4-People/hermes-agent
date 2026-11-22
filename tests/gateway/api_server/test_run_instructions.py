"""Run-owned system instructions survive admission recovery without becoming session defaults."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from aiohttp.test_utils import TestClient, TestServer
import pytest

from agent.codex_runtime import _codex_developer_instructions
from agent.background_review import _same_model_parity_kwargs
from agent.run_instructions import effective_system_prompt
from gateway.run_turn_runner import TurnRunner
from gateway.session_api_turn import admit_api_turn, api_execution, api_settings, prepare_api_execution
from gateway.turn_context import TurnContext
from persistence.core.errors import AuthorityRefusal


def _model_prompts(api, owner, ref, payload):
    prepared = prepare_api_execution(owner, ref, payload)
    context = api_execution.set(prepared)
    try:
        runner = SimpleNamespace(session_authority=None, _consume_pending_native_image_paths=lambda key: [])
        turn = TurnRunner(runner, TurnContext(message='hello', session_id=ref.session_id, session_key='api',
            source=SimpleNamespace(user_id='owner', user_name='Owner', is_bot=False)))
        agent = SimpleNamespace(ephemeral_system_prompt=turn._combined_ephemeral_prompt(), _cached_system_prompt='system')
        seen = []
        def run(message, **kwargs):
            seen.extend((_codex_developer_instructions(agent), _same_model_parity_kwargs(agent)['ephemeral_system_prompt']))
            return {'final_response': 'ok'}
        agent.run_conversation = run
        turn._run_conversation_with_approval(agent, [], None, None, None)
        seen.append(effective_system_prompt(agent, 'system'))
        return seen
    finally:
        api_execution.reset(context)


def test_queued_recovery_and_internal_inputs_keep_the_normal_prompt(api, owner):
    base = 'Use the workspace form.'
    once = 'Use this approved area for this turn.'
    common = dict(session_id='conversation', conversation_history=[])
    _, ref, _ = admit_api_turn(api, **common, request_id='first', user_message='first', ephemeral_system_prompt=base)
    scoped = dict(**common, request_id='second', user_message='hello', run_instructions=once)
    _, _, row = admit_api_turn(api, **scoped)
    assert once not in str(api_settings(owner, ref))
    assert api_settings(owner, ref)['ephemeral_system_prompt'] == base
    owner.sessions.clear()
    from gateway.session_api import restore_api_session
    restore_api_session(owner, ref.session_id)
    stored = owner.persistence.admissions.get(admission_id=row['admission_id'])
    assert _model_prompts(api, owner, ref, stored.payload) == ['system\n\n' + base + '\n\n' + once, base, 'system\n\n' + base]
    assert _model_prompts(api, owner, ref, {'text': 'internal'}) == ['system\n\n' + base, base, 'system\n\n' + base]

    # Changing the durable default later must not alter a previously accepted request's identity or input.
    _, _, cleared = admit_api_turn(api, **common, request_id='third', user_message='new', ephemeral_system_prompt='', run_instructions='one')
    assert _model_prompts(api, owner, ref, cleared['payload']) == ['system\n\none', '', 'system']
    assert admit_api_turn(api, **scoped)[2]['admission_id'] == row['admission_id']
    with pytest.raises(AuthorityRefusal, match='admission_conflict'):
        admit_api_turn(api, **{**scoped, 'run_instructions':'changed'})
    assert api_settings(owner, ref)['ephemeral_system_prompt'] == ''


@pytest.mark.asyncio
async def test_run_http_prompt_and_idempotent_execution(monkeypatch):
    from tests.gateway.test_api_server_runs import _create_runs_app, _make_adapter

    adapter = _make_adapter()
    adapter._api_key = 'owner-test'
    agent = MagicMock()
    calls = []
    def create(**kwargs):
        agent.ephemeral_system_prompt = kwargs['ephemeral_system_prompt']
        agent._cached_system_prompt = 'system'
        return agent
    def run(**kwargs):
        calls.append(_codex_developer_instructions(agent))
        return {'final_response':'ok'}
    agent.run_conversation.side_effect = run
    monkeypatch.setattr(adapter, '_create_agent', create)
    app = _create_runs_app(adapter)
    app.router.add_get('/v1/capabilities', adapter._handle_capabilities)
    headers = {'Authorization':'Bearer owner-test', 'Idempotency-Key':'test-run'}
    body = {'input':'question', 'instructions':'durable', 'run_instructions':'one send'}
    async with TestClient(TestServer(app)) as client:
        assert (await client.post('/v1/runs', json=body)).status == 401
        caps = await (await client.get('/v1/capabilities', headers=headers)).json()
        assert caps['features']['run_scoped_instructions'] is True
        for invalid in (None, {}, [], False, 5, '\ud800'):
            assert (await client.post('/v1/runs', json={**body, 'run_instructions':invalid}, headers=headers)).status == 400
        reply = await client.post('/v1/runs', json=body, headers=headers)
        assert reply.status == 202
        ident = (await reply.json())['run_id']
        async with client.get(f'/v1/runs/{ident}/events', headers=headers) as events:
            await events.read()
        assert (await (await client.post('/v1/runs', json=body, headers=headers)).json())['run_id'] == ident
        assert (await client.post('/v1/runs', json={**body, 'run_instructions':'changed'}, headers=headers)).status == 409
    assert calls == ['system\n\ndurable\n\none send']
    assert _codex_developer_instructions(agent) == 'system\n\ndurable'
