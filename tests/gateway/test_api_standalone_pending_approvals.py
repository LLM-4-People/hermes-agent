"""Reconnect to live standalone approval waiters through the authenticated runs endpoint."""
import asyncio
from types import SimpleNamespace

from aiohttp.test_utils import TestClient, TestServer
import pytest

from gateway.platforms import api_server, api_server_runs
from tests.gateway.test_api_server_runs import _claim_run, _create_runs_app, _make_adapter
from tools import approval
from tools.approval_gateway_wait import _await_gateway_decision


@pytest.mark.asyncio
async def test_standalone_status_tracks_only_this_runs_unanswered_native_requests():
    api = _make_adapter(api_key="test-key")
    headers = {"Authorization": "Bearer test-key"}
    run, other = "run-approval-owner", "run-other-owner"
    workers = []
    secret = "sk-" + "X" * 44
    for sid in (run, other):
        _claim_run(api, sid)
        api._run_approval_sessions[sid] = sid
        api._set_run_status(sid, "running")
    api._run_owners[other] = "another-profile"
    async with TestClient(TestServer(_create_runs_app(api))) as client:
        async def status():
            response = await client.get(f"/v1/runs/{run}", headers=headers)
            assert response.status == 200
            return await response.json()

        async def ask(sid, description, **options):
            events = asyncio.Queue()
            notify = api_server_runs._make_approval_notify(api, SimpleNamespace(run_id=sid, queue=events),
                                                           _api_server=api_server)
            data = {"description": description, "command": f"echo {description}", "private_context": "internal-only", **options}
            workers.append(asyncio.create_task(asyncio.to_thread(_await_gateway_decision, sid, notify, data)))
            return (await asyncio.wait_for(events.get(), timeout=10))["request_id"]

        async def answer(request_id):
            return await client.post(f"/v1/runs/{run}/approval", headers=headers, json={"request_id": request_id, "choice": "deny"})

        try:
            first = await ask(run, "first " + secret, allow_session=False)
            second = await ask(run, "second", allow_permanent=False)
            foreign = await ask(other, "other-profile-control")
            assert (await client.get(f"/v1/runs/{run}")).status == 401
            assert (await client.get(f"/v1/runs/{other}", headers=headers)).status == 404
            assert (await answer(foreign)).status == 409
            snapshot = await status()
            controls = snapshot["pending_controls"]
            assert [prompt["request_id"] for prompt in controls] == [first, second]
            assert [prompt["choices"] for prompt in controls] == [["once", "deny"], ["once", "session", "deny"]]
            assert all(p["event"] == "approval.request" and p["run_id"] == run for p in controls)
            assert all("execution_generation" not in p and "prompt_id" not in p for p in controls)
            assert secret not in str(snapshot) and foreign not in str(snapshot) and "internal-only" not in str(snapshot)
            assert (await answer(second)).status == 200
            for retained in await asyncio.gather(status(), status()):
                assert retained["status"] == "waiting_for_approval"
                assert [p["request_id"] for p in retained["pending_controls"]] == [first]
                assert retained["approval"]["request_id"] == first
            assert (await answer(second)).status == 409
            assert approval.withdraw_gateway_approval(run, first, "client closed")
            empty = await status()
            assert empty["pending_controls"] == [] and "approval" not in empty and empty["status"] == "running"
            assert [p["request_id"] for p in approval.list_gateway_approvals(other)] == [foreign]
            # Retained status can survive the local queue owner. It cannot certify a live list.
            api._run_approval_sessions.pop(run)
            api._set_run_status(run, "waiting_for_approval", approval={"request_id": first})
            remote = await status()
            assert "approval" not in remote and "pending_controls" not in remote
            api._set_run_status(run, "completed")
            assert (await status())["pending_controls"] == []
        finally:
            for sid in (run, other):
                approval.unregister_gateway_notify(sid)
            await asyncio.gather(*workers)


@pytest.mark.asyncio
@pytest.mark.parametrize(("permissions", "choices"), [(("status",), []), (("status", "approve"), ["once", "deny"])])
async def test_room_grant_only_offers_choices_it_can_submit(permissions, choices):
    from gateway import hosted_rooms
    from gateway.hosted_room_peer import decode_room_grant, issue_room_grant
    from persistence_assembly import open_persistence, profile_scope

    api = _make_adapter(api_key="test-key")
    token = issue_room_grant(api._room_grant_secret(), grant_id="grant-poll", room_id="room-poll", home_install_id="home",
                             authority_gateway_id="home", authority_epoch=1, member_id="member",
                             target_install_id=hosted_rooms.local_authority_gateway_id(), target_profile="default",
                             permissions=permissions)
    claims = decode_room_grant(api._room_grant_secret(), token, permission="status")
    with open_persistence(profile_scope()) as store:
        hosted_rooms.reserve_peer_room(store, placement="shared", claims=claims, expires_at=claims["status_expires_at"])
    run = "run-granted"
    api._run_owners[run] = api_server_runs.room_run_scope(claims)
    api._run_approval_sessions[run] = run
    api._set_run_status(run, "running")
    events = asyncio.Queue()
    notify = api_server_runs._make_approval_notify(api, SimpleNamespace(run_id=run, queue=events), _api_server=api_server)
    worker = asyncio.create_task(asyncio.to_thread(_await_gateway_decision, run, notify, {"command": "echo room"}))
    try:
        request_id = (await asyncio.wait_for(events.get(), timeout=10))["request_id"]
        async with TestClient(TestServer(_create_runs_app(api))) as client:
            r = await client.get(f"/v1/runs/{run}", headers={"Authorization": f"HermesRoom {token}"})
            assert r.status == 200, await r.text()
            prompt, = (await r.json())["pending_controls"]
            assert prompt["request_id"] == request_id and prompt["choices"] == choices
    finally:
        approval.unregister_gateway_notify(run)
        await worker
