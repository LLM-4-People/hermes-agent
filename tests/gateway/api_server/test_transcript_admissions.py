"""Match identical API turns to their own visible human rows through the real transcript endpoint."""
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from tests.fixtures.storage_seed import seed_admission, seed_messages, seed_session


@pytest.mark.asyncio
async def test_visible_input_admission_requires_matching_session_and_origin(api, owner):
    store = owner.persistence
    sid = seed_session(store, "guide-thread", "api_server")
    other = seed_session(store, "other-thread", "api_server")
    local = seed_session(store, "terminal-thread", "cli")

    def receipt(session, key, principal="api"):
        return seed_admission(store, session, key, "terminal", "completed", principal_id=principal)

    first, second = receipt(sid, "first"), receipt(sid, "second")
    other_turn, wrong_source = receipt(other, "other"), receipt(local, "also-api")
    different_actor = receipt(sid, "local-actor", "interactive")

    def message(role, inbound=None, **fields):
        return {"role": role, "content": "Identical text.", "platform_message_id": inbound, **fields}

    seed_messages(store, sid, message("user", first), message("assistant", first),
                  message("user", second), message("assistant"), message("user", other_turn),
                  message("user", different_actor), message("user", "ordinary-platform-id"),
                  message("user", second, observed=True), message("user"),
                  message("user", first, display_kind="hidden"))
    seed_messages(store, local, message("user", wrong_source))
    app = web.Application()
    api._api_key = "fixture-key"
    app.router.add_get("/api/sessions/{session_id}/messages", api._handle_session_messages)
    async with TestClient(TestServer(app)) as client:
        headers = {"Authorization": "Bearer fixture-key"}
        path = f"/api/sessions/{sid}/messages"
        assert (await client.get(path + "?visible_only=true")).status == 401
        async def read(path, query):
            response = await client.get(path, params=query, headers=headers)
            assert response.status == 200
            return (await response.json())["data"]

        visible = await read(path, {"visible_only": "true", "order": "oldest"})
        assert [row.get("api_admission_id") for row in visible] == [first, None, second] + [None] * 6
        assert all("api_admission_id" not in row for row in visible if row["role"] == "assistant")
        assert all("platform_message_id" not in row for row in visible)
        assert (await read(path, {"visible_only": "true", "limit": 1, "offset": 2}))[0] == visible[2]
        raw = await read(path, {})
        assert len(raw) == len(visible) + 1 and raw[-1]["display_kind"] == "hidden"
        assert all("api_admission_id" not in row for row in raw)
        foreign = await read(f"/api/sessions/{local}/messages", {"visible_only": "true"})
        assert "api_admission_id" not in foreign[0]


def test_read_only_profile_from_before_the_admission_ledger(tmp_path):
    import sqlite3
    from contextlib import closing
    from persistence.conversation.messages import TranscriptRead
    from persistence.sqlite._holder import StoreHolder
    from persistence.sqlite.conversation._store.messages import append_messages_batch
    from persistence.sqlite.conversation._store.sessions import create_session
    from persistence.sqlite.conversation._store.transcript_pages import get_transcript_page

    path = tmp_path / "legacy.db"
    with closing(StoreHolder(path)) as writer:
        create_session(writer, "thread", source="api_server")
        append_messages_batch(writer, "thread", [{"role": "user", "content": "Historic message."}])
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("DROP TABLE session_admissions")
        conn.commit()
    with closing(StoreHolder(path, read_only=True)) as reader:
        rows, total, _ = get_transcript_page(reader, "thread", TranscriptRead(), visible_only=True)
        assert total == 1 and rows[0]["content"] == "Historic message."
        assert "api_admission_id" not in rows[0]
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='session_admissions'").fetchone() is None
