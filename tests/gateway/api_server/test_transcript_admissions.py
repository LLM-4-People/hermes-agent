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


def test_page_resolves_a_repeated_marker_independently_of_page_length(owner):
    import sqlite3
    from contextlib import closing

    from persistence.conversation.messages import NewMessage, TranscriptRead
    from persistence.sqlite._holder import StoreHolder
    from persistence.sqlite.conversation._store.transcript_pages import get_transcript_page
    from persistence_assembly import store_location

    store = owner.persistence
    sid = seed_session(store, "rewound-copies", "api_server")
    marker = seed_admission(store, sid, "request", "terminal", "completed", principal_id="api")
    def generations(prefix, count):
        return [NewMessage(role="user", content=f"{prefix} {i}", platform_message_id=marker) for i in range(count)]

    store.messages.append_batch(sid, generations("Historic", 1400))
    rewind = store.messages.rewind_user_turn(sid, 0)
    assert rewind.rewound_count == 1400
    store.messages.append_batch(sid, generations("Current", 70))

    def page_work(limit):
        ticks = 0

        def count_steps():
            nonlocal ticks
            ticks += 1
            return 0

        with closing(StoreHolder(store_location(store.scope), read_only=True)) as reader:
            reader._conn.set_progress_handler(count_steps, 100)
            try:
                rows, total, _ = get_transcript_page(reader, sid, TranscriptRead(limit=limit), visible_only=True)
            finally:
                reader._conn.set_progress_handler(None, 0)
        assert total == 70 and len(rows) == limit
        assert all(row.get("api_admission_id") is None for row in rows)
        return ticks

    # Both pages traverse the same old generations for their single ambiguous marker.
    wide = page_work(60)
    assert wide < 4 * page_work(1)

    # Adding another session's ledger entries must not change that scan into a ledger walk.
    other = seed_session(store, "other-session", "api_server")
    with closing(sqlite3.connect(store_location(store.scope))) as conn:
        conn.executemany(
            """INSERT INTO session_admissions (admission_id, request_id, principal_id, target_session_id,
                   lineage_json, payload_json, payload_digest, intent, status, generation)
               VALUES (?, ?, 'api', ?, '[]', '{}', 'test', 'queue', 'terminal', 1)""",
            ((f"unrelated-{i}", str(i), other) for i in range(4000)),
        )
        conn.commit()
    assert page_work(60) < 1.5 * wide


def test_rewound_copies_do_not_obscure_a_single_active_admission(owner):
    from persistence.conversation.messages import NewMessage, TranscriptRead, TranscriptView

    store = owner.persistence
    sid = seed_session(store, "rewound-then-active", "api_server")
    marker = seed_admission(store, sid, "request", "terminal", "completed", principal_id="api")
    store.messages.append_batch(sid, [NewMessage(role="user", content=f"Old {i}", platform_message_id=marker) for i in range(5)])
    assert store.messages.rewind_user_turn(sid, 0).rewound_count == 5
    store.messages.append_batch(sid, [NewMessage(role="user", content="Live", platform_message_id=marker)])

    visible = store.messages.read_transcript_page(sid, TranscriptRead(), visible_only=True)
    assert visible.total == 1 and visible.rows[0].api_admission_id == marker
    with_history = store.messages.read_transcript_page(sid, TranscriptRead(view=TranscriptView.ALL), visible_only=True)
    assert [row.api_admission_id for row in with_history.rows] == [None] * 5 + [marker]
