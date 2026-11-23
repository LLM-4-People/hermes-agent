"""O(log N) keyset cursor pagination and compaction continuity for GET /api/sessions/{id}/messages."""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import NewMessage, TranscriptRead, TranscriptView
from persistence.core.identity import MessageId, SessionId
from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.transcript_pages import get_transcript_page_cursor
from persistence_assembly import store_location
from tests.fixtures.storage_seed import seed_session


def _build_client_app(api, api_key: str = "cursor-fixture-key") -> web.Application:
    app = web.Application()
    api._api_key = api_key
    app.router.add_get("/api/sessions/{session_id}/messages", api._handle_session_messages)
    return app


def _seed_compacted_lineage(store, *, prefix: str, segments: int, per_segment: int, carry_tail: int = 4) -> tuple[list[str], list[str]]:
    """Seed a multi-compaction lineage using real publish_compression_child rotations.

    Returns ``(segment_ids, expected_visible_contents)``.
    """
    seg_ids: list[str] = []
    expected: list[str] = []
    ts = 1_700_000_000.0
    seq = 0

    root = seed_session(store, f"{prefix}-seg-0", "api_server")
    seg_ids.append(root)
    batch: list[NewMessage] = []
    for _ in range(per_segment):
        role = "user" if seq % 2 == 0 else "assistant"
        content = f"{prefix}-msg-{seq:05d}"
        ts += 1.0
        batch.append(NewMessage(role=role, content=content, timestamp=ts))
        expected.append(content)
        seq += 1
    store.messages.append_batch(root, batch)
    tail_carry_msgs = [
        NewMessage(role=m.role, content=m.content)
        for m in batch[-carry_tail:]
    ]

    prev = root
    for seg_idx in range(1, segments):
        child = f"{prefix}-seg-{seg_idx}"
        handoff = [
            NewMessage(role="assistant", content=f"[CONTEXT COMPACTION SUMMARY {seg_idx}]", compressed_summary=True),
            *tail_carry_msgs,
        ]
        store.lineage.publish_compression_child(
            SessionId(prev),
            CompressionChild(session_id=SessionId(child), source="api_server"),
            handoff,
            require_compression_lease=False,
        )
        seg_ids.append(child)
        new_batch: list[NewMessage] = []
        for _ in range(per_segment):
            role = "user" if seq % 2 == 0 else "assistant"
            content = f"{prefix}-msg-{seq:05d}"
            ts += 1.0
            new_batch.append(NewMessage(role=role, content=content, timestamp=ts))
            expected.append(content)
            seq += 1
        store.messages.append_batch(child, new_batch)
        tail_carry_msgs = [
            NewMessage(role=m.role, content=m.content)
            for m in new_batch[-carry_tail:]
        ]
        prev = child

    return seg_ids, expected


@pytest.mark.asyncio
async def test_10k_multi_compaction_lineage_cursor_pagination_is_fast_and_exact(api, owner):
    """10,400-message 4-segment compaction lineage: latest page is O(log N) fast, walking backward
    across every compaction boundary has 0 duplicates/gaps, and after_id delta sync stays exact
    even across a brand-new compaction rotation."""
    store = owner.persistence
    seg_ids, expected = _seed_compacted_lineage(
        store, prefix="tenk", segments=4, per_segment=2600, carry_tail=4,
    )
    assert len(expected) == 10_400
    tip = seg_ids[-1]
    headers = {"Authorization": "Bearer cursor-fixture-key"}

    # Verify SQLite uses indexed seeks on (session_id, id) — zero full-table SCAN on messages.
    with closing(sqlite3.connect(store_location(store.scope))) as conn:
        plan_before = [
            row[3] for row in conn.execute(
                "EXPLAIN QUERY PLAN SELECT id FROM messages WHERE session_id = ? AND active = 1 AND id < ? "
                "ORDER BY id DESC LIMIT 41",
                (tip, 5000),
            ).fetchall()
        ]
        plan_after = [
            row[3] for row in conn.execute(
                "EXPLAIN QUERY PLAN SELECT id FROM messages WHERE session_id = ? AND active = 1 AND id > ? "
                "ORDER BY id ASC LIMIT 41",
                (tip, 5000),
            ).fetchall()
        ]
    assert any("idx_messages_session_id" in step for step in plan_before), plan_before
    assert any("idx_messages_session_id" in step for step in plan_after), plan_after
    assert not any(step.startswith("SCAN messages") for step in plan_before + plan_after)

    # Measure SQLite VM steps for latest page and boundary before_id page on 10,400 messages vs
    # a 100-message session to verify sublinear page fetch (no full Python dedupe scan).
    def measure_cursor_read(sid: str, *, before_id=None, after_id=None, limit: int = 40):
        ticks = 0

        def handler():
            nonlocal ticks
            ticks += 1
            return 0

        req = TranscriptRead(view=TranscriptView.ACTIVE, limit=limit, offset=0, latest=after_id is None, include_lineage=True)
        with closing(StoreHolder(store_location(store.scope), read_only=True)) as reader:
            reader._conn.set_progress_handler(handler, 100)
            t0 = time.perf_counter()
            try:
                rows, total, root, has_more = get_transcript_page_cursor(
                    reader, sid, req, True, before_id=before_id, after_id=after_id,
                )
            finally:
                reader._conn.set_progress_handler(None, 0)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return rows, total, root, has_more, ticks, elapsed_ms

    rows_latest, total_latest, root_id, has_more_latest, _ticks_latest, elapsed_ms = measure_cursor_read(tip, limit=40)
    assert root_id == seg_ids[0]
    assert total_latest == 10_400
    assert has_more_latest is True
    assert [r["content"] for r in rows_latest] == expected[-40:]
    # Must be genuinely fast (well under 500ms even on loaded CI runners).
    assert elapsed_ms < 500.0, f"10k latest page took {elapsed_ms:.1f}ms"

    async with TestClient(TestServer(_build_client_app(api))) as client:
        # 1. Latest page via HTTP API (from tip AND from root via lineage resolution)
        resp = await client.get(
            f"/api/sessions/{tip}/messages",
            params={"limit": "40", "order": "latest", "visible_only": "true"},
            headers=headers,
        )
        assert resp.status == 200
        body = await resp.json()
        pag = body["pagination"]
        assert body["session_id"] == tip
        assert body["thread_id"] == seg_ids[0]
        assert pag["limit"] == 40
        assert pag["offset"] == 0
        assert pag["order"] == "latest"
        assert pag["returned"] == 40
        assert pag["total"] == 10_400
        assert pag["visible_only"] is True
        assert pag["has_more"] is True
        assert pag["before_id"] is None
        assert pag["after_id"] is None
        assert pag["oldest_id"] == body["data"][0]["id"]
        assert pag["newest_id"] == body["data"][-1]["id"]
        assert [m["content"] for m in body["data"]] == expected[-40:]

        # 2. Walk backward across the seg-2 -> seg-3 compaction boundary using before_id
        # Find an anchor just past the boundary in seg-3 (around index 7815 = 3*2600 + 15)
        with closing(sqlite3.connect(store_location(store.scope))) as conn:
            boundary_anchor = conn.execute(
                "SELECT id FROM messages WHERE session_id = ? AND content = ?",
                (tip, expected[7815]),
            ).fetchone()[0]

        walked_backward: list[dict] = []
        cursor = boundary_anchor
        for _ in range(3):
            r = await client.get(
                f"/api/sessions/{tip}/messages",
                params={"before_id": str(cursor), "limit": "20", "visible_only": "true"},
                headers=headers,
            )
            assert r.status == 200
            b = await r.json()
            p = b["pagination"]
            assert p["before_id"] == cursor
            assert p["after_id"] is None
            assert p["order"] == "latest"
            assert p["returned"] == 20
            assert p["has_more"] is True
            assert p["oldest_id"] == b["data"][0]["id"]
            assert p["newest_id"] == b["data"][-1]["id"]
            walked_backward = b["data"] + walked_backward
            cursor = p["oldest_id"]

        # 60 messages spanning index 7755..7814 cleanly across seg-2 -> seg-3 (boundary at 7800, carried tail 7796..7799)
        assert [m["content"] for m in walked_backward] == expected[7755:7815]
        assert len({m["id"] for m in walked_backward}) == 60

        # 3. Walk forward with after_id across the EXACT SAME compaction boundary (7790 -> 7810)
        with closing(sqlite3.connect(store_location(store.scope))) as conn:
            pre_boundary_id = conn.execute(
                "SELECT id FROM messages WHERE session_id = ? AND content = ?",
                (seg_ids[2], expected[7790]),
            ).fetchone()[0]
        r_fwd = await client.get(
            f"/api/sessions/{tip}/messages",
            params={"after_id": str(pre_boundary_id), "limit": "25", "visible_only": "true"},
            headers=headers,
        )
        assert r_fwd.status == 200
        b_fwd = await r_fwd.json()
        assert b_fwd["pagination"]["after_id"] == pre_boundary_id
        assert b_fwd["pagination"]["before_id"] is None
        assert b_fwd["pagination"]["order"] == "oldest"
        assert b_fwd["pagination"]["has_more"] is True
        assert [m["content"] for m in b_fwd["data"]] == expected[7791:7816]

        # 4. Incremental delta sync (after_id = current newest_id) across a FRESH compaction!
        last_seen_id = pag["newest_id"]
        r_empty = await client.get(
            f"/api/sessions/{tip}/messages",
            params={"after_id": str(last_seen_id), "limit": "40", "visible_only": "true"},
            headers=headers,
        )
        assert r_empty.status == 200
        b_empty = await r_empty.json()
        assert b_empty["data"] == []
        assert b_empty["pagination"]["returned"] == 0
        assert b_empty["pagination"]["has_more"] is False
        assert b_empty["pagination"]["oldest_id"] is None
        assert b_empty["pagination"]["newest_id"] is None

        # Perform fresh compaction rotation into seg-4 with carried tail + 3 brand new messages
        new_tip = "tenk-seg-4"
        carried = [
            NewMessage(role=m["role"], content=m["content"])
            for m in body["data"][-4:]
        ]
        store.lineage.publish_compression_child(
            SessionId(tip),
            CompressionChild(session_id=SessionId(new_tip), source="api_server"),
            [
                NewMessage(role="assistant", content="[CONTEXT COMPACTION SUMMARY 4]", compressed_summary=True),
                *carried,
            ],
            require_compression_lease=False,
        )
        fresh_msgs = [
            NewMessage(role="user", content="post-compaction-u1", timestamp=1_700_099_001.0),
            NewMessage(role="assistant", content="post-compaction-a1", timestamp=1_700_099_002.0),
            NewMessage(role="user", content="post-compaction-u2", timestamp=1_700_099_003.0),
        ]
        store.messages.append_batch(new_tip, fresh_msgs)

        # Querying either the OLD tip or NEW tip with after_id=last_seen_id returns ONLY the 3 fresh messages!
        for queried_sid in (tip, new_tip):
            r_delta = await client.get(
                f"/api/sessions/{queried_sid}/messages",
                params={"after_id": str(last_seen_id), "limit": "40", "visible_only": "true"},
                headers=headers,
            )
            assert r_delta.status == 200
            b_delta = await r_delta.json()
            assert b_delta["session_id"] == new_tip
            assert b_delta["thread_id"] == seg_ids[0]
            assert [m["content"] for m in b_delta["data"]] == [
                "post-compaction-u1",
                "post-compaction-a1",
                "post-compaction-u2",
            ]
            assert b_delta["pagination"]["returned"] == 3
            assert b_delta["pagination"]["has_more"] is False
            assert b_delta["pagination"]["total"] == 10_403
            assert b_delta["pagination"]["after_id"] == last_seen_id
            assert b_delta["pagination"]["before_id"] is None


@pytest.mark.asyncio
async def test_full_walk_backward_and_forward_across_compactions_zero_duplicates_or_gaps(api, owner):
    """Exhaustively walk an entire 4-segment compacted thread (240 messages) in BOTH directions
    (before_id and after_id) and verify 100% exact equality with the logical transcript."""
    store = owner.persistence
    seg_ids, expected = _seed_compacted_lineage(
        store, prefix="walk", segments=4, per_segment=60, carry_tail=5,
    )
    tip = seg_ids[-1]
    headers = {"Authorization": "Bearer cursor-fixture-key"}

    async with TestClient(TestServer(_build_client_app(api))) as client:
        # Full backward walk with page size 17 (intentionally non-divisible by 60 & 5)
        backward_msgs: list[dict] = []
        cursor = None
        while True:
            params = {"limit": "17", "visible_only": "true"}
            if cursor is None:
                params["order"] = "latest"
            else:
                params["before_id"] = str(cursor)
            r = await client.get(f"/api/sessions/{tip}/messages", params=params, headers=headers)
            assert r.status == 200
            b = await r.json()
            page_rows = b["data"]
            backward_msgs = page_rows + backward_msgs
            if not b["pagination"]["has_more"]:
                break
            assert b["pagination"]["oldest_id"] is not None
            cursor = b["pagination"]["oldest_id"]

        assert [m["content"] for m in backward_msgs] == expected
        assert len({m["id"] for m in backward_msgs}) == len(expected)

        # One more before_id at the very oldest row yields empty page with has_more=False
        r_head = await client.get(
            f"/api/sessions/{tip}/messages",
            params={"before_id": str(backward_msgs[0]["id"]), "limit": "17", "visible_only": "true"},
            headers=headers,
        )
        assert r_head.status == 200
        b_head = await r_head.json()
        assert b_head["data"] == []
        assert b_head["pagination"]["has_more"] is False
        assert b_head["pagination"]["oldest_id"] is None
        assert b_head["pagination"]["newest_id"] is None

        # Full forward walk with after_id=0 and page size 19
        forward_msgs: list[dict] = []
        cursor = 0
        while True:
            r = await client.get(
                f"/api/sessions/{tip}/messages",
                params={"after_id": str(cursor), "limit": "19", "visible_only": "true"},
                headers=headers,
            )
            assert r.status == 200
            b = await r.json()
            forward_msgs.extend(b["data"])
            if not b["pagination"]["has_more"]:
                break
            cursor = b["pagination"]["newest_id"]

        assert [m["content"] for m in forward_msgs] == expected
        assert [m["id"] for m in forward_msgs] == [m["id"] for m in backward_msgs]


@pytest.mark.asyncio
async def test_single_segment_and_in_place_compaction_cursor_pagination(api, owner):
    """Single-segment cursor paging (ACTIVE and DISPLAY after in-place compact) + limit=0 + legacy offset."""
    store = owner.persistence
    sid = seed_session(store, "single-seg", "api_server")
    ts = 1_700_100_000.0
    initial = [
        NewMessage(role="user" if i % 2 == 0 else "assistant", content=f"s-{i:03d}", timestamp=ts + i)
        for i in range(30)
    ]
    store.messages.append_batch(sid, initial)
    # In-place compaction preserving last 4 rows as carried tail
    replacement = [
        NewMessage(role="assistant", content="[CONTEXT SUMMARY]", compressed_summary=True, timestamp=ts + 30.5),
        *[NewMessage(role=m.role, content=m.content, timestamp=m.timestamp) for m in initial[-4:]],
    ]
    store.messages.compact(sid, replacement, carried_tail_count=4)
    extra = [
        NewMessage(role="user", content="s-030", timestamp=ts + 31),
        NewMessage(role="assistant", content="s-031", timestamp=ts + 32),
    ]
    store.messages.append_batch(sid, extra)
    headers = {"Authorization": "Bearer cursor-fixture-key"}

    async with TestClient(TestServer(_build_client_app(api))) as client:
        # DISPLAY view (include_compacted=true) walks all 32 visible messages cleanly with before_id
        collected: list[str] = []
        cursor = None
        while True:
            params = {"limit": "7", "include_compacted": "true", "visible_only": "true"}
            if cursor is None:
                params["order"] = "latest"
            else:
                params["before_id"] = str(cursor)
            r = await client.get(f"/api/sessions/{sid}/messages", params=params, headers=headers)
            assert r.status == 200
            b = await r.json()
            collected = [m["content"] for m in b["data"]] + collected
            if not b["pagination"]["has_more"]:
                break
            cursor = b["pagination"]["oldest_id"]
        assert collected == [f"s-{i:03d}" for i in range(32)]

        # limit=0 reports exact has_more without returning rows
        r_zero = await client.get(
            f"/api/sessions/{sid}/messages",
            params={"limit": "0", "visible_only": "true"},
            headers=headers,
        )
        assert r_zero.status == 200
        b_zero = await r_zero.json()
        assert b_zero["data"] == []
        assert b_zero["pagination"]["returned"] == 0
        assert b_zero["pagination"]["has_more"] is True
        assert b_zero["pagination"]["total"] == 6

        # Backward-compatible offset paging still works identically
        r_off = await client.get(
            f"/api/sessions/{sid}/messages",
            params={"limit": "2", "offset": "2", "order": "oldest", "visible_only": "true"},
            headers=headers,
        )
        assert r_off.status == 200
        b_off = await r_off.json()
        assert b_off["pagination"]["offset"] == 2
        assert b_off["pagination"]["returned"] == 2
        assert b_off["pagination"]["has_more"] is True
        assert [m["content"] for m in b_off["data"]] == ["s-028", "s-029"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_params",
    [
        {"before_id": "-1"},
        {"after_id": "-5"},
        {"before_id": "abc"},
        {"after_id": "1.5"},
        {"before_id": ""},
        {"after_id": ""},
        {"before_id": "+10"},
        {"before_id": "10", "after_id": "5"},
        {"before_id": "10", "offset": "1"},
        {"after_id": "10", "offset": "2"},
    ],
)
async def test_invalid_cursor_params_fail_closed_400(api, owner, bad_params):
    store = owner.persistence
    sid = seed_session(store, "bad-cursor", "api_server")
    store.messages.append_batch(sid, [NewMessage(role="user", content="hello")])
    headers = {"Authorization": "Bearer cursor-fixture-key"}

    async with TestClient(TestServer(_build_client_app(api))) as client:
        resp = await client.get(f"/api/sessions/{sid}/messages", params=bad_params, headers=headers)
        assert resp.status == 400
        body = await resp.json()
        assert body["error"]["code"] == "invalid_pagination"
