"""Count and page the client transcript without loading message payloads outside the requested page."""

from __future__ import annotations

import json
from typing import Optional

from persistence.conversation.messages import TranscriptView
from session_rules.turns import transcript_message_visible
from .messages import (
    _active_clause, _decode_content, _decode_display_metadata, _dedupe_display_generations, _display_dedupe_key, _display_identity,
    _legacy_display_ids, _message_column_names, _resume_lineage_ids, _row_to_message_dict,
)
from .messages_sql import DISPLAY_VISIBLE_SQL, _DISPLAY_ACTIVE_CLAUSE, _DISPLAY_INDEX_MISSING_SQL
from .sql_common import _in_one_read
from .timeline import _indexed_display_rows_sql


def _visible(role, kind, metadata, summary, content):
    return transcript_message_visible({
        "role": role, "display_kind": kind, "display_metadata": _decode_display_metadata(metadata),
        "_compressed_summary": bool(summary), "content": _decode_content(content),
    })


# Ordinary rows need only their role and display kind. Check the shared projection for flagged or
# legacy content-marked carriers. JSON escape sequences can encode a marker in a structured row.
_VISIBLE = """CASE WHEN role = 'session_meta' THEN 0
    WHEN _compressed_summary OR instr(content, '[CONTEXT') OR instr(content, '\\u')
    THEN transcript_visible(role, display_kind, display_metadata, _compressed_summary, content)
    ELSE display_kind IS NOT 'hidden' END"""


def _placeholders(count: int) -> str:
    return ",".join("?" for _ in range(count))


def _validate_cursors(request, before_id: Optional[int], after_id: Optional[int]) -> tuple[Optional[int], Optional[int]]:
    eff_after = after_id if after_id is not None else (None if request.after_id is None else int(request.after_id))
    eff_before = None if before_id is None else int(before_id)
    if eff_before is not None and eff_after is not None:
        raise ValueError("before_id and after_id are mutually exclusive")
    if (eff_before is not None or eff_after is not None) and request.offset:
        raise ValueError("before_id and after_id are incompatible with non-zero offset")
    if (eff_before is not None and eff_before < 0) or (eff_after is not None and eff_after < 0):
        raise ValueError("cursor ids must be non-negative")
    return eff_before, eff_after


def _resolve_segment_display_anchor(conn, sid: str, cursor_id: int) -> int:
    row = conn.execute(
        "SELECT display_order FROM messages WHERE session_id = ? AND id = ? LIMIT 1",
        (sid, cursor_id),
    ).fetchone()
    if row is not None and row[0] is not None:
        return int(row[0])
    return cursor_id


def _segment_cursor_page(holder, conn, sid, request, visible_only, before_id: Optional[int], after_id: Optional[int]):
    """O(log N + limit) indexed keyset page for a single session segment."""
    probe_limit = -1 if request.limit is None else (request.limit + 1)
    vis_clause = f" AND ({_VISIBLE})" if visible_only else ""

    if request.view is TranscriptView.DISPLAY:
        columns = _message_column_names(holder, conn)
        if not {"display_order", "display_identity"}.issubset(columns) or conn.execute(_DISPLAY_INDEX_MISSING_SQL, (sid,)).fetchone():
            legacy = _legacy_display_ids(holder, conn, sid, _DISPLAY_ACTIVE_CLAUSE)
            if visible_only:
                rows = conn.execute(
                    f"SELECT value FROM json_each(?) p JOIN messages m ON m.id = p.value WHERE {_VISIBLE} ORDER BY p.key",
                    (json.dumps(legacy),),
                ).fetchall()
                all_ids = [int(r[0]) for r in rows]
            else:
                all_ids = [int(i) for i in legacy]
            return _slice_id_list(all_ids, request, before_id, after_id)

        source = _indexed_display_rows_sql()
        base_params: dict[str, object] = {"sid": sid}
        count_sql = source + " SELECT COUNT(*) FROM display_rows"
        if visible_only:
            count_sql += f" JOIN messages m ON m.id = row_id WHERE {_VISIBLE}"
        total = int(conn.execute(count_sql, base_params).fetchone()[0])

        cursor_id = before_id if before_id is not None else after_id
        anchor_sort = None if cursor_id is None else _resolve_segment_display_anchor(conn, sid, cursor_id)
        if before_id is not None:
            cond, direction = "sort_id < :anchor", "DESC"
            base_params["anchor"] = anchor_sort
        elif after_id is not None:
            cond, direction = "sort_id > :anchor", "ASC"
            base_params["anchor"] = anchor_sort
        else:
            cond, direction = "1=1", ("DESC" if request.latest else "ASC")

        base_params["probe_limit"] = probe_limit
        if visible_only:
            page_sql = (
                f"{source} SELECT row_id FROM display_rows JOIN messages m ON m.id = row_id "
                f"WHERE {cond} AND ({_VISIBLE}) ORDER BY sort_id {direction} LIMIT :probe_limit"
            )
        else:
            page_sql = (
                f"{source} SELECT row_id FROM display_rows "
                f"WHERE {cond} ORDER BY sort_id {direction} LIMIT :probe_limit"
            )
        fetched = [int(r[0]) for r in conn.execute(page_sql, base_params).fetchall()]
    else:
        active = " AND active = 1" if request.view is TranscriptView.ACTIVE else ""
        total = int(conn.execute(
            f"SELECT COUNT(*) FROM messages WHERE session_id = :sid{active}{vis_clause}",
            {"sid": sid},
        ).fetchone()[0])
        params: dict[str, object] = {"sid": sid, "probe_limit": probe_limit}
        if before_id is not None:
            key_clause, direction = " AND id < :cursor", "DESC"
            params["cursor"] = before_id
        elif after_id is not None:
            key_clause, direction = " AND id > :cursor", "ASC"
            params["cursor"] = after_id
        else:
            key_clause, direction = "", ("DESC" if request.latest else "ASC")
        # Uses idx_messages_session_id (session_id, id) for O(log N + limit) range seek.
        fetched = [int(r[0]) for r in conn.execute(
            f"SELECT id FROM messages WHERE session_id = :sid{active}{key_clause}{vis_clause} "
            f"ORDER BY id {direction} LIMIT :probe_limit",
            params,
        ).fetchall()]

    if request.limit is None:
        has_more = False
        ids = fetched
    elif request.limit == 0:
        has_more = bool(fetched)
        ids = []
    else:
        has_more = len(fetched) > request.limit
        ids = fetched[:request.limit]
    if before_id is not None or (after_id is None and request.latest):
        ids = ids[::-1]
    return ids, total, has_more


def _segment_page(holder, conn, sid, request, visible_only, before_id: Optional[int] = None, after_id: Optional[int] = None):
    if before_id is not None or after_id is not None or (request.latest and request.offset == 0):
        return _segment_cursor_page(holder, conn, sid, request, visible_only, before_id, after_id)

    params = {"sid": sid, "after": request.after_id, "limit": request.limit, "offset": request.offset}
    if request.view is TranscriptView.DISPLAY:
        columns = _message_column_names(holder, conn)
        if not {"display_order", "display_identity"}.issubset(columns) or conn.execute(_DISPLAY_INDEX_MISSING_SQL, (sid,)).fetchone():
            params["legacy"] = json.dumps(_legacy_display_ids(holder, conn, sid, _DISPLAY_ACTIVE_CLAUSE))
            source = "WITH display_rows AS (SELECT value AS row_id, key AS sort_id FROM json_each(:legacy))"
        else:
            source = _indexed_display_rows_sql()
    else:
        active = " AND active = 1" if request.view is TranscriptView.ACTIVE else ""
        after = " AND id > :after" if request.after_id is not None else ""
        source = f"WITH display_rows AS (SELECT id AS row_id, id AS sort_id FROM messages WHERE session_id = :sid{active}{after})"
    query = source + " SELECT row_id, sort_id FROM display_rows"
    if visible_only:
        query += f" JOIN messages m ON m.id = row_id WHERE {_VISIBLE}"
    direction = "DESC" if request.latest else "ASC"
    if visible_only:
        # Materialize just the candidate IDs once so the count and page share the visibility scan,
        # even when the requested offset lies past the end.
        sql = f"""WITH eligible AS MATERIALIZED ({query}), page AS (
                      SELECT * FROM eligible ORDER BY sort_id {direction} LIMIT COALESCE(:limit, -1) OFFSET :offset)
                  SELECT p.row_id, t.total FROM (SELECT COUNT(*) AS total FROM eligible) t
                  LEFT JOIN page p ON 1 ORDER BY p.sort_id"""
        rows = conn.execute(sql, params).fetchall()
        ids = [r[0] for r in rows if r[0] is not None]
        total = int(rows[0][1])
    else:
        total = int(conn.execute("SELECT COUNT(*) FROM (" + query + ")", params).fetchone()[0])
        selected = conn.execute(query + f" ORDER BY sort_id {direction} LIMIT COALESCE(:limit, -1) OFFSET :offset", params)
        ids = [row[0] for row in selected]
        if request.latest:
            ids = ids[::-1]
    has_more = False if request.limit is None else (request.offset + len(ids)) < total
    return ids, total, has_more


def _slice_id_list(ids: list[int], request, before_id: Optional[int], after_id: Optional[int],
                   first_id_by_row: Optional[dict[int, int]] = None, anchor_first_id: Optional[int] = None):
    total = len(ids)
    if before_id is not None:
        cutoff = anchor_first_id if anchor_first_id is not None else before_id
        if first_id_by_row is not None:
            filtered = [i for i in ids if first_id_by_row.get(i, i) < cutoff]
        else:
            filtered = [i for i in ids if i < cutoff]
        if request.limit is None:
            return filtered, total, False
        if request.limit == 0:
            return [], total, bool(filtered)
        has_more = len(filtered) > request.limit
        return filtered[-request.limit:], total, has_more
    if after_id is not None:
        cutoff = anchor_first_id if anchor_first_id is not None else after_id
        if first_id_by_row is not None:
            filtered = [i for i in ids if first_id_by_row.get(i, i) > cutoff]
        else:
            filtered = [i for i in ids if i > cutoff]
        if request.limit is None:
            return filtered, total, False
        if request.limit == 0:
            return [], total, bool(filtered)
        has_more = len(filtered) > request.limit
        return filtered[:request.limit], total, has_more
    if request.latest:
        end = max(0, total - request.offset)
        start = 0 if request.limit is None else max(0, end - request.limit)
        selected = ids[start:end]
        has_more = False if request.limit is None else start > 0
        return selected, total, has_more
    end = None if request.limit is None else request.offset + request.limit
    selected = ids[request.offset:end]
    has_more = False if request.limit is None else (request.offset + len(selected)) < total
    return selected, total, has_more


def _identity_rows(holder, cursor):
    """Project each legacy key before retaining it. A summary may have an identity folded within a different active set."""
    for row in cursor:
        if row["display_identity"] is None or row["_compressed_summary"]:
            identity = _display_identity(_display_dedupe_key(holder, row))
            row = dict(row)
            row.update(display_identity=identity, content=None, tool_calls=None)
        yield row


def _lineage_fast_eligible(holder, conn, segments: list[str], request) -> bool:
    """True when all segments have indexed display_identity/display_order and no edge cases require full Python dedupe."""
    columns = _message_column_names(holder, conn)
    if not {"display_order", "display_identity"}.issubset(columns):
        return False
    ph = _placeholders(len(segments))
    # Uses idx_messages_display_backfill on (session_id) WHERE display_order IS NULL OR display_identity IS NULL.
    if conn.execute(
        f"SELECT 1 FROM messages WHERE session_id IN ({ph}) AND (active = 1 OR compacted = 1) "
        f"AND (display_order IS NULL OR display_identity IS NULL) LIMIT 1",
        segments,
    ).fetchone():
        return False
    if request.view is TranscriptView.DISPLAY:
        # Removed rows or compacted tool copies can hide ancestor/later copies via _dedupe_display_generations.
        if conn.execute(
            f"SELECT 1 FROM messages WHERE session_id IN ({ph}) AND active = 0 AND compacted = 0 LIMIT 1",
            segments,
        ).fetchone():
            return False
        if conn.execute(
            f"SELECT 1 FROM messages WHERE session_id IN ({ph}) AND compacted = 1 AND role = 'tool' LIMIT 1",
            segments,
        ).fetchone():
            return False
    return True


def _resolve_lineage_anchor_first_id(conn, segments: list[str], cursor_id: int, *, active_sql: str, use_display_order: bool) -> tuple[int, Optional[int]]:
    """Resolve the logical first_id of cursor_id across compaction segments in O(k log N)."""
    seg_index = {sid: idx for idx, sid in enumerate(segments)}
    row = conn.execute(
        "SELECT id, session_id, display_order, display_identity FROM messages WHERE id = ?",
        (cursor_id,),
    ).fetchone()
    if row is None or row["session_id"] not in seg_index:
        return cursor_id, None
    first_id = (
        int(row["display_order"])
        if use_display_order and row["display_order"] is not None
        else int(row["id"])
    )
    ident = row["display_identity"]
    if ident:
        ph = _placeholders(len(segments))
        order_expr = "COALESCE(display_order, id)" if use_display_order else "id"
        found = conn.execute(
            f"SELECT MIN({order_expr}) FROM messages "
            f"WHERE session_id IN ({ph}) AND display_identity = ?{active_sql}",
            (*segments, ident),
        ).fetchone()[0]
        if found is not None and int(found) < first_id:
            first_id = int(found)
    return first_id, seg_index[row["session_id"]]


def _earlier_first_ids(conn, older_segments: list[str], identities: list[object], *, active_sql: str, use_display_order: bool) -> dict[object, int]:
    if not older_segments or not identities:
        return {}
    s_ph = _placeholders(len(older_segments))
    i_ph = _placeholders(len(identities))
    order_expr = "COALESCE(display_order, id)" if use_display_order else "id"
    rows = conn.execute(
        f"SELECT display_identity, MIN({order_expr}) FROM messages "
        f"WHERE session_id IN ({s_ph}) AND display_identity IN ({i_ph}){active_sql} "
        f"GROUP BY display_identity",
        (*older_segments, *identities),
    ).fetchall()
    return {r[0]: int(r[1]) for r in rows if r[0] is not None and r[1] is not None}


def _winning_representatives(conn, target_segments: list[str], identities: list[object], active_sql: str, shown_sql: str) -> dict[object, tuple[int, bool]]:
    if not target_segments or not identities:
        return {}
    s_ph = _placeholders(len(target_segments))
    i_ph = _placeholders(len(identities))
    rows = conn.execute(
        f"SELECT display_identity, id, ({shown_sql}) AS shown FROM messages "
        f"WHERE session_id IN ({s_ph}) AND display_identity IN ({i_ph}){active_sql}{DISPLAY_VISIBLE_SQL} "
        f"ORDER BY active DESC, id DESC",
        (*target_segments, *identities),
    ).fetchall()
    winners: dict[object, tuple[int, bool]] = {}
    for ident, row_id, shown in rows:
        if ident is not None and ident not in winners:
            winners[ident] = (int(row_id), bool(shown))
    return winners


def _collect_lineage_batch(
    conn,
    *,
    sid: str,
    older_segs: list[str],
    newer_segs: list[str],
    batch: list[tuple],
    active_sql: str,
    shown_sql: str,
    display: bool,
    backward: bool,
    anchor_first_id: Optional[int],
    emitted_identities: set[object],
    emitted_ids: set[int],
    collected: list[tuple[int, int]],
) -> bool:
    """Resolve one segment batch into `(first_id, winning_row_id)` entries; returns True if any carried copy appeared."""
    batch_idents = list(dict.fromkeys(r[2] for r in batch if r[2] is not None))
    earlier_map = (
        _earlier_first_ids(conn, older_segs, batch_idents, active_sql=active_sql, use_display_order=display)
        if older_segs else {}
    )
    newer_map = _winning_representatives(conn, newer_segs, batch_idents, active_sql, shown_sql) if newer_segs else {}
    in_seg_winners = _winning_representatives(conn, [sid], batch_idents, active_sql, shown_sql) if display else {}

    saw_carried_copy = False
    for row_id, display_order, ident, shown in batch:
        row_id = int(row_id)
        ident_key: object = ident if ident is not None else ("id", row_id)
        if ident_key in emitted_identities:
            continue
        local_first = int(display_order) if display and display_order is not None else row_id
        first_id = min(local_first, earlier_map.get(ident_key, local_first))
        if first_id < row_id:
            saw_carried_copy = True
        if anchor_first_id is not None and (first_id >= anchor_first_id if backward else first_id <= anchor_first_id):
            continue

        if ident_key in newer_map:
            win_id, win_shown = newer_map[ident_key]
        elif display and ident_key in in_seg_winners:
            win_id, win_shown = in_seg_winners[ident_key]
        else:
            win_id, win_shown = row_id, bool(shown)

        emitted_identities.add(ident_key)
        if win_shown and win_id not in emitted_ids:
            emitted_ids.add(win_id)
            collected.append((first_id, win_id))
    return saw_carried_copy


def _fetch_segment_batch(conn, sid: str, *, active_sql: str, shown_sql: str,
                         backward: bool, seg_cursor: Optional[int], batch_size: int) -> list[tuple]:
    if backward and seg_cursor is None:
        return conn.execute(
            f"SELECT id, display_order, display_identity, ({shown_sql}) AS shown "
            f"FROM messages WHERE session_id = ?{active_sql}{DISPLAY_VISIBLE_SQL} "
            f"ORDER BY id DESC LIMIT ?",
            (sid, batch_size),
        ).fetchall()
    op, direction = ("<", "DESC") if backward else (">", "ASC")
    return conn.execute(
        f"SELECT id, display_order, display_identity, ({shown_sql}) AS shown "
        f"FROM messages WHERE session_id = ?{active_sql}{DISPLAY_VISIBLE_SQL} AND id {op} ? "
        f"ORDER BY id {direction} LIMIT ?",
        (sid, seg_cursor, batch_size),
    ).fetchall()


def _initial_segment_cursor(conn, sid: str, idx: int, *, active_sql: str, backward: bool,
                            before_id: Optional[int], cursor_id: Optional[int],
                            anchor_first_id: Optional[int], anchor_seg_idx: Optional[int]) -> tuple[bool, Optional[int]]:
    """Return ``(should_skip, initial_seg_cursor)`` for segment ``idx`` using O(log N) boundary probes."""
    if backward:
        if before_id is None:
            return False, None
        if anchor_seg_idx is not None and idx > anchor_seg_idx:
            return True, None
        min_row = conn.execute(
            f"SELECT MIN(id) FROM messages WHERE session_id = ?{active_sql}", (sid,),
        ).fetchone()[0]
        if min_row is None or int(min_row) >= cursor_id:
            return True, None
        return False, cursor_id
    if anchor_first_id is None:
        return False, 0
    max_row = conn.execute(
        f"SELECT MAX(id) FROM messages WHERE session_id = ?{active_sql}", (sid,),
    ).fetchone()[0]
    if max_row is None or int(max_row) <= anchor_first_id:
        return True, None
    return False, anchor_first_id


def _finalize_lineage_collection(collected: list[tuple[int, int]], limit: Optional[int], total: int, *, backward: bool):
    collected.sort(key=lambda pair: pair[0])
    if limit is None:
        return [row_id for _, row_id in collected], total, False
    if limit == 0:
        return [], total, bool(collected)
    has_more = len(collected) > limit
    selected = collected[-limit:] if backward else collected[:limit]
    return [row_id for _, row_id in selected], total, has_more


def _lineage_fast_keyset_page(conn, segments: list[str], request, visible_only: bool,
                              before_id: Optional[int], after_id: Optional[int]):
    """O(k log N + limit) indexed keyset pagination across multi-segment compaction lineages."""
    display = request.view is TranscriptView.DISPLAY
    active_sql = (
        " AND active = 1" if request.view is TranscriptView.ACTIVE
        else (" AND (active = 1 OR compacted = 1)" if display else "")
    )
    shown_sql = _VISIBLE if visible_only else "1"
    s_ph = _placeholders(len(segments))
    vis_filter = f" AND ({_VISIBLE})" if visible_only else ""

    # Exact total across lineage in pure SQLite C (zero Python row materialization).
    total = int(conn.execute(
        f"SELECT COUNT(DISTINCT display_identity) FROM messages "
        f"WHERE session_id IN ({s_ph}){active_sql}{DISPLAY_VISIBLE_SQL}{vis_filter}",
        segments,
    ).fetchone()[0])

    probe_target = None if request.limit is None else (1 if request.limit == 0 else request.limit + 1)
    batch_size = 64 if probe_target is None else max(64, min(probe_target * 2, 512))
    backward = before_id is not None or (after_id is None and request.latest)
    cursor_id = before_id if before_id is not None else after_id
    anchor_first_id, anchor_seg_idx = (
        _resolve_lineage_anchor_first_id(conn, segments, cursor_id, active_sql=active_sql, use_display_order=display)
        if cursor_id is not None else (None, None)
    )

    collected: list[tuple[int, int]] = []
    emitted_identities: set[object] = set()
    emitted_ids: set[int] = set()
    indices = range(len(segments) - 1, -1, -1) if backward else range(len(segments))

    for idx in indices:
        sid = segments[idx]
        skip, seg_cursor = _initial_segment_cursor(
            conn, sid, idx, active_sql=active_sql, backward=backward,
            before_id=before_id, cursor_id=cursor_id,
            anchor_first_id=anchor_first_id, anchor_seg_idx=anchor_seg_idx,
        )
        if skip:
            continue

        while True:
            batch = _fetch_segment_batch(
                conn, sid, active_sql=active_sql, shown_sql=shown_sql,
                backward=backward, seg_cursor=seg_cursor, batch_size=batch_size,
            )
            if not batch:
                break
            seg_cursor = int(batch[-1][0])
            saw_carried = _collect_lineage_batch(
                conn, sid=sid, older_segs=segments[:idx], newer_segs=segments[idx + 1:],
                batch=batch, active_sql=active_sql, shown_sql=shown_sql,
                display=display, backward=backward, anchor_first_id=anchor_first_id,
                emitted_identities=emitted_identities, emitted_ids=emitted_ids, collected=collected,
            )
            if (probe_target is not None and len(collected) >= probe_target and not saw_carried) or len(batch) < batch_size:
                break

        if probe_target is not None and len(collected) >= probe_target:
            break

    return _finalize_lineage_collection(collected, request.limit, total, backward=backward)


def _lineage_page(holder, conn, segments, request, visible_only,
                  before_id: Optional[int] = None, after_id: Optional[int] = None):
    use_fast = (
        (before_id is not None or after_id is not None or request.offset == 0)
        and _lineage_fast_eligible(holder, conn, segments, request)
    )
    if use_fast:
        return _lineage_fast_keyset_page(conn, segments, request, visible_only, before_id, after_id)

    # Full dedupe fallback for non-zero offset paging, legacy unindexed rows, or removed-in-display edge cases.
    display = request.view is TranscriptView.DISPLAY
    active = _active_clause(request.view is TranscriptView.ALL or display, False)
    shown = _VISIBLE if visible_only else "1"
    columns = _message_column_names(holder, conn)
    identity = "display_identity" if "display_identity" in columns else "NULL"
    order = "display_order" if "display_order" in columns else "NULL"
    rows = conn.execute(f"""
        SELECT id, session_id, active, compacted, role, timestamp, message_uid, absorbed_message_uids,
               {order} AS display_order, {identity} AS display_identity, display_kind, display_metadata, _compressed_summary,
               CASE WHEN {identity} IS NULL OR _compressed_summary THEN content END AS content,
               CASE WHEN {identity} IS NULL OR _compressed_summary THEN tool_calls END AS tool_calls, tool_name, tool_call_id,
               {shown} AS shown
          FROM messages WHERE session_id IN (SELECT value FROM json_each(?)){active} ORDER BY id""", (json.dumps(segments),))
    first_ids: dict[int, int] = {}
    kept = _dedupe_display_generations(holder, _identity_rows(holder, rows), drop_removed=display,
                                       first_ids=first_ids, indexed_keys=True)
    ids = [int(row["id"]) for row in kept if row["shown"]]
    anchor_first_id = None
    cursor_id = before_id if before_id is not None else after_id
    if cursor_id is not None:
        anchor_first_id = first_ids.get(cursor_id)
        if anchor_first_id is None:
            active_sql = (
                " AND active = 1" if request.view is TranscriptView.ACTIVE
                else (" AND (active = 1 OR compacted = 1)" if display else "")
            )
            anchor_first_id, _ = _resolve_lineage_anchor_first_id(
                conn, segments, cursor_id, active_sql=active_sql, use_display_order=display,
            )
    return _slice_id_list(ids, request, before_id, after_id, first_id_by_row=first_ids, anchor_first_id=anchor_first_id)


# Rewound generations may share a platform id. Inspect the existing message index once per
# distinct admission on this page, stopping after the second eligible input proves ambiguity.
# CROSS JOIN keeps those page markers outside the indexed ledger probe as the ledger grows.
_ELIGIBLE_INPUTS = """FROM messages twin
    WHERE twin.session_id=a.target_session_id AND twin.platform_message_id=a.admission_id
      AND twin.role='user' AND twin.observed=0 AND (twin.active=1 OR twin.compacted=1)
      AND twin.display_kind IS NOT 'hidden'"""
_PAGE_WITH_INPUTS = f"""WITH page AS MATERIALIZED (SELECT key, value FROM json_each(?)),
    markers AS MATERIALIZED (
        SELECT DISTINCT m.session_id, m.platform_message_id FROM page p JOIN messages m ON m.id=p.value
        WHERE m.role='user' AND m.observed=0 AND m.platform_message_id IS NOT NULL),
    owners AS MATERIALIZED (
        SELECT a.target_session_id, a.admission_id, (SELECT twin.id {_ELIGIBLE_INPUTS} LIMIT 1) AS input_row
        FROM markers k CROSS JOIN session_admissions a ON a.admission_id=k.platform_message_id
        JOIN sessions s ON s.id=k.session_id
        WHERE a.target_session_id=k.session_id AND a.principal_id='api' AND a.generation IS NOT NULL
          AND s.source='api_server' AND NOT EXISTS (SELECT 1 {_ELIGIBLE_INPUTS} LIMIT 1 OFFSET 1))
    SELECT m.*, o.admission_id AS api_admission_id FROM page p JOIN messages m ON m.id=p.value
    LEFT JOIN owners o ON o.admission_id=m.platform_message_id AND o.target_session_id=m.session_id
                      AND m.role='user' AND m.observed=0 AND (o.input_row IS NULL OR o.input_row=m.id)
    ORDER BY p.key"""


def get_transcript_page_cursor(holder, sid, request, visible_only, *,
                               before_id: Optional[int] = None, after_id: Optional[int] = None):
    eff_before, eff_after = _validate_cursors(request, before_id, after_id)
    segments = _resume_lineage_ids(holder, sid) if request.include_lineage else [sid]

    def read(conn):
        conn.create_function("transcript_visible", 5, _visible, deterministic=True)
        if len(segments) == 1:
            ids, total, has_more = _segment_page(holder, conn, sid, request, visible_only, eff_before, eff_after)
        else:
            ids, total, has_more = _lineage_page(holder, conn, segments, request, visible_only, eff_before, eff_after)
        sql = (_PAGE_WITH_INPUTS if visible_only and holder.has_session_admissions else
               "SELECT m.* FROM json_each(?) p JOIN messages m ON m.id=p.value ORDER BY p.key")
        rows = conn.execute(sql, (json.dumps(ids),))
        return [_row_to_message_dict(holder, row, warn_context="transcript page", summary_flag=True)
                for row in rows], total, segments[0], has_more
    return _in_one_read(holder, read)


def get_transcript_page(holder, sid, request, visible_only):
    rows, total, root, _has_more = get_transcript_page_cursor(holder, sid, request, visible_only)
    return rows, total, root
