"""Count and page the client transcript without loading message payloads outside the requested page."""

from __future__ import annotations

import json

from persistence.conversation.messages import TranscriptView
from session_rules.turns import transcript_message_visible
from .messages import (
    _active_clause, _decode_content, _decode_display_metadata, _dedupe_display_generations, _display_dedupe_key, _display_identity,
    _legacy_display_ids, _message_column_names, _resume_lineage_ids, _row_to_message_dict,
)
from .messages_sql import _DISPLAY_ACTIVE_CLAUSE, _DISPLAY_INDEX_MISSING_SQL
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


def _segment_page(holder, conn, sid, request, visible_only):
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
        return [r[0] for r in rows if r[0] is not None], rows[0][1]
    total = conn.execute("SELECT COUNT(*) FROM (" + query + ")", params).fetchone()[0]
    selected = conn.execute(query + f" ORDER BY sort_id {direction} LIMIT COALESCE(:limit, -1) OFFSET :offset", params)
    ids = [row[0] for row in selected]
    return (ids[::-1] if request.latest else ids), total


def _identity_rows(holder, cursor):
    """Project each legacy key before retaining it. A summary may have an identity folded within a different active set."""
    for row in cursor:
        if row["display_identity"] is None or row["_compressed_summary"]:
            identity = _display_identity(_display_dedupe_key(holder, row))
            row = dict(row)
            row.update(display_identity=identity, content=None, tool_calls=None)
        yield row


def _lineage_page(holder, conn, segments, request, visible_only):
    # Merged segments share copied generations across sessions. The canonical deduper operates on
    # stored fixed-width identities. Legacy rows lacking an identity still supply their original key.
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
    kept = _dedupe_display_generations(holder, _identity_rows(holder, rows), drop_removed=display, indexed_keys=True)
    ids = [row["id"] for row in kept if row["shown"]]
    total = len(ids)
    if request.latest:
        end = max(0, total - request.offset)
        return ids[0 if request.limit is None else max(0, end - request.limit):end], total
    return ids[request.offset:None if request.limit is None else request.offset + request.limit], total


def get_transcript_page(holder, sid, request, visible_only):
    segments = _resume_lineage_ids(holder, sid) if request.include_lineage else [sid]

    def read(conn):
        conn.create_function("transcript_visible", 5, _visible, deterministic=True)
        if len(segments) == 1:
            ids, total = _segment_page(holder, conn, sid, request, visible_only)
        else:
            ids, total = _lineage_page(holder, conn, segments, request, visible_only)
        # Inspect only this page's selected human inputs. The gateway records the canonical
        # admission as their inbound platform id; other platforms use that column for their own ids.
        evidence = """, CASE WHEN m.role='user' AND m.platform_message_id IS NOT NULL AND m.observed=0
                              THEN (SELECT a.admission_id FROM session_admissions a
                                    JOIN sessions s ON s.id=a.target_session_id
                                    WHERE a.admission_id=m.platform_message_id AND a.target_session_id=m.session_id
                                      AND a.principal_id='api' AND a.generation IS NOT NULL AND s.source='api_server'
                                      AND NOT EXISTS (SELECT 1 FROM messages twin
                                          WHERE twin.session_id=m.session_id AND twin.platform_message_id=m.platform_message_id
                                            AND twin.id!=m.id AND twin.role='user' AND twin.observed=0
                                            AND (twin.active=1 OR twin.compacted=1) AND twin.display_kind IS NOT 'hidden'))
                         END AS api_admission_id""" if visible_only else ""
        rows = conn.execute(f"SELECT m.*{evidence} FROM json_each(?) p JOIN messages m ON m.id = p.value ORDER BY p.key",
                            (json.dumps(ids),))
        return [_row_to_message_dict(holder, row, warn_context="transcript page", summary_flag=True)
                for row in rows], total, segments[0]
    return _in_one_read(holder, read)
