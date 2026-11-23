"""Privacy-safe, habit-adaptive suggestions tracker for ``APIServerAdapter``.

Stores ONLY anonymized ``(category, action_key, count, last_used_at)`` rows inside
the active profile's ``<HERMES_HOME>/habits.db`` (zero user messages, titles, URLs,
coordinates, or PII). Suggestions are NEVER surfaced until a pattern has genuinely
repeated at least ``HABIT_MIN_THRESHOLD`` (3) times.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import functools
import logging
import re
import sqlite3
import time
from typing import Any, Iterator, Optional

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]

from gateway.i18n import resolve_request_lang, t as _gt, user_lang_scope
from hermes_constants import (
    assert_named_profile_home_live,
    get_hermes_home,
    mkdir_under_hermes_home,
)
from tools.mobile_sqlite import open_mobile_db

logger = logging.getLogger("gateway.platforms.api_server")

HABIT_MIN_THRESHOLD = 3
_MAX_SUGGESTIONS_LIMIT = 20
_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS habit_counts (
    category TEXT NOT NULL,
    action_key TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    last_used_at REAL NOT NULL,
    PRIMARY KEY (category, action_key)
);
CREATE INDEX IF NOT EXISTS idx_habit_counts_rank ON habit_counts(count DESC, last_used_at DESC);
"""

CAPABILITY_FEATURES: dict[str, Any] = {
    "habit_suggestions": True,
}

CAPABILITY_ENDPOINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("habit_suggestions", ("GET", "/api/mobile/habits/suggestions", "habit_suggestions")),
    ("habit_event", ("POST", "/api/mobile/habits/events", "habit_suggestions")),
    ("habit_reset", ("DELETE", "/api/mobile/habits", "habit_suggestions")),
)


@contextmanager
def _connect_habits_db(*, write: bool = False) -> Iterator[sqlite3.Connection]:
    home = get_hermes_home()
    assert_named_profile_home_live(home)
    db_path = home / "habits.db"
    if write:
        mkdir_under_hermes_home(db_path.parent)
    with open_mobile_db(
        db_path,
        schema_key="habits",
        schema_sql=_SCHEMA_SQL,
        db_label="habits.db",
        write=write,
    ) as conn:
        yield conn


def _validate_key(raw: Any, field: str) -> tuple[Optional[str], Optional[str]]:
    if not isinstance(raw, str) or not raw.strip():
        return None, _gt("habits.invalid_key_empty", field=field)
    norm = raw.strip().lower()
    if not _KEY_RE.fullmatch(norm):
        return None, _gt("habits.invalid_key_format", field=field)
    return norm, None


def record_habit_event(category: str, action_key: str) -> dict[str, Any]:
    """Increment an anonymized ``(category, action_key)`` counter for the active profile."""
    now = time.time()
    with _connect_habits_db(write=True) as conn:
        conn.execute(
            """
            INSERT INTO habit_counts (category, action_key, count, last_used_at)
            VALUES (?, ?, 1, ?)
            ON CONFLICT(category, action_key)
            DO UPDATE SET count = count + 1, last_used_at = excluded.last_used_at
            """,
            (category, action_key, now),
        )
        row = conn.execute(
            "SELECT category, action_key, count, last_used_at FROM habit_counts WHERE category = ? AND action_key = ?",
            (category, action_key),
        ).fetchone()
    count = int(row["count"])
    return {
        "category": row["category"],
        "action": row["action_key"],
        "count": count,
        "last_used_at": row["last_used_at"],
        "learned": count >= HABIT_MIN_THRESHOLD,
        "threshold": HABIT_MIN_THRESHOLD,
    }


def list_habit_suggestions(
    *,
    category: Optional[str] = None,
    limit: int = 8,
    min_count: int = HABIT_MIN_THRESHOLD,
) -> dict[str, Any]:
    """Return ONLY learned patterns (strictly ``count >= max(3, min_count)``)."""
    effective_min = max(HABIT_MIN_THRESHOLD, min_count)
    capped_limit = max(1, min(limit, _MAX_SUGGESTIONS_LIMIT))
    with _connect_habits_db(write=False) as conn:
        if category:
            rows = conn.execute(
                """
                SELECT category, action_key, count, last_used_at
                FROM habit_counts
                WHERE category = ? AND count >= ?
                ORDER BY count DESC, last_used_at DESC
                LIMIT ?
                """,
                (category, effective_min, capped_limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT category, action_key, count, last_used_at
                FROM habit_counts
                WHERE count >= ?
                ORDER BY count DESC, last_used_at DESC
                LIMIT ?
                """,
                (effective_min, capped_limit),
            ).fetchall()
    return {
        "object": "list",
        "threshold": effective_min,
        "data": [
            {
                "category": row["category"],
                "action": row["action_key"],
                "count": int(row["count"]),
                "last_used_at": row["last_used_at"],
            }
            for row in rows
        ],
    }


def reset_habits() -> dict[str, Any]:
    """Clear all learned habit counts for the active profile."""
    with _connect_habits_db(write=True) as conn:
        cur = conn.execute("DELETE FROM habit_counts")
        deleted = cur.rowcount
    return {"object": "hermes.habits.reset", "cleared": int(deleted)}


async def handle_habit_suggestions(adapter, request: web.Request) -> web.Response:
    """GET /api/mobile/habits/suggestions — return learned patterns (count >= 3 only)."""
    from gateway.platforms.api_server import _error_response

    with user_lang_scope(resolve_request_lang(request.headers)):
        if auth_err := adapter._check_auth(request):
            return auth_err
        raw_cat = (request.query.get("category") or "").strip()
        category: Optional[str] = None
        if raw_cat:
            category, cat_err = _validate_key(raw_cat, "category")
            if cat_err is not None:
                return _error_response(cat_err, 400, code="invalid_habit_category")
        try:
            limit = int(request.query.get("limit", "8"))
            min_count = int(request.query.get("min_count", str(HABIT_MIN_THRESHOLD)))
        except ValueError:
            return _error_response(_gt("habits.invalid_int_query"), 400, code="invalid_habit_query")
        if limit < 1 or limit > _MAX_SUGGESTIONS_LIMIT:
            return _error_response(
                _gt("habits.invalid_limit", max_limit=_MAX_SUGGESTIONS_LIMIT),
                400,
                code="invalid_habit_query",
            )
        min_count = max(min_count, HABIT_MIN_THRESHOLD)
        payload = await asyncio.to_thread(list_habit_suggestions, category=category, limit=limit, min_count=min_count)
        return web.json_response(payload)


async def handle_record_habit_event(adapter, request: web.Request) -> web.Response:
    """POST /api/mobile/habits/events — record an anonymized category/action signal."""
    from gateway.platforms.api_server import _error_response

    with user_lang_scope(resolve_request_lang(request.headers)):
        if auth_err := adapter._check_auth(request):
            return auth_err
        body, err = await adapter._read_json_body(request)
        if err is not None:
            return err
        unknown = sorted(set(body) - {"category", "action"})
        if unknown:
            return _error_response(
                _gt("habits.disallowed_fields", fields=", ".join(unknown)),
                400,
                code="disallowed_habit_fields",
            )
        category, cat_err = _validate_key(body.get("category"), "category")
        if cat_err is not None:
            return _error_response(cat_err, 400, code="invalid_habit_category")
        action_key, act_err = _validate_key(body.get("action"), "action")
        if act_err is not None:
            return _error_response(act_err, 400, code="invalid_habit_action")
        assert category is not None and action_key is not None
        result = await asyncio.to_thread(record_habit_event, category, action_key)
        return web.json_response({"object": "hermes.habit.event", **result}, status=201)


async def handle_reset_habits(adapter, request: web.Request) -> web.Response:
    """DELETE /api/mobile/habits — clear learned habits for the active profile."""
    if auth_err := adapter._check_auth(request):
        return auth_err
    result = await asyncio.to_thread(reset_habits)
    return web.json_response(result)


def _http_routes(adapter) -> list[tuple]:
    return [
        ("GET", "/api/mobile/habits/suggestions", functools.partial(handle_habit_suggestions, adapter)),
        ("POST", "/api/mobile/habits/events", functools.partial(handle_record_habit_event, adapter)),
        ("DELETE", "/api/mobile/habits", functools.partial(handle_reset_habits, adapter)),
    ]
