"""Single source of truth for high-concurrency mobile SQLite access.

Centralizes WAL configuration, ``busy_timeout``, ``synchronous=NORMAL`` (outside
macOS), one-time per-inode schema initialization, lock-free concurrent WAL
readers, and per-DB write serialization across ``notes.db``, ``shares.db``,
``habits.db``, and ``mobile_state.db``.
"""

from __future__ import annotations

from contextlib import contextmanager, suppress
import os
from pathlib import Path
import sqlite3
import sys
import threading
from typing import Callable, Iterator, Optional

import sqlite_support


DEFAULT_BUSY_TIMEOUT_MS = 15000
DEFAULT_CONNECT_TIMEOUT_S = DEFAULT_BUSY_TIMEOUT_MS / 1000.0

_DB_LOCKS: dict[str, threading.RLock] = {}
_DB_LOCKS_GUARD = threading.Lock()

# Cache of (resolved_path, schema_key, st_dev, st_ino) -> active journal_mode.
_SCHEMA_INIT_CACHE: dict[tuple[str, str, int, int], str] = {}
_SCHEMA_INIT_GUARD = threading.Lock()


def lock_for(db_path: Path) -> threading.RLock:
    """Return a process-wide reentrant write lock for *db_path*."""
    key = str(db_path.resolve())
    with _DB_LOCKS_GUARD:
        lock = _DB_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _DB_LOCKS[key] = lock
        return lock


def _stat_identity(db_path: Path) -> Optional[tuple[int, int]]:
    try:
        st = db_path.stat()
    except OSError:
        return None
    return (st.st_dev, st.st_ino)


def _cached_mode(resolved_path: str, schema_key: str, db_path: Path) -> Optional[str]:
    identity = _stat_identity(db_path)
    if identity is None:
        return None
    with _SCHEMA_INIT_GUARD:
        return _SCHEMA_INIT_CACHE.get((resolved_path, schema_key, identity[0], identity[1]))


def _record_initialized(resolved_path: str, schema_key: str, db_path: Path, mode: str) -> None:
    identity = _stat_identity(db_path)
    if identity is None:
        return
    with _SCHEMA_INIT_GUARD:
        _SCHEMA_INIT_CACHE[(resolved_path, schema_key, identity[0], identity[1])] = mode


def configure_sqlite_connection(
    conn: sqlite3.Connection,
    *,
    db_label: str,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    read_only: bool = False,
    apply_wal: bool = True,
) -> str:
    """Apply concurrency-safe PRAGMAs to a SQLite connection."""
    conn.execute(f"PRAGMA busy_timeout = {max(0, int(busy_timeout_ms))}")
    if read_only:
        with suppress(sqlite3.OperationalError):
            conn.execute("PRAGMA query_only = ON")
        return "readonly"
    mode = sqlite_support.apply_wal_with_fallback(conn, db_label=db_label) if apply_wal else "wal"
    if mode == "wal" and sys.platform != "darwin":
        with suppress(sqlite3.OperationalError):
            conn.execute("PRAGMA synchronous = NORMAL")
    return mode


def _ensure_initialized(
    db_path: Path,
    resolved_path: str,
    *,
    schema_key: str,
    schema_sql: str,
    db_label: str,
    migrate: Optional[Callable[[sqlite3.Connection], None]],
    busy_timeout_ms: int,
    conn: Optional[sqlite3.Connection] = None,
) -> str:
    """Initialize WAL mode, schema DDL, migrations, and 0600 permissions once per inode."""
    cached = _cached_mode(resolved_path, schema_key, db_path)
    if cached is not None:
        return cached
    with lock_for(db_path):
        cached = _cached_mode(resolved_path, schema_key, db_path)
        if cached is not None:
            return cached
        owned_conn = conn is None
        active = conn
        if active is None:
            active = sqlite3.connect(
                str(db_path),
                timeout=max(1.0, busy_timeout_ms / 1000.0),
                isolation_level=None,
            )
            active.row_factory = sqlite3.Row
        try:
            mode = configure_sqlite_connection(
                active, db_label=db_label, busy_timeout_ms=busy_timeout_ms, apply_wal=True
            )
            active.executescript(schema_sql)
            if migrate is not None:
                migrate(active)
            with suppress(OSError):
                if db_path.exists():
                    os.chmod(db_path, 0o600)
            _record_initialized(resolved_path, schema_key, db_path, mode)
            return mode
        finally:
            if owned_conn:
                active.close()


@contextmanager
def open_mobile_db(
    db_path: Path,
    *,
    schema_key: str,
    schema_sql: str,
    db_label: str,
    write: bool = False,
    migrate: Optional[Callable[[sqlite3.Connection], None]] = None,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
) -> Iterator[sqlite3.Connection]:
    """Open a profile-scoped mobile SQLite store with high-concurrency semantics.

    - Reads on a missing DB yield an in-memory empty schema without touching disk.
    - Reads on an existing DB run lock-free under WAL (no process-wide read mutex).
    - Writes serialize per-DB with ``BEGIN IMMEDIATE``, ``busy_timeout``, and cached DDL.
    """
    resolved_path = str(db_path.resolve())
    timeout_s = max(1.0, busy_timeout_ms / 1000.0)
    if not write:
        if not db_path.is_file():
            conn = sqlite3.connect(":memory:")
            conn.row_factory = sqlite3.Row
            conn.executescript(schema_sql)
            if migrate is not None:
                migrate(conn)
            try:
                yield conn
            finally:
                conn.close()
            return
        _ensure_initialized(
            db_path,
            resolved_path,
            schema_key=schema_key,
            schema_sql=schema_sql,
            db_label=db_label,
            migrate=migrate,
            busy_timeout_ms=busy_timeout_ms,
        )
        conn = sqlite3.connect(str(db_path), timeout=timeout_s, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            configure_sqlite_connection(
                conn,
                db_label=db_label,
                busy_timeout_ms=busy_timeout_ms,
                read_only=True,
                apply_wal=False,
            )
            yield conn
        finally:
            conn.close()
        return

    with lock_for(db_path):
        conn = sqlite3.connect(str(db_path), timeout=timeout_s, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            cached = _cached_mode(resolved_path, schema_key, db_path)
            if cached is None:
                _ensure_initialized(
                    db_path,
                    resolved_path,
                    schema_key=schema_key,
                    schema_sql=schema_sql,
                    db_label=db_label,
                    migrate=migrate,
                    busy_timeout_ms=busy_timeout_ms,
                    conn=conn,
                )
            else:
                configure_sqlite_connection(
                    conn,
                    db_label=db_label,
                    busy_timeout_ms=busy_timeout_ms,
                    apply_wal=False,
                )
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            if conn.in_transaction:
                conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                with suppress(sqlite3.Error):
                    conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()


@contextmanager
def open_mobile_db_readonly(
    db_path: Path,
    *,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
) -> Iterator[sqlite3.Connection]:
    """Open an existing mobile SQLite file strictly read-only with ``busy_timeout``."""
    conn = sqlite3.connect(
        f"file:{db_path}?mode=ro",
        uri=True,
        timeout=max(1.0, busy_timeout_ms / 1000.0),
    )
    conn.row_factory = sqlite3.Row
    try:
        configure_sqlite_connection(
            conn,
            db_label=db_path.name,
            busy_timeout_ms=busy_timeout_ms,
            read_only=True,
            apply_wal=False,
        )
        yield conn
    finally:
        conn.close()
