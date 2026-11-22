"""AI-driven Smart Notes store (profile-scoped SQLite) with a forgiving, small-LLM-friendly
``mobile_notes`` tool.

All multi-user same-server sharing & ACL enforcement delegates directly to
``tools.mobile_sharing`` (the Single Source of Truth Universal Resource Sharing
Engine) — zero duplicated ACL tables or permission logic!
"""

from __future__ import annotations

from contextlib import contextmanager, suppress
import json
import logging
import os
from pathlib import Path
import re
import secrets
import sqlite3
import threading
import time
from typing import Any, Iterator, Optional

from hermes_constants import (
    assert_named_profile_home_live,
    get_hermes_home,
    mkdir_under_hermes_home,
)
from tools.mobile_sharing import (
    ShareError,
    active_profile_name,
    check_incoming_access,
    grant_share,
    list_incoming_shares,
    list_outgoing_shares,
    normalize_share_permission,
    revoke_all_shares_for_resource,
    revoke_share_by_target,
)
from tools.registry import registry

logger = logging.getLogger(__name__)

MAX_NOTE_TITLE_CHARS = 200
MAX_NOTE_CONTENT_CHARS = 65_536
MAX_NOTE_TAGS = 20
MAX_TAG_CHARS = 40
_NOTE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,79}$")
_SLUG_STRIP_RE = re.compile(r"[^a-z0-9]+")
_NOTE_SHARE_PERMISSIONS = frozenset({"read", "edit"})

_ACTION_ALIASES = {
    "create": "create",
    "add": "create",
    "new": "create",
    "save": "create",
    "update": "update",
    "edit": "update",
    "replace": "update",
    "append": "append",
    "add_text": "append",
    "get": "get",
    "read": "get",
    "view": "get",
    "show": "get",
    "list": "list",
    "ls": "list",
    "search": "search",
    "find": "search",
    "query": "search",
    "tag": "tag",
    "tags": "tag",
    "pin": "pin",
    "unpin": "unpin",
    "delete": "delete",
    "remove": "delete",
    "rm": "delete",
    "share": "share",
    "unshare": "unshare",
    "revoke": "unshare",
}

_DB_LOCKS_GUARD = threading.Lock()
_DB_LOCKS: dict[str, threading.RLock] = {}

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS notes (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    tags_json TEXT NOT NULL DEFAULT '[]',
    pinned INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    updated_by TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_notes_updated ON notes(pinned DESC, updated_at DESC);
"""


class NotesError(Exception):
    """Typed Smart Notes error with HTTP status, machine code, and LLM self-correction hint."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        code: str = "invalid_note_request",
        hint: Optional[str] = None,
        example: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.hint = hint
        self.example = example


def _wrap_share_error(exc: ShareError) -> NotesError:
    return NotesError(
        exc.message,
        status_code=exc.status_code,
        code=exc.code,
        hint=exc.hint,
        example=exc.example,
    )


def _lock_for(db_path: Path) -> threading.RLock:
    key = str(db_path.resolve(strict=False))
    with _DB_LOCKS_GUARD:
        lock = _DB_LOCKS.get(key)
        if lock is None:
            lock = _DB_LOCKS[key] = threading.RLock()
        return lock


def _notes_db_path(home: Optional[Path] = None) -> Path:
    base = home if home is not None else get_hermes_home()
    assert_named_profile_home_live(base)
    return base / "notes.db"


@contextmanager
def _connect_notes_db(home: Optional[Path] = None, *, write: bool = False) -> Iterator[sqlite3.Connection]:
    db_path = _notes_db_path(home)
    if write:
        mkdir_under_hermes_home(db_path.parent)
    with _lock_for(db_path):
        if not write and not db_path.is_file():
            conn = sqlite3.connect(":memory:")
            conn.row_factory = sqlite3.Row
            conn.executescript(_SCHEMA_SQL)
            try:
                yield conn
            finally:
                conn.close()
            return
        conn = sqlite3.connect(str(db_path), timeout=10.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(_SCHEMA_SQL)
            with suppress(OSError):
                if db_path.exists():
                    os.chmod(db_path, 0o600)
            if write:
                conn.execute("BEGIN IMMEDIATE")
            yield conn
            if write and conn.in_transaction:
                conn.execute("COMMIT")
        except BaseException:
            if write and conn.in_transaction:
                with suppress(sqlite3.Error):
                    conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()


def _validate_utf8(value: Any, field: str, *, max_chars: int, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise NotesError(
            f"'{field}' must be a string.",
            code=f"invalid_{field}",
            hint=f"Pass '{field}' as plain text.",
        )
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise NotesError(f"'{field}' must be valid UTF-8.", code=f"invalid_{field}") from exc
    if "\x00" in value:
        raise NotesError(f"'{field}' cannot contain NUL bytes.", code=f"invalid_{field}")
    cleaned = value.strip() if field == "title" else value
    if not allow_empty and not cleaned.strip():
        raise NotesError(
            f"'{field}' cannot be empty.",
            code=f"missing_{field}",
            hint=f"Provide a non-empty '{field}'.",
        )
    if len(cleaned) > max_chars:
        raise NotesError(
            f"'{field}' exceeds {max_chars} characters.",
            status_code=413 if field == "content" else 400,
            code=f"{field}_too_long",
        )
    return cleaned


def normalize_tags(raw: Any) -> list[str]:
    """Accept a list or comma-separated string of tags (small-LLM friendly)."""
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, list):
        items = []
        for item in raw:
            if not isinstance(item, str):
                raise NotesError(
                    "'tags' items must be strings.",
                    code="invalid_tags",
                    hint='Pass tags as ["work", "ideas"] or "work, ideas".',
                )
            if item.strip():
                items.append(item.strip())
    else:
        raise NotesError(
            "'tags' must be a list of strings or a comma-separated string.",
            code="invalid_tags",
            hint='Pass tags as ["work", "ideas"].',
        )
    deduped: list[str] = []
    seen: set[str] = set()
    for tag in items:
        try:
            tag.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise NotesError("Tags must be valid UTF-8.", code="invalid_tags") from exc
        norm = tag.lower()[:MAX_TAG_CHARS]
        if norm and norm not in seen:
            seen.add(norm)
            deduped.append(norm)
        if len(deduped) > MAX_NOTE_TAGS:
            raise NotesError(f"At most {MAX_NOTE_TAGS} tags are allowed per note.", code="too_many_tags")
    return deduped


def slugify_note_id(title: str) -> str:
    slug = _SLUG_STRIP_RE.sub("-", title.strip().lower()).strip("-")[:48].rstrip("-")
    return slug or f"note-{secrets.token_hex(4)}"


def _validate_note_id(raw: Any) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise NotesError(
            "note_id is required.",
            code="invalid_note_id",
            hint="Pass the note's 'id' from list/search.",
        )
    cleaned = raw.strip().lower()
    if not _NOTE_ID_RE.fullmatch(cleaned):
        raise NotesError(
            f"Invalid note_id '{raw}'. Use lowercase letters, numbers, '-' or '_'.",
            code="invalid_note_id",
            hint="Use a slug like 'meeting-notes' or the id returned by create/list.",
        )
    return cleaned


def _choose_unique_id(conn: sqlite3.Connection, requested_id: Optional[str], title: str) -> str:
    if requested_id is not None and str(requested_id).strip():
        note_id = _validate_note_id(requested_id)
        if conn.execute("SELECT 1 FROM notes WHERE id = ?", (note_id,)).fetchone():
            raise NotesError(f"Note '{note_id}' already exists.", status_code=409, code="note_exists")
        return note_id
    base = slugify_note_id(title)
    candidate = base
    for suffix in range(2, 100):
        if not conn.execute("SELECT 1 FROM notes WHERE id = ?", (candidate,)).fetchone():
            return candidate
        candidate = f"{base[:64]}-{suffix}"
    return f"{base[:60]}-{secrets.token_hex(4)}"


def _row_to_note(
    row: sqlite3.Row,
    *,
    owner_profile: str,
    access: str,
    shares: Optional[list[dict[str, Any]]] = None,
    share_id: Optional[str] = None,
) -> dict[str, Any]:
    try:
        tags = json.loads(row["tags_json"] or "[]")
        if not isinstance(tags, list):
            tags = []
    except json.JSONDecodeError:
        tags = []
    payload: dict[str, Any] = {
        "id": row["id"],
        "title": row["title"],
        "content": row["content"],
        "tags": tags,
        "pinned": bool(row["pinned"]),
        "archived": bool(row["archived"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "updated_by": row["updated_by"] or owner_profile,
        "owner_profile": owner_profile,
        "access": access,
        "can_edit": access in ("owner", "edit"),
        "can_share": access == "owner",
    }
    if shares is not None:
        payload["shares"] = shares
    if share_id is not None:
        payload["share_id"] = share_id
    return payload


# ── Core CRUD + Delegated Universal Sharing ACL Operations (SSoT) ────────────


def list_notes(
    *,
    query: Optional[str] = None,
    tag: Optional[str] = None,
    pinned_only: bool = False,
    include_archived: bool = False,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """List the active profile's own notes with optional search/tag/pin filtering."""
    owner = active_profile_name()
    clauses: list[str] = []
    params: list[Any] = []
    if not include_archived:
        clauses.append("archived = 0")
    if pinned_only:
        clauses.append("pinned = 1")
    if query and query.strip():
        like = f"%{query.strip().lower()}%"
        clauses.append("(lower(title) LIKE ? OR lower(content) LIKE ? OR lower(tags_json) LIKE ?)")
        params.extend([like, like, like])
    wanted_tag = tag.strip().lower() if tag and tag.strip() else None
    if wanted_tag:
        clauses.append("EXISTS (SELECT 1 FROM json_each(notes.tags_json) WHERE lower(value) = ?)")
        params.append(wanted_tag)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    shares_by_note: dict[str, list[dict[str, Any]]] = {}
    for s in list_outgoing_shares(resource_kind="note"):
        shares_by_note.setdefault(s["resource_id"], []).append(s)
    with _connect_notes_db(write=False) as conn:
        rows = conn.execute(
            f"SELECT * FROM notes {where} ORDER BY pinned DESC, updated_at DESC LIMIT ? OFFSET ?",
            (*params, max(1, min(limit, 200)), max(0, offset)),
        ).fetchall()
        return [
            _row_to_note(row, owner_profile=owner, access="owner", shares=shares_by_note.get(row["id"], []))
            for row in rows
        ]


def list_shared_with_me(*, limit: int = 50) -> list[dict[str, Any]]:
    """List notes explicitly shared with the active profile via ``tools.mobile_sharing``."""
    from tools.mobile_sharing import resolve_profile_home_if_exists

    incoming = list_incoming_shares(resource_kind="note")[: max(1, min(limit, 200))]
    notes: list[dict[str, Any]] = []
    for item in incoming:
        owner_home = resolve_profile_home_if_exists(item["owner_profile"])
        if owner_home is None:
            continue
        with _connect_notes_db(owner_home, write=False) as oconn:
            row = oconn.execute(
                "SELECT * FROM notes WHERE id = ? AND archived = 0",
                (item["resource_id"],),
            ).fetchone()
        if row is not None:
            notes.append(
                _row_to_note(
                    row,
                    owner_profile=item["owner_profile"],
                    access=item["permission"],
                    share_id=item["share_id"],
                )
            )
    return notes


def get_note(note_id: str, *, owner_profile: Optional[str] = None) -> dict[str, Any]:
    """Read an owned note or a note explicitly shared via ``tools.mobile_sharing``."""
    clean_id = _validate_note_id(note_id)
    current = active_profile_name()
    if owner_profile is None or owner_profile == current:
        with _connect_notes_db(write=False) as conn:
            row = conn.execute("SELECT * FROM notes WHERE id = ?", (clean_id,)).fetchone()
            if row is not None:
                shares = list_outgoing_shares(resource_kind="note", resource_id=clean_id)
                return _row_to_note(row, owner_profile=current, access="owner", shares=shares)
    shared = check_incoming_access(current, "note", clean_id, owner_hint=owner_profile)
    if shared is None:
        raise NotesError(f"Note '{clean_id}' not found.", status_code=404, code="note_not_found")
    owner_name, owner_home, permission, share_id = shared
    with _connect_notes_db(owner_home, write=False) as owner_conn:
        row = owner_conn.execute("SELECT * FROM notes WHERE id = ? AND archived = 0", (clean_id,)).fetchone()
    if row is None:
        raise NotesError(f"Note '{clean_id}' not found.", status_code=404, code="note_not_found")
    return _row_to_note(row, owner_profile=owner_name, access=permission, share_id=share_id)


def create_note(
    *,
    title: str,
    content: str = "",
    tags: Any = None,
    pinned: bool = False,
    note_id: Optional[str] = None,
) -> dict[str, Any]:
    """Create a new note in the active profile's SQLite store."""
    clean_title = _validate_utf8(title, "title", max_chars=MAX_NOTE_TITLE_CHARS)
    clean_content = _validate_utf8(content, "content", max_chars=MAX_NOTE_CONTENT_CHARS, allow_empty=True)
    clean_tags = normalize_tags(tags)
    if not isinstance(pinned, bool):
        raise NotesError("'pinned' must be a boolean.", code="invalid_pinned")
    owner = active_profile_name()
    now = time.time()
    with _connect_notes_db(write=True) as conn:
        chosen_id = _choose_unique_id(conn, note_id, clean_title)
        conn.execute(
            """
            INSERT INTO notes (id, title, content, tags_json, pinned, archived, created_at, updated_at, updated_by)
            VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?)
            """,
            (chosen_id, clean_title, clean_content, json.dumps(clean_tags), int(pinned), now, now, owner),
        )
        row = conn.execute("SELECT * FROM notes WHERE id = ?", (chosen_id,)).fetchone()
        return _row_to_note(row, owner_profile=owner, access="owner", shares=[])


def _apply_note_mutation(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    actor: str,
    is_owner: bool,
    title: Optional[str] = None,
    content: Optional[str] = None,
    append: Optional[str] = None,
    tags: Any = None,
    has_tags: bool = False,
    pinned: Optional[bool] = None,
    archived: Optional[bool] = None,
) -> sqlite3.Row:
    new_title = _validate_utf8(title, "title", max_chars=MAX_NOTE_TITLE_CHARS) if title is not None else row["title"]
    if content is not None and append is not None:
        raise NotesError("Provide either 'content' or 'append', not both.", code="conflicting_note_update")
    if content is not None:
        new_content = _validate_utf8(content, "content", max_chars=MAX_NOTE_CONTENT_CHARS, allow_empty=True)
    elif append is not None:
        add_text = _validate_utf8(append, "append", max_chars=MAX_NOTE_CONTENT_CHARS)
        existing = row["content"] or ""
        joined = f"{existing}\n{add_text}" if existing else add_text
        if len(joined) > MAX_NOTE_CONTENT_CHARS:
            raise NotesError(
                f"Appending exceeds {MAX_NOTE_CONTENT_CHARS} characters.",
                status_code=413,
                code="content_too_long",
            )
        new_content = joined
    else:
        new_content = row["content"]
    new_tags_json = json.dumps(normalize_tags(tags)) if has_tags else row["tags_json"]
    if not is_owner and (pinned is not None or archived is not None):
        raise NotesError(
            "Only the note owner can pin or archive a shared note.",
            status_code=403,
            code="note_owner_required",
        )
    if pinned is not None and not isinstance(pinned, bool):
        raise NotesError("'pinned' must be a boolean.", code="invalid_pinned")
    if archived is not None and not isinstance(archived, bool):
        raise NotesError("'archived' must be a boolean.", code="invalid_archived")
    new_pinned = int(pinned) if pinned is not None else row["pinned"]
    new_archived = int(archived) if archived is not None else row["archived"]
    now = time.time()
    conn.execute(
        """
        UPDATE notes
        SET title = ?, content = ?, tags_json = ?, pinned = ?, archived = ?, updated_at = ?, updated_by = ?
        WHERE id = ?
        """,
        (new_title, new_content, new_tags_json, new_pinned, new_archived, now, actor, row["id"]),
    )
    return conn.execute("SELECT * FROM notes WHERE id = ?", (row["id"],)).fetchone()


def update_note(
    note_id: str,
    *,
    title: Optional[str] = None,
    content: Optional[str] = None,
    append: Optional[str] = None,
    tags: Any = None,
    has_tags: bool = False,
    pinned: Optional[bool] = None,
    archived: Optional[bool] = None,
    owner_profile: Optional[str] = None,
) -> dict[str, Any]:
    """Update an owned note or a shared note with ``edit`` permission."""
    clean_id = _validate_note_id(note_id)
    current = active_profile_name()
    if owner_profile is None or owner_profile == current:
        with _connect_notes_db(write=True) as conn:
            row = conn.execute("SELECT * FROM notes WHERE id = ?", (clean_id,)).fetchone()
            if row is not None:
                updated = _apply_note_mutation(
                    conn,
                    row,
                    actor=current,
                    is_owner=True,
                    title=title,
                    content=content,
                    append=append,
                    tags=tags,
                    has_tags=has_tags,
                    pinned=pinned,
                    archived=archived,
                )
                shares = list_outgoing_shares(resource_kind="note", resource_id=clean_id)
                return _row_to_note(updated, owner_profile=current, access="owner", shares=shares)
    shared = check_incoming_access(current, "note", clean_id, owner_hint=owner_profile)
    if shared is None:
        raise NotesError(f"Note '{clean_id}' not found.", status_code=404, code="note_not_found")
    owner_name, owner_home, permission, share_id = shared
    if permission != "edit":
        raise NotesError(
            f"Note '{clean_id}' is shared read-only; edit access is required.",
            status_code=403,
            code="note_read_only",
        )
    with _connect_notes_db(owner_home, write=True) as owner_conn:
        # Re-verify live grant before committing edit.
        recheck = check_incoming_access(current, "note", clean_id, owner_hint=owner_name)
        if recheck is None or recheck[2] != "edit":
            raise NotesError(f"Edit permission for note '{clean_id}' was revoked.", status_code=403, code="note_read_only")
        row = owner_conn.execute("SELECT * FROM notes WHERE id = ? AND archived = 0", (clean_id,)).fetchone()
        if row is None:
            raise NotesError(f"Note '{clean_id}' not found.", status_code=404, code="note_not_found")
        updated = _apply_note_mutation(
            owner_conn,
            row,
            actor=current,
            is_owner=False,
            title=title,
            content=content,
            append=append,
            tags=tags,
            has_tags=has_tags,
            pinned=pinned,
            archived=archived,
        )
        return _row_to_note(updated, owner_profile=owner_name, access="edit", share_id=share_id)


def delete_note(note_id: str) -> dict[str, Any]:
    """Delete an owned note and revoke all universal shares for it (owner-only)."""
    clean_id = _validate_note_id(note_id)
    current = active_profile_name()
    with _connect_notes_db(write=True) as conn:
        row = conn.execute("SELECT id FROM notes WHERE id = ?", (clean_id,)).fetchone()
        if row is not None:
            conn.execute("DELETE FROM notes WHERE id = ?", (clean_id,))
            deleted = True
        else:
            deleted = False
    if not deleted:
        if check_incoming_access(current, "note", clean_id) is not None:
            raise NotesError("Only the note owner can delete a shared note.", status_code=403, code="note_owner_required")
        raise NotesError(f"Note '{clean_id}' not found.", status_code=404, code="note_not_found")
    revoke_all_shares_for_resource("note", clean_id)
    return {"id": clean_id, "deleted": True}


def share_note(note_id: str, target_profile: str, permission: str = "read") -> dict[str, Any]:
    """Delegate note sharing to ``tools.mobile_sharing.grant_share`` (SSoT)."""
    clean_id = _validate_note_id(note_id)
    try:
        perm = normalize_share_permission(permission, allowed=_NOTE_SHARE_PERMISSIONS)
        share = grant_share("note", clean_id, target_profile, perm)
    except ShareError as exc:
        raise _wrap_share_error(exc) from exc
    return {
        "note_id": clean_id,
        "share_id": share["share_id"],
        "owner_profile": share["owner_profile"],
        "target_profile": share["target_profile"],
        "permission": share["permission"],
        "shares": list_outgoing_shares(resource_kind="note", resource_id=clean_id),
    }


def unshare_note(note_id: str, target_profile: str) -> dict[str, Any]:
    """Delegate note share revocation to ``tools.mobile_sharing`` (SSoT)."""
    clean_id = _validate_note_id(note_id)
    try:
        res = revoke_share_by_target("note", clean_id, target_profile)
    except ShareError as exc:
        raise _wrap_share_error(exc) from exc
    return {
        "note_id": clean_id,
        "share_id": res["share_id"],
        "target_profile": target_profile,
        "revoked": True,
        "shares": list_outgoing_shares(resource_kind="note", resource_id=clean_id),
    }


# ── Small-LLM-Friendly Tool Entry Point ──────────────────────────────────────


def _tool_create(args: dict[str, Any]) -> dict[str, Any]:
    title = args.get("title")
    content = args.get("content") or ""
    if not title and isinstance(content, str) and content.strip():
        first_line = content.strip().splitlines()[0].lstrip("# ").strip()
        title = first_line[:80] or "Quick Note"
    if not title:
        raise NotesError(
            "title is required to create a note.",
            code="missing_title",
            hint="Provide a short 'title' (and optional 'content' / 'tags').",
            example={"action": "create", "title": "Grocery list", "content": "- Milk\n- Coffee", "tags": ["home"]},
        )
    return {"success": True, "note": create_note(
        title=title,
        content=content,
        tags=args.get("tags"),
        pinned=bool(args.get("pinned", False)),
        note_id=args.get("note_id"),
    )}


def _tool_update(args: dict[str, Any]) -> dict[str, Any]:
    return {"success": True, "note": update_note(
        args.get("note_id", ""),
        title=args.get("title"),
        content=args.get("content"),
        tags=args.get("tags"),
        has_tags="tags" in args and args.get("tags") is not None,
        pinned=args.get("pinned"),
    )}


def _tool_append(args: dict[str, Any]) -> dict[str, Any]:
    text = args.get("content") if args.get("content") is not None else args.get("text")
    return {"success": True, "note": update_note(args.get("note_id", ""), append=text)}


def _tool_get(args: dict[str, Any]) -> dict[str, Any]:
    return {"success": True, "note": get_note(args.get("note_id", ""))}


def _tool_list(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": True,
        "notes": list_notes(
            tag=args.get("tag"),
            pinned_only=bool(args.get("pinned", False)),
            limit=int(args.get("limit") or 25),
        ),
        "shared_with_me": list_shared_with_me(limit=25),
    }


def _tool_search(args: dict[str, Any]) -> dict[str, Any]:
    q = str(args.get("query") or args.get("q") or "").strip()
    if not q:
        raise NotesError(
            "query is required for search.",
            code="missing_query",
            hint="Pass a keyword in 'query'.",
            example={"action": "search", "query": "grocery"},
        )
    return {"success": True, "query": q, "notes": list_notes(query=q, limit=int(args.get("limit") or 25))}


def _tool_tag(args: dict[str, Any]) -> dict[str, Any]:
    return {"success": True, "note": update_note(args.get("note_id", ""), tags=args.get("tags"), has_tags=True)}


def _tool_pin(args: dict[str, Any]) -> dict[str, Any]:
    return {"success": True, "note": update_note(args.get("note_id", ""), pinned=True)}


def _tool_unpin(args: dict[str, Any]) -> dict[str, Any]:
    return {"success": True, "note": update_note(args.get("note_id", ""), pinned=False)}


def _tool_delete(args: dict[str, Any]) -> dict[str, Any]:
    return {"success": True, **delete_note(args.get("note_id", ""))}


def _tool_share(args: dict[str, Any]) -> dict[str, Any]:
    return {"success": True, **share_note(
        args.get("note_id", ""),
        str(args.get("target_profile") or ""),
        str(args.get("permission") or "read"),
    )}


def _tool_unshare(args: dict[str, Any]) -> dict[str, Any]:
    return {"success": True, **unshare_note(args.get("note_id", ""), str(args.get("target_profile") or ""))}


_TOOL_DISPATCH = {
    "create": _tool_create,
    "update": _tool_update,
    "append": _tool_append,
    "get": _tool_get,
    "list": _tool_list,
    "search": _tool_search,
    "tag": _tool_tag,
    "pin": _tool_pin,
    "unpin": _tool_unpin,
    "delete": _tool_delete,
    "share": _tool_share,
    "unshare": _tool_unshare,
}


def mobile_notes_tool(args: dict[str, Any], **_kwargs: Any) -> str:
    """Flat, forgiving Smart Notes tool for small and large LLMs alike."""
    raw_action = str((args or {}).get("action") or "").strip().lower()
    action = _ACTION_ALIASES.get(raw_action)
    if action not in _TOOL_DISPATCH:
        return json.dumps({
            "success": False,
            "error": f"Unknown action '{raw_action}'.",
            "hint": "Use one of: create, update, append, get, list, search, tag, pin, unpin, delete, share, unshare.",
            "example": {"action": "create", "title": "Trip plan", "content": "Pack passport", "tags": ["travel"]},
        }, ensure_ascii=False)
    try:
        return json.dumps(_TOOL_DISPATCH[action](args or {}), ensure_ascii=False)
    except NotesError as exc:
        payload: dict[str, Any] = {"success": False, "error": exc.message, "code": exc.code}
        if exc.hint:
            payload["hint"] = exc.hint
        if exc.example:
            payload["example"] = exc.example
        return json.dumps(payload, ensure_ascii=False)
    except ValueError:
        logger.warning("mobile_notes tool invalid parameter on action=%s", action, exc_info=True)
        return json.dumps({"success": False, "error": "Invalid parameter value.", "code": "invalid_parameter"}, ensure_ascii=False)
    except (OSError, sqlite3.Error):
        logger.warning("mobile_notes tool storage failure on action=%s", action, exc_info=True)
        return json.dumps({"success": False, "error": "Storage operation failed.", "code": "storage_error"}, ensure_ascii=False)


MOBILE_NOTES_SCHEMA: dict[str, Any] = {
    "name": "mobile_notes",
    "description": (
        "Create, update, append, search, tag, pin, delete, or explicitly share personal Smart Notes "
        "(saved privately per user profile; sharing uses the universal same-server sharing engine)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "update", "append", "get", "list", "search", "tag", "pin", "unpin", "delete", "share", "unshare"],
                "description": "Operation to perform.",
            },
            "note_id": {"type": "string", "description": "Note slug/id (optional on create; auto-generated from title)."},
            "title": {"type": "string", "description": "Short note title (required for create)."},
            "content": {"type": "string", "description": "Note body markdown/text (or text to append)."},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of short lowercase tags.",
            },
            "pinned": {"type": "boolean", "description": "Pin/unpin the note."},
            "query": {"type": "string", "description": "Search keyword for action='search'."},
            "target_profile": {"type": "string", "description": "Target profile name on this server for share/unshare."},
            "permission": {"type": "string", "enum": ["read", "edit"], "description": "Share permission ('read' or 'edit')."},
            "limit": {"type": "integer", "description": "Max items for list/search."},
        },
        "required": ["action"],
    },
}


registry.register(
    name="mobile_notes",
    toolset="memory",
    schema=MOBILE_NOTES_SCHEMA,
    handler=mobile_notes_tool,
    check_fn=lambda: True,
    emoji="📝",
)

__all__ = [
    "MOBILE_NOTES_SCHEMA",
    "NotesError",
    "create_note",
    "delete_note",
    "get_note",
    "list_notes",
    "list_shared_with_me",
    "mobile_notes_tool",
    "normalize_tags",
    "share_note",
    "slugify_note_id",
    "unshare_note",
    "update_note",
]
