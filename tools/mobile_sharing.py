"""Universal deny-by-default same-server Resource Sharing Engine (SSoT).

One single ACL engine at the lowest choking point shared by ``mobile_notes`` and
every other shareable resource on the Hermes server:
``note``, ``surface``, ``widget``, ``tool``, ``skill``, ``artifact``, ``session``.

Security invariants:
- Deny-by-default: unshared resources are completely invisible cross-profile.
- Authoritative ACLs live ONLY in the owner's ``<HERMES_HOME>/shares.db``
  (``outgoing_shares``). Recipients store ONLY a pointer in ``incoming_shares``
  so owner revocation or resource deletion cuts off access instantaneously.
- Non-owners can NEVER re-share or delete a resource.
- Never enumerates or exposes the list of profiles on the server.
"""

from __future__ import annotations

from contextlib import contextmanager, suppress
import json
import logging
from pathlib import Path
import re
import secrets
import sqlite3
import time
from typing import Any, Iterator, Optional

from hermes_constants import (
    assert_named_profile_home_live,
    get_hermes_home,
    mkdir_under_hermes_home,
)
from tools.mobile_sqlite import open_mobile_db, open_mobile_db_readonly
from tools.registry import registry

logger = logging.getLogger(__name__)

VALID_RESOURCE_KINDS = frozenset({"note", "surface", "widget", "tool", "skill", "artifact", "session"})
VALID_PERMISSIONS = frozenset({"read", "edit", "clone"})

_KIND_ALIASES = {
    "note": "note",
    "notes": "note",
    "surface": "surface",
    "card": "surface",
    "dashboard": "surface",
    "widget": "widget",
    "tool": "tool",
    "skill": "skill",
    "artifact": "artifact",
    "file": "artifact",
    "upload": "artifact",
    "session": "session",
    "thread": "session",
    "discussion": "session",
    "chat": "session",
}

_PERMISSION_ALIASES = {
    "read": "read",
    "ro": "read",
    "view": "read",
    "viewer": "read",
    "readonly": "read",
    "read-only": "read",
    "edit": "edit",
    "rw": "edit",
    "write": "edit",
    "editor": "edit",
    "readwrite": "edit",
    "clone": "clone",
    "copy": "clone",
    "fork": "clone",
    "duplicate": "clone",
}

_ACTION_ALIASES = {
    "share": "share",
    "grant": "share",
    "add": "share",
    "revoke": "revoke",
    "unshare": "revoke",
    "remove": "revoke",
    "dismiss": "revoke",
    "outgoing": "outgoing",
    "list_outgoing": "outgoing",
    "shared_by_me": "outgoing",
    "incoming": "incoming",
    "list_incoming": "incoming",
    "shared_with_me": "incoming",
    "read": "read",
    "get": "read",
    "content": "read",
    "open": "read",
}

_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SHARE_ID_RE = re.compile(r"^shr_[0-9a-f]{24}$")
_RESOURCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,159}$")
_MAX_LABEL_CHARS = 120
_MAX_ARTIFACT_PREVIEW_BYTES = 64 * 1024

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS outgoing_shares (
    share_id TEXT PRIMARY KEY,
    resource_kind TEXT NOT NULL CHECK (resource_kind IN ('note','surface','widget','tool','skill','artifact','session')),
    resource_id TEXT NOT NULL,
    target_profile TEXT NOT NULL,
    permission TEXT NOT NULL CHECK (permission IN ('read','edit','clone')),
    label TEXT NOT NULL DEFAULT '',
    granted_at REAL NOT NULL,
    UNIQUE(resource_kind, resource_id, target_profile)
);
CREATE INDEX IF NOT EXISTS idx_outgoing_kind_id ON outgoing_shares(resource_kind, resource_id);

CREATE TABLE IF NOT EXISTS incoming_shares (
    share_id TEXT NOT NULL,
    owner_profile TEXT NOT NULL,
    resource_kind TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    granted_at REAL NOT NULL,
    PRIMARY KEY (owner_profile, share_id)
);
CREATE INDEX IF NOT EXISTS idx_incoming_lookup ON incoming_shares(resource_kind, resource_id);
"""


class ShareError(Exception):
    """Typed Universal Sharing error with HTTP status, code, and LLM hint."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        code: str = "invalid_share_request",
        hint: Optional[str] = None,
        example: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.hint = hint
        self.example = example


def active_profile_name() -> str:
    """Resolve the active profile name in the current runtime scope."""
    try:
        from agent.file_safety import _resolve_active_profile_name

        return _resolve_active_profile_name() or "default"
    except (ImportError, OSError, RuntimeError):
        return "default"


def resolve_profile_home_if_exists(profile_name: str) -> Optional[Path]:
    """Resolve an explicitly named profile home without ever enumerating server profiles."""
    if not isinstance(profile_name, str):
        return None
    cleaned = profile_name.strip()
    if not _PROFILE_NAME_RE.fullmatch(cleaned):
        return None
    try:
        from hermes_cli.profiles import get_profile_dir

        candidate = Path(get_profile_dir(cleaned))
    except (ImportError, ValueError, OSError, RuntimeError):
        return None
    if candidate.is_symlink() or not candidate.is_dir():
        return None
    return candidate


def _shares_db_path(home: Optional[Path] = None) -> Path:
    base = home if home is not None else get_hermes_home()
    assert_named_profile_home_live(base)
    return base / "shares.db"


@contextmanager
def _connect_shares_db(home: Optional[Path] = None, *, write: bool = False) -> Iterator[sqlite3.Connection]:
    db_path = _shares_db_path(home)
    if write:
        mkdir_under_hermes_home(db_path.parent)
    with open_mobile_db(
        db_path,
        schema_key="shares",
        schema_sql=_SCHEMA_SQL,
        db_label="shares.db",
        write=write,
    ) as conn:
        yield conn


def normalize_resource_kind(raw: Any) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ShareError(
            "'resource_kind' is required.",
            code="invalid_resource_kind",
            hint="Use one of: note, surface, widget, tool, skill, artifact, session.",
            example={
                "action": "share",
                "resource_kind": "note",
                "resource_id": "shopping-list",
                "target_profile": "alice",
                "permission": "read",
            },
        )
    mapped = _KIND_ALIASES.get(raw.strip().lower())
    if mapped not in VALID_RESOURCE_KINDS:
        raise ShareError(
            f"Unsupported resource_kind '{raw}'.",
            code="invalid_resource_kind",
            hint="Use one of: note, surface, widget, tool, skill, artifact, session.",
        )
    return mapped


def normalize_share_permission(raw: Any, *, allowed: frozenset[str] = VALID_PERMISSIONS) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ShareError(
            "'permission' must be 'read', 'edit', or 'clone'.",
            code="invalid_permission",
            hint="Use permission='read', 'edit', or 'clone'.",
        )
    mapped = _PERMISSION_ALIASES.get(raw.strip().lower())
    if mapped not in allowed:
        valid_str = ", ".join(sorted(allowed))
        raise ShareError(
            f"Unsupported permission '{raw}'. Use one of: {valid_str}.",
            code="invalid_permission",
            hint=f"Use one of: {valid_str}.",
        )
    return mapped


def validate_resource_id(raw: Any) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ShareError("'resource_id' is required.", code="invalid_resource_id")
    cleaned = raw.strip()
    if ".." in cleaned or "\\" in cleaned or "\x00" in cleaned or not _RESOURCE_ID_RE.fullmatch(cleaned):
        raise ShareError(
            f"Invalid resource_id '{raw}'.",
            code="invalid_resource_id",
            hint="Pass the exact identifier of the resource (no path traversal).",
        )
    return cleaned


def _validate_share_id(raw: Any) -> str:
    if not isinstance(raw, str) or not _SHARE_ID_RE.fullmatch(raw.strip()):
        raise ShareError("Invalid share_id.", code="invalid_share_id")
    return raw.strip()


def _validate_label(raw: Any) -> str:
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raise ShareError("'label' must be a string.", code="invalid_label")
    try:
        raw.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ShareError("'label' must be valid UTF-8.", code="invalid_label") from exc
    return raw.strip()[:_MAX_LABEL_CHARS]


# ── Resource Metadata & Content Resolvers (Per Kind, Owner-Home Scoped) ──────


def _read_note_resource(owner_home: Path, resource_id: str, *, include_content: bool) -> Optional[dict[str, Any]]:
    db_path = owner_home / "notes.db"
    if not db_path.is_file():
        return None
    try:
        with open_mobile_db_readonly(db_path) as conn:
            row = conn.execute(
                "SELECT id, title, content, tags_json, pinned, updated_at, updated_by FROM notes WHERE id = ? AND archived = 0",
                (resource_id,),
            ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    tags = []
    with suppress(json.JSONDecodeError):
        parsed = json.loads(row["tags_json"] or "[]")
        if isinstance(parsed, list):
            tags = parsed
    meta = {
        "title": row["title"],
        "tags": tags,
        "updated_at": row["updated_at"],
    }
    if not include_content:
        return meta
    return {
        **meta,
        "id": row["id"],
        "content": row["content"],
        "pinned": bool(row["pinned"]),
        "updated_by": row["updated_by"],
    }


def _read_skill_resource(owner_home: Path, resource_id: str, *, include_content: bool) -> Optional[dict[str, Any]]:
    from tools.skills_catalog_ops import read_skill_detail, validate_skill_locator

    if validate_skill_locator(resource_id) is not None:
        return None
    from agent.runtime_scope import bind_profile_scope

    with bind_profile_scope(owner_home):
        detail, err = read_skill_detail(resource_id, platform="api_server")
    if err is not None or detail is None:
        return None
    meta = {
        "title": detail["name"],
        "description": detail.get("description", ""),
        "category": detail.get("category"),
        "tags": detail.get("tags", []),
    }
    if not include_content:
        return meta
    return {
        **meta,
        "frontmatter": detail.get("frontmatter", {}),
        "body": detail.get("body", ""),
        "raw": detail.get("raw", ""),
        "supporting_files": detail.get("supporting_files", []),
    }


def _read_session_resource(owner_home: Path, resource_id: str, *, include_content: bool) -> Optional[dict[str, Any]]:
    from persistence.conversation.messages import TranscriptRead, TranscriptView
    from persistence_assembly import open_persistence, profile_scope

    db_path = owner_home / "state.db"
    if not db_path.is_file():
        return None
    store = open_persistence(profile_scope(home=owner_home), read_only=True)
    try:
        sess = store.sessions.get(resource_id)
        if sess is None:
            return None
        meta = {
            "title": sess.title or resource_id,
            "model": sess.model,
            "started_at": sess.started_at,
            "message_count": sess.message_count,
        }
        if not include_content:
            return meta
        page = store.messages.read_transcript_page(
            resource_id,
            TranscriptRead(view=TranscriptView.ACTIVE, limit=500, offset=0, latest=False, include_lineage=True),
            visible_only=True,
        )
        messages = [
            {
                "id": int(row.id) if row.id is not None else None,
                "role": row.role,
                "content": row.as_wire_row().get("content"),
                "timestamp": row.timestamp,
            }
            for row in page.rows
        ]
        return {**meta, "session_id": resource_id, "messages": messages}
    except Exception:
        logger.debug("Shared session read failed for %s", resource_id, exc_info=True)
        return None
    finally:
        store.close()


def _match_surface_entry(surfaces: Any, resource_id: str) -> Optional[Any]:
    if not isinstance(surfaces, dict):
        return None
    if resource_id in surfaces:
        return surfaces[resource_id]
    atelier_key = f"atelier:{resource_id}"
    if atelier_key in surfaces:
        return surfaces[atelier_key]
    suffix = f":{resource_id}"
    for key in sorted(surfaces):
        spec = surfaces[key]
        if (isinstance(spec, dict) and spec.get("surface_id") == resource_id) or str(key).endswith(suffix):
            return spec
    return None


def _match_keyed_entry(bucket: Any, resource_id: str, id_field: str) -> Optional[Any]:
    if not isinstance(bucket, dict):
        return None
    if resource_id in bucket:
        return bucket[resource_id]
    for key in sorted(bucket):
        spec = bucket[key]
        if isinstance(spec, dict) and spec.get(id_field) == resource_id:
            return spec
    return None


def _lookup_ui_spec(state: dict[str, Any], resource_kind: str, resource_id: str) -> Optional[Any]:
    if resource_kind == "surface":
        return _match_surface_entry(state.get("surfaces"), resource_id)
    if resource_kind == "widget":
        return _match_keyed_entry(state.get("widgets"), resource_id, "widget_id")
    if resource_kind == "tool":
        found = _match_surface_entry(state.get("surfaces"), resource_id)
        if found is not None:
            return found
        return _match_keyed_entry(state.get("quick_actions"), resource_id, "action_id")
    return None


def _read_mobile_ui_resource(
    owner_home: Path, resource_kind: str, resource_id: str, *, include_content: bool
) -> Optional[dict[str, Any]]:
    """Read an existing surface, widget, or tool from ``mobile_state.db`` (fail-closed)."""
    db_path = owner_home / "mobile_state.db"
    if not db_path.is_file():
        return None
    try:
        with open_mobile_db_readonly(db_path) as conn:
            row = conn.execute("SELECT state_json FROM ui_active_state LIMIT 1").fetchone()
        if not row or not row["state_json"]:
            return None
        state = json.loads(row["state_json"])
        if not isinstance(state, dict):
            return None
    except (sqlite3.Error, json.JSONDecodeError):
        logger.debug("mobile_state.db read failed for %s/%s", resource_kind, resource_id, exc_info=True)
        return None

    spec = _lookup_ui_spec(state, resource_kind, resource_id)
    if spec is None:
        return None
    title = str(spec.get("title") or spec.get("label") or resource_id) if isinstance(spec, dict) else resource_id
    return {"title": title, "spec": spec} if include_content else {"title": title}


def _read_artifact_resource(owner_home: Path, resource_id: str, *, include_content: bool) -> Optional[dict[str, Any]]:
    from hermes_cli.workspace_ops import looks_binary, path_is_under

    uploads_root = owner_home / "uploads"
    if not uploads_root.is_dir():
        return None
    try:
        resolved_root = uploads_root.resolve(strict=True)
    except OSError:
        return None
    meta_match = next(
        (p for p in uploads_root.rglob(f"{resource_id}.json") if p.is_file() and not p.is_symlink()),
        None,
    )
    if meta_match is None:
        return None
    try:
        resolved_meta = meta_match.resolve(strict=True)
        if not path_is_under(resolved_root, resolved_meta):
            return None
        meta = json.loads(resolved_meta.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    stored_name = str(meta.get("stored_name") or "")
    file_path = resolved_meta.parent / stored_name
    if not stored_name or file_path.is_symlink() or not file_path.is_file():
        return None
    try:
        resolved_file = file_path.resolve(strict=True)
    except OSError:
        return None
    if not path_is_under(resolved_root, resolved_file):
        return None
    info = {
        "title": str(meta.get("filename") or resource_id),
        "filename": str(meta.get("filename") or stored_name),
        "mime_type": str(meta.get("mime_type") or "application/octet-stream"),
        "size": int(meta.get("size") or 0),
        "sha256": str(meta.get("sha256") or ""),
    }
    if not include_content:
        return info
    try:
        with resolved_file.open("rb") as handle:
            raw = handle.read(_MAX_ARTIFACT_PREVIEW_BYTES + 1)
    except OSError:
        return None
    truncated = len(raw) > _MAX_ARTIFACT_PREVIEW_BYTES
    sample = raw[:_MAX_ARTIFACT_PREVIEW_BYTES]
    if looks_binary(sample):
        return {**info, "is_binary": True, "text_preview": None, "truncated": truncated}
    try:
        text = sample.decode("utf-8")
    except UnicodeDecodeError:
        return {**info, "is_binary": True, "text_preview": None, "truncated": truncated}
    return {**info, "is_binary": False, "text_preview": text, "truncated": truncated}


def _resolve_resource(
    owner_home: Path,
    resource_kind: str,
    resource_id: str,
    *,
    include_content: bool,
) -> Optional[dict[str, Any]]:
    if resource_kind == "note":
        return _read_note_resource(owner_home, resource_id, include_content=include_content)
    if resource_kind == "skill":
        return _read_skill_resource(owner_home, resource_id, include_content=include_content)
    if resource_kind == "session":
        return _read_session_resource(owner_home, resource_id, include_content=include_content)
    if resource_kind == "artifact":
        return _read_artifact_resource(owner_home, resource_id, include_content=include_content)
    if resource_kind in ("surface", "widget", "tool"):
        return _read_mobile_ui_resource(owner_home, resource_kind, resource_id, include_content=include_content)
    return None


# ── Core Universal Sharing ACL API ───────────────────────────────────────────


def _outgoing_row_dict(row: sqlite3.Row, owner_profile: str) -> dict[str, Any]:
    return {
        "share_id": row["share_id"],
        "owner_profile": owner_profile,
        "resource_kind": row["resource_kind"],
        "resource_id": row["resource_id"],
        "target_profile": row["target_profile"],
        "permission": row["permission"],
        "label": row["label"] or "",
        "granted_at": row["granted_at"],
    }


def grant_share(
    resource_kind: str,
    resource_id: str,
    target_profile: str,
    permission: str = "read",
    *,
    label: str = "",
) -> dict[str, Any]:
    """Explicitly grant ``target_profile`` access to an owned resource (owner-only)."""
    kind = normalize_resource_kind(resource_kind)
    res_id = validate_resource_id(resource_id)
    perm = normalize_share_permission(permission)
    clean_label = _validate_label(label)
    current = active_profile_name()
    target = str(target_profile or "").strip()
    if not target or not _PROFILE_NAME_RE.fullmatch(target):
        raise ShareError(
            "Invalid target_profile name.",
            status_code=400,
            code="invalid_target_profile",
            hint="Provide the exact profile name of another user on this Hermes server.",
        )
    if target == current:
        raise ShareError("Cannot share a resource with your own profile.", status_code=400, code="cannot_share_with_self")
    target_home = resolve_profile_home_if_exists(target)
    if target_home is None:
        raise ShareError("Target profile was not found on this server.", status_code=404, code="target_profile_not_found")

    owner_home = get_hermes_home()
    meta = _resolve_resource(owner_home, kind, res_id, include_content=False)
    if meta is None:
        if check_incoming_access(current, kind, res_id) is not None:
            raise ShareError(
                "Only the resource owner can share it with other users.",
                status_code=403,
                code="share_owner_required",
            )
        raise ShareError(f"{kind.capitalize()} '{res_id}' not found.", status_code=404, code="resource_not_found")

    effective_label = clean_label or str(meta.get("title") or res_id)[:_MAX_LABEL_CHARS]
    now = time.time()
    with _connect_shares_db(write=True) as conn:
        existing = conn.execute(
            "SELECT share_id FROM outgoing_shares WHERE resource_kind = ? AND resource_id = ? AND target_profile = ?",
            (kind, res_id, target),
        ).fetchone()
        share_id = str(existing["share_id"]) if existing else f"shr_{secrets.token_hex(12)}"
        conn.execute(
            """
            INSERT INTO outgoing_shares (share_id, resource_kind, resource_id, target_profile, permission, label, granted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(resource_kind, resource_id, target_profile)
            DO UPDATE SET permission = excluded.permission, label = excluded.label, granted_at = excluded.granted_at
            """,
            (share_id, kind, res_id, target, perm, effective_label, now),
        )
        row = conn.execute("SELECT * FROM outgoing_shares WHERE share_id = ?", (share_id,)).fetchone()
        share_dict = _outgoing_row_dict(row, current)

    with _connect_shares_db(target_home, write=True) as tconn:
        tconn.execute(
            """
            INSERT INTO incoming_shares (share_id, owner_profile, resource_kind, resource_id, granted_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(owner_profile, share_id)
            DO UPDATE SET granted_at = excluded.granted_at
            """,
            (share_id, current, kind, res_id, now),
        )
    logger.info("Shared %s/%s from '%s' -> '%s' (%s, id=%s)", kind, res_id, current, target, perm, share_id)
    return share_dict


def list_outgoing_shares(
    *,
    resource_kind: Optional[str] = None,
    resource_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """List resources the active profile has shared with others."""
    owner = active_profile_name()
    clauses: list[str] = []
    params: list[Any] = []
    if resource_kind is not None:
        clauses.append("resource_kind = ?")
        params.append(normalize_resource_kind(resource_kind))
    if resource_id is not None:
        clauses.append("resource_id = ?")
        params.append(validate_resource_id(resource_id))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connect_shares_db(write=False) as conn:
        rows = conn.execute(
            f"SELECT * FROM outgoing_shares {where} ORDER BY granted_at DESC",
            tuple(params),
        ).fetchall()
    return [_outgoing_row_dict(row, owner) for row in rows]


def check_incoming_access(
    current_profile: str,
    resource_kind: str,
    resource_id: str,
    *,
    owner_hint: Optional[str] = None,
) -> Optional[tuple[str, Path, str, str]]:
    """Verify LIVE against owner's ``outgoing_shares`` whether ``current_profile`` has access.

    Returns ``(owner_profile, owner_home, permission, share_id)`` or None.
    """
    with _connect_shares_db(write=False) as conn:
        if owner_hint:
            ptrs = conn.execute(
                "SELECT share_id, owner_profile FROM incoming_shares WHERE resource_kind = ? AND resource_id = ? AND owner_profile = ?",
                (resource_kind, resource_id, owner_hint),
            ).fetchall()
        else:
            ptrs = conn.execute(
                "SELECT share_id, owner_profile FROM incoming_shares WHERE resource_kind = ? AND resource_id = ? ORDER BY granted_at DESC",
                (resource_kind, resource_id),
            ).fetchall()
    stale: list[tuple[str, str]] = []
    found: Optional[tuple[str, Path, str, str]] = None
    for ptr in ptrs:
        owner_name = str(ptr["owner_profile"])
        share_id = str(ptr["share_id"])
        owner_home = resolve_profile_home_if_exists(owner_name)
        if owner_home is None:
            stale.append((owner_name, share_id))
            continue
        with _connect_shares_db(owner_home, write=False) as oconn:
            grant = oconn.execute(
                "SELECT permission FROM outgoing_shares WHERE share_id = ? AND target_profile = ?",
                (share_id, current_profile),
            ).fetchone()
        if grant is None:
            stale.append((owner_name, share_id))
            continue
        found = (owner_name, owner_home, str(grant["permission"]), share_id)
        break
    if stale:
        with suppress(OSError, sqlite3.Error):
            with _connect_shares_db(write=True) as conn:
                conn.executemany("DELETE FROM incoming_shares WHERE owner_profile = ? AND share_id = ?", stale)
    return found


def list_incoming_shares(*, resource_kind: Optional[str] = None) -> list[dict[str, Any]]:
    """List live shares granted to the active profile, with safe resolved metadata."""
    current = active_profile_name()
    kind_filter = normalize_resource_kind(resource_kind) if resource_kind is not None else None
    with _connect_shares_db(write=False) as conn:
        if kind_filter:
            ptrs = conn.execute(
                "SELECT * FROM incoming_shares WHERE resource_kind = ? ORDER BY granted_at DESC",
                (kind_filter,),
            ).fetchall()
        else:
            ptrs = conn.execute("SELECT * FROM incoming_shares ORDER BY granted_at DESC").fetchall()

    items: list[dict[str, Any]] = []
    stale: list[tuple[str, str]] = []
    for ptr in ptrs:
        owner_name = str(ptr["owner_profile"])
        share_id = str(ptr["share_id"])
        kind = str(ptr["resource_kind"])
        res_id = str(ptr["resource_id"])
        owner_home = resolve_profile_home_if_exists(owner_name)
        if owner_home is None:
            stale.append((owner_name, share_id))
            continue
        with _connect_shares_db(owner_home, write=False) as oconn:
            grant = oconn.execute(
                "SELECT permission, label, granted_at FROM outgoing_shares WHERE share_id = ? AND target_profile = ?",
                (share_id, current),
            ).fetchone()
        if grant is None:
            stale.append((owner_name, share_id))
            continue
        meta = _resolve_resource(owner_home, kind, res_id, include_content=False)
        if meta is None:
            stale.append((owner_name, share_id))
            continue
        items.append({
            "share_id": share_id,
            "owner_profile": owner_name,
            "resource_kind": kind,
            "resource_id": res_id,
            "permission": str(grant["permission"]),
            "label": str(grant["label"] or meta.get("title") or res_id),
            "granted_at": grant["granted_at"],
            "metadata": meta,
        })
    if stale:
        with suppress(OSError, sqlite3.Error):
            with _connect_shares_db(write=True) as conn:
                conn.executemany("DELETE FROM incoming_shares WHERE owner_profile = ? AND share_id = ?", stale)
    return items


def read_incoming_share_content(share_id: str) -> dict[str, Any]:
    """Serve authorized resource content for an active incoming share (instant revoke respected)."""
    clean_id = _validate_share_id(share_id)
    current = active_profile_name()
    with _connect_shares_db(write=False) as conn:
        ptr = conn.execute(
            "SELECT owner_profile, resource_kind, resource_id FROM incoming_shares WHERE share_id = ?",
            (clean_id,),
        ).fetchone()
    if ptr is None:
        raise ShareError("Shared resource not found.", status_code=404, code="share_not_found")
    owner_name = str(ptr["owner_profile"])
    kind = str(ptr["resource_kind"])
    res_id = str(ptr["resource_id"])
    owner_home = resolve_profile_home_if_exists(owner_name)
    if owner_home is None:
        raise ShareError("Shared resource is no longer available.", status_code=404, code="share_not_found")
    with _connect_shares_db(owner_home, write=False) as oconn:
        grant = oconn.execute(
            "SELECT permission, label, granted_at FROM outgoing_shares WHERE share_id = ? AND target_profile = ?",
            (clean_id, current),
        ).fetchone()
    if grant is None:
        with suppress(OSError, sqlite3.Error):
            with _connect_shares_db(write=True) as conn:
                conn.execute("DELETE FROM incoming_shares WHERE owner_profile = ? AND share_id = ?", (owner_name, clean_id))
        raise ShareError("Access to this shared resource has been revoked.", status_code=403, code="share_revoked")

    content = _resolve_resource(owner_home, kind, res_id, include_content=True)
    if content is None:
        raise ShareError("Shared resource no longer exists.", status_code=404, code="resource_not_found")
    return {
        "share_id": clean_id,
        "owner_profile": owner_name,
        "resource_kind": kind,
        "resource_id": res_id,
        "permission": str(grant["permission"]),
        "label": str(grant["label"] or res_id),
        "granted_at": grant["granted_at"],
        "content": content,
    }


def revoke_or_dismiss_share(share_id: str) -> dict[str, Any]:
    """Owner instant revoke OR recipient dismiss for ``share_id``."""
    clean_id = _validate_share_id(share_id)
    current = active_profile_name()
    # 1. Owner revoke path (release owner lock/txn before opening target DB to avoid AB-BA deadlock)
    revoked_target: Optional[str] = None
    with _connect_shares_db(write=True) as conn:
        row = conn.execute(
            "SELECT share_id, resource_kind, resource_id, target_profile FROM outgoing_shares WHERE share_id = ?",
            (clean_id,),
        ).fetchone()
        if row is not None:
            revoked_target = str(row["target_profile"])
            conn.execute("DELETE FROM outgoing_shares WHERE share_id = ?", (clean_id,))
    if revoked_target is not None:
        target_home = resolve_profile_home_if_exists(revoked_target)
        if target_home is not None:
            with suppress(OSError, sqlite3.Error):
                with _connect_shares_db(target_home, write=True) as tconn:
                    tconn.execute(
                        "DELETE FROM incoming_shares WHERE owner_profile = ? AND share_id = ?",
                        (current, clean_id),
                    )
        logger.info("Owner '%s' revoked share %s from '%s'", current, clean_id, revoked_target)
        return {"share_id": clean_id, "action": "revoked", "target_profile": revoked_target}

    # 2. Recipient dismiss path
    with _connect_shares_db(write=True) as conn:
        ptr = conn.execute(
            "SELECT owner_profile FROM incoming_shares WHERE share_id = ?",
            (clean_id,),
        ).fetchone()
        if ptr is None:
            raise ShareError("Share not found.", status_code=404, code="share_not_found")
        owner_name = str(ptr["owner_profile"])
        conn.execute("DELETE FROM incoming_shares WHERE owner_profile = ? AND share_id = ?", (owner_name, clean_id))
    owner_home = resolve_profile_home_if_exists(owner_name)
    if owner_home is not None:
        with suppress(OSError, sqlite3.Error):
            with _connect_shares_db(owner_home, write=True) as oconn:
                oconn.execute(
                    "DELETE FROM outgoing_shares WHERE share_id = ? AND target_profile = ?",
                    (clean_id, current),
                )
    logger.info("Recipient '%s' dismissed share %s from '%s'", current, clean_id, owner_name)
    return {"share_id": clean_id, "action": "dismissed", "owner_profile": owner_name}


def revoke_share_by_target(resource_kind: str, resource_id: str, target_profile: str) -> dict[str, Any]:
    """Owner revoke by ``(resource_kind, resource_id, target_profile)`` (used by ``/api/notes/{id}/shares/{target}``)."""
    kind = normalize_resource_kind(resource_kind)
    res_id = validate_resource_id(resource_id)
    target = str(target_profile or "").strip()
    if not target or not _PROFILE_NAME_RE.fullmatch(target):
        raise ShareError("Invalid target_profile name.", status_code=400, code="invalid_target_profile")
    current = active_profile_name()
    with _connect_shares_db(write=False) as conn:
        row = conn.execute(
            "SELECT share_id FROM outgoing_shares WHERE resource_kind = ? AND resource_id = ? AND target_profile = ?",
            (kind, res_id, target),
        ).fetchone()
    if row is None:
        if check_incoming_access(current, kind, res_id) is not None:
            raise ShareError(
                "Only the resource owner can revoke shares.",
                status_code=403,
                code="share_owner_required",
            )
        raise ShareError(f"No share found for '{target}' on {kind} '{res_id}'.", status_code=404, code="share_not_found")
    return revoke_or_dismiss_share(str(row["share_id"]))


def revoke_all_shares_for_resource(resource_kind: str, resource_id: str) -> int:
    """Clean up every outgoing share & recipient pointer when an owner deletes a resource."""
    current = active_profile_name()
    removed: list[tuple[str, str]] = []
    with _connect_shares_db(write=True) as conn:
        rows = conn.execute(
            "SELECT share_id, target_profile FROM outgoing_shares WHERE resource_kind = ? AND resource_id = ?",
            (resource_kind, resource_id),
        ).fetchall()
        removed = [(str(r["share_id"]), str(r["target_profile"])) for r in rows]
        conn.execute(
            "DELETE FROM outgoing_shares WHERE resource_kind = ? AND resource_id = ?",
            (resource_kind, resource_id),
        )
    for share_id, target in removed:
        target_home = resolve_profile_home_if_exists(target)
        if target_home is not None:
            with suppress(OSError, sqlite3.Error):
                with _connect_shares_db(target_home, write=True) as tconn:
                    tconn.execute(
                        "DELETE FROM incoming_shares WHERE owner_profile = ? AND share_id = ?",
                        (current, share_id),
                    )
    return len(removed)


# ── Forgiving Small-LLM Tool (``mobile_share``) ──────────────────────────────


def _tool_share(args: dict[str, Any]) -> dict[str, Any]:
    return {"success": True, "share": grant_share(
        args.get("resource_kind", ""),
        args.get("resource_id", ""),
        str(args.get("target_profile") or ""),
        str(args.get("permission") or "read"),
        label=str(args.get("label") or ""),
    )}


def _tool_revoke(args: dict[str, Any]) -> dict[str, Any]:
    if args.get("share_id"):
        return {"success": True, **revoke_or_dismiss_share(str(args["share_id"]))}
    return {"success": True, **revoke_share_by_target(
        args.get("resource_kind", ""),
        args.get("resource_id", ""),
        str(args.get("target_profile") or ""),
    )}


def _tool_outgoing(args: dict[str, Any]) -> dict[str, Any]:
    return {"success": True, "shares": list_outgoing_shares(
        resource_kind=args.get("resource_kind"),
        resource_id=args.get("resource_id"),
    )}


def _tool_incoming(args: dict[str, Any]) -> dict[str, Any]:
    return {"success": True, "shares": list_incoming_shares(resource_kind=args.get("resource_kind"))}


def _tool_read(args: dict[str, Any]) -> dict[str, Any]:
    return {"success": True, **read_incoming_share_content(str(args.get("share_id") or ""))}


_TOOL_DISPATCH = {
    "share": _tool_share,
    "revoke": _tool_revoke,
    "outgoing": _tool_outgoing,
    "incoming": _tool_incoming,
    "read": _tool_read,
}


def mobile_share_tool(args: dict[str, Any], **_kwargs: Any) -> str:
    """Forgiving universal sharing tool for small and large LLMs alike."""
    raw_action = str((args or {}).get("action") or "").strip().lower()
    action = _ACTION_ALIASES.get(raw_action)
    if action not in _TOOL_DISPATCH:
        return json.dumps({
            "success": False,
            "error": f"Unknown action '{raw_action}'.",
            "hint": "Use one of: share, revoke, outgoing, incoming, read.",
            "example": {
                "action": "share",
                "resource_kind": "note",
                "resource_id": "shopping-list",
                "target_profile": "alice",
                "permission": "read",
            },
        }, ensure_ascii=False)
    try:
        return json.dumps(_TOOL_DISPATCH[action](args or {}), ensure_ascii=False)
    except ShareError as exc:
        payload: dict[str, Any] = {"success": False, "error": exc.message, "code": exc.code}
        if exc.hint:
            payload["hint"] = exc.hint
        if exc.example:
            payload["example"] = exc.example
        return json.dumps(payload, ensure_ascii=False)
    except ValueError:
        logger.warning("mobile_share tool invalid parameter on action=%s", action, exc_info=True)
        return json.dumps({"success": False, "error": "Invalid parameter value.", "code": "invalid_parameter"}, ensure_ascii=False)
    except (OSError, sqlite3.Error):
        logger.warning("mobile_share tool storage failure on action=%s", action, exc_info=True)
        return json.dumps({"success": False, "error": "Storage operation failed.", "code": "storage_error"}, ensure_ascii=False)


MOBILE_SHARE_SCHEMA: dict[str, Any] = {
    "name": "mobile_share",
    "description": (
        "Explicitly share or revoke access to any owned resource (note, surface, widget, tool, skill, "
        "artifact, or session) with another named user profile on this same Hermes server."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["share", "revoke", "outgoing", "incoming", "read"],
                "description": "Operation to perform.",
            },
            "resource_kind": {
                "type": "string",
                "enum": ["note", "surface", "widget", "tool", "skill", "artifact", "session"],
                "description": "Type of resource to share or filter.",
            },
            "resource_id": {"type": "string", "description": "Identifier of the resource."},
            "target_profile": {"type": "string", "description": "Target user profile name on this server."},
            "permission": {
                "type": "string",
                "enum": ["read", "edit", "clone"],
                "description": "Permission to grant ('read', 'edit', or 'clone').",
            },
            "label": {"type": "string", "description": "Optional friendly display label."},
            "share_id": {"type": "string", "description": "Share ID for revoke or read."},
        },
        "required": ["action"],
    },
}

registry.register(
    name="mobile_share",
    toolset="memory",
    schema=MOBILE_SHARE_SCHEMA,
    handler=mobile_share_tool,
    check_fn=lambda: True,
    emoji="🤝",
)

__all__ = [
    "MOBILE_SHARE_SCHEMA",
    "VALID_PERMISSIONS",
    "VALID_RESOURCE_KINDS",
    "ShareError",
    "active_profile_name",
    "check_incoming_access",
    "grant_share",
    "list_incoming_shares",
    "list_outgoing_shares",
    "mobile_share_tool",
    "normalize_resource_kind",
    "normalize_share_permission",
    "read_incoming_share_content",
    "resolve_profile_home_if_exists",
    "revoke_all_shares_for_resource",
    "revoke_or_dismiss_share",
    "revoke_share_by_target",
    "validate_resource_id",
]
