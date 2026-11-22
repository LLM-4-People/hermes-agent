"""Framework-agnostic workspace browsing & read-only Git operations (SSoT).

Strictly jailed to a session's authorized working directory: canonical path
containment, symlink-escape rejection, secret/credential/.git blocking, bounded
tree traversal, bounded UTF-8 text reads, and read-only Git diffs via
``hermes_cli.web_git``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Optional
import urllib.parse

from agent.file_safety import (
    HOME_CREDENTIAL_DIRS,
    HOME_CREDENTIAL_FILES,
    deny_key,
    deny_path_under,
    resolved_home_credential_stores,
)
from hermes_cli.home_data_layout import CREDENTIAL_STORE_DIRS, HOME_ENV_FILES, SECRET_CACHE_STORES
from hermes_cli.web_git import file_diff_vs_head, repo_status, review_commit_context, review_list

WORKSPACE_FILE_MAX_BYTES = 256 * 1024  # 256 KiB hard cap per file read
WORKSPACE_TREE_MAX_DEPTH = 3
WORKSPACE_TREE_MAX_ENTRIES = 200
WORKSPACE_DIFF_MAX_CHARS = 120_000
_LINK_CHAIN_MAX_HOPS = 40

_SKIPPED_TREE_DIRS = frozenset({
    ".git",
    ".hg",
    ".svn",
    ".cache",
    ".next",
    ".turbo",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "venv",
})

_SECRET_BASENAMES = frozenset({
    "auth.json",
    "auth.lock",
    "credentials",
    ".anthropic_oauth.json",
    "google_token.json",
    "google_client_secret.json",
    "google_oauth_pending.json",
    "google_oauth.json",
    "webhook_subscriptions.json",
    "vault.key",
    "vault.json.enc",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
    *HOME_ENV_FILES,
    *(PurePosixPath(store).name.lower() for store in SECRET_CACHE_STORES),
    *(name.lower() for name in HOME_CREDENTIAL_FILES),
})

_SECRET_EXTENSIONS = frozenset({".pem", ".key", ".p12", ".pfx", ".kdbx", ".jks"})

_SECRET_DIR_SEQUENCES = (
    *(tuple(PurePosixPath(store).parts) for store in CREDENTIAL_STORE_DIRS if store != "vault"),
    *(tuple(d.lower().split("/")) for d in HOME_CREDENTIAL_DIRS),
    (".git",),
)

_LANGUAGE_BY_EXT = {
    ".c": "c",
    ".conf": "ini",
    ".cpp": "cpp",
    ".css": "css",
    ".csv": "csv",
    ".go": "go",
    ".gradle": "groovy",
    ".graphql": "graphql",
    ".h": "c",
    ".hpp": "cpp",
    ".html": "html",
    ".java": "java",
    ".js": "javascript",
    ".json": "json",
    ".jsx": "jsx",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".lua": "lua",
    ".md": "markdown",
    ".mjs": "javascript",
    ".py": "python",
    ".rb": "ruby",
    ".rs": "rust",
    ".sh": "shell",
    ".sql": "sql",
    ".svg": "xml",
    ".toml": "toml",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".txt": "text",
    ".xml": "xml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".zsh": "shell",
}

_DRIVE_PREFIX_RE = re.compile(r"^[A-Za-z]:")


class WorkspaceError(Exception):
    """Typed workspace read error mapped cleanly to HTTP responses."""

    def __init__(self, message: str, *, status_code: int = 400, code: str = "workspace_error"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


def path_is_under(root: Path, target: Path) -> bool:
    """True when ``target`` equals ``root`` or is strictly inside ``root``."""
    return target == root or root in target.parents


def detect_language(path: Path) -> str:
    """Best-effort syntax language name from filename extension."""
    return _LANGUAGE_BY_EXT.get(path.suffix.lower(), "text")


def looks_binary(data: bytes) -> bool:
    """True if ``data`` contains NUL bytes or a high ratio of control bytes."""
    if not data:
        return False
    if b"\x00" in data:
        return True
    suspicious = sum(1 for byte in data if byte < 32 and byte not in (9, 10, 13))
    return (suspicious / len(data)) > 0.12


def _is_secret_filename(name: str) -> bool:
    lowered = name.lower()
    if lowered == ".env" or lowered.startswith(".env.") or lowered == ".envrc":
        return True
    if lowered in _SECRET_BASENAMES:
        return True
    return Path(lowered).suffix in _SECRET_EXTENSIONS


def _has_secret_component(path: Path) -> bool:
    if path.name and _is_secret_filename(path.name):
        return True
    parts = tuple(part.lower() for part in path.parts)
    return any(
        parts[i : i + len(seq)] == seq
        for seq in _SECRET_DIR_SEQUENCES
        for i in range(len(parts) - len(seq) + 1)
    )


def _walk_symlink_hops(start: Path) -> Optional[list[Path]]:
    """Return every path traversed by a symlink chain from ``start`` (None on loop)."""
    hops: list[Path] = [start]
    current = start
    for hop in range(_LINK_CHAIN_MAX_HOPS + 1):
        try:
            link_target = os.readlink(current)
        except OSError:
            return hops
        if hop == _LINK_CHAIN_MAX_HOPS:
            return None
        current = Path(os.path.dirname(str(current)), link_target)
        hops.append(current)
        with_parent = Path(os.path.realpath(current.parent), current.name)
        hops.append(with_parent)
    return None


def is_secret_workspace_path(root: Path, candidate: Path, resolved: Path) -> bool:
    """Fail-closed secret/credential/.git check over raw path, symlink hops, and realpath."""
    hops = _walk_symlink_hops(candidate)
    if hops is None:
        return True
    for hop in (candidate, resolved, *hops):
        if _has_secret_component(hop):
            return True
    try:
        real_str = os.path.realpath(resolved)
        files, dirs = resolved_home_credential_stores()
    except (OSError, ValueError):
        return True
    if deny_key(real_str) in {deny_key(f) for f in files}:
        return True
    if any(deny_path_under(real_str, d) for d in dirs):
        return True
    # Protect Hermes config.yaml if the workspace happens to point inside HERMES_HOME.
    if resolved.name.lower() == "config.yaml":
        try:
            from hermes_constants import get_hermes_home

            home = get_hermes_home().resolve(strict=False)
            if path_is_under(home, resolved):
                return True
        except (ImportError, OSError, RuntimeError):
            return True
    return False


def _validate_rel_path(rel_path: Optional[str]) -> str:
    text = str(rel_path or "").strip()
    if not text or text == ".":
        return ""
    if "\x00" in text or "\r" in text or "\n" in text:
        raise WorkspaceError("Invalid path", status_code=400, code="invalid_workspace_path")
    unquoted = urllib.parse.unquote(text)
    for variant in (text, unquoted):
        if variant.startswith("/") or variant.startswith("\\") or _DRIVE_PREFIX_RE.match(variant):
            raise WorkspaceError("Path must be relative to the session workspace", status_code=400, code="invalid_workspace_path")
        segments = re.split(r"[\\/]", variant)
        if any(seg == ".." for seg in segments):
            raise WorkspaceError("Path traversal ('..') is not allowed", status_code=400, code="workspace_traversal")
    return text.replace("\\", "/").strip("/")


def _check_intermediate_symlinks(root: Path, rel_norm: str) -> None:
    """Reject any path component whose symlink escapes ``root``."""
    if not rel_norm:
        return
    current = root
    for segment in rel_norm.split("/"):
        current = current / segment
        if current.is_symlink():
            try:
                hop_target = current.resolve(strict=False)
            except (OSError, RuntimeError) as exc:
                raise WorkspaceError("Unresolvable symlink", status_code=403, code="workspace_escape") from exc
            if not path_is_under(root, hop_target):
                raise WorkspaceError("Symlink escapes the session workspace", status_code=403, code="workspace_escape")
            if is_secret_workspace_path(root, current, hop_target):
                raise WorkspaceError("Access to sensitive path is blocked", status_code=403, code="workspace_secret_blocked")


def resolve_workspace_path(
    root_dir: str | Path,
    rel_path: Optional[str] = "",
    *,
    require_exists: bool = True,
) -> tuple[Path, Path, str]:
    """Resolve ``rel_path`` strictly inside ``root_dir``.

    Returns ``(resolved_root, resolved_target, normalized_rel_path)``.
    """
    try:
        root = Path(root_dir).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise WorkspaceError("Session workspace directory is unavailable", status_code=404, code="workspace_unavailable") from exc
    if not root.is_dir():
        raise WorkspaceError("Session workspace is not a directory", status_code=404, code="workspace_unavailable")

    rel_norm = _validate_rel_path(rel_path)
    if not rel_norm:
        return root, root, ""

    candidate = root / rel_norm
    _check_intermediate_symlinks(root, rel_norm)
    try:
        resolved = candidate.resolve(strict=require_exists)
    except FileNotFoundError as exc:
        raise WorkspaceError(f"Path not found: {rel_norm}", status_code=404, code="workspace_not_found") from exc
    except (OSError, RuntimeError) as exc:
        raise WorkspaceError("Invalid workspace path", status_code=400, code="invalid_workspace_path") from exc

    if not path_is_under(root, resolved):
        raise WorkspaceError("Path escapes the session workspace", status_code=403, code="workspace_escape")
    if is_secret_workspace_path(root, candidate, resolved):
        raise WorkspaceError("Access to sensitive path is blocked", status_code=403, code="workspace_secret_blocked")
    return root, resolved, rel_norm


def _git_status_map(root: Path) -> dict[str, str]:
    status = repo_status(str(root))
    if not status:
        return {}
    mapping: dict[str, str] = {}
    for row in status.get("files") or ():
        rel = str(row.get("path") or "").strip("/")
        if not rel:
            continue
        if row.get("conflicted"):
            code = "U"
        elif row.get("untracked"):
            code = "?"
        elif row.get("staged") and row.get("unstaged"):
            code = "MM"
        elif row.get("staged"):
            code = "A"
        else:
            code = "M"
        mapping[rel] = code
    return mapping


def list_workspace_tree(
    root_dir: str | Path,
    rel_path: Optional[str] = "",
    *,
    depth: int = 2,
    limit: int = WORKSPACE_TREE_MAX_ENTRIES,
) -> dict[str, Any]:
    """Bounded, read-only directory tree strictly jailed to ``root_dir``."""
    if depth < 1 or depth > WORKSPACE_TREE_MAX_DEPTH:
        raise WorkspaceError(f"depth must be between 1 and {WORKSPACE_TREE_MAX_DEPTH}", status_code=400, code="invalid_depth")
    if limit < 1 or limit > WORKSPACE_TREE_MAX_ENTRIES:
        raise WorkspaceError(f"limit must be between 1 and {WORKSPACE_TREE_MAX_ENTRIES}", status_code=400, code="invalid_limit")

    root, start_dir, rel_norm = resolve_workspace_path(root_dir, rel_path, require_exists=True)
    if not start_dir.is_dir():
        raise WorkspaceError("Target path is not a directory", status_code=400, code="not_a_directory")

    git_map = _git_status_map(root)
    entries: list[dict[str, Any]] = []
    truncated = False

    def _walk(current_dir: Path, current_depth: int) -> None:
        nonlocal truncated
        try:
            children = sorted(os.scandir(current_dir), key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower()))
        except OSError:
            return
        for entry in children:
            if len(entries) >= limit:
                truncated = True
                return
            name = entry.name
            if name in _SKIPPED_TREE_DIRS:
                continue
            entry_path = Path(entry.path)
            try:
                resolved = entry_path.resolve(strict=True)
                st = resolved.stat()
            except (OSError, RuntimeError):
                continue
            if not path_is_under(root, resolved):
                continue
            if is_secret_workspace_path(root, entry_path, resolved):
                continue
            is_dir = stat.S_ISDIR(st.st_mode)
            rel_entry = resolved.relative_to(root).as_posix()
            entries.append({
                "path": rel_entry,
                "name": name,
                "kind": "directory" if is_dir else "file",
                "size": None if is_dir else st.st_size,
                "language": None if is_dir else detect_language(resolved),
                "git_status": git_map.get(rel_entry),
            })
            if is_dir and current_depth < depth and not entry.is_symlink():
                _walk(resolved, current_depth + 1)

    _walk(start_dir, 1)
    return {
        "root_label": root.name or str(root),
        "path": rel_norm,
        "entries": entries,
        "truncated": truncated,
    }


def read_workspace_file(
    root_dir: str | Path,
    rel_path: str,
    *,
    start_line: int = 1,
    end_line: Optional[int] = None,
    max_bytes: int = WORKSPACE_FILE_MAX_BYTES,
) -> dict[str, Any]:
    """Read a bounded UTF-8 text file inside ``root_dir``."""
    if not str(rel_path or "").strip():
        raise WorkspaceError("path query parameter is required", status_code=400, code="missing_path")
    if start_line < 1:
        raise WorkspaceError("start_line must be >= 1", status_code=400, code="invalid_line_range")
    if end_line is not None and end_line < start_line:
        raise WorkspaceError("end_line must be >= start_line", status_code=400, code="invalid_line_range")

    root, resolved, _ = resolve_workspace_path(root_dir, rel_path, require_exists=True)
    if not resolved.is_file():
        raise WorkspaceError("Target path is not a regular file", status_code=400, code="not_a_file")

    try:
        size_bytes = resolved.stat().st_size
        with resolved.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
    except PermissionError as exc:
        raise WorkspaceError("Permission denied reading file", status_code=403, code="permission_denied") from exc
    except OSError as exc:
        raise WorkspaceError("Could not read workspace file", status_code=500, code="read_failed") from exc

    truncated = len(raw) > max_bytes
    if truncated:
        raw = raw[:max_bytes]
    if looks_binary(raw):
        raise WorkspaceError("Binary files cannot be read as text", status_code=415, code="binary_file_unsupported")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WorkspaceError("File is not valid UTF-8 text", status_code=415, code="binary_file_unsupported") from exc

    all_lines = text.splitlines()
    total_lines = len(all_lines)
    slice_end = total_lines if end_line is None else min(end_line, total_lines)
    selected = all_lines[start_line - 1 : slice_end] if start_line <= total_lines else []
    rel_posix = resolved.relative_to(root).as_posix()
    return {
        "path": rel_posix,
        "language": detect_language(resolved),
        "total_lines": total_lines,
        "start_line": start_line,
        "end_line": slice_end if selected else start_line - 1,
        "lines": selected,
        "content": "\n".join(selected),
        "truncated": truncated,
        "size_bytes": size_bytes,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def read_workspace_diff(
    root_dir: str | Path,
    *,
    file_path: Optional[str] = None,
    max_diff_chars: int = WORKSPACE_DIFF_MAX_CHARS,
) -> dict[str, Any]:
    """Read-only Git status and unified diff inside ``root_dir`` (secret paths filtered)."""
    root, _, _ = resolve_workspace_path(root_dir, "", require_exists=True)
    status = repo_status(str(root))
    if status is None:
        return {
            "is_git_repo": False,
            "branch": None,
            "ahead": 0,
            "behind": 0,
            "files": [],
            "diff": "",
            "truncated": False,
        }

    def _allowed_path(candidate: Path) -> bool:
        try:
            resolved = candidate.resolve(strict=False)
        except (OSError, RuntimeError):
            return False
        return path_is_under(root, resolved) and not is_secret_workspace_path(root, candidate, resolved)

    review = review_list(str(root), scope="working", base_ref=None)
    safe_files = []
    for item in review.get("files") or ():
        rel = str(item.get("path") or "")
        if not rel or not _allowed_path(root / rel):
            continue
        safe_files.append({
            "path": rel,
            "status": item.get("status", "M"),
            "additions": int(item.get("added") or 0),
            "deletions": int(item.get("removed") or 0),
            "staged": bool(item.get("staged")),
        })

    if file_path is not None and str(file_path).strip():
        _, _, rel_norm = resolve_workspace_path(root, file_path, require_exists=False)
        diff_text = file_diff_vs_head(str(root), rel_norm)
    else:
        diff_text = review_commit_context(str(root), path_allowed=_allowed_path).get("diff", "")

    truncated = len(diff_text) > max_diff_chars
    if truncated:
        diff_text = diff_text[:max_diff_chars]

    return {
        "is_git_repo": True,
        "branch": status.get("branch"),
        "ahead": int(status.get("ahead") or 0),
        "behind": int(status.get("behind") or 0),
        "files": safe_files,
        "diff": diff_text,
        "truncated": truncated,
    }


__all__ = [
    "WORKSPACE_DIFF_MAX_CHARS",
    "WORKSPACE_FILE_MAX_BYTES",
    "WORKSPACE_TREE_MAX_DEPTH",
    "WORKSPACE_TREE_MAX_ENTRIES",
    "WorkspaceError",
    "detect_language",
    "is_secret_workspace_path",
    "list_workspace_tree",
    "looks_binary",
    "path_is_under",
    "read_workspace_diff",
    "read_workspace_file",
    "resolve_workspace_path",
]
