"""Structured, secret-redacted visual approval preview and unified diff builder.

Single source of truth for turning raw gateway approval payloads (dangerous commands,
protected/approval-required file writes, ACP edit proposals, or pre-attached previews)
into bounded, secret-redacted ``preview`` objects suitable for SSE ``approval.request``
events and ``GET /v1/runs/{id}`` ``pending_controls[]``.
"""

from __future__ import annotations

import difflib
import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger("tools.approval")

_MAX_DIFF_INPUT_CHARS = 200_000
_MAX_DIFF_HUNKS = 20
_MAX_DIFF_LINES = 200
_MAX_DIFF_LINE_CHARS = 500
_MAX_DIFF_TOTAL_CHARS = 32_000
_MAX_TITLE_CHARS = 160
_MAX_SUMMARY_CHARS = 240
_MAX_COMMAND_CHARS = 2_000

_VALID_KINDS = frozenset({
    "terminal", "file_patch", "file_write", "skill_write",
    "device_action", "ui_patch", "mcp_elicitation",
})
_VALID_RISKS = frozenset({"low", "medium", "high"})
_HIGH_RISK_PATTERNS = frozenset({
    "protected_instruction_file", "ssh_config_write", "shell_rc_write",
})
_HIGH_RISK_COMMAND_RE = re.compile(
    r"(?:^|[;&|]\s*)(?:sudo\b|rm\s+-[a-zA-Z]*[rf]|mkfs\b|dd\s+if=|chmod\s+(?:-R\s+)?777\b)"
    r"|\|\s*(?:ba|z|da)?sh\b|>\s*/(?:etc|dev)/",
    re.IGNORECASE,
)
_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")

_LANGUAGE_BY_EXT: dict[str, str] = {
    ".py": "python", ".pyi": "python", ".java": "java", ".kt": "kotlin", ".kts": "kotlin",
    ".ts": "typescript", ".tsx": "tsx", ".js": "javascript", ".jsx": "javascript",
    ".mjs": "javascript", ".cjs": "javascript", ".json": "json", ".yaml": "yaml",
    ".yml": "yaml", ".md": "markdown", ".sh": "bash", ".bash": "bash", ".zsh": "bash",
    ".xml": "xml", ".gradle": "groovy", ".groovy": "groovy", ".rs": "rust", ".go": "go",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".sql": "sql", ".toml": "toml", ".html": "html", ".htm": "html", ".css": "css",
    ".scss": "scss", ".rb": "ruby", ".php": "php", ".swift": "swift", ".dart": "dart",
    ".ini": "ini", ".cfg": "ini", ".properties": "properties", ".proto": "protobuf",
}
_LANGUAGE_BY_BASENAME: dict[str, str] = {
    "dockerfile": "dockerfile", "makefile": "makefile", "cmakelists.txt": "cmake",
}


def infer_language(path: str) -> str:
    """Infer syntax-highlighting language tag from *path* (defaults to ``"text"``)."""
    clean = str(path or "").strip()
    if not clean:
        return "text"
    base = os.path.basename(clean).lower()
    if base in _LANGUAGE_BY_BASENAME:
        return _LANGUAGE_BY_BASENAME[base]
    _, ext = os.path.splitext(base)
    return _LANGUAGE_BY_EXT.get(ext, "text")


def _redact(text: Any) -> str:
    """Force-redact secrets before any truncation so a cut can never leak a secret prefix."""
    from agent.redact import redact_approval_display
    return redact_approval_display(str(text or ""), force=True)


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[: max(1, limit - 3)] + "...", True


def _redact_diff_line(raw_line: str) -> tuple[str, bool]:
    """Preserve the unified-diff prefix (``+``/``-``/`` ``), redact body first, then bound length."""
    if raw_line and raw_line[0] in ("+", "-", " "):
        prefix, body = raw_line[0], raw_line[1:]
    else:
        prefix, body = " ", raw_line
    redacted = _redact(body)
    bounded, cut = _truncate(redacted, _MAX_DIFF_LINE_CHARS)
    return prefix + bounded, cut


def _bound_input_lines(text: Optional[str]) -> tuple[list[str], bool]:
    if not text:
        return [], False
    if len(text) <= _MAX_DIFF_INPUT_CHARS:
        return text.splitlines(), False
    return text[:_MAX_DIFF_INPUT_CHARS].splitlines(), True


def _append_bounded_hunks(
    raw_hunks: list[tuple[int, int, list[str]]],
) -> tuple[list[dict[str, Any]], int, int, bool]:
    """Apply hunk/line/char caps and secret redaction to parsed hunks."""
    hunks: list[dict[str, Any]] = []
    additions = deletions = total_lines = total_chars = 0
    truncated = False

    for old_start, new_start, raw_lines in raw_hunks:
        if len(hunks) >= _MAX_DIFF_HUNKS or total_lines >= _MAX_DIFF_LINES or total_chars >= _MAX_DIFF_TOTAL_CHARS:
            truncated = True
            break
        out_lines: list[str] = []
        for raw_line in raw_lines:
            if total_lines >= _MAX_DIFF_LINES or total_chars >= _MAX_DIFF_TOTAL_CHARS:
                truncated = True
                break
            line, line_cut = _redact_diff_line(raw_line)
            if line_cut:
                truncated = True
            if total_chars + len(line) > _MAX_DIFF_TOTAL_CHARS and out_lines:
                truncated = True
                break
            out_lines.append(line)
            total_lines += 1
            total_chars += len(line)
            if line.startswith("+"):
                additions += 1
            elif line.startswith("-"):
                deletions += 1
        if out_lines:
            hunks.append({
                "old_start": max(1, int(old_start)),
                "new_start": max(1, int(new_start)),
                "lines": out_lines,
            })

    return hunks, additions, deletions, truncated


def _parse_unified_diff_lines(diff_lines: list[str]) -> list[tuple[int, int, list[str]]]:
    raw_hunks: list[tuple[int, int, list[str]]] = []
    cur_old, cur_new, cur_lines = 1, 1, []
    in_hunk = False

    for line in diff_lines:
        if line.startswith("--- ") or line.startswith("+++ "):
            continue
        match = _HUNK_HEADER_RE.match(line)
        if match:
            if in_hunk and cur_lines:
                raw_hunks.append((cur_old, cur_new, cur_lines))
            cur_old, cur_new = int(match.group(1)), int(match.group(2))
            cur_lines = []
            in_hunk = True
            continue
        if in_hunk and line and line[0] in ("+", "-", " "):
            cur_lines.append(line)

    if in_hunk and cur_lines:
        raw_hunks.append((cur_old, cur_new, cur_lines))
    return raw_hunks


def _parse_patch_text_hunks(patch_text: str) -> tuple[list[tuple[int, int, list[str]]], bool]:
    """Parse a raw V4A or unified patch body into hunks when before/after snapshots are absent."""
    lines, cut = _bound_input_lines(patch_text)
    if any(_HUNK_HEADER_RE.match(line) for line in lines):
        return _parse_unified_diff_lines(lines), cut
    body_lines = [
        line for line in lines
        if line and line[0] in ("+", "-", " ")
        and not line.startswith("+++") and not line.startswith("---")
    ]
    return ([(1, 1, body_lines)] if body_lines else []), cut


def build_diff_preview(
    path: str,
    old_text: Optional[str] = None,
    new_text: Optional[str] = None,
    *,
    patch_text: Optional[str] = None,
) -> dict[str, Any]:
    """Build a bounded, secret-redacted structured unified diff preview."""
    redacted_path, path_cut = _truncate(_redact(path), _MAX_TITLE_CHARS)
    language = infer_language(path)

    if patch_text is not None and old_text is None:
        raw_hunks, input_cut = _parse_patch_text_hunks(patch_text)
    else:
        old_lines, old_cut = _bound_input_lines(old_text)
        new_lines, new_cut = _bound_input_lines(new_text)
        input_cut = old_cut or new_cut
        diff_iter = list(difflib.unified_diff(old_lines, new_lines, fromfile="a", tofile="b", lineterm="", n=3))
        raw_hunks = _parse_unified_diff_lines(diff_iter)

    hunks, additions, deletions, cap_cut = _append_bounded_hunks(raw_hunks)
    return {
        "path": redacted_path,
        "language": language,
        "hunks": hunks,
        "additions": additions,
        "deletions": deletions,
        "truncated": bool(path_cut or input_cut or cap_cut),
    }


def _sanitize_caller_diff(raw_diff: Any) -> Optional[dict[str, Any]]:
    """Validate, force-redact, and size-bound a pre-attached ``preview.diff`` dict."""
    if not isinstance(raw_diff, dict):
        return None
    path = str(raw_diff.get("path") or "")
    redacted_path, path_cut = _truncate(_redact(path), _MAX_TITLE_CHARS)
    language = str(raw_diff.get("language") or "").strip().lower() or infer_language(path)

    raw_hunks_in = raw_diff.get("hunks")
    parsed_hunks: list[tuple[int, int, list[str]]] = []
    if isinstance(raw_hunks_in, list):
        for item in raw_hunks_in:
            if not isinstance(item, dict):
                continue
            lines = item.get("lines")
            if not isinstance(lines, list):
                continue
            old_start = item.get("old_start", 1)
            new_start = item.get("new_start", 1)
            try:
                old_i, new_i = int(old_start), int(new_start)
            except (TypeError, ValueError):
                old_i, new_i = 1, 1
            parsed_hunks.append((old_i, new_i, [str(line) for line in lines]))

    hunks, additions, deletions, cap_cut = _append_bounded_hunks(parsed_hunks)
    return {
        "path": redacted_path,
        "language": language,
        "hunks": hunks,
        "additions": additions,
        "deletions": deletions,
        "truncated": bool(raw_diff.get("truncated") or path_cut or cap_cut),
    }


def _is_skill_path(path: str) -> bool:
    norm = str(path or "").replace("\\", "/").lower()
    return norm.endswith("/skill.md") or norm == "skill.md" or "/skills/" in norm


def _infer_kind(data: dict[str, Any], edit: Optional[dict[str, Any]], caller_preview: Optional[dict[str, Any]]) -> str:
    explicit = caller_preview.get("kind") if isinstance(caller_preview, dict) else None
    if isinstance(explicit, str) and explicit in _VALID_KINDS:
        return explicit
    if isinstance(edit, dict):
        tool_name = str(edit.get("tool_name") or "").lower()
        path = str(edit.get("path") or "")
        if _is_skill_path(path):
            return "skill_write"
        return "file_patch" if tool_name == "patch" else "file_write"
    pattern_key = str(data.get("pattern_key") or "").lower()
    if pattern_key == "mcp_elicitation":
        return "mcp_elicitation"
    if pattern_key in _HIGH_RISK_PATTERNS:
        return "file_write"
    return "terminal"


def _infer_risk(
    data: dict[str, Any], *, kind: str, command: str, diff: Optional[dict[str, Any]],
    caller_preview: Optional[dict[str, Any]],
) -> str:
    explicit = caller_preview.get("risk") if isinstance(caller_preview, dict) else None
    if isinstance(explicit, str) and explicit in _VALID_RISKS:
        return explicit
    pattern_key = str(data.get("pattern_key") or "").lower()
    if data.get("smart_denied") or pattern_key in _HIGH_RISK_PATTERNS:
        return "high"
    if kind == "terminal" and _HIGH_RISK_COMMAND_RE.search(command):
        return "high"
    if isinstance(diff, dict):
        deletions = int(diff.get("deletions") or 0)
        additions = int(diff.get("additions") or 0)
        if deletions >= 100:
            return "high"
        if deletions == 0 and additions <= 20:
            return "low"
        return "medium"
    return "low" if kind == "mcp_elicitation" else "medium"


def _default_title(kind: str, diff: Optional[dict[str, Any]], command: str) -> str:
    if isinstance(diff, dict) and diff.get("path"):
        verb = "Patch" if kind == "file_patch" else "Write"
        return f"{verb} {diff['path']}"
    if kind == "mcp_elicitation":
        return "Confirm action"
    if kind in {"file_write", "file_patch", "skill_write"}:
        return command or "Approve file change"
    return "Approve command"


def _default_summary(description: str, diff: Optional[dict[str, Any]]) -> str:
    if description:
        return description
    if isinstance(diff, dict) and diff.get("path"):
        base = os.path.basename(diff["path"]) or diff["path"]
        return f"+{diff.get('additions', 0)} / -{diff.get('deletions', 0)} lines in {base}"
    return ""


def _diff_from_edit(edit: dict[str, Any]) -> Optional[dict[str, Any]]:
    path = str(edit.get("path") or "")
    old_text = edit.get("old_text")
    new_text = edit.get("new_text")
    tool_name = str(edit.get("tool_name") or "").lower()
    if not path and old_text is None and new_text is None:
        return None
    if tool_name == "patch" and old_text is None and isinstance(new_text, str):
        return build_diff_preview(path, None, None, patch_text=new_text)
    return build_diff_preview(
        path,
        str(old_text) if old_text is not None else None,
        str(new_text) if new_text is not None else "",
    )


def build_approval_preview(approval_data: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Build a bounded, secret-redacted structured ``preview`` dict for an approval payload."""
    data = approval_data if isinstance(approval_data, dict) else {}
    caller_preview = data.get("preview") if isinstance(data.get("preview"), dict) else None
    edit = data.get("edit") if isinstance(data.get("edit"), dict) else None

    raw_command = str(data.get("command") or (caller_preview.get("command") if caller_preview else "") or "")
    raw_description = str(data.get("description") or "")
    command, _ = _truncate(_redact(raw_command), _MAX_COMMAND_CHARS)
    description, _ = _truncate(_redact(raw_description), _MAX_SUMMARY_CHARS)

    diff: Optional[dict[str, Any]] = None
    try:
        if caller_preview and caller_preview.get("diff") is not None:
            diff = _sanitize_caller_diff(caller_preview["diff"])
        elif edit is not None:
            diff = _diff_from_edit(edit)
    except Exception:
        logger.debug("Failed to build structured approval diff preview", exc_info=True)
        diff = None

    kind = _infer_kind(data, edit, caller_preview)
    risk = _infer_risk(data, kind=kind, command=raw_command, diff=diff, caller_preview=caller_preview)

    raw_title = (
        caller_preview.get("title")
        if caller_preview and isinstance(caller_preview.get("title"), str) and caller_preview["title"].strip()
        else _default_title(kind, diff, command)
    )
    raw_summary = (
        caller_preview.get("summary")
        if caller_preview and isinstance(caller_preview.get("summary"), str) and caller_preview["summary"].strip()
        else _default_summary(description, diff)
    )
    title, _ = _truncate(_redact(raw_title), _MAX_TITLE_CHARS)
    summary, _ = _truncate(_redact(raw_summary), _MAX_SUMMARY_CHARS)

    return {
        "kind": kind,
        "title": title,
        "summary": summary,
        "risk": risk,
        "command": command,
        "diff": diff,
    }


def attach_approval_preview(event: dict[str, Any]) -> dict[str, Any]:
    """Attach redacted ``preview`` to *event* and drop raw unredacted ``edit`` payloads."""
    event["preview"] = build_approval_preview(event)
    event.pop("edit", None)
    return event
