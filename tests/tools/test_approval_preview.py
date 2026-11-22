"""Deterministic positive and negative tests for structured visual approval diffs."""

from __future__ import annotations

import threading

from tools.approval import register_gateway_notify, resolve_gateway_approval, unregister_gateway_notify
from tools.approval_context import reset_current_session_key, set_current_session_key
from tools.approval_prompt import build_approval_preview, build_diff_preview, infer_language
from tools.file_tools import write_file_tool


def test_file_patch_preview_produces_structured_unified_diff():
    edit = {
        "tool_name": "patch",
        "path": "app/src/main/java/com/hermes/MainActivity.java",
        "old_text": "class MainActivity {\n    int count = 1;\n    void run() {}\n}\n",
        "new_text": "class MainActivity {\n    int count = 2;\n    int retries = 3;\n    void run() {}\n}\n",
    }
    preview = build_approval_preview({
        "command": "Edit app/src/main/java/com/hermes/MainActivity.java",
        "description": "Update retry settings",
        "edit": edit,
    })

    assert preview["kind"] == "file_patch"
    assert preview["risk"] == "medium"
    assert preview["title"].startswith("Patch ")
    diff = preview["diff"]
    assert diff is not None
    assert diff["path"] == "app/src/main/java/com/hermes/MainActivity.java"
    assert diff["language"] == "java"
    assert diff["additions"] == 2
    assert diff["deletions"] == 1
    assert diff["truncated"] is False
    assert len(diff["hunks"]) == 1
    hunk = diff["hunks"][0]
    assert hunk["old_start"] >= 1
    assert hunk["new_start"] >= 1
    assert "-    int count = 1;" in hunk["lines"]
    assert "+    int count = 2;" in hunk["lines"]
    assert "+    int retries = 3;" in hunk["lines"]


def test_new_file_skill_write_and_terminal_risk_classification():
    skill_preview = build_approval_preview({
        "command": "Write skills/review/SKILL.md",
        "description": "Create skill",
        "edit": {
            "tool_name": "write_file",
            "path": "/home/user/.hermes/skills/review/SKILL.md",
            "old_text": None,
            "new_text": "# Review\n\nCheck diffs carefully.\n",
        },
    })
    assert skill_preview["kind"] == "skill_write"
    assert skill_preview["risk"] == "low"
    assert skill_preview["diff"]["language"] == "markdown"
    assert skill_preview["diff"]["additions"] == 3
    assert skill_preview["diff"]["deletions"] == 0

    high_term = build_approval_preview({
        "command": "rm -rf /tmp/build && echo done",
        "description": "Recursive delete",
        "pattern_key": "rm_rf",
    })
    assert high_term["kind"] == "terminal"
    assert high_term["risk"] == "high"
    assert high_term["diff"] is None

    low_mcp = build_approval_preview({
        "command": "Allow calendar lookup",
        "description": "MCP elicitation",
        "pattern_key": "mcp_elicitation",
    })
    assert low_mcp["kind"] == "mcp_elicitation"
    assert low_mcp["risk"] == "low"
    assert low_mcp["diff"] is None


def test_secrets_are_redacted_before_truncation_in_diff_and_caller_preview():
    secret = "sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0U1v2"
    # Place secret right across the 500-char line cap to prove redaction happens BEFORE truncation.
    padded_secret_line = ("x" * 485) + " token=" + secret + " trailing"
    diff = build_diff_preview("config.py", "token = 'old'\n", padded_secret_line + "\n")

    assert diff["language"] == "python"
    assert diff["truncated"] is True
    serialized = str(diff)
    assert secret not in serialized
    assert "A1b2C3d4E5f6" not in serialized

    # Pre-attached caller preview with raw secret & bogus kind/risk is re-sanitized and normalized.
    sanitized = build_approval_preview({
        "command": f"echo {secret}",
        "description": f"Uses {secret}",
        "preview": {
            "kind": "invalid_kind",
            "risk": "extreme",
            "title": f"Leak {secret}",
            "summary": f"Summary {secret}",
            "diff": {
                "path": f"dir/{secret}/file.ts",
                "hunks": [{"old_start": "bad", "new_start": 1, "lines": [f"+const key = '{secret}';"]}],
            },
        },
    })
    assert sanitized["kind"] == "terminal"
    assert sanitized["risk"] in {"low", "medium", "high"}
    assert secret not in str(sanitized)
    assert "A1b2C3d4E5f6" not in str(sanitized)
    assert sanitized["diff"]["language"] == "typescript"


def test_oversized_diff_is_bounded_and_marked_truncated():
    old_lines = [f"line_{i} = {i}" for i in range(600)]
    new_lines = [f"line_{i} = {i + 1}" for i in range(600)]
    diff = build_diff_preview("big.py", "\n".join(old_lines), "\n".join(new_lines))

    assert diff["truncated"] is True
    total_lines = sum(len(h["lines"]) for h in diff["hunks"])
    assert total_lines <= 200
    assert len(diff["hunks"]) <= 20
    assert infer_language("Dockerfile") == "dockerfile"
    assert infer_language("") == "text"


def test_malformed_approval_inputs_never_raise():
    for raw in (None, {}, {"edit": "not-a-dict"}, {"preview": {"diff": "bad"}}):
        preview = build_approval_preview(raw)
        assert preview["kind"] in {"terminal", "file_patch", "file_write", "skill_write", "device_action", "ui_patch", "mcp_elicitation"}
        assert preview["risk"] in {"low", "medium", "high"}


def test_protected_instruction_write_attaches_visual_diff_to_gateway_approval(tmp_path):
    target = tmp_path / "SOUL.md"
    target.write_text("# Soul\nOriginal persona.\n", encoding="utf-8")
    session_key = "test-protected-diff-session"
    seen: list[dict] = []

    def notify(data: dict) -> None:
        seen.append(data)
        threading.Thread(
            target=resolve_gateway_approval,
            args=(session_key, "deny"),
            kwargs={"request_id": data["request_id"]},
            daemon=True,
        ).start()

    token = set_current_session_key(session_key)
    register_gateway_notify(session_key, notify)
    try:
        result = write_file_tool(str(target), "# Soul\nUpdated persona.\n", task_id="default")
    finally:
        unregister_gateway_notify(session_key)
        reset_current_session_key(token)

    assert "BLOCKED" in result
    assert len(seen) == 1
    preview = seen[0]["preview"]
    assert preview["kind"] == "file_write"
    assert preview["risk"] == "high"
    assert preview["diff"] is not None
    assert preview["diff"]["language"] == "markdown"
    assert preview["diff"]["additions"] == 1
    assert preview["diff"]["deletions"] == 1
    assert "-Original persona." in preview["diff"]["hunks"][0]["lines"]
    assert "+Updated persona." in preview["diff"]["hunks"][0]["lines"]
