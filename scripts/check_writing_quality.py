#!/usr/bin/env python3
"""Fail-closed writing-quality & anti-AI-slop gate for LLM-4-People/hermes-agent.

Scans fork-owned documentation, user-facing mobile tool prose, prompt_builder identity,
and the gateway/i18n.py EN/FR catalog to enforce:
  1. Zero Wikipedia "Signs of AI Writing" cliché vocabulary, filler openers, or robotic sign-offs.
  2. Exact EN/FR key and format-placeholder parity in gateway/i18n.py.
  3. Zero mixed-language leakage in English-only docs or per-language catalog entries.

Pure stdlib; skips fenced code blocks, inline code spans, and URLs to keep false positives at zero.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re
import string
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.i18n import CATALOG, find_slop_hits

_EXTRA_SLOP_WORDS_EN: tuple[str, ...] = (
    "comprehensive",
    "realm",
    "intricate",
    "embark",
    "cutting-edge",
    "streamline",
    "revolutionary",
)

_EXTRA_SLOP_PHRASES_EN: tuple[str, ...] = (
    "not merely",
    "unlock the power",
    "elevate your",
    "seamless experience",
)

_NEGATIVE_PARALLELISM_RE = re.compile(
    r"\b(?:it's|it is)\s+not\s+just\b|\bnot\s+just\b[^.\n]{1,40}\b(?:it's|it is|but also)\b"
)

_EXTRA_SLOP_PHRASES_FR: tuple[str, ...] = (
    "plonger dans",
    "sans couture",
    "révolutionnaire",
    "n'hésitez pas si",
    "n’hésitez pas si",
)

_DOC_TARGETS: tuple[str, ...] = (
    "README.md",
    "docs/MOBILE_COMPANION.md",
    "DOWNSTREAM.md",
)

_FENCE_RE = re.compile(r"(?ms)^```.*?^```[ \t]*$")
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_URL_RE = re.compile(r"https?://\S+")


def strip_markdown_code(text: str) -> str:
    cleaned = _FENCE_RE.sub("", text)
    cleaned = _INLINE_CODE_RE.sub("", cleaned)
    cleaned = _URL_RE.sub("", cleaned)
    return cleaned


def scan_prose_for_slop(text: str, *, lang: str = "en") -> list[str]:
    hits = list(find_slop_hits(text, lang=lang))
    lower = text.lower()
    if lang == "en":
        for phrase in _EXTRA_SLOP_PHRASES_EN:
            if phrase in lower and phrase not in hits:
                hits.append(phrase)
        if _NEGATIVE_PARALLELISM_RE.search(lower) and "negative parallelism" not in hits:
            hits.append("negative parallelism")
        for word in _EXTRA_SLOP_WORDS_EN:
            if re.search(rf"\b{re.escape(word)}\b", lower) and word not in hits:
                hits.append(word)
    else:
        for phrase in _EXTRA_SLOP_PHRASES_FR:
            if phrase in lower and phrase not in hits:
                hits.append(phrase)
    return hits


def _extract_placeholders(template: str) -> set[str]:
    return {field for _, field, _, _ in string.Formatter().parse(template) if field is not None}


def _check_catalog() -> list[str]:
    violations: list[str] = []
    en_keys = set(CATALOG["en"].keys())
    fr_keys = set(CATALOG["fr"].keys())
    if en_keys != fr_keys:
        violations.append(f"gateway/i18n.py: EN/FR key mismatch: {sorted(en_keys ^ fr_keys)}")
    for lang in ("en", "fr"):
        for key, value in CATALOG[lang].items():
            if not value.strip():
                violations.append(f"gateway/i18n.py[{lang}][{key}]: empty translation")
                continue
            hits = scan_prose_for_slop(value, lang=lang)
            if hits:
                violations.append(f"gateway/i18n.py[{lang}][{key}]: banned slop {hits}")
    for key in sorted(en_keys & fr_keys):
        en_ph = _extract_placeholders(CATALOG["en"][key])
        fr_ph = _extract_placeholders(CATALOG["fr"][key])
        if en_ph != fr_ph:
            violations.append(f"gateway/i18n.py[{key}]: placeholder mismatch EN={sorted(en_ph)} vs FR={sorted(fr_ph)}")
    return violations


def _eval_str_expr(node: ast.AST, env: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _eval_str_expr(node.left, env)
        right = _eval_str_expr(node.right, env)
        if left is not None and right is not None:
            return left + right
    return None


def _check_prompt_builder(root: Path) -> list[str]:
    """Parse agent/prompt_builder.py with pure-stdlib AST (zero third-party imports)."""
    path = root / "agent" / "prompt_builder.py"
    if not path.is_file():
        return ["agent/prompt_builder.py: file not found"]
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except SyntaxError as exc:
        return [f"agent/prompt_builder.py: syntax error {exc}"]

    env: dict[str, str] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            val = _eval_str_expr(stmt.value, env)
            if val is not None:
                env[stmt.targets[0].id] = val

    violations: list[str] = []
    for name in ("ANTI_SLOP_STYLE_RULES", "DEFAULT_AGENT_IDENTITY"):
        text = env.get(name)
        if not text:
            violations.append(f"agent/prompt_builder.py:{name}: missing or non-string constant")
            continue
        hits = scan_prose_for_slop(text, lang="en")
        if hits:
            violations.append(f"agent/prompt_builder.py:{name}: banned slop {hits}")
    return violations


def _check_docs(root: Path) -> list[str]:
    violations: list[str] = []
    for rel in _DOC_TARGETS:
        path = root / rel
        if not path.is_file():
            continue
        prose = strip_markdown_code(path.read_text(encoding="utf-8-sig"))
        for lineno, line in enumerate(prose.splitlines(), start=1):
            hits = scan_prose_for_slop(line, lang="en")
            if hits:
                violations.append(f"{rel}:{lineno}: banned slop {hits} -> {line.strip()[:90]}")
    return violations


def _check_mobile_tools(root: Path) -> list[str]:
    violations: list[str] = []
    tools_dir = root / "tools"
    if not tools_dir.is_dir():
        return violations
    for path in sorted(tools_dir.glob("mobile_*.py")):
        source = path.read_text(encoding="utf-8-sig")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            violations.append(f"tools/{path.name}: syntax error {exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for k, v in zip(node.keys, node.values):
                    if (
                        isinstance(k, ast.Constant)
                        and k.value in ("description", "message", "hint", "summary")
                        and isinstance(v, ast.Constant)
                        and isinstance(v.value, str)
                    ):
                        hits = scan_prose_for_slop(strip_markdown_code(v.value), lang="en")
                        if hits:
                            violations.append(
                                f"tools/{path.name}:{v.lineno}: banned slop {hits} in '{k.value}'"
                            )
    return violations


def run_all_checks(root: Path = ROOT) -> list[str]:
    violations: list[str] = []
    violations.extend(_check_catalog())
    violations.extend(_check_prompt_builder(root))
    violations.extend(_check_docs(root))
    violations.extend(_check_mobile_tools(root))
    return violations


def main() -> int:
    violations = run_all_checks(ROOT)
    if violations:
        for item in violations:
            sys.stderr.write(f"WRITING_QUALITY_FAIL: {item}\n")
        return 1
    sys.stdout.write("check_writing_quality: OK (0 violations)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
