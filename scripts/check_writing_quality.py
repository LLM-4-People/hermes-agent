#!/usr/bin/env python3
"""Fail-closed writing-quality, sentence-case & anti-AI-slop gate for LLM-4-People/hermes-agent.

Scans fork-owned documentation, user-facing mobile tool prose, mobile companion prompt,
mobile_doctor CLI output, prompt_builder identity, and the gateway/i18n.py EN/FR catalog to enforce:
  1. Zero Wikipedia "Signs of AI Writing" cliché vocabulary, filler openers, or robotic sign-offs.
  2. Strict sentence case on Markdown headings, catalog labels, mobile tool titles, and CLI headers.
  3. Exact EN/FR key and format-placeholder parity in gateway/i18n.py.
  4. Zero mixed-language leakage in English-only docs or per-language catalog entries.

Pure stdlib (`python3 -I` safe); skips fenced code blocks, inline code spans, and URLs.
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

from gateway.i18n import CATALOG, detect_text_lang, find_slop_hits

_EXTRA_SLOP_WORDS_EN: tuple[str, ...] = (
    "comprehensive",
    "realm",
    "intricate",
    "embark",
    "cutting-edge",
    "streamline",
    "revolutionary",
    "seamless",
    "supercharge",
    "bespoke",
    "next-level",
    "groundbreaking",
    "transformative",
    "synergy",
    "vibrant",
    "unleash",
    "effortless",
    "effortlessly",
    "orchestrate",
    "reimagine",
    "unlocks",
)

_EXTRA_SLOP_PHRASES_EN: tuple[str, ...] = (
    "not merely",
    "unlock the power",
    "elevate your",
    "seamless experience",
)

_NEGATIVE_PARALLELISM_RE = re.compile(
    r"\b(?:it's|it is)\s+not\s+just\b"
    r"|\bnot\s+just\b[^.\n]{1,40}\b(?:it's|it is|but also)\b"
    r"|\bnot\s+only\b[^.\n]{1,60}\bbut\s+also\b"
)

_EXTRA_SLOP_PHRASES_FR: tuple[str, ...] = (
    "plonger dans",
    "sans couture",
    "révolutionnaire",
    "n'hésitez pas si",
    "n’hésitez pas si",
    "décupler",
    "sublimer",
    "sur-mesure",
)

_DOC_TARGETS: tuple[str, ...] = (
    "README.md",
    "docs/MOBILE_COMPANION.md",
    "DOWNSTREAM.md",
)

# Stop scanning README.md for fork sentence-case / expanded rules once upstream content begins.
_UPSTREAM_README_MARKER = "## Upstream Hermes Agent"

_FENCE_RE = re.compile(r"(?ms)^```.*?^```[ \t]*$")
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_URL_RE = re.compile(r"https?://\S+")
_PLACEHOLDER_RE = re.compile(r"\{[^{}]+\}")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_LEADING_NUM_RE = re.compile(r"^(?:\d+(?:\.\d+)*\.\s+|Step\s+\d+\s*[—:-]\s*)")
_CLAUSE_SPLIT_RE = re.compile(r"(?:\.\s+|\?\s+|!\s+|\s+—\s+|\s+-\s+|:\s+)")
_WORD_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9][A-Za-zÀ-ÖØ-öø-ÿ0-9'’-]*")
_ACRONYM_RE = re.compile(r"^[A-Z0-9][A-Z0-9+/_-]*$")

# True proper nouns, product/brand names, languages, and technical names exempt from sentence-case checks.
_PROPER_NOUNS: frozenset[str] = frozenset({
    "Hermes",
    "Mobile",
    "Agent",
    "Android",
    "GitHub",
    "Pages",
    "Releases",
    "Docker",
    "Compose",
    "SQLite",
    "UnifiedPush",
    "Keystore",
    "Passkey",
    "Caddy",
    "Nginx",
    "Traefik",
    "Tailscale",
    "Serve",
    "Cosign",
    "BuildKit",
    "Codex",
    "Python",
    "Gradle",
    "Signal",
    "OpenAI",
    "Anthropic",
    "NousResearch",
    "Nous",
    "Research",
    "Telegram",
    "Discord",
    "Slack",
    "WhatsApp",
    "Linux",
    "Daytona",
    "Modal",
    "Vercel",
    "Sandbox",
    "Singularity",
    "Apptainer",
    "Honcho",
    "French",
    "English",
    "Français",
    "Atelier",
    "Activité",
    "Cerveau",
    "Montréal",
    "Wikipedia",
    "CalVer",
    "SemVer",
    "TalkBack",
    "BiometricPrompt",
    "Markdown",
    "Google",
    "Apple",
    "Windows",
    "PowerShell",
    "Desktop",
    "People",
    "Wi-Fi",
    "Bluetooth",
    "CodeQL",
    "I",
})


def strip_markdown_code(text: str) -> str:
    """Strip fenced/inline code and URLs while preserving exact 1-based line numbers."""
    cleaned = _FENCE_RE.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    cleaned = _INLINE_CODE_RE.sub("", cleaned)
    cleaned = _URL_RE.sub("", cleaned)
    return cleaned


def find_mixed_language_leak(text: str, *, expected_lang: str) -> str | None:
    """Return an error reason when prose in ``text`` switches to the opposite EN/FR language."""
    cleaned = _HTML_TAG_RE.sub(" ", _PLACEHOLDER_RE.sub(" ", text))
    detected = detect_text_lang(cleaned)
    if detected is not None and detected != expected_lang:
        return f"mixed-language leakage (expected {expected_lang}, detected {detected})"
    return None


def _is_exempt_capitalized_word(token: str) -> bool:
    base = token.strip("'’-")
    if not base:
        return True
    if base in _PROPER_NOUNS:
        return True
    if _ACRONYM_RE.match(base):
        return True
    # Hyphenated tokens like "LLM-4-People" or "Small-LLM": reject any non-exempt Title-Cased sub-part.
    if "-" in base:
        parts = [p for p in base.split("-") if p]
        if len(parts) >= 2:
            for idx, part in enumerate(parts):
                if not any(ch.isalpha() for ch in part):
                    continue
                if part[0].isupper() and any(ch.islower() for ch in part[1:]):
                    if part not in _PROPER_NOUNS and idx > 0:
                        return False
            return True
    return False


def find_title_case_violations(text: str) -> list[str]:
    """Return non-exempt Title-Cased words in a heading, title, or UI label."""
    cleaned = _MD_LINK_RE.sub(r"\1", text)
    cleaned = _INLINE_CODE_RE.sub(" ", cleaned)
    cleaned = _URL_RE.sub(" ", cleaned)
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)
    cleaned = _PLACEHOLDER_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("**", "").replace("__", "").replace("*", "")
    cleaned = cleaned.strip()
    if not cleaned:
        return []

    offenders: list[str] = []
    for raw_clause in _CLAUSE_SPLIT_RE.split(cleaned):
        clause = _LEADING_NUM_RE.sub("", raw_clause.strip())
        if not clause:
            continue
        tokens = [t.strip("'’-") for t in _WORD_TOKEN_RE.findall(clause)]
        tokens = [t for t in tokens if t]
        for idx, token in enumerate(tokens):
            if not any(ch.isalpha() for ch in token):
                continue
            has_title_part = any(
                part and part[0].isupper() and any(ch.islower() for ch in part[1:])
                for part in token.split("-")
            )
            if not has_title_part:
                continue
            if idx == 0:
                if "-" in token and not _is_exempt_capitalized_word(token):
                    offenders.append(token)
                continue
            if not _is_exempt_capitalized_word(token):
                offenders.append(token)
    return offenders


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
    if text.count("—") >= 3 and "em-dash clutter" not in hits:
        hits.append("em-dash clutter")
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
            leak = find_mixed_language_leak(value, expected_lang=lang)
            if leak:
                violations.append(f"gateway/i18n.py[{lang}][{key}]: {leak}")
            first_line = value.splitlines()[0].strip()
            if not first_line.startswith("/") and not first_line.startswith("⚠️"):
                tc = find_title_case_violations(first_line)
                if tc:
                    violations.append(f"gateway/i18n.py[{lang}][{key}]: title case {tc} in {first_line!r}")
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
    rules = env.get("ANTI_SLOP_STYLE_RULES", "")
    if rules and "sentence case" not in rules.lower():
        violations.append("agent/prompt_builder.py:ANTI_SLOP_STYLE_RULES: missing sentence-case rule")
    return violations


def _check_docs(root: Path) -> list[str]:
    violations: list[str] = []
    for rel in _DOC_TARGETS:
        path = root / rel
        if not path.is_file():
            continue
        raw_text = path.read_text(encoding="utf-8-sig")
        if rel == "README.md" and _UPSTREAM_README_MARKER in raw_text:
            # Scan only the fork-owned section of README.md so upstream NousResearch prose never conflicts.
            raw_text = raw_text.split(_UPSTREAM_README_MARKER, 1)[0]
        prose = strip_markdown_code(raw_text)
        for lineno, line in enumerate(prose.splitlines(), start=1):
            hits = scan_prose_for_slop(line, lang="en")
            if hits:
                violations.append(f"{rel}:{lineno}: banned slop {hits} -> {line.strip()[:90]}")
            leak = find_mixed_language_leak(line, expected_lang="en")
            if leak:
                violations.append(f"{rel}:{lineno}: {leak} -> {line.strip()[:90]}")
            heading_match = _HEADING_RE.match(line)
            if heading_match:
                heading_text = heading_match.group(2)
                tc = find_title_case_violations(heading_text)
                if tc:
                    violations.append(f"{rel}:{lineno}: title-case heading {tc} -> {heading_text[:90]}")
    return violations


def _scan_mobile_tool_dict(path: Path, node: ast.Dict) -> list[str]:
    violations: list[str] = []
    for k, v in zip(node.keys, node.values):
        if not (
            isinstance(k, ast.Constant)
            and k.value in ("description", "message", "hint", "summary", "title", "label")
            and isinstance(v, ast.Constant)
            and isinstance(v.value, str)
        ):
            continue
        clean_val = strip_markdown_code(v.value)
        hits = scan_prose_for_slop(clean_val, lang="en")
        if hits:
            violations.append(f"tools/{path.name}:{v.lineno}: banned slop {hits} in '{k.value}'")
        if k.value in ("title", "label") and not clean_val.startswith("http"):
            tc = find_title_case_violations(clean_val)
            if tc:
                violations.append(
                    f"tools/{path.name}:{v.lineno}: title case {tc} in '{k.value}' -> {clean_val[:80]}"
                )
    return violations


def _check_mobile_tools(root: Path) -> list[str]:
    violations: list[str] = []
    tools_dir = root / "tools"
    if not tools_dir.is_dir():
        return violations
    for path in sorted(tools_dir.glob("mobile_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except SyntaxError as exc:
            violations.append(f"tools/{path.name}: syntax error {exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                violations.extend(_scan_mobile_tool_dict(path, node))
    return violations


def _iter_static_strings(tree: ast.AST) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append((node.lineno, node.value))
        elif isinstance(node, ast.JoinedStr):
            parts = [v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str)]
            joined = " ".join(p for p in parts if p.strip())
            if joined:
                out.append((node.lineno, joined))
    return out


def _check_doctor_report_titles(rel: str, tree: ast.AST) -> list[str]:
    violations: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "format_report_text"):
            continue
        for sub in ast.walk(node):
            if not (isinstance(sub, ast.Constant) and isinstance(sub.value, str)):
                continue
            val = sub.value.strip()
            if len(val) < 8 or val.startswith("[") or val.isupper():
                continue
            tc = find_title_case_violations(val)
            if tc:
                violations.append(f"{rel}:{sub.lineno}: title case {tc} -> {val[:90]}")
    return violations


def _check_companion_prompt_and_doctor(root: Path) -> list[str]:
    """Pure-stdlib AST scan of agent/mobile_companion_prompt.py and scripts/mobile_doctor.py."""
    violations: list[str] = []
    for rel in ("agent/mobile_companion_prompt.py", "scripts/mobile_doctor.py"):
        path = root / rel
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except SyntaxError as exc:
            violations.append(f"{rel}: syntax error {exc}")
            continue
        for lineno, raw in _iter_static_strings(tree):
            if len(raw.strip()) < 4 or raw.startswith("http"):
                continue
            cleaned = strip_markdown_code(raw)
            hits = scan_prose_for_slop(cleaned, lang="en")
            if hits:
                violations.append(f"{rel}:{lineno}: banned slop {hits} -> {cleaned.strip()[:90]}")
        if rel == "scripts/mobile_doctor.py":
            violations.extend(_check_doctor_report_titles(rel, tree))
    return violations


_DECORATIVE_SLOP_EMOJI_RE = re.compile(r"[♻✨🚀🎉🔥⌛✏]")

_AUDITED_LOCALE_SUBSECTIONS: frozenset[str] = frozenset({
    "startup",
    "shutdown",
    "restart",
    "reset",
    "cron_store",
    "bundles",
    "approvals",
    "platform",
    "session",
    "progress",
    "busy",
    "reload_mcp",
})

_NO_EMDASH_LOCALE_KEYS: frozenset[str] = frozenset({
    "gateway.startup.online",
    "gateway.startup.restarted",
    "gateway.startup.db_unavailable",
    "gateway.shutdown.notice_shutdown",
    "gateway.shutdown.notice_restart",
    "gateway.restart.in_progress",
    "gateway.restart.restarting",
    "gateway.busy.drain_queued",
    "gateway.busy.drain_rejected",
    "gateway.busy.drain_rejected_new_work",
    "gateway.session.auto_reset_context_exhausted",
    "gateway.progress.timeout_header",
})

_FR_ENGLISH_LEAK_RE = re.compile(
    r"\b(?:gateway\s+en\s+ligne|gateway\s+redémarré|le\s+gateway|du\s+gateway|ce\s+gateway|endpoint\s*:)\b",
    re.IGNORECASE,
)

_GATEWAY_RUNTIME_TARGETS: tuple[str, ...] = (
    "gateway/delivery_ledger.py",
    "persistence/authority/coordination.py",
    "gateway/session_stall.py",
    "gateway/session_ingress.py",
    "gateway/relay/adapter.py",
    "gateway/relay/prompt_resolvers.py",
)

_YAML_KEY_RE = re.compile(r"^( +)([A-Za-z0-9_]+):(?:\s*(.*))?$")


def _unquote_yaml_scalar(raw: str) -> str:
    val = raw.strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
        val = val[1:-1]
    return val.replace('\\"', '"').replace("\\n", "\n")


def _parse_locale_gateway_entries(path: Path) -> list[tuple[int, str, str]]:
    """Pure-stdlib extractor for ``gateway.<subsection>.<key>`` entries from locales YAML."""
    entries: list[tuple[int, str, str]] = []
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    in_gateway = False
    section: str | None = None
    block_key: str | None = None
    block_lineno = 0
    block_indent = 0
    block_lines: list[str] = []

    def _flush_block() -> None:
        nonlocal block_key, block_lines
        if block_key is not None:
            entries.append((block_lineno, block_key, "\n".join(block_lines).strip()))
            block_key = None
            block_lines = []

    for lineno, line in enumerate(lines, start=1):
        if not in_gateway:
            if line.startswith("gateway:"):
                in_gateway = True
            continue
        if line and not line[0].isspace() and not line.startswith("#"):
            _flush_block()
            break
        if block_key is not None:
            stripped = line.strip()
            indent = len(line) - len(line.lstrip(" "))
            if not stripped or indent >= block_indent:
                block_lines.append(stripped)
                continue
            _flush_block()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _YAML_KEY_RE.match(line)
        if not match:
            continue
        indent = len(match.group(1))
        key_name = match.group(2)
        rest = (match.group(3) or "").strip()
        if indent == 2:
            section = key_name if not rest else None
            continue
        if indent == 4 and section in _AUDITED_LOCALE_SUBSECTIONS:
            full_key = f"gateway.{section}.{key_name}"
            if rest in ("|", "|-", "|2-", ">", ">-"):
                block_key = full_key
                block_lineno = lineno
                block_indent = indent + 2
                block_lines = []
            elif rest:
                entries.append((lineno, full_key, _unquote_yaml_scalar(rest)))
    _flush_block()
    return entries


def _check_gateway_locales(root: Path) -> list[str]:
    """Scan locales/{en,fr}.yaml gateway notices for decorative emojis, em-dashes, slop & FR leaks."""
    violations: list[str] = []
    for lang in ("en", "fr"):
        rel = f"locales/{lang}.yaml"
        path = root / rel
        if not path.is_file():
            continue
        for lineno, key, value in _parse_locale_gateway_entries(path):
            cleaned = strip_markdown_code(value)
            if _DECORATIVE_SLOP_EMOJI_RE.search(cleaned):
                violations.append(f"{rel}:{lineno} ({key}): decorative AI-slop emoji -> {cleaned[:90]!r}")
            if key in _NO_EMDASH_LOCALE_KEYS and "—" in cleaned:
                violations.append(f"{rel}:{lineno} ({key}): em-dash in gateway status notice -> {cleaned[:90]!r}")
            hits = scan_prose_for_slop(cleaned, lang=lang)
            if hits:
                violations.append(f"{rel}:{lineno} ({key}): banned slop {hits}")
            if lang == "fr" and _FR_ENGLISH_LEAK_RE.search(cleaned):
                violations.append(f"{rel}:{lineno} ({key}): English leak in French locale -> {cleaned[:90]!r}")
            if key in ("gateway.bundles.header", "gateway.bundles.header_plain", "gateway.startup.online", "gateway.startup.restarted"):
                tc = find_title_case_violations(cleaned)
                if tc:
                    violations.append(f"{rel}:{lineno} ({key}): title case {tc} -> {cleaned[:90]!r}")
    return violations


def _check_gateway_runtime_strings(root: Path) -> list[str]:
    """Pure-stdlib AST scan of gateway delivery/stall/ingress/relay user-facing runtime strings."""
    violations: list[str] = []
    for rel in _GATEWAY_RUNTIME_TARGETS:
        path = root / rel
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except SyntaxError as exc:
            violations.append(f"{rel}: syntax error {exc}")
            continue
        for stmt in tree.body:
            if not isinstance(stmt, ast.Assign):
                continue
            names = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
            if not any(n.isupper() or n.startswith("_EXEC_APPROVAL_LABELS") or n.startswith("_SLASH_CONFIRM_LABELS") for n in names):
                continue
            for lineno, raw in _iter_static_strings(stmt.value):
                if _DECORATIVE_SLOP_EMOJI_RE.search(raw):
                    violations.append(f"{rel}:{lineno}: decorative AI-slop emoji in constant -> {raw[:90]!r}")
                if "—" in raw:
                    violations.append(f"{rel}:{lineno}: em-dash in gateway runtime notice -> {raw[:90]!r}")
    return violations


def run_all_checks(root: Path = ROOT) -> list[str]:
    violations: list[str] = []
    violations.extend(_check_catalog())
    violations.extend(_check_prompt_builder(root))
    violations.extend(_check_docs(root))
    violations.extend(_check_mobile_tools(root))
    violations.extend(_check_companion_prompt_and_doctor(root))
    violations.extend(_check_gateway_locales(root))
    violations.extend(_check_gateway_runtime_strings(root))
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
