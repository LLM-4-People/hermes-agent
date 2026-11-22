#!/usr/bin/env python3
"""Fail when tracked content contains an operator-specific identifier.

The identifiers are deliberately **not** committed: publishing the denylist would leak exactly
what it protects. They are supplied at the lowest choke point, by ``--denylist-file`` or the
``HERMES_PUBLIC_PII_DENYLIST`` environment variable (newline-separated). Each non-blank,
non-comment entry is a literal or a Python regular expression prefixed with ``re:``. Matching is
case-insensitive, and every rule is matched independently, so backreferences keep their own
numbering.

The scan reads tracked files directly, so it has exactly one regex dialect (Python's) and works
the same on every platform. Exit codes: 0 clean (or no denylist configured, with a visible
notice), 1 findings, 2 misuse. Findings never echo the matched text: the log itself must not
become the leak.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DENYLIST = Path(
    os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "hermes" / "pii-denylist"
_ENV = "HERMES_PUBLIC_PII_DENYLIST"
_REQUIRE_ENV = "HERMES_PUBLIC_PII_REQUIRE"
# Generated blobs (inline source maps, data URIs) are not reviewable text and produce random
# substring hits for short identifiers; a long base64 line is skipped and counted, never silent.
_GENERATED_LINE_MIN = 2000
_BINARY_PROBE = 8192


@dataclass(frozen=True)
class Rule:
    index: int
    pattern: str
    regex: bool

    def expression(self) -> str:
        return self.pattern if self.regex else re.escape(self.pattern)


def parse_denylist(text: str) -> list[Rule]:
    """One rule per meaningful line; ``re:`` opts into a regular expression."""
    rules: list[Rule] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        regex = line.startswith("re:")
        pattern = line[3:] if regex else line
        if not pattern:
            raise ValueError("empty denylist entry")
        if regex:
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                raise ValueError(f"invalid denylist regex at entry {len(rules) + 1}: {exc}") from exc
        rules.append(Rule(len(rules) + 1, pattern, regex))
    return rules


def load_denylist(path: Path | None, env_value: str | None) -> tuple[list[Rule], str]:
    """Rules plus where they came from. Environment wins over the file."""
    if env_value is not None and env_value.strip():
        return parse_denylist(env_value), _ENV
    candidate = path or DEFAULT_DENYLIST
    try:
        return parse_denylist(candidate.read_text(encoding="utf-8-sig")), str(candidate)
    except FileNotFoundError:
        return [], str(candidate)


def _tracked_files(repo: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {proc.stderr.strip()}")
    return [name for name in proc.stdout.split("\0") if name]


def scan(repo: Path, rules: Sequence[Rule]) -> tuple[list[str], int]:
    """``(matches sorted by path/line, generated lines skipped)``."""
    # One compiled pattern per rule: an alternation would renumber numeric backreferences.
    patterns = [re.compile(rule.expression(), re.IGNORECASE) for rule in rules]
    hits: list[str] = []
    skipped = 0
    for rel in _tracked_files(repo):
        path = repo / rel
        if path.is_symlink():  # git stores the link target, not the target's content
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:_BINARY_PROBE]:
            continue
        for lineno, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
            if not any(pattern.search(line) for pattern in patterns):
                continue
            if len(line) > _GENERATED_LINE_MIN and "base64," in line:
                skipped += 1
                continue
            hits.append(f"{rel}:{lineno}")
    return hits, skipped


def _report(hits: Iterable[str], source: str, rule_count: int) -> None:
    hits = list(hits)
    print(
        f"public-pii: {len(hits)} location(s) contain a protected identifier "
        f"({rule_count} rule(s) from {source}); text is redacted:",
        file=sys.stderr,
    )
    for hit in hits:
        print(f"  {hit}", file=sys.stderr)
    print("Remove the identifier or narrow the denylist entry.", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=REPO, help="repository to scan")
    parser.add_argument(
        "--denylist-file",
        type=Path,
        help=f"file of literal or 're:' entries (default: {DEFAULT_DENYLIST})",
    )
    parser.add_argument(
        "--require-denylist",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=f"fail when no denylist is configured (default: ${_REQUIRE_ENV})",
    )
    args = parser.parse_args(argv)

    require = args.require_denylist
    if require is None:
        require = os.environ.get(_REQUIRE_ENV, "").strip().lower() in {"1", "true", "yes"}

    try:
        rules, source = load_denylist(args.denylist_file, os.environ.get(_ENV))
    except (OSError, ValueError) as exc:
        print(f"public-pii: {exc}", file=sys.stderr)
        return 2

    if not rules:
        notice = f"public-pii: no denylist configured at {source}; literal scan skipped"
        if require:
            print(f"{notice}; {_REQUIRE_ENV} requires it", file=sys.stderr)
            return 2
        print(notice, file=sys.stderr)
        return 0

    try:
        hits, skipped = scan(args.repo, rules)
    except (OSError, RuntimeError) as exc:
        print(f"public-pii: {exc}", file=sys.stderr)
        return 2

    if hits:
        _report(hits, source, len(rules))
        if skipped:
            print(f"public-pii: {skipped} generated base64 line(s) skipped", file=sys.stderr)
        return 1
    print(
        f"public-pii: {len(rules)} rule(s) from {source}; no matches"
        + (f" ({skipped} generated base64 line(s) skipped)" if skipped else ""),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
