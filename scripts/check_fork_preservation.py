#!/usr/bin/env python3
"""Fail-closed fork preservation guard for automated upstream syncs.

Enforces three invariants on a candidate merge of ``upstream/main`` into the
fork's ``origin/main`` so automated sync can never silently overwrite, clobber,
or revert fork changes:

1. Ancestry lock: ``base_ref`` (``origin/main``) must be a strict ancestor of
   ``merged_ref`` (``HEAD``).
2. Fork-exclusive immutability: every file added exclusively by the fork since
   ``merge-base(base_ref, upstream_ref)`` plus explicit protected fork paths
   must have an empty diff between ``base_ref`` and ``merged_ref``.
3. Shared-file overlap guard: if ``fork_changed ∩ upstream_changed`` since
   ``merge-base`` is non-empty, fail closed even when ``git merge`` resolved
   the overlap without textual conflict markers.

Zero third-party dependencies (Python standard library only). Exit codes:
  0  all invariants hold (safe to proceed with verification/PR)
  1  one or more preservation invariants failed (human resolution required)
  2  invalid invocation or unreadable git repository/refs
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
import subprocess
import sys

# Explicit fork-owned paths that upstream must never alter or delete.
PROTECTED_EXACT_PATHS: frozenset[str] = frozenset({
    "README.md",
    "DOWNSTREAM.md",
    "docs/MOBILE_COMPANION.md",
    "scripts/downstream-verify.sh",
    "scripts/mobile_doctor.py",
    "scripts/check_fork_preservation.py",
    "gateway/i18n.py",
    ".github/workflows/sync-upstream.yml",
    ".github/workflows/downstream-ci.yml",
    ".github/workflows/docker-publish.yml",
    ".github/workflows/pii-guard.yml",
    ".github/workflows/codeql.yml",
    ".github/workflows/dependency-review.yml",
})

PROTECTED_GLOB_PATTERNS: tuple[str, ...] = (
    "tools/mobile_*.py",
)

_GIT_TIMEOUT_SECONDS = 60


class GitCommandError(RuntimeError):
    """Raised when a required git inspection command fails."""


@dataclass(frozen=True)
class PreservationResult:
    ok: bool
    base_sha: str
    upstream_sha: str
    merged_sha: str
    merge_base_sha: str
    ancestry_ok: bool
    protected_violations: tuple[str, ...]
    shared_overlaps: tuple[str, ...]

    def format_report(self) -> str:
        lines = [
            "Fork preservation guard blocked automated upstream sync:",
            f"- base ({self.base_sha[:12]}), upstream ({self.upstream_sha[:12]}), merged ({self.merged_sha[:12]}), merge-base ({self.merge_base_sha[:12]})",
        ]
        if not self.ancestry_ok:
            lines.append("")
            lines.append("Ancestry lock violation:")
            lines.append("  - base_ref is NOT an ancestor of merged_ref (fork commits would be dropped or rewritten).")
        if self.protected_violations:
            lines.append("")
            lines.append(f"Protected / fork-exclusive files altered or deleted ({len(self.protected_violations)}):")
            lines.extend(f"  - {path}" for path in self.protected_violations)
        if self.shared_overlaps:
            lines.append("")
            lines.append(f"Shared files modified by both fork and upstream ({len(self.shared_overlaps)}):")
            lines.extend(f"  - {path}" for path in self.shared_overlaps)
        lines.append("")
        lines.append("Resolve manually on a branch with human review so no fork changes are lost.")
        return "\n".join(lines) + "\n"


def _run_git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
        timeout=_GIT_TIMEOUT_SECONDS,
        check=False,
    )
    if check and proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise GitCommandError(f"git {' '.join(args)} failed: {detail}")
    return proc


def _resolve_commit(repo: Path, ref: str) -> str:
    return _run_git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").stdout.strip()


def _is_ancestor(repo: Path, ancestor_sha: str, descendant_sha: str) -> bool:
    proc = _run_git(repo, "merge-base", "--is-ancestor", ancestor_sha, descendant_sha, check=False)
    if proc.returncode == 0:
        return True
    if proc.returncode == 1:
        return False
    detail = proc.stderr.strip() or f"exit {proc.returncode}"
    raise GitCommandError(f"git merge-base --is-ancestor failed: {detail}")


def _diff_paths(repo: Path, rev_range: str, *, diff_filter: str | None = None) -> set[str]:
    args = ["diff", "--no-renames", "--name-only", "-z"]
    if diff_filter:
        args.append(f"--diff-filter={diff_filter}")
    args.append(rev_range)
    raw = _run_git(repo, *args).stdout
    return {entry for entry in raw.split("\0") if entry}


def _tree_paths(repo: Path, commit_sha: str) -> set[str]:
    raw = _run_git(repo, "ls-tree", "-r", "--name-only", "-z", commit_sha).stdout
    return {entry for entry in raw.split("\0") if entry}


def _matches_protected_glob(path: str) -> bool:
    return any(fnmatch(path, pattern) for pattern in PROTECTED_GLOB_PATTERNS)


def check_preservation(
    *,
    repo: Path,
    base_ref: str,
    upstream_ref: str,
    merged_ref: str,
) -> PreservationResult:
    """Evaluate all three fork-preservation invariants for a candidate merge."""
    base_sha = _resolve_commit(repo, base_ref)
    upstream_sha = _resolve_commit(repo, upstream_ref)
    merged_sha = _resolve_commit(repo, merged_ref)
    merge_base_sha = _run_git(repo, "merge-base", base_sha, upstream_sha).stdout.strip()
    if not merge_base_sha:
        raise GitCommandError(f"No common merge-base between {base_ref} and {upstream_ref}.")

    ancestry_ok = _is_ancestor(repo, base_sha, merged_sha)

    fork_added = _diff_paths(repo, f"{merge_base_sha}..{base_sha}", diff_filter="A")
    fork_changed = _diff_paths(repo, f"{merge_base_sha}..{base_sha}")
    upstream_changed = _diff_paths(repo, f"{merge_base_sha}..{upstream_sha}")
    merged_vs_base = _diff_paths(repo, f"{base_sha}..{merged_sha}")

    base_tree = _tree_paths(repo, base_sha)
    merged_tree = _tree_paths(repo, merged_sha)
    present_paths = base_tree | merged_tree

    explicit_protected = {path for path in PROTECTED_EXACT_PATHS if path in present_paths}
    glob_protected = {path for path in present_paths if _matches_protected_glob(path)}
    protected_paths = fork_added | explicit_protected | glob_protected

    protected_violations = tuple(sorted(protected_paths & merged_vs_base))
    shared_overlaps = tuple(sorted(fork_changed & upstream_changed))

    ok = ancestry_ok and not protected_violations and not shared_overlaps
    return PreservationResult(
        ok=ok,
        base_sha=base_sha,
        upstream_sha=upstream_sha,
        merged_sha=merged_sha,
        merge_base_sha=merge_base_sha,
        ancestry_ok=ancestry_ok,
        protected_violations=protected_violations,
        shared_overlaps=shared_overlaps,
    )


def _write_report(report_file: Path, content: str) -> None:
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(content, encoding="utf-8", newline="\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_fork_preservation.py",
        description="Fail-closed fork preservation guard for automated upstream merges.",
    )
    parser.add_argument("--repo", type=Path, default=Path("."), help="Git repository path (default: .)")
    parser.add_argument("--base-ref", required=True, help="Fork base ref (e.g. origin/main)")
    parser.add_argument("--upstream-ref", required=True, help="Upstream ref (e.g. upstream/main)")
    parser.add_argument("--merged-ref", default="HEAD", help="Candidate merged ref (default: HEAD)")
    parser.add_argument("--report-file", type=Path, help="Optional file path to write failure details")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = check_preservation(
            repo=args.repo,
            base_ref=args.base_ref,
            upstream_ref=args.upstream_ref,
            merged_ref=args.merged_ref,
        )
    except (GitCommandError, OSError, subprocess.TimeoutExpired) as exc:
        msg = f"check_fork_preservation: {exc}\n"
        print(msg, file=sys.stderr, end="")
        if args.report_file is not None:
            try:
                _write_report(args.report_file, msg)
            except OSError:
                pass
        return 2

    if result.ok:
        print(
            f"check_fork_preservation: OK (ancestry verified, 0 protected violations, "
            f"0 shared-file overlaps; base={result.base_sha[:12]} merged={result.merged_sha[:12]})"
        )
        return 0

    report = result.format_report()
    print(report, file=sys.stderr, end="")
    if args.report_file is not None:
        try:
            _write_report(args.report_file, report)
        except OSError as exc:
            print(f"check_fork_preservation: failed to write report file: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
