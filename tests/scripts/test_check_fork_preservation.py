"""Real-git integration tests for scripts/check_fork_preservation.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_fork_preservation.py"


@pytest.fixture(scope="module")
def guard():
    spec = importlib.util.spec_from_file_location("check_fork_preservation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_fork_preservation"] = module
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        stdin=subprocess.DEVNULL,
        check=True,
    )
    return proc.stdout.strip()


def _commit_file(repo: Path, rel_path: str, content: str, message: str) -> str:
    target = repo / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    _git(repo, "add", rel_path)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _init_fork_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Test Bot")
    _git(repo, "config", "user.email", "test@example.com")

    # Shared upstream base commit
    _commit_file(repo, "README.md", "Upstream README\n", "base: readme")
    _commit_file(
        repo,
        "shared.py",
        "line1 = 'a'\nline2 = 'b'\nline3 = 'c'\nline4 = 'd'\nline5 = 'e'\n",
        "base: shared module",
    )
    _commit_file(repo, "upstream_only.py", "x = 1\n", "base: upstream_only")

    # Create upstream branch at common ancestor
    _git(repo, "branch", "upstream-main")

    # Add fork-exclusive file + fork customization to shared.py on fork-main
    _commit_file(repo, "docs/MOBILE_COMPANION.md", "# Companion\n", "fork: add companion doc")
    _commit_file(repo, "tools/mobile_custom.py", "VALUE = 1\n", "fork: add mobile tool")
    _commit_file(repo, "fork_only.py", "FORK = True\n", "fork: add fork_only")
    _commit_file(
        repo,
        "shared.py",
        "line1 = 'fork-edit'\nline2 = 'b'\nline3 = 'c'\nline4 = 'd'\nline5 = 'e'\n",
        "fork: customize top of shared.py",
    )
    _git(repo, "branch", "fork-main")
    return repo


def test_disjoint_upstream_merge_passes(guard, tmp_path: Path):
    repo = _init_fork_repo(tmp_path)
    _git(repo, "checkout", "upstream-main")
    _commit_file(repo, "upstream_only.py", "x = 2\n", "upstream: edit upstream_only")
    _commit_file(repo, "upstream_new.py", "y = 3\n", "upstream: add upstream_new")

    _git(repo, "checkout", "-b", "sync", "fork-main")
    _git(repo, "merge", "--no-edit", "upstream-main")

    report_path = tmp_path / "report.txt"
    rc = guard.main(
        [
            "--repo",
            str(repo),
            "--base-ref",
            "fork-main",
            "--upstream-ref",
            "upstream-main",
            "--merged-ref",
            "HEAD",
            "--report-file",
            str(report_path),
        ]
    )
    assert rc == 0
    assert not report_path.exists()


def test_upstream_modifying_or_deleting_fork_exclusive_file_fails_closed(guard, tmp_path: Path):
    repo = _init_fork_repo(tmp_path)
    _git(repo, "checkout", "upstream-main")
    _commit_file(repo, "upstream_only.py", "x = 2\n", "upstream: touch upstream_only")

    # Simulate a merge commit that tampers with a fork-exclusive file AND a protected README.md
    _git(repo, "checkout", "-b", "sync", "fork-main")
    _git(repo, "merge", "--no-commit", "--no-ff", "upstream-main")
    (repo / "fork_only.py").unlink()
    (repo / "README.md").write_text("Overwritten by upstream\n", encoding="utf-8")
    (repo / "tools" / "mobile_custom.py").write_text("VALUE = 999\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "merge: tampered protected paths")

    report_path = tmp_path / "report.txt"
    rc = guard.main(
        [
            "--repo",
            str(repo),
            "--base-ref",
            "fork-main",
            "--upstream-ref",
            "upstream-main",
            "--merged-ref",
            "HEAD",
            "--report-file",
            str(report_path),
        ]
    )
    assert rc == 1
    report = report_path.read_text(encoding="utf-8-sig")
    assert "fork_only.py" in report
    assert "README.md" in report
    assert "tools/mobile_custom.py" in report


def test_clean_auto_merge_on_shared_file_overlap_still_fails_closed(guard, tmp_path: Path):
    repo = _init_fork_repo(tmp_path)
    # Upstream edits the BOTTOM of shared.py while fork edited the TOP -> git merge auto-resolves cleanly!
    _git(repo, "checkout", "upstream-main")
    _commit_file(
        repo,
        "shared.py",
        "line1 = 'a'\nline2 = 'b'\nline3 = 'c'\nline4 = 'd'\nline5 = 'upstream-edit'\n",
        "upstream: edit bottom of shared.py",
    )

    _git(repo, "checkout", "-b", "sync", "fork-main")
    _git(repo, "merge", "--no-edit", "upstream-main")  # Succeeds with ZERO conflict markers

    report_path = tmp_path / "report.txt"
    rc = guard.main(
        [
            "--repo",
            str(repo),
            "--base-ref",
            "fork-main",
            "--upstream-ref",
            "upstream-main",
            "--merged-ref",
            "HEAD",
            "--report-file",
            str(report_path),
        ]
    )
    assert rc == 1
    report = report_path.read_text(encoding="utf-8-sig")
    assert "Shared files modified by both fork and upstream" in report
    assert "shared.py" in report


def test_non_ancestor_rewritten_base_fails_closed(guard, tmp_path: Path):
    repo = _init_fork_repo(tmp_path)
    _git(repo, "checkout", "upstream-main")
    _commit_file(repo, "upstream_only.py", "x = 2\n", "upstream: advance without fork")

    report_path = tmp_path / "report.txt"
    rc = guard.main(
        [
            "--repo",
            str(repo),
            "--base-ref",
            "fork-main",
            "--upstream-ref",
            "upstream-main",
            "--merged-ref",
            "upstream-main",
            "--report-file",
            str(report_path),
        ]
    )
    assert rc == 1
    report = report_path.read_text(encoding="utf-8-sig")
    assert "Ancestry lock violation" in report


def test_invalid_ref_returns_exit_2(guard, tmp_path: Path):
    repo = _init_fork_repo(tmp_path)
    report_path = tmp_path / "err.txt"
    rc = guard.main(
        [
            "--repo",
            str(repo),
            "--base-ref",
            "refs/heads/does-not-exist",
            "--upstream-ref",
            "upstream-main",
            "--merged-ref",
            "fork-main",
            "--report-file",
            str(report_path),
        ]
    )
    assert rc == 2
    assert report_path.is_file()


def test_sync_upstream_workflow_invokes_trusted_guard():
    wf_path = ROOT / ".github" / "workflows" / "sync-upstream.yml"
    raw = wf_path.read_text(encoding="utf-8-sig")
    assert "git show origin/main:scripts/check_fork_preservation.py" in raw
    assert "--base-ref origin/main" in raw
    assert "--upstream-ref upstream/main" in raw
    assert "--merged-ref HEAD" in raw
    # Trusted capture must appear before switching to automation/upstream-sync and merging.
    assert raw.index("git show origin/main:scripts/check_fork_preservation.py") < raw.index(
        "merge_or_abort upstream/main"
    )
    assert raw.index("merge_or_abort upstream/main") < raw.index(
        "--base-ref origin/main"
    )
