"""Tests for mechanically generated fork docs & README (scripts/docs_fork.py)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "docs_fork.py"
PRESERVATION_SCRIPT = ROOT / "scripts" / "check_fork_preservation.py"


@pytest.fixture(scope="module")
def docs_fork():
    spec = importlib.util.spec_from_file_location("docs_fork", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["docs_fork"] = module
    spec.loader.exec_module(module)
    return module


def _copy_minimal_repo(tmp_path: Path) -> Path:
    """Copy only the files introspected and rendered by docs_fork.py into an isolated tmp repo."""
    repo = tmp_path / "repo"
    repo.mkdir()
    rel_files = [
        "README.md",
        "DOWNSTREAM.md",
        "docs/MOBILE_COMPANION.md",
        "docs/sources/fork_facts.json",
        "docs/sources/README_FORK.md.in",
        "docs/sources/MOBILE_COMPANION.md.in",
        "docs/sources/DOWNSTREAM.md.in",
        "docs/assets/social-preview.png",
        "docs/assets/mobile-companion-preview.png",
        "gateway/platforms/api_server.py",
        "gateway/platforms/api_server_ui.py",
        "gateway/platforms/api_server_memory.py",
        "gateway/platforms/api_server_skills.py",
        "gateway/platforms/api_server_workspace.py",
        "gateway/platforms/api_server_profile.py",
        "gateway/platforms/api_server_notes.py",
        "gateway/platforms/api_server_sharing.py",
        "gateway/platforms/api_server_habits.py",
        "gateway/platforms/api_server_auth_guard.py",
        "gateway/i18n.py",
        "hermes_cli/tools_config.py",
        "scripts/fork-owned-workflows.txt",
        ".github/workflows/downstream-ci.yml",
    ]
    for rel in rel_files:
        src = ROOT / rel
        dst = repo / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    for src in (ROOT / "tools").glob("mobile_*.py"):
        dst = repo / "tools" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    for src in (ROOT / "locales").glob("*.yaml"):
        dst = repo / "locales" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    return repo


def test_real_repo_passes_isolated_cli_check():
    proc = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--check"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        stdin=subprocess.DEVNULL,
        check=False,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "0 drift" in proc.stdout


def test_write_is_idempotent_and_preserves_upstream_readme(docs_fork, tmp_path: Path):
    repo = _copy_minimal_repo(tmp_path)
    readme_path = repo / "README.md"
    before_readme = readme_path.read_text(encoding="utf-8-sig")
    marker = docs_fork.UPSTREAM_README_HEADING
    assert marker in before_readme
    upstream_tail_before = before_readme.split(marker, 1)[1]

    # Running --write on an already-synced repo writes 0 files.
    written_first = docs_fork.run_write(repo)
    assert written_first == []
    assert docs_fork.run_check(repo) == []

    # Mutate only inside the fork section, regenerate, and confirm upstream content is untouched.
    readme_path.write_text(
        before_readme.replace("3-minute Docker quickstart", "stale heading"),
        encoding="utf-8",
    )
    assert docs_fork.run_check(repo)  # drift detected

    written_second = docs_fork.run_write(repo)
    assert readme_path in written_second
    assert docs_fork.run_write(repo) == []  # second pass is strictly idempotent
    after_readme = readme_path.read_text(encoding="utf-8-sig")
    assert after_readme.split(marker, 1)[1] == upstream_tail_before
    assert docs_fork.README_BEGIN_MARKER in after_readme
    assert docs_fork.README_END_MARKER in after_readme


def test_check_detects_generated_doc_drift(docs_fork, tmp_path: Path):
    repo = _copy_minimal_repo(tmp_path)
    target = repo / "docs" / "MOBILE_COMPANION.md"
    target.write_text(target.read_text(encoding="utf-8-sig") + "\nManual stale edit\n", encoding="utf-8")

    errors = docs_fork.run_check(repo)
    assert any("MOBILE_COMPANION.md" in err for err in errors)
    assert docs_fork.main(["--check", "--root", str(repo)]) == 1


def test_check_detects_stale_live_fact_mismatch(docs_fork, tmp_path: Path):
    repo = _copy_minimal_repo(tmp_path)
    facts_path = repo / "docs" / "sources" / "fork_facts.json"
    facts = json.loads(facts_path.read_text(encoding="utf-8-sig"))
    facts["live_facts"]["active_capability_count"] = 999
    facts_path.write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")

    errors = docs_fork.run_check(repo)
    assert any("stale live_facts.active_capability_count" in err for err in errors)
    assert docs_fork.main(["--check", "--root", str(repo)]) == 1

    # --write reconciles live_facts back to live source truth.
    assert docs_fork.main(["--write", "--root", str(repo)]) == 0
    assert docs_fork.main(["--check", "--root", str(repo)]) == 0


def test_check_detects_source_code_change_without_regeneration(docs_fork, tmp_path: Path):
    repo = _copy_minimal_repo(tmp_path)
    guard_path = repo / "gateway" / "platforms" / "api_server_auth_guard.py"
    raw = guard_path.read_text(encoding="utf-8-sig")
    assert "DEFAULT_MAX_CONCURRENT_RUNS = 32" in raw
    guard_path.write_text(
        raw.replace("DEFAULT_MAX_CONCURRENT_RUNS = 32", "DEFAULT_MAX_CONCURRENT_RUNS = 64", 1),
        encoding="utf-8",
    )

    errors = docs_fork.run_check(repo)
    assert any("max_concurrent_runs" in err for err in errors)
    assert any("README.md" in err for err in errors)


def test_readme_malformed_markers_fail_closed(docs_fork, tmp_path: Path):
    repo = _copy_minimal_repo(tmp_path)
    readme_path = repo / "README.md"
    readme_path.write_text(f"{docs_fork.README_BEGIN_MARKER}\nUnclosed fork docs\n", encoding="utf-8")

    with pytest.raises(docs_fork.DocsForkError, match="malformed"):
        docs_fork.run_check(repo)
    assert docs_fork.main(["--check", "--root", str(repo)]) == 2


def test_unknown_template_placeholder_fails_closed(docs_fork, tmp_path: Path):
    repo = _copy_minimal_repo(tmp_path)
    tpl = repo / "docs" / "sources" / "DOWNSTREAM.md.in"
    tpl.write_text(tpl.read_text(encoding="utf-8-sig") + "\n{{nonexistent_fact}}\n", encoding="utf-8")

    with pytest.raises(docs_fork.DocsForkError, match="nonexistent_fact"):
        docs_fork.run_check(repo)


def test_missing_referenced_asset_fails_closed(docs_fork, tmp_path: Path):
    repo = _copy_minimal_repo(tmp_path)
    (repo / "docs" / "assets" / "social-preview.png").unlink()

    with pytest.raises(docs_fork.DocsForkError, match="social-preview.png"):
        docs_fork.run_check(repo)


def test_fork_preservation_protects_docs_fork_and_sources():
    spec = importlib.util.spec_from_file_location("check_fork_preservation", PRESERVATION_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_fork_preservation"] = mod
    spec.loader.exec_module(mod)

    assert "scripts/docs_fork.py" in mod.PROTECTED_EXACT_PATHS
    assert "tests/scripts/test_docs_fork.py" in mod.PROTECTED_EXACT_PATHS
    assert mod._matches_protected_glob("docs/sources/fork_facts.json")
    assert mod._matches_protected_glob("docs/sources/README_FORK.md.in")
