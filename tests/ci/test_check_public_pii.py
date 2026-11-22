"""The public-tree identifier guard: precise, case-insensitive, and redacted in its own output."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ci" / "check_public_pii.py"


def _repo(tmp_path: Path, files: dict[str, bytes]) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    return repo


def _run(repo: Path, denylist: str | None, *, require: bool = False, tmp_path: Path):
    argv = [sys.executable, str(SCRIPT), "--repo", str(repo)]
    if denylist is not None:
        path = tmp_path / "denylist.txt"
        path.write_text(denylist)
        argv += ["--denylist-file", str(path)]
    if require:
        argv.append("--require-denylist")
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def test_literal_match_is_case_insensitive_and_redacted(tmp_path):
    secret = "corp-internal.example"
    repo = _repo(tmp_path, {
        "docs/notes.md": f"route: {secret.upper()}/v1\n".encode(),
        "src/app.py": b"BASE = 'https://public.example/v1'\n",
    })
    result = _run(repo, f"{secret}\n", tmp_path=tmp_path)

    assert result.returncode == 1
    assert "docs/notes.md:1" in result.stderr
    assert "src/app.py" not in result.stderr
    assert secret not in result.stderr  # the log must not become the leak
    assert secret.upper() not in result.stderr


def test_regex_entry_matches_and_reports_each_location(tmp_path):
    repo = _repo(tmp_path, {
        "a.txt": b"token-1234\ntoken-9999\n",
        "b.txt": b"token-12\n",
    })
    result = _run(repo, "re:token-[0-9]{4}\n", tmp_path=tmp_path)

    assert result.returncode == 1
    assert "a.txt:1" in result.stderr
    assert "a.txt:2" in result.stderr
    assert "b.txt" not in result.stderr


def test_binary_files_and_untracked_files_are_ignored(tmp_path):
    repo = _repo(tmp_path, {"blob.bin": b"\x89PNG\r\n\x1a\nsecret-value\x00"})
    (repo / "scratch.txt").write_text("secret-value")
    result = _run(repo, "secret-value\n", tmp_path=tmp_path)

    assert result.returncode == 0
    assert "no matches" in result.stderr


def test_generated_base64_lines_are_counted_not_flagged(tmp_path):
    """A short identifier can appear by chance inside an inline source map; that is a generated
    blob, not reviewable text, and the skip is reported rather than silent."""
    repo = _repo(tmp_path, {
        "bundle.tsx": b"//# sourceMappingURL=data:application/json;base64," + b"A" * 3000 + b"\n",
    })
    result = _run(repo, "aaa\n", tmp_path=tmp_path)

    assert result.returncode == 0
    assert "1 generated base64 line(s) skipped" in result.stderr


def test_no_denylist_skips_visibly_or_fails_when_required(tmp_path):
    repo = _repo(tmp_path, {"a.txt": b"anything\n"})

    skipped = _run(repo, None, tmp_path=tmp_path)
    assert skipped.returncode == 0
    assert "no denylist configured" in skipped.stderr

    required = _run(repo, None, require=True, tmp_path=tmp_path)
    assert required.returncode == 2
    assert "no denylist configured" in required.stderr


def test_invalid_regex_is_misuse_not_a_silent_pass(tmp_path):
    repo = _repo(tmp_path, {"a.txt": b"anything\n"})
    result = _run(repo, "re:[unclosed\n", tmp_path=tmp_path)

    assert result.returncode == 2
    assert "invalid denylist regex" in result.stderr


def test_regexes_use_the_same_python_dialect_that_validates_them(tmp_path):
    """``\\d`` is a digit in Python and a literal ``d`` in POSIX ERE: one dialect, or the gate lies."""
    repo = _repo(tmp_path, {"a.txt": b"token-1234\n"})
    result = _run(repo, "re:token-\\d{4}\n", tmp_path=tmp_path)

    assert result.returncode == 1
    assert "a.txt:1" in result.stderr


def test_inline_flags_and_backreferences_follow_python_semantics(tmp_path):
    """Each rule compiles on its own: an inline flag works, and a numeric backreference keeps its
    own group numbering even when another rule also has a group."""
    repo = _repo(tmp_path, {"a.txt": b"bb\ncorp-internal\n"})
    result = _run(repo, "re:(z)\nre:(b)\\1\nre:(?i)CORP-INTERNAL\n", tmp_path=tmp_path)

    assert result.returncode == 1
    assert "a.txt:1" in result.stderr
    assert "a.txt:2" in result.stderr
