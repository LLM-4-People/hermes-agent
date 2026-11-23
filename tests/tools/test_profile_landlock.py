"""Real Linux kernel Landlock LSM + agent/file_safety.py cross-profile isolation tests."""

from __future__ import annotations

import errno
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from agent.file_safety import (
    FOREIGN_PROFILE_DENIAL,
    foreign_profile_search_exclusions,
    get_read_block_error,
    get_write_denied_error,
    is_foreign_profile_path,
)
from agent.prompt_builder import DEFAULT_AGENT_IDENTITY, PROFILE_ISOLATION_RULES
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations
from tools.file_tools import clear_file_ops_cache, patch_tool, read_file_tool, search_tool, write_file_tool
from tools.profile_landlock import (
    compute_allowed_roots,
    handled_fs_mask_for_abi,
    query_landlock_abi,
    wrap_argv_for_profile_sandbox,
)

ROOT = Path(__file__).resolve().parents[2]
LANDLOCK_SCRIPT = ROOT / "tools" / "profile_landlock.py"
PRESERVATION_SCRIPT = ROOT / "scripts" / "check_fork_preservation.py"

_LANDLOCK_ABI = query_landlock_abi()


@pytest.fixture
def multi_profile_tree(tmp_path: Path):
    """Build a realistic <root> + <root>/profiles/{allison,bob} layout with symlink escapes."""
    root_home = tmp_path / "opt_data"
    allison_home = root_home / "profiles" / "allison"
    bob_home = root_home / "profiles" / "bob"
    outside = tmp_path / "outside"

    for d in (root_home / "skills", allison_home / "notes", bob_home / "notes", outside):
        d.mkdir(parents=True, exist_ok=True)

    (root_home / "config.yaml").write_text("default_owner: true\n", encoding="utf-8")
    (root_home / ".env").write_text("API_SERVER_KEY=default-secret-123\n", encoding="utf-8")
    (root_home / "state.db").write_text("default-sqlite-placeholder\n", encoding="utf-8")
    (root_home / "skills" / "ok.txt").write_text("default-own-skill\n", encoding="utf-8")

    (allison_home / "config.yaml").write_text("signal_phone: '+15550199999'\n", encoding="utf-8")
    (allison_home / ".env").write_text("API_SERVER_KEY=allison-secret-456\n", encoding="utf-8")
    (allison_home / "state.db").write_text("allison-sqlite-placeholder\n", encoding="utf-8")
    (allison_home / "notes" / "diary.md").write_text("ALLISON_PRIVATE_MARKER\n", encoding="utf-8")

    (bob_home / "config.yaml").write_text("signal_phone: '+15550188888'\n", encoding="utf-8")
    (bob_home / "notes" / "todo.md").write_text("BOB_PRIVATE_MARKER\n", encoding="utf-8")

    (outside / "shared.txt").write_text("public-temp-file\n", encoding="utf-8")

    link_to_allison = outside / "escape_allison.yaml"
    link_to_allison.symlink_to(allison_home / "config.yaml")

    link_to_default_env = outside / "escape_default_env"
    link_to_default_env.symlink_to(root_home / ".env")

    link_to_profiles_dir = outside / "escape_profiles_dir"
    link_to_profiles_dir.symlink_to(root_home / "profiles")

    return {
        "root": root_home,
        "allison": allison_home,
        "bob": bob_home,
        "outside": outside,
        "link_to_allison": link_to_allison,
        "link_to_default_env": link_to_default_env,
        "link_to_profiles_dir": link_to_profiles_dir,
    }


@pytest.fixture
def active_profile_scope():
    tokens = []

    def _activate(home: Path):
        clear_file_ops_cache()
        tok = set_hermes_home_override(home)
        tokens.append(tok)
        return home

    yield _activate

    for tok in reversed(tokens):
        reset_hermes_home_override(tok)
    clear_file_ops_cache()


def _run_landlocked_python(current_home: Path, hermes_root: Path, code: str, *, mode: str = "strict") -> subprocess.CompletedProcess[str]:
    argv = [
        sys.executable,
        "-S",
        "-I",
        str(LANDLOCK_SCRIPT),
        "--current-home",
        str(current_home),
        "--hermes-root",
        str(hermes_root),
        "--mode",
        mode,
        "--",
        sys.executable,
        "-I",
        "-c",
        code,
    ]
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        encoding="utf-8",
        stdin=subprocess.DEVNULL,
        check=False,
    )


def test_handled_fs_mask_respects_abi_versions():
    assert handled_fs_mask_for_abi(0) == 0
    assert handled_fs_mask_for_abi(1) == 0x1FFF
    assert handled_fs_mask_for_abi(2) == 0x3FFF
    assert handled_fs_mask_for_abi(3) == 0x7FFF
    assert handled_fs_mask_for_abi(7) == 0x7FFF


def test_compute_allowed_roots_carve_out_structure(multi_profile_tree):
    root = multi_profile_tree["root"]
    allison = multi_profile_tree["allison"]
    outside = multi_profile_tree["outside"]

    # Default profile: allows root/skills and outside, NEVER root/profiles
    rw_def, ro_def = compute_allowed_roots(str(root), str(root))
    assert str(root / "skills") in rw_def
    assert str(outside) in rw_def
    assert str(root / "profiles") not in rw_def
    assert str(allison) not in rw_def
    assert ro_def == []

    # Ancestor-symlink attack: a symlink at an ancestor level pointing back to an ancestor of root
    # (like pytest's `pytest-current -> pytest-0`) MUST be excluded so Landlock never leaks root.
    ancestor_link = root.parent / "loop_to_parent"
    ancestor_link.symlink_to(root.parent)
    try:
        rw_with_loop, _ = compute_allowed_roots(str(root), str(root))
        assert str(ancestor_link) not in rw_with_loop
    finally:
        ancestor_link.unlink()

    # Named profile: allows ONLY allison inside root (plus outside), NEVER root/.env or bob
    rw_allison, _ = compute_allowed_roots(str(allison), str(root))
    assert str(allison) in rw_allison
    assert str(outside) in rw_allison
    assert str(root) not in rw_allison
    assert str(root / ".env") not in rw_allison
    assert str(root / "config.yaml") not in rw_allison
    assert str(multi_profile_tree["bob"]) not in rw_allison


def test_wrap_argv_noop_when_off_or_single_profile(tmp_path: Path, monkeypatch):
    single_root = tmp_path / "single"
    single_root.mkdir()
    cmd = ["/bin/echo", "hi"]
    # Single profile without profiles/ dir -> zero wrapper overhead
    assert wrap_argv_for_profile_sandbox(cmd, current_home=str(single_root), hermes_root=str(single_root), mode="auto") == cmd

    # Explicit off mode -> bypassed even with profiles/
    (single_root / "profiles").mkdir()
    assert wrap_argv_for_profile_sandbox(cmd, current_home=str(single_root), hermes_root=str(single_root), mode="off") == cmd

    if sys.platform.startswith("linux"):
        wrapped = wrap_argv_for_profile_sandbox(cmd, current_home=str(single_root), hermes_root=str(single_root), mode="auto")
        assert wrapped[1:3] == ["-S", "-I"]
        assert Path(wrapped[3]).resolve() == LANDLOCK_SCRIPT.resolve()
        assert wrapped[-2:] == cmd
        # Idempotent: never double-wraps
        assert wrap_argv_for_profile_sandbox(wrapped, current_home=str(single_root), hermes_root=str(single_root), mode="auto") == wrapped


@pytest.mark.skipif(_LANDLOCK_ABI <= 0, reason="Linux kernel Landlock LSM not available on this host")
def test_real_kernel_landlock_isolates_default_profile(multi_profile_tree):
    root = multi_profile_tree["root"]
    allison = multi_profile_tree["allison"]
    outside = multi_profile_tree["outside"]

    probe = f"""
import errno, json, os, pathlib
results = {{}}

def probe_read(name, path):
    try:
        data = pathlib.Path(path).read_text(encoding='utf-8')
        results[name] = ('ok', data.strip())
    except PermissionError as exc:
        results[name] = ('eacces', exc.errno)

def probe_listdir(name, path):
    try:
        entries = sorted(os.listdir(path))
        results[name] = ('ok', entries)
    except PermissionError as exc:
        results[name] = ('eacces', exc.errno)

# Must be DENIED by Linux kernel (direct, dynamic string concat, .., symlink, listdir)
probe_read('allison_config', {str(allison / 'config.yaml')!r})
probe_read('allison_concat', ''.join([{str(root)!r}, '/pro', 'files/all', 'ison/config.yaml']))
probe_read('allison_dotdot', {str(root / 'skills' / '..' / 'profiles' / 'allison' / 'config.yaml')!r})
probe_read('allison_symlink', {str(multi_profile_tree['link_to_allison'])!r})
probe_listdir('profiles_dir', {str(root / 'profiles')!r})

# Must SUCCEED (default's own skills + outside workspace)
probe_read('own_skill', {str(root / 'skills' / 'ok.txt')!r})
probe_read('outside_file', {str(outside / 'shared.txt')!r})
print(json.dumps(results))
"""
    proc = _run_landlocked_python(root, root, probe, mode="strict")
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    res = json.loads(proc.stdout)

    for blocked_key in ("allison_config", "allison_concat", "allison_dotdot", "allison_symlink", "profiles_dir"):
        assert res[blocked_key] == ["eacces", errno.EACCES], f"Kernel did not block {blocked_key}: {res[blocked_key]}"

    assert res["own_skill"] == ["ok", "default-own-skill"]
    assert res["outside_file"] == ["ok", "public-temp-file"]


@pytest.mark.skipif(_LANDLOCK_ABI <= 0, reason="Linux kernel Landlock LSM not available on this host")
def test_real_kernel_landlock_isolates_named_profile(multi_profile_tree):
    root = multi_profile_tree["root"]
    allison = multi_profile_tree["allison"]
    bob = multi_profile_tree["bob"]
    outside = multi_profile_tree["outside"]

    probe = f"""
import errno, json, os, pathlib
results = {{}}

def probe_read(name, path):
    try:
        data = pathlib.Path(path).read_text(encoding='utf-8')
        results[name] = ('ok', data.strip())
    except PermissionError as exc:
        results[name] = ('eacces', exc.errno)

# Must be DENIED by Linux kernel (host root .env/config/state.db, sibling bob, symlink)
probe_read('root_env', {str(root / '.env')!r})
probe_read('root_config', {str(root / 'config.yaml')!r})
probe_read('root_state', {str(root / 'state.db')!r})
probe_read('bob_config', {str(bob / 'config.yaml')!r})
probe_read('root_symlink', {str(multi_profile_tree['link_to_default_env'])!r})

# Must SUCCEED (allison's own profile read + write, plus outside workspace)
probe_read('own_diary', {str(allison / 'notes' / 'diary.md')!r})
probe_read('outside_file', {str(outside / 'shared.txt')!r})
new_note = pathlib.Path({str(allison / 'notes' / 'created.txt')!r})
new_note.write_text('allison-wrote-this', encoding='utf-8')
results['own_write'] = ('ok', new_note.read_text(encoding='utf-8'))
print(json.dumps(results))
"""
    proc = _run_landlocked_python(allison, root, probe, mode="strict")
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    res = json.loads(proc.stdout)

    for blocked_key in ("root_env", "root_config", "root_state", "bob_config", "root_symlink"):
        assert res[blocked_key] == ["eacces", errno.EACCES], f"Kernel did not block {blocked_key}: {res[blocked_key]}"

    assert res["own_diary"] == ["ok", "ALLISON_PRIVATE_MARKER"]
    assert res["outside_file"] == ["ok", "public-temp-file"]
    assert res["own_write"] == ["ok", "allison-wrote-this"]


@pytest.mark.skipif(_LANDLOCK_ABI <= 0, reason="Linux kernel Landlock LSM not available on this host")
def test_local_environment_subprocesses_enforce_kernel_landlock(multi_profile_tree, active_profile_scope):
    root = multi_profile_tree["root"]
    allison = multi_profile_tree["allison"]
    outside = multi_profile_tree["outside"]

    active_profile_scope(root)
    env = LocalEnvironment(cwd=str(outside))
    try:
        denied = env.execute(f"cat {allison / 'config.yaml'}", cwd=str(outside))
        assert denied["returncode"] != 0
        assert "15550199999" not in denied["output"]

        allowed = env.execute(f"cat {root / 'skills' / 'ok.txt'}", cwd=str(outside))
        assert allowed["returncode"] == 0
        assert "default-own-skill" in allowed["output"]
    finally:
        env.cleanup()


def test_file_safety_and_tools_enforce_foreign_profile_boundary(multi_profile_tree, active_profile_scope):
    root = multi_profile_tree["root"]
    allison = multi_profile_tree["allison"]
    bob = multi_profile_tree["bob"]
    outside = multi_profile_tree["outside"]

    # --- Default profile active ---
    active_profile_scope(root)
    assert is_foreign_profile_path(allison / "config.yaml") is True
    assert is_foreign_profile_path(root / "profiles") is True
    assert is_foreign_profile_path(root / "skills" / "ok.txt") is False
    assert foreign_profile_search_exclusions(root) == ["profiles"]

    assert get_read_block_error(str(allison / "config.yaml")) == FOREIGN_PROFILE_DENIAL
    assert get_read_block_error(str(multi_profile_tree["link_to_allison"])) == FOREIGN_PROFILE_DENIAL
    assert get_write_denied_error(str(allison / "notes" / "new.md")) == FOREIGN_PROFILE_DENIAL

    # read_file_tool, write_file_tool, patch_tool, and search_tool all fail closed
    for target in (
        str(allison / "config.yaml"),
        str(multi_profile_tree["link_to_allison"]),
        str(root / "skills" / ".." / "profiles" / "allison" / "config.yaml"),
    ):
        resp = json.loads(read_file_tool(target, task_id="landlock-default"))
        assert FOREIGN_PROFILE_DENIAL in resp.get("error", "")

    write_resp = json.loads(write_file_tool(str(allison / "notes" / "pwn.md"), "x", task_id="landlock-default"))
    assert FOREIGN_PROFILE_DENIAL in write_resp.get("error", "")

    patch_resp = json.loads(
        patch_tool(
            mode="replace",
            path=str(allison / "notes" / "diary.md"),
            old_string="ALLISON_PRIVATE_MARKER",
            new_string="NOPE",
            task_id="landlock-default",
        )
    )
    assert FOREIGN_PROFILE_DENIAL in patch_resp.get("error", "")

    for bad_root in (str(root / "profiles"), str(allison)):
        sr = json.loads(search_tool("signal_phone", target="content", path=bad_root, task_id="landlock-default"))
        assert FOREIGN_PROFILE_DENIAL in sr.get("error", "")

    # Recursive searches on default root prune profiles/ completely
    content_sr = json.loads(search_tool("ALLISON_PRIVATE_MARKER", target="content", path=str(root), task_id="landlock-default"))
    assert not content_sr.get("matches")
    assert "ALLISON_PRIVATE_MARKER" not in json.dumps(content_sr)

    files_sr = json.loads(search_tool("*", target="files", path=str(root), task_id="landlock-default"))
    listed = json.dumps(files_sr)
    assert "allison" not in listed
    assert "bob" not in listed
    assert "ok.txt" in listed

    # Direct ShellFileOperations check
    ops = ShellFileOperations(LocalEnvironment(cwd=str(outside)), cwd=str(outside))
    try:
        assert ops.read_file(str(allison / "config.yaml")).error == FOREIGN_PROFILE_DENIAL
        assert ops.write_file(str(allison / "x.txt"), "x").error == FOREIGN_PROFILE_DENIAL
        assert ops.delete_file(str(allison / "notes" / "diary.md")).error == FOREIGN_PROFILE_DENIAL
        assert ops.search("ALLISON", path=str(allison)).error == FOREIGN_PROFILE_DENIAL
    finally:
        ops.env.cleanup()

    # Positive: default reads its own skill and outside file
    assert "default-own-skill" in json.loads(read_file_tool(str(root / "skills" / "ok.txt"), task_id="landlock-default")).get("content", "")
    assert "public-temp-file" in json.loads(read_file_tool(str(outside / "shared.txt"), task_id="landlock-default")).get("content", "")

    # --- Named profile (allison) active ---
    active_profile_scope(allison)
    assert is_foreign_profile_path(root / ".env") is True
    assert is_foreign_profile_path(root / "config.yaml") is True
    assert is_foreign_profile_path(bob / "config.yaml") is True
    assert is_foreign_profile_path(allison / "notes" / "diary.md") is False

    for target in (str(root / ".env"), str(root / "config.yaml"), str(bob / "config.yaml")):
        assert FOREIGN_PROFILE_DENIAL in json.loads(read_file_tool(target, task_id="landlock-allison")).get("error", "")

    assert "ALLISON_PRIVATE_MARKER" in json.loads(read_file_tool(str(allison / "notes" / "diary.md"), task_id="landlock-allison")).get("content", "")


def test_prompt_builder_and_fork_preservation_include_landlock():
    assert PROFILE_ISOLATION_RULES in DEFAULT_AGENT_IDENTITY
    assert "strictly private" in DEFAULT_AGENT_IDENTITY.lower()

    spec = importlib.util.spec_from_file_location("check_fork_preservation", PRESERVATION_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_fork_preservation"] = mod
    spec.loader.exec_module(mod)

    assert "tools/profile_landlock.py" in mod.PROTECTED_EXACT_PATHS
    assert "tests/tools/test_profile_landlock.py" in mod.PROTECTED_EXACT_PATHS
