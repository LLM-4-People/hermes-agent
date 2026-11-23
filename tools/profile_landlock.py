#!/usr/bin/env python3
"""Linux kernel Landlock LSM subprocess sandbox for cross-profile isolation.

Zero third-party dependencies (pure stdlib: ``ctypes``, ``errno``, ``os``, ``stat``, ``sys``).
Runnable directly as a thread-safe ``python3 -S -I`` exec wrapper so multithreaded gateways
never use ``preexec_fn``::

    python3 -S -I /abs/path/to/tools/profile_landlock.py \\
        --current-home /opt/data/profiles/allison \\
        --hermes-root /opt/data \\
        --mode auto \\
        -- /bin/bash -c "cat /opt/data/.env"

Algorithm (ancestor-sibling carve-out):
  1. Walk from ``/`` down to ``hermes_root``; at every ancestor directory, allow every direct
     sibling except the component leading toward ``hermes_root`` (skipping symlinks whose
     resolved target lands inside ``hermes_root``).
  2. Inside ``hermes_root``:
     - Default profile (``current_home == hermes_root``): allow every direct entry of
       ``hermes_root`` EXCEPT ``profiles`` (skipping symlinks targeting ``profiles/``).
       The Linux kernel VFS then denies ``<hermes_root>/profiles`` and every sibling profile
       inside it with ``EACCES`` across all syscalls, symlinks, ``..`` traversals, and child
       processes.
     - Named profile (``<hermes_root>/profiles/<slug>``): allow ONLY ``current_home`` with
       full read/write access (plus read+execute only on ``<hermes_root>/tools`` and
       ``<hermes_root>/bin`` when present so PM-installed interpreters remain runnable).
       Every root file (``.env``, ``config.yaml``, ``state.db``, ``memories/``, ``sessions/``)
       and every sibling profile is denied by the kernel with ``EACCES``.
"""

from __future__ import annotations

import ctypes
import errno
import os
import stat
import struct
import sys

# Linux syscall numbers (identical on x86_64 and aarch64).
_SYS_LANDLOCK_CREATE_RULESET = 444
_SYS_LANDLOCK_ADD_RULE = 445
_SYS_LANDLOCK_RESTRICT_SELF = 446

_PR_SET_NO_NEW_PRIVS = 38
_LANDLOCK_CREATE_RULESET_VERSION = 1 << 0
_LANDLOCK_RULE_PATH_BENEATH = 1

# Landlock filesystem access flags (include/uapi/linux/landlock.h).
_FS_EXECUTE = 1 << 0
_FS_WRITE_FILE = 1 << 1
_FS_READ_FILE = 1 << 2
_FS_READ_DIR = 1 << 3
_FS_REMOVE_DIR = 1 << 4
_FS_REMOVE_FILE = 1 << 5
_FS_MAKE_CHAR = 1 << 6
_FS_MAKE_DIR = 1 << 7
_FS_MAKE_REG = 1 << 8
_FS_MAKE_SOCK = 1 << 9
_FS_MAKE_FIFO = 1 << 10
_FS_MAKE_BLOCK = 1 << 11
_FS_MAKE_SYM = 1 << 12
_FS_REFER = 1 << 13       # ABI >= 2
_FS_TRUNCATE = 1 << 14    # ABI >= 3

# Cap handled FS mask at ABI 3 (EXECUTE..TRUNCATE) so ABI 5+ IOCTL_DEV never interferes with PTYs.
_ABI_1_MASK = 0x1FFF
_ABI_2_MASK = 0x3FFF
_ABI_3_MASK = 0x7FFF

# Bits valid on non-directory file descriptors (passing dir-only bits on a regular file returns EINVAL).
_FILE_VALID_MASK = _FS_EXECUTE | _FS_WRITE_FILE | _FS_READ_FILE | _FS_TRUNCATE
_READ_EXEC_DIR_MASK = _FS_EXECUTE | _FS_READ_FILE | _FS_READ_DIR
_READ_EXEC_FILE_MASK = _FS_EXECUTE | _FS_READ_FILE

_O_PATH = getattr(os, "O_PATH", 0x200000)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0x80000)

_SANDBOX_OFF_VALUES = frozenset({"off", "disabled", "0", "false", "no"})
_SANDBOX_STRICT_VALUES = frozenset({"strict", "enforce", "required", "1", "true"})


class _LandlockRulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


def _pack_path_beneath_attr(allowed_access: int, parent_fd: int) -> ctypes.Array[ctypes.c_char]:
    """Pack ``struct landlock_path_beneath_attr`` (``__u64 allowed_access; __s32 parent_fd``)
    without ``_pack_`` so Python 3.14+ emits zero ``DeprecationWarning``."""
    return ctypes.create_string_buffer(struct.pack("=Qi", allowed_access, parent_fd), 12)


def _load_libc() -> ctypes.CDLL:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    libc.prctl.restype = ctypes.c_int
    return libc


def query_landlock_abi() -> int:
    """Return kernel Landlock ABI version (>= 1), or 0 when unsupported / non-Linux."""
    if not sys.platform.startswith("linux"):
        return 0
    try:
        libc = _load_libc()
        abi = int(
            libc.syscall(
                ctypes.c_long(_SYS_LANDLOCK_CREATE_RULESET),
                ctypes.c_void_p(0),
                ctypes.c_size_t(0),
                ctypes.c_uint32(_LANDLOCK_CREATE_RULESET_VERSION),
            )
        )
        return max(0, abi)
    except OSError:
        return 0


def handled_fs_mask_for_abi(abi: int) -> int:
    """Return the Landlock ``handled_access_fs`` bitmask supported by *abi*."""
    if abi >= 3:
        return _ABI_3_MASK
    if abi == 2:
        return _ABI_2_MASK
    if abi == 1:
        return _ABI_1_MASK
    return 0


def _is_under_posix(path: str, base: str) -> bool:
    norm_p = os.path.normpath(path)
    norm_b = os.path.normpath(base)
    if norm_p == norm_b:
        return True
    prefix = "/" if norm_b == "/" else norm_b + "/"
    return norm_p.startswith(prefix)


def compute_allowed_roots(current_home: str, hermes_root: str) -> tuple[list[str], list[str]]:
    """Compute ``(rw_paths, ro_exec_paths)`` using the ancestor-sibling carve-out algorithm.

    - Walks from ``/`` down to ``real_root``: at each ancestor directory, allows every direct
      sibling entry except the segment leading to ``real_root`` (skipping symlinks whose target
      lands inside ``real_root``).
    - Inside ``real_root``:
      - Default profile (``real_current == real_root``): allows every direct entry of
        ``real_root`` except ``profiles`` (skipping symlinks targeting ``profiles``).
      - Named profile (``real_current`` under ``real_root/profiles/<slug>``): allows ONLY
        ``real_current`` (R/W) plus read-execute-only ``real_root/tools`` and ``real_root/bin``
        if they exist so PM-installed interpreters remain usable.
    """
    real_current = os.path.realpath(current_home)
    real_root = os.path.realpath(hermes_root)
    is_named = real_current != real_root and _is_under_posix(real_current, os.path.join(real_root, "profiles"))
    profiles_dir = os.path.realpath(os.path.join(real_root, "profiles"))

    rw_paths: list[str] = []
    ro_exec_paths: list[str] = []

    # 1. Walk from '/' down to real_root, allowing all sibling subtrees outside real_root.
    parts = [p for p in real_root.split("/") if p]
    anc = "/"
    for seg in parts:
        try:
            with os.scandir(anc) as it:
                for entry in it:
                    if entry.name == seg:
                        continue
                    entry_path = entry.path
                    if entry.is_symlink():
                        try:
                            target = os.path.realpath(entry_path)
                        except OSError:
                            continue
                        if not os.path.exists(target):
                            continue
                        # Never allow a symlink pointing INTO real_root OR to any ANCESTOR of real_root
                        # (e.g., `pytest-current -> pytest-0` or `/link -> /opt`), because Landlock rules
                        # inherit downward from the symlink's resolved target.
                        if (
                            _is_under_posix(target, real_root)
                            or _is_under_posix(real_root, target)
                            or _is_under_posix(target, profiles_dir)
                        ):
                            continue
                    rw_paths.append(entry_path)
        except OSError:
            pass
        anc = "/" + seg if anc == "/" else anc + "/" + seg

    # 2. Carve out inside real_root.
    if not is_named:
        # Default profile: allow every direct entry of real_root EXCEPT 'profiles'.
        try:
            with os.scandir(real_root) as it:
                for entry in it:
                    if entry.name == "profiles":
                        continue
                    entry_path = entry.path
                    if entry.is_symlink():
                        try:
                            target = os.path.realpath(entry_path)
                        except OSError:
                            continue
                        if (
                            not os.path.exists(target)
                            or _is_under_posix(target, profiles_dir)
                            or _is_under_posix(profiles_dir, target)
                        ):
                            continue
                    rw_paths.append(entry_path)
        except OSError:
            pass
    else:
        # Named profile: allow ONLY current_home (R/W) + read-exec-only runtime tool dirs if present.
        if os.path.exists(real_current):
            rw_paths.append(real_current)
        for runtime_dir in ("tools", "bin"):
            cand = os.path.join(real_root, runtime_dir)
            if os.path.isdir(cand) and not os.path.islink(cand):
                ro_exec_paths.append(cand)

    return rw_paths, ro_exec_paths


def _add_path_rule(
    libc: ctypes.CDLL,
    ruleset_fd: int,
    path: str,
    *,
    dir_access: int,
    file_access: int,
) -> None:
    try:
        fd = os.open(path, _O_PATH | _O_CLOEXEC)
    except OSError:
        return
    try:
        st = os.fstat(fd)
        allowed = dir_access if stat.S_ISDIR(st.st_mode) else file_access
        if not allowed:
            return
        attr = _pack_path_beneath_attr(allowed, fd)
        libc.syscall(
            ctypes.c_long(_SYS_LANDLOCK_ADD_RULE),
            ctypes.c_int(ruleset_fd),
            ctypes.c_int(_LANDLOCK_RULE_PATH_BENEATH),
            attr,
            ctypes.c_uint32(0),
        )
    except OSError:
        return
    finally:
        os.close(fd)


def apply_landlock(current_home: str, hermes_root: str) -> tuple[bool, str]:
    """Enforce Landlock LSM rules on the current process. Returns ``(applied, detail)``."""
    if not sys.platform.startswith("linux"):
        return False, "Landlock is only supported on Linux"

    abi = query_landlock_abi()
    if abi <= 0:
        return False, "Linux kernel Landlock LSM is unavailable"

    handled_fs = handled_fs_mask_for_abi(abi)
    file_rw_mask = handled_fs & _FILE_VALID_MASK
    ro_dir_mask = handled_fs & _READ_EXEC_DIR_MASK
    ro_file_mask = handled_fs & _READ_EXEC_FILE_MASK

    libc = _load_libc()
    ruleset_attr = _LandlockRulesetAttr(handled_access_fs=handled_fs)
    ruleset_fd = int(
        libc.syscall(
            ctypes.c_long(_SYS_LANDLOCK_CREATE_RULESET),
            ctypes.byref(ruleset_attr),
            ctypes.c_size_t(ctypes.sizeof(ruleset_attr)),
            ctypes.c_uint32(0),
        )
    )
    if ruleset_fd < 0:
        err = ctypes.get_errno()
        return False, f"landlock_create_ruleset failed: errno={err} ({os.strerror(err)})"

    try:
        real_current = os.path.realpath(current_home)
        real_root = os.path.realpath(hermes_root)
        is_named = real_current != real_root and _is_under_posix(real_current, os.path.join(real_root, "profiles"))

        rw_paths, ro_exec_paths = compute_allowed_roots(current_home, hermes_root)
        for path in rw_paths:
            _add_path_rule(libc, ruleset_fd, path, dir_access=handled_fs, file_access=file_rw_mask)
        for path in ro_exec_paths:
            _add_path_rule(libc, ruleset_fd, path, dir_access=ro_dir_mask, file_access=ro_file_mask)
        if not is_named:
            # Default profile: allow opendir(hermes_root) ONLY (no READ_FILE, WRITE_FILE, EXECUTE,
            # MAKE_*, REMOVE_*, or TRUNCATE) so `ls` / `search_files` on default's own root work
            # while every file open/read/write/exec/delete inside `<hermes_root>/profiles/*` stays
            # blocked by the Linux kernel with EACCES. Named profiles receive ZERO access on root.
            _add_path_rule(libc, ruleset_fd, real_root, dir_access=handled_fs & _FS_READ_DIR, file_access=0)

        rc_prctl = int(
            libc.prctl(
                ctypes.c_int(_PR_SET_NO_NEW_PRIVS),
                ctypes.c_ulong(1),
                ctypes.c_ulong(0),
                ctypes.c_ulong(0),
                ctypes.c_ulong(0),
            )
        )
        if rc_prctl != 0:
            err = ctypes.get_errno()
            return False, f"prctl(PR_SET_NO_NEW_PRIVS) failed: errno={err} ({os.strerror(err)})"

        rc_restrict = int(
            libc.syscall(
                ctypes.c_long(_SYS_LANDLOCK_RESTRICT_SELF),
                ctypes.c_int(ruleset_fd),
                ctypes.c_uint32(0),
            )
        )
        if rc_restrict != 0:
            err = ctypes.get_errno()
            return False, f"landlock_restrict_self failed: errno={err} ({os.strerror(err)})"
    finally:
        os.close(ruleset_fd)

    return True, f"Landlock ABI v{abi} applied"


def _resolve_sandbox_mode() -> str:
    env_mode = os.environ.get("HERMES_PROFILE_SANDBOX", "").strip().lower()
    if env_mode:
        if env_mode in _SANDBOX_OFF_VALUES:
            return "off"
        if env_mode in _SANDBOX_STRICT_VALUES:
            return "strict"
        return "auto"
    try:
        from hermes_cli.config import cfg_get, load_config_readonly
        cfg_val = str(cfg_get(load_config_readonly(), "security", "profile_sandbox", default="auto") or "auto").strip().lower()
        if cfg_val in _SANDBOX_OFF_VALUES:
            return "off"
        if cfg_val in _SANDBOX_STRICT_VALUES:
            return "strict"
    except Exception:
        pass
    return "auto"


def _active_profile_and_root() -> tuple[str, str, bool]:
    """Return ``(current_home, hermes_root, is_named_profile)`` for the active profile scope."""
    from hermes_constants import get_hermes_home

    current = os.path.realpath(str(get_hermes_home().expanduser()))
    parent = os.path.dirname(current)
    grandparent = os.path.dirname(parent)
    if os.path.basename(parent) == "profiles" and grandparent and grandparent != parent:
        return current, os.path.realpath(grandparent), True
    return current, current, False


def wrap_argv_for_profile_sandbox(
    argv: list[str],
    *,
    current_home: str | None = None,
    hermes_root: str | None = None,
    mode: str | None = None,
) -> list[str]:
    """Wrap *argv* in the ``-S -I`` Landlock exec trampoline when multi-profile isolation applies.

    No-op on non-Linux hosts, when ``mode == "off"``, or on single-profile hosts with no
    ``<hermes_root>/profiles`` directory.
    """
    if not argv or not sys.platform.startswith("linux"):
        return argv

    resolved_mode = (mode or _resolve_sandbox_mode()).strip().lower()
    if resolved_mode in _SANDBOX_OFF_VALUES:
        return argv

    if current_home is None or hermes_root is None:
        cur, root, is_named = _active_profile_and_root()
    else:
        cur = os.path.realpath(current_home)
        root = os.path.realpath(hermes_root)
        is_named = cur != root and _is_under_posix(cur, os.path.join(root, "profiles"))

    if not is_named and not os.path.isdir(os.path.join(root, "profiles")):
        return argv

    # Avoid double-wrapping if already wrapped.
    wrapper_path = os.path.realpath(__file__)
    if len(argv) >= 4 and os.path.realpath(argv[3]) == wrapper_path:
        return argv

    return [
        sys.executable,
        "-S",
        "-I",
        wrapper_path,
        "--current-home",
        cur,
        "--hermes-root",
        root,
        "--mode",
        "strict" if resolved_mode in _SANDBOX_STRICT_VALUES else "auto",
        "--",
        *argv,
    ]


def _parse_wrapper_argv(argv: list[str]) -> tuple[str, str, str, list[str]]:
    current_home = ""
    hermes_root = ""
    mode = "auto"
    i = 0
    n = len(argv)
    while i < n:
        arg = argv[i]
        if arg == "--":
            return current_home, hermes_root, mode, argv[i + 1 :]
        if arg == "--current-home" and i + 1 < n:
            current_home = argv[i + 1]
            i += 2
        elif arg == "--hermes-root" and i + 1 < n:
            hermes_root = argv[i + 1]
            i += 2
        elif arg == "--mode" and i + 1 < n:
            mode = argv[i + 1]
            i += 2
        else:
            sys.stderr.write(f"profile_landlock: unexpected argument {arg!r}\n")
            raise SystemExit(2)
    sys.stderr.write("profile_landlock: missing '--' command separator\n")
    raise SystemExit(2)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    current_home, hermes_root, mode, cmd = _parse_wrapper_argv(args)
    if not current_home or not hermes_root or not cmd:
        sys.stderr.write("profile_landlock: --current-home, --hermes-root, and target command are required\n")
        return 2

    if mode not in _SANDBOX_OFF_VALUES:
        applied, detail = apply_landlock(current_home, hermes_root)
        if not applied:
            if mode in _SANDBOX_STRICT_VALUES:
                sys.stderr.write(f"profile_landlock: strict mode failed: {detail}\n")
                return 126
            # Auto mode: warn once on stderr only if Landlock failed unexpectedly on an enabled kernel.
            if "unavailable" not in detail:
                sys.stderr.write(f"profile_landlock: warning: {detail}\n")

    try:
        os.execvpe(cmd[0], cmd, os.environ)
    except OSError as exc:
        code = 127 if exc.errno == errno.ENOENT else 126
        sys.stderr.write(f"profile_landlock: exec {cmd[0]!r} failed: {exc}\n")
        return code


if __name__ == "__main__":
    raise SystemExit(main())
