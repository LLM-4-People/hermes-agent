#!/usr/bin/env python3
"""One-command admin setup and pairing CLI for Hermes Mobile.

Usage:
    python3 scripts/mobile_setup.py --pair --public-url https://hermes.example.com --qr
    python3 scripts/mobile_setup.py --invite alex --lang fr --public-url https://hermes.example.com --qr
    python3 scripts/mobile_setup.py --doctor --public-url https://hermes.example.com

Pure Python standard library (zero external dependencies, ``python3 -I`` safe).
Includes an ISO/IEC 18004 Byte-mode EC-L QR encoder and compact Unicode half-block renderer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from typing import Any
import zlib

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ANDROID_RELEASE_URL = "https://github.com/LLM-4-People/hermes-mobile/releases/latest"

# ── Pure-Python ISO/IEC 18004 QR Code Generator (Version 1-10, Byte Mode, EC-L) ──

# (data_codewords, ec_codewords_per_block, num_blocks, alignment_centers)
_QR_VERSIONS: dict[int, tuple[int, int, int, tuple[int, ...]]] = {
    1: (19, 7, 1, ()),
    2: (34, 10, 1, (6, 18)),
    3: (55, 15, 1, (6, 22)),
    4: (80, 20, 1, (6, 26)),
    5: (108, 26, 1, (6, 30)),
    6: (136, 18, 2, (6, 34)),
    7: (156, 20, 2, (6, 22, 38)),
    8: (194, 24, 2, (6, 24, 42)),
    9: (232, 30, 2, (6, 26, 46)),
    10: (274, 18, 4, (6, 28, 50)),
}

_GF_EXP = [0] * 512
_GF_LOG = [0] * 256
_x = 1
for _i in range(255):
    _GF_EXP[_i] = _x
    _GF_LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11D
for _i in range(255, 512):
    _GF_EXP[_i] = _GF_EXP[_i - 255]


def _gf_mul(a: int, b: int) -> int:
    return 0 if (a == 0 or b == 0) else _GF_EXP[_GF_LOG[a] + _GF_LOG[b]]


def _rs_generator(degree: int) -> list[int]:
    poly = [1]
    for i in range(degree):
        nxt = [0] * (len(poly) + 1)
        factor = _GF_EXP[i]
        for j, coef in enumerate(poly):
            nxt[j] ^= coef
            nxt[j + 1] ^= _gf_mul(coef, factor)
        poly = nxt
    return poly


def _rs_remainder(data: list[int], ec_len: int) -> list[int]:
    gen = _rs_generator(ec_len)
    rem = [0] * ec_len
    for byte in data:
        factor = byte ^ rem[0]
        rem = rem[1:] + [0]
        for i in range(ec_len):
            rem[i] ^= _gf_mul(gen[i + 1], factor)
    return rem


def _choose_qr_version(byte_len: int) -> int:
    for version, (cap, _, _, _) in _QR_VERSIONS.items():
        header_bits = 4 + (8 if version <= 9 else 16)
        if (header_bits + byte_len * 8 + 7) // 8 <= cap:
            return version
    raise ValueError(f"QR payload too long ({byte_len} bytes; max supported is 271 bytes).")


def _encode_qr_codewords(payload: bytes, version: int) -> list[int]:
    cap, ec_per_block, blocks, _ = _QR_VERSIONS[version]
    bits: list[int] = []

    def push(val: int, width: int) -> None:
        for shift in range(width - 1, -1, -1):
            bits.append((val >> shift) & 1)

    push(0b0100, 4)  # Byte mode
    push(len(payload), 8 if version <= 9 else 16)
    for b in payload:
        push(b, 8)
    rem = cap * 8 - len(bits)
    push(0, min(4, max(0, rem)))
    while len(bits) % 8 != 0:
        bits.append(0)
    data = [
        sum(bits[i + k] << (7 - k) for k in range(8))
        for i in range(0, len(bits), 8)
    ]
    pad_bytes = (0xEC, 0x11)
    pad_idx = 0
    while len(data) < cap:
        data.append(pad_bytes[pad_idx % 2])
        pad_idx += 1

    # Split into equal-or-plus-1 RS blocks per ISO/IEC 18004 Table 9.
    short_len = cap // blocks
    num_long = cap % blocks
    num_short = blocks - num_long
    data_blocks: list[list[int]] = []
    ec_blocks: list[list[int]] = []
    pos = 0
    for idx in range(blocks):
        blen = short_len + (1 if idx >= num_short else 0)
        blk = data[pos : pos + blen]
        pos += blen
        data_blocks.append(blk)
        ec_blocks.append(_rs_remainder(blk, ec_per_block))

    interleaved: list[int] = []
    for col in range(short_len + (1 if num_long else 0)):
        for blk in data_blocks:
            if col < len(blk):
                interleaved.append(blk[col])
    for col in range(ec_per_block):
        for ec in ec_blocks:
            interleaved.append(ec[col])
    return interleaved


def _place_finder(matrix: list[list[int]], func: list[list[bool]], r0: int, c0: int) -> None:
    size = len(matrix)
    for dr in range(-1, 8):
        for dc in range(-1, 8):
            r, c = r0 + dr, c0 + dc
            if not (0 <= r < size and 0 <= c < size):
                continue
            dark = (
                0 <= dr <= 6
                and 0 <= dc <= 6
                and (dr in (0, 6) or dc in (0, 6) or (2 <= dr <= 4 and 2 <= dc <= 4))
            )
            matrix[r][c] = 1 if dark else 0
            func[r][c] = True


def _place_alignment(matrix: list[list[int]], func: list[list[bool]], cr: int, cc: int) -> None:
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            r, c = cr + dr, cc + dc
            matrix[r][c] = 1 if (max(abs(dr), abs(dc)) != 1) else 0
            func[r][c] = True


def _format_bits(ec_bits: int, mask: int) -> int:
    data = (ec_bits << 3) | mask
    rem = data
    for _ in range(10):
        rem = (rem << 1) ^ (0x537 if (rem & 0x200) else 0)
    return ((data << 10) | rem) ^ 0x5412


def _version_bits(version: int) -> int:
    rem = version
    for _ in range(12):
        rem = (rem << 1) ^ (0x1F25 if (rem & 0x800) else 0)
    return (version << 12) | rem


def _write_format_and_version(matrix: list[list[int]], func: list[list[bool]], version: int, mask: int) -> None:
    size = len(matrix)
    fmt = _format_bits(0b01, mask)  # EC-L = 01
    # Copy 1 around top-left finder (ISO/IEC 18004 Figure 25): LSB=bit 0 at (0, 8), MSB=bit 14 at (8, 0).
    for i in range(6):
        matrix[i][8] = (fmt >> i) & 1
        func[i][8] = True
    matrix[7][8] = (fmt >> 6) & 1
    matrix[8][8] = (fmt >> 7) & 1
    matrix[8][7] = (fmt >> 8) & 1
    func[7][8] = func[8][8] = func[8][7] = True
    for i in range(9, 15):
        matrix[8][14 - i] = (fmt >> i) & 1
        func[8][14 - i] = True
    # Copy 2 across top-right (bits 0..7) and bottom-left (bits 8..14).
    for i in range(8):
        matrix[8][size - 1 - i] = (fmt >> i) & 1
        func[8][size - 1 - i] = True
    for i in range(8, 15):
        matrix[size - 15 + i][8] = (fmt >> i) & 1
        func[size - 15 + i][8] = True
    # Fixed dark module.
    matrix[size - 8][8] = 1
    func[size - 8][8] = True
    # Version information blocks for Version 7-10 (ISO/IEC 18004 Figure 26).
    if version >= 7:
        vbits = _version_bits(version)
        for i in range(18):
            bit = (vbits >> i) & 1
            a, b = size - 11 + (i % 3), i // 3
            matrix[b][a] = matrix[a][b] = bit
            func[b][a] = func[a][b] = True


def _mask_bit(mask: int, r: int, c: int) -> bool:
    if mask == 0:
        return (r + c) % 2 == 0
    if mask == 1:
        return r % 2 == 0
    if mask == 2:
        return c % 3 == 0
    if mask == 3:
        return (r + c) % 3 == 0
    if mask == 4:
        return (r // 2 + c // 3) % 2 == 0
    if mask == 5:
        return (r * c) % 2 + (r * c) % 3 == 0
    if mask == 6:
        return ((r * c) % 2 + (r * c) % 3) % 2 == 0
    return ((r + c) % 2 + (r * c) % 3) % 2 == 0


def _qr_penalty_score(matrix: list[list[int]]) -> int:
    """Compute ISO/IEC 18004 section 7.8.3.1 mask penalty score (N1 + N2 + N3 + N4)."""
    size = len(matrix)
    score = 0
    finder_a = (1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0)
    finder_b = (0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1)

    # N1 (adjacent runs >= 5) and N3 (finder-like 1:1:3:1:1 patterns) across rows & columns.
    for lines in (matrix, [[matrix[r][c] for r in range(size)] for c in range(size)]):
        for line in lines:
            run_val = line[0]
            run_len = 1
            for cell in line[1:]:
                if cell == run_val:
                    run_len += 1
                else:
                    if run_len >= 5:
                        score += 3 + (run_len - 5)
                    run_val = cell
                    run_len = 1
            if run_len >= 5:
                score += 3 + (run_len - 5)
            for idx in range(size - 10):
                window = tuple(line[idx : idx + 11])
                if window == finder_a or window == finder_b:
                    score += 40

    # N2 (2x2 same-color blocks).
    for r in range(size - 1):
        row0, row1 = matrix[r], matrix[r + 1]
        for c in range(size - 1):
            v = row0[c]
            if v == row0[c + 1] == row1[c] == row1[c + 1]:
                score += 3

    # N4 (dark/light balance deviation from 50% in 5% steps).
    dark = sum(sum(row) for row in matrix)
    total = size * size
    k = (abs(dark * 2 - total) * 10) // total
    score += k * 10
    return score


def encode_qr_matrix(text: str) -> list[list[bool]]:
    """Encode UTF-8 text into a scannable ISO/IEC 18004 QR matrix (list of bool rows)."""
    payload = text.encode("utf-8")
    version = _choose_qr_version(len(payload))
    size = 17 + 4 * version
    base_matrix = [[0] * size for _ in range(size)]
    func = [[False] * size for _ in range(size)]

    _place_finder(base_matrix, func, 0, 0)
    _place_finder(base_matrix, func, 0, size - 7)
    _place_finder(base_matrix, func, size - 7, 0)

    for i in range(8, size - 8):
        bit = 1 if i % 2 == 0 else 0
        base_matrix[6][i] = base_matrix[i][6] = bit
        func[6][i] = func[i][6] = True

    centers = _QR_VERSIONS[version][3]
    last_center = len(centers) - 1
    for ri, r in enumerate(centers):
        for ci, c in enumerate(centers):
            # Skip only the three corners occupied by the finder patterns (keep timing-row centers on V7-10).
            if (ri == 0 and ci == 0) or (ri == 0 and ci == last_center) or (ri == last_center and ci == 0):
                continue
            _place_alignment(base_matrix, func, r, c)

    # Reserve format & version function modules before placing data bits.
    _write_format_and_version(base_matrix, func, version, 0)

    codewords = _encode_qr_codewords(payload, version)
    bit_stream = [(cw >> s) & 1 for cw in codewords for s in range(7, -1, -1)]
    bit_idx = 0
    col = size - 1
    upward = True
    while col > 0:
        if col == 6:
            col -= 1
        row_iter = range(size - 1, -1, -1) if upward else range(size)
        for r in row_iter:
            for dc in (0, -1):
                c = col + dc
                if func[r][c]:
                    continue
                base_matrix[r][c] = bit_stream[bit_idx] if bit_idx < len(bit_stream) else 0
                bit_idx += 1
        col -= 2
        upward = not upward

    # Evaluate all 8 ISO/IEC 18004 masks and pick the lowest-penalty pattern for reliable scanning.
    best_matrix: list[list[int]] | None = None
    best_penalty = 1 << 30
    for mask in range(8):
        candidate = [
            [
                base_matrix[r][c] ^ (1 if (not func[r][c] and _mask_bit(mask, r, c)) else 0)
                for c in range(size)
            ]
            for r in range(size)
        ]
        _write_format_and_version(candidate, func, version, mask)
        penalty = _qr_penalty_score(candidate)
        if penalty < best_penalty:
            best_penalty = penalty
            best_matrix = candidate

    assert best_matrix is not None
    return [[bool(cell) for cell in row] for row in best_matrix]


def render_qr_unicode(text: str, *, border: int = 4, invert: bool = False) -> str:
    """Render *text* as a compact Unicode half-block QR code suitable for terminals.

    Uses the ISO/IEC 18004 standard 4-module quiet zone by default. Set ``invert=True``
    for dark-mode terminals so foreground blocks draw the white background and empty cells
    draw dark modules.
    """
    core = encode_qr_matrix(text)
    size = len(core)
    padded_width = size + border * 2
    grid = [[False] * padded_width for _ in range(padded_width)]
    for r in range(size):
        for c in range(size):
            grid[r + border][c + border] = core[r][c]

    lines: list[str] = []
    for r in range(0, padded_width, 2):
        top = grid[r]
        bottom = grid[r + 1] if r + 1 < padded_width else [False] * padded_width
        chars: list[str] = []
        for t_cell, b_cell in zip(top, bottom):
            t_fg = (not t_cell) if invert else t_cell
            b_fg = (not b_cell) if invert else b_cell
            if t_fg and b_fg:
                chars.append("█")
            elif t_fg:
                chars.append("▀")
            elif b_fg:
                chars.append("▄")
            else:
                chars.append(" ")
        lines.append("".join(chars))
    return "\n".join(lines)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(data, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def encode_qr_png(matrix: list[list[bool]], *, scale: int = 10, border: int = 4) -> bytes:
    """Encode a boolean QR matrix into an 8-bit grayscale black-on-white PNG (stdlib only)."""
    if scale < 1:
        raise ValueError("QR PNG scale must be at least 1.")
    if border < 0:
        raise ValueError("QR PNG border must be non-negative.")
    size = len(matrix)
    dim = (size + border * 2) * scale
    white_row = b"\x00" + (b"\xff" * dim)
    raw_rows: list[bytes] = [white_row] * (border * scale)
    border_px = b"\xff" * (border * scale)
    dark_px = b"\x00" * scale
    light_px = b"\xff" * scale
    for row in matrix:
        scanline = b"\x00" + border_px + b"".join(dark_px if cell else light_px for cell in row) + border_px
        raw_rows.extend([scanline] * scale)
    raw_rows.extend([white_row] * (border * scale))

    ihdr = struct.pack(">IIBBBBB", dim, dim, 8, 0, 0, 0, 0)
    idat = zlib.compress(b"".join(raw_rows), level=9)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


def write_qr_png(text: str, path: str | Path, *, scale: int = 10, border: int = 4) -> Path:
    """Atomically write a black-on-white QR PNG for *text* with strict ``0600`` permissions."""
    dest = Path(path).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    png_bytes = encode_qr_png(encode_qr_matrix(text), scale=scale, border=border)
    fd, tmp_name = tempfile.mkstemp(prefix=".hermes_qr_", suffix=".png", dir=str(dest.parent))
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(png_bytes)
        os.replace(tmp_name, dest)
        os.chmod(dest, 0o600)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return dest


def _resolve_root(raw_home: str | None) -> Path:
    if raw_home:
        return Path(raw_home).expanduser().resolve()
    env_home = os.environ.get("HERMES_HOME", "").strip()
    if env_home:
        return Path(env_home).expanduser().resolve()
    return (Path.home() / ".hermes").resolve()


def _format_ticket_text(
    ticket: dict[str, Any],
    *,
    show_qr: bool,
    invert_qr: bool = False,
    png_path: Path | None = None,
) -> str:
    lines = [
        f"Hermes Mobile pairing ready for '{ticket['display_name']}' ({ticket['profile_slug']})",
        f"Pairing link: {ticket['deep_link']}",
        f"Expires in: {ticket['ttl_seconds']}s (single-use)",
        f"Android APK: {ANDROID_RELEASE_URL}",
    ]
    if png_path is not None:
        lines.append(f"QR PNG (0600): {png_path}")
    if show_qr:
        lines.append("")
        lines.append(render_qr_unicode(ticket["deep_link"], invert=invert_qr))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One-command setup and pairing wizard for Hermes Mobile.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pair", action="store_true", help="Mint a single-use pairing link for an existing profile.")
    mode.add_argument("--invite", metavar="SLUG", help="Create an isolated user profile and mint a single-use invite link.")
    mode.add_argument("--doctor", action="store_true", help="Run the Hermes Mobile companion doctor check.")
    parser.add_argument("--profile", default="default", help="Target profile slug for --pair (default: default).")
    parser.add_argument("--display-name", default="", help="Display name for --invite or --pair.")
    parser.add_argument("--lang", default="auto", choices=("auto", "en", "fr"), help="Preferred language for --invite.")
    parser.add_argument("--admin", action="store_true", help="Grant admin role to the profile created by --invite.")
    parser.add_argument("--ttl", type=int, default=300, help="Ticket expiry in seconds (30-900, default 300).")
    parser.add_argument(
        "--public-url",
        default="",
        help="Public HTTPS server URL (defaults to HERMES_PUBLIC_URL, API_SERVER_PUBLIC_URL, or config.yaml).",
    )
    parser.add_argument("--hermes-home", default=None, help="Override HERMES_HOME root directory.")
    parser.add_argument("--qr", action="store_true", help="Print a scannable Unicode QR code in the terminal.")
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert dark and light QR blocks for dark-mode terminals.",
    )
    parser.add_argument(
        "--png",
        metavar="PATH",
        default="",
        help="Write a black-on-white QR PNG file with 0600 permissions to PATH.",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="Emit machine-readable JSON output.")
    args = parser.parse_args(argv)

    from gateway.mobile_pairing import PairingError, PairingStore, resolve_public_host

    root = _resolve_root(args.hermes_home)
    try:
        public_url = resolve_public_host(args.public_url, root_home=root)
    except PairingError as exc:
        sys.stderr.write(f"error ({exc.code}): {exc.message}\n")
        return 1

    if args.doctor:
        from scripts import mobile_doctor

        doctor_args: list[str] = []
        if args.hermes_home:
            doctor_args.extend(["--hermes-home", args.hermes_home])
        if public_url:
            doctor_args.extend(["--public-url", public_url])
        if args.as_json:
            doctor_args.append("--json")
        return mobile_doctor.main(doctor_args)

    from gateway.mobile_profiles import (
        ProfileManagementError,
        create_managed_profile,
        get_profile_summary,
    )

    try:
        if args.invite:
            created = create_managed_profile(
                caller_slug="default",
                slug=args.invite,
                display_name=args.display_name or args.invite,
                preferred_language=args.lang,
                is_admin=bool(args.admin),
                create_invite=True,
                host=public_url,
                ttl_seconds=args.ttl,
                root_home=root,
            )
            ticket = created["pairing_ticket"]
            payload: dict[str, Any] = {
                "profile": created["profile"],
                "pairing_ticket": ticket,
                "android_apk": ANDROID_RELEASE_URL,
            }
        else:
            slug = args.profile.strip().lower() or "default"
            summary = get_profile_summary(slug, root_home=root)
            if summary is None:
                raise ProfileManagementError(f"Profile '{slug}' not found.", status_code=404, code="profile_not_found")
            ticket = PairingStore(root).create_ticket(
                profile_slug=slug,
                display_name=args.display_name or summary.get("display_name") or slug,
                mode="device",
                created_by="default",
                ttl_seconds=args.ttl,
                host=public_url,
            )
            payload = {"profile": summary, "pairing_ticket": ticket, "android_apk": ANDROID_RELEASE_URL}
    except (ProfileManagementError, PairingError) as exc:
        sys.stderr.write(f"error ({exc.code}): {exc.message}\n")
        return 1

    png_path: Path | None = None
    if args.png:
        try:
            png_path = write_qr_png(ticket["deep_link"], args.png)
        except OSError as exc:
            sys.stderr.write(f"error (png_write_failed): {exc}\n")
            return 1
        payload["qr_png"] = str(png_path)

    if args.as_json:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write(
            _format_ticket_text(
                ticket,
                show_qr=bool(args.qr),
                invert_qr=bool(args.invert),
                png_path=png_path,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
