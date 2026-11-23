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
import sys
from typing import Any

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
    for i in range(6):
        bit = (fmt >> i) & 1
        matrix[8][i] = bit
        matrix[size - 1 - i][8] = bit
        func[8][i] = func[size - 1 - i][8] = True
    for idx, (r1, c1, r2, c2) in enumerate(
        ((8, 7, size - 7, 8), (8, 8, 8, size - 8), (7, 8, 8, size - 7)),
        start=6,
    ):
        bit = (fmt >> idx) & 1
        matrix[r1][c1] = matrix[r2][c2] = bit
        func[r1][c1] = func[r2][c2] = True
    for i in range(9, 15):
        bit = (fmt >> i) & 1
        matrix[14 - i][8] = bit
        matrix[8][size - 15 + i] = bit
        func[14 - i][8] = func[8][size - 15 + i] = True
    matrix[size - 8][8] = 1
    func[size - 8][8] = True
    if version >= 7:
        vbits = _version_bits(version)
        for i in range(18):
            bit = (vbits >> i) & 1
            a, b = i // 3, size - 11 + (i % 3)
            matrix[a][b] = matrix[b][a] = bit
            func[a][b] = func[b][a] = True


def encode_qr_matrix(text: str) -> list[list[bool]]:
    """Encode UTF-8 text into a scannable ISO/IEC 18004 QR matrix (list of bool rows)."""
    payload = text.encode("utf-8")
    version = _choose_qr_version(len(payload))
    size = 17 + 4 * version
    matrix = [[0] * size for _ in range(size)]
    func = [[False] * size for _ in range(size)]

    _place_finder(matrix, func, 0, 0)
    _place_finder(matrix, func, 0, size - 7)
    _place_finder(matrix, func, size - 7, 0)

    for i in range(8, size - 8):
        bit = 1 if i % 2 == 0 else 0
        matrix[6][i] = matrix[i][6] = bit
        func[6][i] = func[i][6] = True

    centers = _QR_VERSIONS[version][3]
    for r in centers:
        for c in centers:
            if not func[r][c]:
                _place_alignment(matrix, func, r, c)

    mask = 0  # (r + c) % 2 == 0
    _write_format_and_version(matrix, func, version, mask)

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
                bit = bit_stream[bit_idx] if bit_idx < len(bit_stream) else 0
                bit_idx += 1
                if (r + c) % 2 == 0:
                    bit ^= 1
                matrix[r][c] = bit
        col -= 2
        upward = not upward

    return [[bool(cell) for cell in row] for row in matrix]


def render_qr_unicode(text: str, *, border: int = 2) -> str:
    """Render *text* as a compact Unicode half-block QR code suitable for terminals."""
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
        for t_dark, b_dark in zip(top, bottom):
            if t_dark and b_dark:
                chars.append("█")
            elif t_dark:
                chars.append("▀")
            elif b_dark:
                chars.append("▄")
            else:
                chars.append(" ")
        lines.append("".join(chars))
    return "\n".join(lines)


def _resolve_root(raw_home: str | None) -> Path:
    if raw_home:
        return Path(raw_home).expanduser().resolve()
    env_home = os.environ.get("HERMES_HOME", "").strip()
    if env_home:
        return Path(env_home).expanduser().resolve()
    return (Path.home() / ".hermes").resolve()


def _format_ticket_text(ticket: dict[str, Any], *, show_qr: bool) -> str:
    lines = [
        f"Hermes Mobile pairing ready for '{ticket['display_name']}' ({ticket['profile_slug']})",
        f"Pairing link: {ticket['deep_link']}",
        f"Expires in: {ticket['ttl_seconds']}s (single-use)",
        f"Android APK: {ANDROID_RELEASE_URL}",
    ]
    if show_qr:
        lines.append("")
        lines.append(render_qr_unicode(ticket["deep_link"]))
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
    parser.add_argument("--public-url", default=os.environ.get("HERMES_PUBLIC_URL", ""), help="Public HTTPS server URL.")
    parser.add_argument("--hermes-home", default=None, help="Override HERMES_HOME root directory.")
    parser.add_argument("--qr", action="store_true", help="Print a scannable Unicode QR code in the terminal.")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Emit machine-readable JSON output.")
    args = parser.parse_args(argv)

    if args.doctor:
        from scripts import mobile_doctor

        doctor_args: list[str] = []
        if args.hermes_home:
            doctor_args.extend(["--hermes-home", args.hermes_home])
        if args.public_url:
            doctor_args.extend(["--public-url", args.public_url])
        if args.as_json:
            doctor_args.append("--json")
        return mobile_doctor.main(doctor_args)

    from gateway.mobile_pairing import PairingError, PairingStore
    from gateway.mobile_profiles import (
        ProfileManagementError,
        create_managed_profile,
        get_profile_summary,
    )

    root = _resolve_root(args.hermes_home)
    try:
        if args.invite:
            created = create_managed_profile(
                caller_slug="default",
                slug=args.invite,
                display_name=args.display_name or args.invite,
                preferred_language=args.lang,
                is_admin=bool(args.admin),
                create_invite=True,
                host=args.public_url,
                ttl_seconds=args.ttl,
                root_home=root,
            )
            ticket = created["pairing_ticket"]
            payload = {"profile": created["profile"], "pairing_ticket": ticket, "android_apk": ANDROID_RELEASE_URL}
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
                host=args.public_url,
            )
            payload = {"profile": summary, "pairing_ticket": ticket, "android_apk": ANDROID_RELEASE_URL}
    except (ProfileManagementError, PairingError) as exc:
        sys.stderr.write(f"error ({exc.code}): {exc.message}\n")
        return 1

    if args.as_json:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write(_format_ticket_text(ticket, show_qr=bool(args.qr)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
