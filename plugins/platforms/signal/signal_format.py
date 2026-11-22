"""Shared Signal formatting helpers: Markdown → Signal native formatting lives here so both the live
adapter and the standalone send paths emit the same bodyRanges."""

from __future__ import annotations

import re

from agent.markdown_tables import is_table_divider, realign_markdown_tables

# Signal has no fixed-width client area; this budgets for a phone screen in the app's
# monospace face. Tables wider than this fall back to realign_markdown_tables()'s
# vertical Key: value rendering rather than soft-wrapping mid-cell.
_TABLE_WIDTH = 40

# Any fence length (CommonMark): code_fence() wraps a command that contains ``` in a longer run,
# and a ``` inside it must not close the block.
_CODE_BLOCK_RE = re.compile(
    r"(?P<fence>`{3,}|~{3,})[a-zA-Z0-9_+-]*\n?(?P<body>.*?)(?P=fence)(?![`~])", re.DOTALL
)
_FENCE_SPLIT_RE = re.compile(r"((`{3,}|~{3,}).*?\2(?![`~]))", re.DOTALL)


def _split_fenced(source: str) -> list[str]:
    """``[prose, fenced, prose, ...]``: the fence pattern has a nested backreference group."""
    parts, last = [], 0
    for match in _FENCE_SPLIT_RE.finditer(source):
        parts += [source[last:match.start()], match.group(1)]
        last = match.end()
    return parts + [source[last:]]
_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
# Signal has no hyperlink text range: bodyRanges styles only bold/italic/strike/mono, and the client
# auto-links bare URLs. A Markdown link therefore becomes ``label (url)`` so the label stays readable
# and the target stays tappable.
_HTTP_TARGET_RE = re.compile(r"https?://", re.IGNORECASE)
_INLINE_CODE_SPAN_RE = re.compile(r"(`[^`\n]*`)")
_BLOCK_MARKER_RE = re.compile(r"^(?:#{1,6}\s+|[-*+]\s+)")
_INLINE_PATTERNS = [
    # First match wins and spans never nest. Inline code goes first so its text stays verbatim; an
    # enclosing pair then loses ("**run `x`**" keeps its asterisks, where main kept the backticks).
    (re.compile(r"`(.+?)`"), "MONOSPACE"),
    (re.compile(r"\*\*(.+?)\*\*", re.DOTALL), "BOLD"),
    (re.compile(r"__(.+?)__", re.DOTALL), "BOLD"),
    (re.compile(r"~~(.+?)~~", re.DOTALL), "STRIKETHROUGH"),
    (re.compile(r"(?<!\*)\*(?!\*| )(.+?)(?<!\*)\*(?!\*)"), "ITALIC"),
    (re.compile(r"(?<!\w)_(?!_)(.+?)(?<!_)_(?!\w)"), "ITALIC")]


def _utf16_len(s: str) -> int:
    """Length of *s* in UTF-16 code units."""
    return len(s.encode("utf-16-le")) // 2


def _display_url(url: str) -> str:
    """Comparable form: scheme, ``www.`` and a trailing slash do not make two URLs different.

    The path stays case-sensitive; only the scheme/www prefix is compared case-insensitively.
    """
    value = url.strip()
    for prefix in ("https://", "http://", "www."):
        if value[: len(prefix)].lower() == prefix:
            value = value[len(prefix) :]
    return value.rstrip("/")


def _is_escaped(text: str, index: int) -> bool:
    """True when ``text[index]`` is preceded by an odd number of backslashes."""
    backslashes = 0
    while index - backslashes - 1 >= 0 and text[index - backslashes - 1] == "\\":
        backslashes += 1
    return backslashes % 2 == 1


def _unescape_markdown(text: str) -> str:
    return re.sub(r"\\([\\`*{}\[\]()#+\-.!_>~|])", r"\1", text)


def _parse_link_target(text: str, start: int) -> tuple[str, int] | None:
    """Parse ``destination [title]`` starting just after ``](``.

    Returns ``(target, index_of_closing_paren)`` or ``None`` for an invalid construct. A bare
    destination ends at whitespace or an unescaped ``)`` at depth 0; an angle destination ends at
    ``>`` and may contain spaces. An optional title may follow, quoted or parenthesized.
    """
    n, i = len(text), start
    while i < n and text[i].isspace():
        i += 1
    if i < n and text[i] == "<":
        i += 1
        target_start = i
        while i < n:
            if text[i] == "\\" and i + 1 < n:
                i += 2
                continue
            if text[i] == ">":
                break
            i += 1
        if i >= n:
            return None
        target = text[target_start:i]
        i += 1
    else:
        target_start = i
        depth = 0
        while i < n:
            ch = text[i]
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                if depth == 0:
                    break
                depth -= 1
            elif ch.isspace():
                break
            i += 1
        if depth != 0:
            return None
        target = text[target_start:i]
    while i < n and text[i].isspace():
        i += 1
    if i < n and text[i] in "\"'(":
        quote = text[i]
        closing = ")" if quote == "(" else quote
        i += 1
        while i < n:
            if text[i] == "\\" and i + 1 < n:
                i += 2
                continue
            if text[i] == closing:
                i += 1
                break
            i += 1
        else:
            return None
        while i < n and text[i].isspace():
            i += 1
    if i >= n or text[i] != ")":
        return None
    target = _unescape_markdown(target).replace(" ", "%20")
    if target.count("(") != target.count(")"):
        # An escaped literal paren would otherwise make the bare URL ambiguous in Signal.
        target = target.replace("(", "%28").replace(")", "%29")
    return target, i


def _link_replacement(label: str, target: str) -> str:
    label = label.strip()
    if not _HTTP_TARGET_RE.match(target):
        return label
    if _display_url(label) == _display_url(target):
        return target
    return f"{label} ({target})"


def _starts_block(replacement: str) -> bool:
    return _BLOCK_MARKER_RE.match(replacement) is not None


def _at_line_start(text: str, index: int) -> bool:
    line_start = text.rfind("\n", 0, index) + 1
    indent = text[line_start:index]
    return not indent.strip() and len(indent) <= 3


def _unwrap_links_inline(text: str) -> str:
    """Rewrite ``[label](url)``, honoring escapes, nested brackets and CommonMark destinations."""
    out: list[str] = []
    pos = 0
    while True:
        start = text.find("[", pos)
        while start != -1 and _is_escaped(text, start):
            start = text.find("[", start + 1)
        if start == -1:
            out.append(text[pos:])
            break
        depth, label_end, i = 0, -1, start + 1
        while i < len(text):
            if text[i] == "\\" and i + 1 < len(text):
                i += 2
                continue
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                if depth == 0:
                    label_end = i
                    break
                depth -= 1
            i += 1
        if (
            label_end == -1
            or label_end + 1 >= len(text)
            or text[label_end + 1] != "("
            or (start > 0 and text[start - 1] == "!")  # image syntax is not a link
        ):
            out.append(text[pos : start + 1])
            pos = start + 1
            continue
        parsed = _parse_link_target(text, label_end + 2)
        if parsed is None:
            out.append(text[pos : start + 1])
            pos = start + 1
            continue
        target, end = parsed
        label = text[start + 1 : label_end]
        if "](" in label or "![" in label:
            # A nested link or image is not a plain label; leave the outer construct raw.
            out.append(text[pos : start + 1])
            pos = start + 1
            continue
        replacement = _link_replacement(label, target)
        if _starts_block(replacement) and _at_line_start(text, start):
            # Keep the label from turning into a heading or bullet after the rewrite.
            replacement = "\u200b" + replacement
        out.append(text[pos:start])
        out.append(replacement)
        pos = end + 1
    return "".join(out)


def _unwrap_links(source: str) -> str:
    """Rewrite ``[label](https://url)`` outside fenced and inline code.

    Signal renders raw Markdown link syntax literally; only a bare http(s) URL becomes a tap target.
    Non-http targets (``mailto:``, relative paths) are not auto-linked, so they keep the label only.
    Inline code spans are parked behind placeholders so a link label may contain one.
    """
    parts = _split_fenced(source)
    for idx, part in enumerate(parts):
        if idx % 2:  # fenced block: link-looking text is code
            continue
        spans: list[str] = []

        def hold(match: re.Match[str]) -> str:
            spans.append(match.group(0))
            return f"\x00{len(spans) - 1}\x00"

        masked = _INLINE_CODE_SPAN_RE.sub(hold, part)
        masked = _unwrap_links_inline(masked)
        parts[idx] = re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], masked)
    return "".join(parts)


def _normalize_bullet_markers(source: str) -> str:
    """Replace Markdown bullet markers with plain Unicode bullets (Signal renders ``- item`` literally).
    Fenced code blocks are kept byte-for-byte: list-looking lines inside code are code, not bullets."""
    parts = _split_fenced(source)
    return "".join(re.sub(r"(?m)^([ \t]{0,3})[-*+]\s+", r"\1• ", part) if idx % 2 == 0 else part
                   for idx, part in enumerate(parts))


def _fence_tables(source: str) -> str:
    """Re-align every GFM table outside a code fence and wrap it in a fence, so the code-block pass
    in markdown_to_signal() records one MONOSPACE range over the aligned block and shifts every
    later range for it. Fenced regions are left alone: a pipe table inside a code block is code."""
    parts = _split_fenced(source)
    return "".join(part if idx % 2 else _fence_unfenced_tables(part) for idx, part in enumerate(parts))


def _fence_unfenced_tables(text: str) -> str:
    if "|" not in text:
        return text
    lines, out = text.split("\n"), []
    i, n = 0, len(lines)
    while i < n:
        # Same block rule as realign_markdown_tables(): header row, divider, contiguous pipe rows.
        if "|" in lines[i] and i + 1 < n and is_table_divider(lines[i + 1]):
            j = i + 2
            while j < n and "|" in lines[j] and lines[j].strip():
                j += 1
            out += ["```", realign_markdown_tables("\n".join(lines[i:j]), _TABLE_WIDTH), "```"]
            i = j
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def markdown_to_signal(text: str) -> tuple[str, list[str]]:
    """Convert markdown to plain text + Signal textStyles list. Signal uses ``bodyRanges`` (signal-cli
    ``textStyle`` / ``textStyles`` params) as ``start:length:STYLE`` with positions in UTF-16 code units.
    Supported styles: BOLD, ITALIC, STRIKETHROUGH, MONOSPACE."""
    text = _fence_tables(
        _normalize_bullet_markers(_unwrap_links(re.sub(r"\n{3,}", "\n\n", text).strip()))
    )
    styles: list[tuple[int, int, str]] = []
    code_blocks: list[tuple[int, int]] = []
    # One pass over the original text: code a longer fence kept may hold ``` runs, never re-parsed.
    unwrapped, last = "", 0
    for match in _CODE_BLOCK_RE.finditer(text):
        inner = match.group("body").rstrip("\n")
        unwrapped += text[last : match.start()]
        styles.append((len(unwrapped), len(inner), "MONOSPACE"))
        code_blocks.append((len(unwrapped), len(unwrapped) + len(inner)))
        unwrapped += inner
        last = match.end()
    text = unwrapped + text[last:]
    new_text, last_end = "", 0
    heading_removals: list[tuple[int, int]] = []
    heading_styles: list[tuple[int, int, str]] = []
    for match in _HEADING_RE.finditer(text):
        ms, me = match.start(), match.end()
        if any(cs <= ms < ce for cs, ce in code_blocks):
            continue
        new_text += text[last_end:ms]
        eol = text.find("\n", me)
        if eol == -1:
            eol = len(text)
        heading_text = text[me:eol]
        heading_styles.append((len(new_text), len(heading_text), "BOLD"))
        new_text += heading_text
        heading_removals.append((ms, me - ms))
        last_end = eol
    if heading_removals:
        text = new_text + text[last_end:]
        def _adjust_heading(pos: int) -> int:
            shift = 0
            for r_pos, r_len in heading_removals:
                if r_pos >= pos:
                    break
                shift += min(r_len, pos - r_pos)
            return pos - shift

        code_blocks = [(_adjust_heading(s), _adjust_heading(e)) for s, e in code_blocks]
        adjusted_monospace = [(_adjust_heading(s), _adjust_heading(s + l) - _adjust_heading(s), st)
                              for s, l, st in styles if st == "MONOSPACE"]
        styles = adjusted_monospace + heading_styles
    else:
        styles.extend(heading_styles)
    # Inline markers: seed occupied with code blocks so code text is never restyled. Each pattern
    # scans the gaps between code blocks separately: a marker inside code must not open a match
    # that swallows (and so drops) a real pair in the prose after it.
    all_matches: list[tuple[int, int, int, int, str]] = []
    occupied: list[tuple[int, int]] = list(code_blocks)
    bounds = [0, *(b for block in sorted(code_blocks) for b in block), len(text)]
    gaps = [(bounds[i], bounds[i + 1]) for i in range(0, len(bounds), 2) if bounds[i] < bounds[i + 1]]
    for pattern, style in _INLINE_PATTERNS:
        for match in (m for gs, ge in gaps for m in pattern.finditer(text, gs, ge)):
            ms, me = match.start(), match.end()
            if not any(ms < oe and me > os for os, oe in occupied):
                all_matches.append((ms, me, match.start(1), match.end(1), style))
                occupied.append((ms, me))
    all_matches.sort()
    # Strip the markers, recording (pos, len) removals so earlier block/heading ranges can be
    # shifted, and capturing inline ranges in the stripped text.
    result, last_end = "", 0
    removals: list[tuple[int, int]] = []
    inline_styles: list[tuple[int, int, str]] = []
    for ms, me, g1s, g1e, style in all_matches:
        if g1s > ms:
            removals.append((ms, g1s - ms))
        if me > g1e:
            removals.append((g1e, me - g1e))
        result += text[last_end:ms]
        inner = text[g1s:g1e]
        inline_styles.append((len(result), len(inner), style))
        result += inner
        last_end = me
    removals.sort()

    def _adjust(pos: int) -> int:
        shift = 0
        for remove_pos, remove_len in removals:
            if remove_pos >= pos:
                break
            shift += min(remove_len, pos - remove_pos)
        return pos - shift

    adjusted_prior = [(_adjust(start), _adjust(start + length) - _adjust(start), style)
                      for start, length, style in styles if _adjust(start + length) > _adjust(start)]
    text = result + text[last_end:]
    style_strings: list[str] = []
    for cp_start, cp_len, style_type in sorted(adjusted_prior + inline_styles):
        if 0 <= cp_start and cp_start + cp_len <= len(text):
            u16_start, u16_len = _utf16_len(text[:cp_start]), _utf16_len(text[cp_start : cp_start + cp_len])
            style_strings.append(f"{u16_start}:{u16_len}:{style_type}")
    return text, style_strings
