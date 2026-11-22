"""Tests for Signal markdown_to_signal() formatting.

Covers the markdown-to-bodyRanges conversion pipeline: bold, italic,
strikethrough, monospace, code blocks, headings, links, and — critically —
the false-positive regressions that caused spurious italics in production.
"""

import pytest

from plugins.platforms.signal.signal_format import markdown_to_signal

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _m2s(text: str):
    """Shorthand: return (plain_text, styles)."""
    return markdown_to_signal(text)

def _style_types(styles: list[str]) -> list[str]:
    """Extract just the STYLE part from '0:4:BOLD' strings."""
    return [s.rsplit(":", 1)[1] for s in styles]

def _find_style(styles: list[str], style_type: str) -> list[str]:
    """Return only styles matching a given type."""
    return [s for s in styles if s.endswith(f":{style_type}")]

# ===========================================================================
# Basic formatting
# ===========================================================================

class TestMarkdownToSignalBasic:
    """Core formatting: bold, italic, strikethrough, monospace."""

    def test_bold_double_asterisk(self):
        text, styles = _m2s("hello **world**")
        assert text == "hello world"
        assert len(styles) == 1
        assert styles[0].endswith(":BOLD")

    def test_italic_single_asterisk(self):
        text, styles = _m2s("hello *world*")
        assert text == "hello world"
        assert len(styles) == 1
        assert styles[0].endswith(":ITALIC")

    def test_strikethrough(self):
        text, styles = _m2s("hello ~~world~~")
        assert text == "hello world"
        assert len(styles) == 1
        assert styles[0].endswith(":STRIKETHROUGH")

    def test_inline_monospace(self):
        text, styles = _m2s("run `ls -la` now")
        assert text == "run ls -la now"
        assert len(styles) == 1
        assert styles[0].endswith(":MONOSPACE")

# ===========================================================================
# Italic false-positive regressions
# ===========================================================================

class TestItalicFalsePositives:
    """Regressions from signal-italic-false-positive-fix.md and
    signal-italic-bullet-list-fix.md."""

    # --- snake_case (original fix) ---

    def test_snake_case_not_italic(self):
        """snake_case identifiers must NOT be italicized."""
        text, styles = _m2s("the config_file is ready")
        assert text == "the config_file is ready"
        assert _find_style(styles, "ITALIC") == []

    # --- Bullet lists (second fix) ---

    def test_hyphen_bullet_list_uses_signal_safe_bullets(self):
        """Signal does not render Markdown list markers; normalize them."""
        md = "- item one\n- item two"
        text, styles = _m2s(md)
        assert text == "• item one\n• item two"
        assert styles == []

    def test_bullet_list_file_paths(self):
        """Real-world case that triggered the bug."""
        md = (
            "* tools/delegate_tool.py — delegation\n"
            "* tools/file_tools.py — file operations\n"
            "* tools/web_tools.py — web operations"
        )
        _text, styles = _m2s(md)
        assert _find_style(styles, "ITALIC") == []

    # --- Cross-line spans (DOTALL removal) ---

    def test_underscore_italic_no_cross_line(self):
        """_foo\\nbar_ must NOT match as italic (no DOTALL)."""
        _text, styles = _m2s("_foo\nbar_")
        assert _find_style(styles, "ITALIC") == []

    def test_star_italic_multiline_response(self):
        """Multi-paragraph response with * should not false-positive."""
        md = (
            "I checked the following files:\n\n"
            "* tools/delegate_tool.py — sub-agent delegation\n"
            "* tools/file_tools.py — file read/write/search\n"
            "* tools/web_tools.py — web search/extract\n\n"
            "Everything looks good."
        )
        _text, styles = _m2s(md)
        assert _find_style(styles, "ITALIC") == []

    # --- Legitimate italic still works ---

    def test_multiple_italic_same_line(self):
        text, styles = _m2s("*foo* and *bar* ok")
        assert text == "foo and bar ok"
        assert len(_find_style(styles, "ITALIC")) == 2

    def test_italic_single_word(self):
        text, styles = _m2s("*word*")
        assert text == "word"
        assert len(_find_style(styles, "ITALIC")) == 1

# ===========================================================================
# Style position accuracy
# ===========================================================================

class TestStylePositions:
    """Verify that start:length positions map to the correct text."""

    def _extract(self, text: str, style_str: str) -> str:
        """Given 'start:length:STYLE', extract the substring from text."""
        # Positions are UTF-16 code units; for ASCII they match code points
        parts = style_str.split(":")
        start, length = int(parts[0]), int(parts[1])
        # Encode to UTF-16-LE, slice, decode back
        encoded = text.encode("utf-16-le")
        extracted = encoded[start * 2 : (start + length) * 2]
        return extracted.decode("utf-16-le")

    def test_bold_position(self):
        text, styles = _m2s("hello **world** end")
        assert len(styles) == 1
        assert self._extract(text, styles[0]) == "world"

# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    """Tricky inputs that have caused issues or could regress."""

    def test_bold_inside_bullet(self):
        """Bold inside a bullet list item."""
        md = "* **important** item\n* normal item"
        _text, styles = _m2s(md)
        assert len(_find_style(styles, "BOLD")) == 1
        assert _find_style(styles, "ITALIC") == []

    def test_lone_asterisk(self):
        """A single * with no pair should not cause issues."""
        text, _styles = _m2s("5 * 3 = 15")
        # Should not crash; any italic match would be a false positive
        assert "5" in text and "15" in text

# ===========================================================================
# signal-markdown-strip-patch: core conversion pipeline
# ===========================================================================

class TestMarkdownStripPatch:
    """Tests for the original signal-markdown-strip-patch.

    Covers: fenced code blocks with language tags, links rewritten to a
    readable label plus a tappable bare URL, headings converted to bold,
    multiple headings, UTF-16 correctness for multi-byte characters, and
    marker stripping completeness.
    """

    def test_fenced_code_block_multiline(self):
        """Multi-line code blocks preserve all lines."""
        md = "```\nline1\nline2\nline3\n```"
        text, _styles = _m2s(md)
        assert "line1" in text
        assert "line2" in text
        assert "line3" in text
        assert "```" not in text

    def test_markdown_link_becomes_label_and_bare_url(self):
        """Signal has no link text range; the raw Markdown must not reach the client."""
        md = "Check [this link](https://example.com) for details"
        text, styles = _m2s(md)
        assert text == "Check this link (https://example.com) for details"
        assert styles == []

    def test_bare_url_unchanged(self):
        text, _styles = _m2s("Voir https://example.com/a-b pour les détails")
        assert text == "Voir https://example.com/a-b pour les détails"

    def test_domain_label_is_not_repeated(self):
        text, _styles = _m2s("[marchedelagare.com](https://www.marchedelagare.com/)")
        assert text == "https://www.marchedelagare.com/"

    def test_domain_label_path_case_is_not_collapsed(self):
        text, _styles = _m2s("[example.com/A](https://example.com/a)")
        assert text == "example.com/A (https://example.com/a)"

    def test_multiple_links(self):
        text, _styles = _m2s("[A](https://a.example/x) et [B](https://b.example/y).")
        assert text == "A (https://a.example/x) et B (https://b.example/y)."

    def test_escaped_paren_does_not_swallow_prose(self):
        text, _styles = _m2s(r"[x](https://e.com/a\(b) and (note) tail)")
        assert text == "x (https://e.com/a%28b) and (note) tail)"

    def test_title_with_paren_does_not_swallow_prose(self):
        text, _styles = _m2s('[x](https://e.com "see ) below") and tail')
        assert text == "x (https://e.com) and tail"

    def test_angle_destination_with_title(self):
        text, _styles = _m2s('[x](<https://example.com/a> "title")')
        assert text == "x (https://example.com/a)"

    def test_angle_destination_space_is_encoded(self):
        text, _styles = _m2s("[x](<https://example.com/a b>)")
        assert text == "x (https://example.com/a%20b)"

    def test_balanced_paren_url(self):
        text, _styles = _m2s("[x](https://example.com/a_(b))")
        assert text == "x (https://example.com/a_(b))"

    def test_link_label_with_inline_code(self):
        text, styles = _m2s("[use `--flag`](https://example.com)")
        assert text == "use --flag (https://example.com)"
        [style] = _find_style(styles, "MONOSPACE")
        start, length, _kind = style.split(":")
        assert text[int(start) : int(start) + int(length)] == "--flag"

    def test_tilde_fence_link_is_not_rewritten(self):
        text, styles = _m2s("~~~\n[doc](https://example.com)\n~~~")
        assert text == "[doc](https://example.com)"
        assert len(styles) == 1 and styles[0].endswith(":MONOSPACE")

    def test_list_nested_fence_link_is_not_rewritten(self):
        md = "- Steps:\n    ```\n    [doc](https://example.com)\n    ```"
        text, styles = _m2s(md)
        assert "[doc](https://example.com)" in text
        assert _find_style(styles, "MONOSPACE")

    def test_blockquote_fence_link_is_not_rewritten(self):
        md = "> ```\n> [doc](https://example.com)\n> ```"
        text, styles = _m2s(md)
        assert "[doc](https://example.com)" in text
        assert _find_style(styles, "MONOSPACE")

    def test_inline_tilde_run_is_not_a_fence(self):
        text, styles = _m2s("Use `~~~` to open a fence and ~~~ to close it.")
        assert text == "Use ~~~ to open a fence and ~~~ to close it."
        assert len(styles) == 1 and styles[0].endswith(":MONOSPACE")

    def test_inline_tilde_run_does_not_swallow_a_link(self):
        text, _styles = _m2s("Use `~~~` to open a fence, [docs](https://e.com), and ~~~ to close it.")
        assert "docs (https://e.com)" in text

    def test_literal_placeholder_text_does_not_crash(self):
        assert _m2s("a \x000\x00 b")[0] == "a \x000\x00 b"
        assert _m2s("a \x00code9\x00 b")[0] == "a \x00code9\x00 b"

    def test_escaped_bracket_is_not_a_link(self):
        text, _styles = _m2s(r"\[label](https://example.com)")
        assert text == r"\[label](https://example.com)"

    @pytest.mark.parametrize("label", ["# 1 issue", "- item"])
    def test_block_marker_label_stays_prose(self, label):
        text, styles = _m2s(f"[{label}](https://example.com)")
        assert text == f"\u200b{label} (https://example.com)"
        assert _find_style(styles, "BOLD") == []

    def test_image_link_is_not_rewritten(self):
        text, _styles = _m2s("[![alt](https://example.com/i.png)](https://example.com)")
        assert text == "[![alt](https://example.com/i.png)](https://example.com)"

    def test_nested_bracket_label(self):
        text, _styles = _m2s("[a [b] c](https://example.com)")
        assert text == "a [b] c (https://example.com)"

    def test_non_http_link_keeps_label_only(self):
        text, _styles = _m2s("Écris à [Marie](mailto:marie@example.com).")
        assert text == "Écris à Marie."

    def test_link_inside_inline_code_is_untouched(self):
        text, styles = _m2s("Code: `[pas un lien](https://example.com)` ici.")
        assert text == "Code: [pas un lien](https://example.com) ici."
        assert len(styles) == 1 and styles[0].endswith(":MONOSPACE")

    def test_link_inside_fenced_code_is_untouched(self):
        md = "```\n[doc](https://example.com)\n```"
        text, styles = _m2s(md)
        assert text == "[doc](https://example.com)"
        assert len(styles) == 1 and styles[0].endswith(":MONOSPACE")

    def test_image_syntax_is_not_rewritten(self):
        text, _styles = _m2s("![alt](https://example.com/i.png)")
        assert text == "![alt](https://example.com/i.png)"

    def test_link_label_keeps_inline_style(self):
        text, styles = _m2s("[**important**](https://example.com)")
        assert text == "important (https://example.com)"
        assert len(styles) == 1
        start, length, kind = styles[0].split(":")
        assert kind == "BOLD" and text[int(start):int(start) + int(length)] == "important"

    def test_heading_h1(self):
        """# H1 becomes bold text."""
        text, styles = _m2s("# Main Title")
        assert text == "Main Title"
        assert len(styles) == 1
        assert styles[0].endswith(":BOLD")

    def test_multiple_headings(self):
        """Multiple headings each become separate bold spans."""
        md = "## First\n\nSome text\n\n## Second"
        text, styles = _m2s(md)
        assert "First" in text
        assert "Second" in text
        assert "##" not in text
        bold_styles = _find_style(styles, "BOLD")
        assert len(bold_styles) == 2

        # ## at end might remain if not at line start — that's ok
        # The important thing is styled markers are stripped

# ===========================================================================
# signal-streaming-patch: SUPPORTS_MESSAGE_EDITING and send() behavior
# ===========================================================================

class TestTableRealignment:
    """GFM pipe tables are re-aligned to a fixed monospace width and wrapped in a
    single MONOSPACE style range, per the Signal table-rendering feature request."""

    @staticmethod
    def _u16_slice(text: str, style: str) -> str:
        start, length = (int(p) for p in style.split(":")[:2])
        u16 = text.encode("utf-16-le")
        return u16[start * 2 : (start + length) * 2].decode("utf-16-le")

    def test_wide_char_table_becomes_one_aligned_monospace_block(self):
        """Cells with CJK/emoji (2 display cells, 1-2 UTF-16 units) still produce pipes that line up
        by display width, one MONOSPACE range covering exactly the block, and correct UTF-16 offsets
        for styles after it."""
        from wcwidth import wcswidth

        text, styles = _m2s("| 名前 | Score |\n|---|---|\n| 東京 | 🚀 |\n| Bob | 1 |\n\n**tail**")
        (mono,) = _find_style(styles, "MONOSPACE")
        block = self._u16_slice(text, mono)
        assert text == block + "\n\ntail"
        rows = block.split("\n")
        assert len(rows) == 4 and all(wcswidth(row) == wcswidth(rows[0]) for row in rows)
        assert self._u16_slice(text, _find_style(styles, "BOLD")[0]) == "tail"

    def test_fenced_pipe_table_is_code_and_later_code_range_survives_realignment(self):
        """A pipe table inside ``` is code and must not be realigned or double-styled; a table's
        padding changes its length, and a code block after it still maps to its own text."""
        text, styles = _m2s("| Name | Age |\n|---|---|\n| Alice | 30 |\n\n```\n| a | b |\n|---|---|\n| 1 | 22 |\n```")
        mono = _find_style(styles, "MONOSPACE")
        assert len(mono) == 2
        assert self._u16_slice(text, mono[0]) == "| Name  | Age |\n|-------|-----|\n| Alice | 30  |"
        assert self._u16_slice(text, mono[1]) == "| a | b |\n|---|---|\n| 1 | 22 |"
        assert "cost | value" == _m2s("cost | value\nnot a table")[0].split("\n")[0]


# ===========================================================================
# Code protection against restyling & command alteration (#146848)
# ===========================================================================

class TestCodeProtection:
    """Commands and text inside fenced code blocks and inline code spans must
    never have formatting markers (*, _, ~, #) stripped or restyled."""

    def test_fenced_code_preserves_wildcards_and_asterisks(self):
        """rm -rf /srv/*/cache/* in fenced code must not be italicized or stripped."""
        md = "```\nrm -rf /srv/*/cache/*\n```"
        text, styles = _m2s(md)
        assert text == "rm -rf /srv/*/cache/*"
        assert len(styles) == 1
        assert styles[0].endswith(":MONOSPACE")
        assert _find_style(styles, "ITALIC") == []

    def test_fenced_code_preserves_underscores(self):
        """find . -name *_old_* -delete must not have _ stripped as italic."""
        md = "```\nfind . -name *_old_* -delete\n```"
        text, styles = _m2s(md)
        assert text == "find . -name *_old_* -delete"
        assert len(styles) == 1
        assert styles[0].endswith(":MONOSPACE")
        assert _find_style(styles, "ITALIC") == []

    def test_fenced_code_preserves_tildes(self):
        """chmod -R 777 ~~/backup~~ must not become strikethrough."""
        md = "```\nchmod -R 777 ~~/backup~~\n```"
        text, styles = _m2s(md)
        assert text == "chmod -R 777 ~~/backup~~"
        assert len(styles) == 1
        assert styles[0].endswith(":MONOSPACE")
        assert _find_style(styles, "STRIKETHROUGH") == []

    def test_fenced_code_preserves_comments(self):
        """Comments starting with # in code blocks must not become bold headings."""
        md = "```python\n# This is a comment\nx = 1\n```"
        text, styles = _m2s(md)
        assert text == "# This is a comment\nx = 1"
        assert len(styles) == 1
        assert styles[0].endswith(":MONOSPACE")
        assert _find_style(styles, "BOLD") == []

    def test_heading_preceding_code_block_preserves_both_ranges(self):
        """A heading before a code block must retain both heading BOLD and code block MONOSPACE."""
        md = "# Heading\n```\nmy_code_block\n```"
        text, styles = _m2s(md)
        assert text == "Heading\nmy_code_block"
        bold = _find_style(styles, "BOLD")
        mono = _find_style(styles, "MONOSPACE")
        assert len(bold) == 1
        assert len(mono) == 1
        assert bold[0] == "0:7:BOLD"
        assert mono[0] == "8:13:MONOSPACE"

    def test_inline_code_preserves_wildcards_and_tildes(self):
        """Inline code spans must not have internal * or ~ restyled."""
        md = "Execute `rm -rf /srv/*/cache/*` and `chmod -R 777 ~~/backup~~`"
        text, styles = _m2s(md)
        assert text == "Execute rm -rf /srv/*/cache/* and chmod -R 777 ~~/backup~~"
        mono = _find_style(styles, "MONOSPACE")
        assert len(mono) == 2
        assert _find_style(styles, "ITALIC") == []
        assert _find_style(styles, "STRIKETHROUGH") == []


_APPROVAL_COMMANDS = [
    "rm -rf /srv/*/cache/*",
    "find . -name *_old_* -delete",
    "chmod -R 777 ~~/backup~~",
    "cp __init__.py **/pkg/",
    "# comment *line*\necho _x_ `y`",
]


@pytest.mark.parametrize("command", _APPROVAL_COMMANDS)
def test_fenced_command_round_trips_amid_styled_prose(command):
    """An approval prompt's fenced command reads byte-identical, and the prose
    around it (heading, bold, italic) is still styled outside the fence."""
    md = f"## Approve?\n```bash\n{command}\n```\nThis is **dangerous** and _final_."
    text, styles = _m2s(md)
    mono = _find_style(styles, "MONOSPACE")
    assert len(mono) == 1
    start, length, _ = mono[0].split(":")
    assert text[int(start) : int(start) + int(length)] == command
    assert text == f"Approve?\n{command}\nThis is dangerous and final."
    assert len(_find_style(styles, "BOLD")) == 2
    assert len(_find_style(styles, "ITALIC")) == 1


@pytest.mark.parametrize("command", [
    "echo " + "`" * 3 + " ; rm -rf /tmp/*/x*",
    "printf '" + "`" * 3 + "' && rm -rf ~~/home~~",
    "a " + "`" * 4 + " b **c** _d_",
    "echo " + "`" * 3 + " ; echo " + "`" * 3,
    "printf '" + "`" * 3 + "x" + "`" * 3 + "' && rm -rf /srv/*/c*",
])
def test_command_with_a_backtick_run_round_trips_inside_its_longer_fence(command):
    """code_fence() wraps a command holding ``` in a longer fence; the ``` inside must not close it."""
    from gateway.platforms.base_exec_approval import code_fence

    fence = code_fence(command)
    md = f"Approve?\n{fence}\n{command}\n{fence}\nThis is **dangerous**."
    text, styles = _m2s(md)
    mono = _find_style(styles, "MONOSPACE")
    assert len(mono) == 1
    start, length, _ = mono[0].split(":")
    assert text[int(start) : int(start) + int(length)] == command
    assert text == f"Approve?\n{command}\nThis is dangerous."


def test_backtick_run_kept_in_a_longer_fence_does_not_pair_with_a_later_fence():
    """A ``` kept inside one block must not open a new block with the next fence in the message."""
    from gateway.platforms.base_exec_approval import code_fence

    command = "echo " + "`" * 3 + " x"
    fence = code_fence(command)
    text, styles = _m2s(f"{fence}\n{command}\n{fence}\nThen **this**:\n```\nlater\n```")
    assert text == f"{command}\nThen this:\nlater"
    spans = [m.split(":") for m in _find_style(styles, "MONOSPACE")]
    assert [text[int(s) : int(s) + int(n)] for s, n, _ in spans] == [command, "later"]
