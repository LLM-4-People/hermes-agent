"""Declarative, validated, versioned, and reversible mobile UI-modding protocol.

Single source of truth for allowed declarative surface v2 schemas, WCAG 2.1 AA theme
contrast validation, profile-scoped SQLite revision storage, live SSE emission hooks,
and the ``mobile_ui_patch`` agent tool.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
import unicodedata
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_constants import get_hermes_home
from tools.mobile_sqlite import lock_for as _lock_for, open_mobile_db
from tools.registry import registry

logger = logging.getLogger("hermes.mobile_ui")

# ── Single Source of Truth Schema Constants ──────────────────────────────────

ALLOWED_SURFACE_KINDS = frozenset({
    "kpi_grid",
    "sparkline",
    "bar_chart",
    "table",
    "form",
    "toggle_group",
    "chip_filter",
    "slider",
    "action_bar",
    "markdown_card",
    "progress_tracker",
    "map_compass",
})

ALLOWED_OP_TYPES = frozenset({
    "upsert_surface",
    "remove_surface",
    "upsert_quick_action",
    "remove_quick_action",
    "upsert_widget",
    "remove_widget",
    "set_theme_override",
    "clear_theme_override",
})

ALLOWED_WIDGET_TEMPLATES = frozenset({"metric", "checklist", "quick_actions", "status", "briefing"})
ALLOWED_SCOPES = frozenset({"thread", "atelier"})
ALLOWED_DENSITIES = frozenset({"compact", "comfortable", "spacious"})
ALLOWED_SURFACE_STYLES = frozenset({"paper", "elevated", "outlined", "glass"})
ALLOWED_QUICK_ACTION_KINDS = frozenset({"prompt", "open_destination", "toggle_surface"})
ALLOWED_SUGGESTED_ACTION_KINDS = frozenset({
    "pin_widget",
    "save_memory",
    "create_reminder",
    "open_surface",
    "save_skill",
    "prompt",
})
ALLOWED_INTERACTIVE_CARD_KINDS = frozenset({
    "checklist",
    "quick_pick",
    "toggle_confirm",
    "customization_preview",
})
ALLOWED_DESTINATIONS = frozenset({"discussions", "atelier", "cerveau", "activite"})

MAX_OPS_PER_PATCH = 16
MAX_PATCH_JSON_BYTES = 65_536
MAX_ID_LEN = 64
MAX_TITLE_LEN = 160
MAX_SUMMARY_LEN = 240
MAX_PROMPT_LEN = 1000
MAX_SURFACES_PER_SCOPE = 24
MAX_QUICK_ACTIONS = 12
MAX_WIDGETS = 16
MAX_WIDGET_ITEMS = 8
MAX_SUGGESTED_ACTIONS = 3
MAX_INTERACTIVE_CARDS = 4
MAX_CARD_ITEMS = 8
MAX_CARD_TEXT_LEN = 160
MIN_WCAG_AA_CONTRAST = 4.5
MIN_FONT_SCALE_STEP = -2
MAX_FONT_SCALE_STEP = 3

DEFAULT_SURFACE_HEX = "#FAF8F5"
DEFAULT_INK_HEX = "#1A1917"
DEFAULT_ACCENT_INK_HEX = "#FFFFFF"

# Guaranteed WCAG 2.1 AA (>= 4.5:1) named accent presets for small-LLM ergonomics.
ACCENT_PALETTE_ALIASES: dict[str, tuple[str, str]] = {
    "teal": ("#0D6E6E", "#FFFFFF"),
    "green": ("#1B6B3A", "#FFFFFF"),
    "blue": ("#1D4ED8", "#FFFFFF"),
    "amber": ("#92400E", "#FFFFFF"),
    "coral": ("#B83227", "#FFFFFF"),
    "indigo": ("#4338CA", "#FFFFFF"),
    "slate": ("#334155", "#FFFFFF"),
}

SURFACE_KIND_ALIASES: dict[str, str] = {
    "kpi": "kpi_grid",
    "metrics": "kpi_grid",
    "stats": "kpi_grid",
    "chart": "sparkline",
    "line_chart": "sparkline",
    "bars": "bar_chart",
    "list": "table",
    "grid": "table",
    "card": "markdown_card",
    "markdown": "markdown_card",
    "text": "markdown_card",
    "progress": "progress_tracker",
    "toggles": "toggle_group",
    "switches": "toggle_group",
    "chips": "chip_filter",
    "filters": "chip_filter",
    "actions": "action_bar",
    "buttons": "action_bar",
    "map": "map_compass",
    "compass": "map_compass",
}

WIDGET_TEMPLATE_ALIASES: dict[str, str] = {
    "list": "checklist",
    "tasks": "checklist",
    "todo": "checklist",
    "todos": "checklist",
    "kpi": "metric",
    "counter": "metric",
    "stats": "metric",
    "number": "metric",
    "actions": "quick_actions",
    "buttons": "quick_actions",
    "shortcuts": "quick_actions",
    "summary": "briefing",
    "daily": "briefing",
    "digest": "briefing",
    "morning": "briefing",
    "state": "status",
    "health": "status",
}

_SAFE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_SLUG_STRIP_RE = re.compile(r"[^a-z0-9_-]+")
_HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_FORBIDDEN_PAYLOAD_RE = re.compile(
    r"(?i)<\s*/?\s*(script|iframe|object|embed|svg)\b|javascript\s*:|vbscript\s*:|data\s*:\s*text/html|intent\s*:|file\s*://"
)
_WIDGET_SECRET_RE = re.compile(
    r"(?i)(?:sk-[a-z0-9_-]{12,}|ghp_[a-z0-9]{20,}|bearer\s+[a-z0-9._-]{16,}|-----begin\s+[a-z ]*private key-----)"
)

_MOBILE_STATE_DB_NAME = "mobile_state.db"

_ui_patch_emitter: ContextVar[Optional[Callable[[dict[str, Any]], None]]] = ContextVar(
    "mobile_ui_patch_emitter", default=None
)

def mobile_state_db_path(profile_home: Optional[Path] = None) -> Path:
    """Resolve the active profile's mobile state SQLite database path."""
    home = Path(profile_home) if profile_home is not None else get_hermes_home()
    return home / _MOBILE_STATE_DB_NAME


def _valid_utf8_text(value: Any, *, max_len: int, allow_empty: bool = False) -> bool:
    if not isinstance(value, str):
        return False
    if len(value) > max_len or (not allow_empty and not value.strip()):
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return not _FORBIDDEN_PAYLOAD_RE.search(value)


def _scan_strings_safe(value: Any, *, depth: int = 0) -> Optional[str]:
    """Recursively verify UTF-8 integrity and reject executable/script injection patterns."""
    if depth > 12:
        return "Payload nesting exceeds maximum depth of 12."
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            return "Payload contains invalid UTF-8 surrogate characters."
        if _FORBIDDEN_PAYLOAD_RE.search(value):
            return "Payload contains forbidden script or executable URI scheme."
        return None
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                return "All JSON object keys must be strings."
            err = _scan_strings_safe(k, depth=depth + 1) or _scan_strings_safe(v, depth=depth + 1)
            if err:
                return err
        return None
    if isinstance(value, list):
        for item in value:
            err = _scan_strings_safe(item, depth=depth + 1)
            if err:
                return err
        return None
    if value is None or isinstance(value, (bool, int, float)):
        return None
    return f"Unsupported JSON value type: {type(value).__name__}"


# ── WCAG 2.1 Contrast Guard ──────────────────────────────────────────────────

def _channel_luminance(byte_val: int) -> float:
    s = byte_val / 255.0
    return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_color: str) -> float:
    """Compute WCAG 2.1 relative luminance for ``#RRGGBB``."""
    if not isinstance(hex_color, str) or not _HEX_COLOR_RE.match(hex_color):
        raise ValueError(f"Invalid hex color: {hex_color!r}")
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return 0.2126 * _channel_luminance(r) + 0.7152 * _channel_luminance(g) + 0.0722 * _channel_luminance(b)


def contrast_ratio(hex_a: str, hex_b: str) -> float:
    """Compute WCAG 2.1 contrast ratio between two ``#RRGGBB`` colors."""
    la = relative_luminance(hex_a)
    lb = relative_luminance(hex_b)
    lighter, darker = (la, lb) if la >= lb else (lb, la)
    return (lighter + 0.05) / (darker + 0.05)


def slugify_id(text: Any, *, fallback: str = "item") -> str:
    """Derive a deterministic safe ID (``^[a-z0-9][a-z0-9_-]{0,63}$``) from any title/label."""
    if isinstance(text, str) and text.strip():
        ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
        slug = _SLUG_STRIP_RE.sub("_", ascii_text).strip("_-")
        if slug and slug[0].isalnum():
            return slug[:MAX_ID_LEN]
    return fallback[:MAX_ID_LEN]


def resolve_accent_color(raw_accent: Any, raw_ink: Any = None) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve a named accent preset (e.g. ``teal``) or ``#RRGGBB`` into WCAG-AA verified hex pair."""
    if isinstance(raw_accent, str):
        preset = ACCENT_PALETTE_ALIASES.get(raw_accent.strip().lower())
        if preset is not None:
            hex_color, default_ink = preset
            ink_color = raw_ink.upper() if isinstance(raw_ink, str) and _HEX_COLOR_RE.match(raw_ink) else default_ink
            if raw_ink is not None and not (isinstance(raw_ink, str) and _HEX_COLOR_RE.match(raw_ink)):
                return None, None, "accent_ink_hex must be a 6-digit hex color like #FFFFFF."
            ratio = contrast_ratio(hex_color, ink_color)
            if ratio < MIN_WCAG_AA_CONTRAST:
                return None, None, (
                    f"WCAG AA contrast failure: {ink_color} on {hex_color} is "
                    f"{ratio:.2f}:1 (minimum {MIN_WCAG_AA_CONTRAST}:1)."
                )
            return hex_color, ink_color, None
    if not isinstance(raw_accent, str) or not _HEX_COLOR_RE.match(raw_accent):
        return (
            None,
            None,
            f"accent_hex must be a 6-digit hex color like #0D6E6E or one of {sorted(ACCENT_PALETTE_ALIASES)}.",
        )
    ink_color = DEFAULT_ACCENT_INK_HEX if raw_ink is None else raw_ink
    if not isinstance(ink_color, str) or not _HEX_COLOR_RE.match(ink_color):
        return None, None, "accent_ink_hex must be a 6-digit hex color like #FFFFFF."
    ratio = contrast_ratio(raw_accent, ink_color)
    if ratio < MIN_WCAG_AA_CONTRAST:
        return None, None, (
            f"WCAG AA contrast failure: {ink_color.upper()} on {raw_accent.upper()} is "
            f"{ratio:.2f}:1 (minimum {MIN_WCAG_AA_CONTRAST}:1)."
        )
    return raw_accent.upper(), ink_color.upper(), None


def validate_theme_override(theme: Any) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Validate declarative theme tokens and enforce WCAG AA (>= 4.5:1) contrast."""
    if not isinstance(theme, dict) or not theme:
        return None, "theme must be a non-empty object."
    allowed_keys = {
        "accent",
        "accent_hex",
        "accent_ink_hex",
        "surface_hex",
        "ink_hex",
        "density",
        "font_scale_step",
        "surface_style",
    }
    unknown = sorted(set(theme) - allowed_keys)
    if unknown:
        return None, f"Unsupported theme keys: {', '.join(unknown)}"

    cleaned: dict[str, Any] = {}
    for color_key in ("surface_hex", "ink_hex"):
        if color_key in theme:
            raw = theme[color_key]
            if not isinstance(raw, str) or not _HEX_COLOR_RE.match(raw):
                return None, f"{color_key} must be a 6-digit hex color like #FAF8F5."
            cleaned[color_key] = raw.upper()

    if "density" in theme:
        if theme["density"] not in ALLOWED_DENSITIES:
            return None, f"density must be one of {sorted(ALLOWED_DENSITIES)}."
        cleaned["density"] = theme["density"]

    if "surface_style" in theme:
        if theme["surface_style"] not in ALLOWED_SURFACE_STYLES:
            return None, f"surface_style must be one of {sorted(ALLOWED_SURFACE_STYLES)}."
        cleaned["surface_style"] = theme["surface_style"]

    if "font_scale_step" in theme:
        step = theme["font_scale_step"]
        if type(step) is not int or not (MIN_FONT_SCALE_STEP <= step <= MAX_FONT_SCALE_STEP):
            return None, f"font_scale_step must be an integer between {MIN_FONT_SCALE_STEP} and {MAX_FONT_SCALE_STEP}."
        cleaned["font_scale_step"] = step

    surface = cleaned.get("surface_hex", DEFAULT_SURFACE_HEX)
    ink = cleaned.get("ink_hex", DEFAULT_INK_HEX)
    text_contrast = contrast_ratio(ink, surface)
    if text_contrast < MIN_WCAG_AA_CONTRAST:
        return None, (
            f"WCAG AA contrast failure: ink {ink} on surface {surface} is "
            f"{text_contrast:.2f}:1 (minimum {MIN_WCAG_AA_CONTRAST}:1)."
        )

    raw_accent = theme.get("accent_hex", theme.get("accent"))
    if raw_accent is not None or "accent_ink_hex" in theme:
        if raw_accent is None:
            return None, "accent_ink_hex requires accent_hex or accent."
        accent_hex, accent_ink, accent_err = resolve_accent_color(raw_accent, theme.get("accent_ink_hex"))
        if accent_err:
            return None, accent_err
        cleaned["accent_hex"] = accent_hex
        cleaned["accent_ink_hex"] = accent_ink

    return cleaned, None


# ── Op Validation ────────────────────────────────────────────────────────────

def _validate_safe_id(value: Any, label: str) -> Optional[str]:
    if not isinstance(value, str) or len(value) > MAX_ID_LEN or not _SAFE_ID_RE.match(value):
        return f"{label} must match ^[a-z0-9][a-z0-9_-]{{0,63}}$."
    return None


def _normalize_widget_items(raw_items: Any, prefix: str) -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:
    if not isinstance(raw_items, list) or len(raw_items) > MAX_WIDGET_ITEMS:
        return None, f"{prefix}.items must be an array of at most {MAX_WIDGET_ITEMS} entries."
    clean_items: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_items):
        if isinstance(item, str) and _valid_utf8_text(item, max_len=80):
            clean_items.append({"label": item.strip()})
            continue
        if not isinstance(item, dict):
            return None, f"{prefix}.items[{idx}] must be an object or short string."
        label = item.get("label")
        if not _valid_utf8_text(label, max_len=80):
            return None, f"{prefix}.items[{idx}].label must be non-empty UTF-8 text (<= 80 chars)."
        clean_item: dict[str, Any] = {"label": label.strip()}
        if item.get("value") is not None:
            raw_val = item["value"]
            val_str = str(raw_val) if isinstance(raw_val, (int, float)) and not isinstance(raw_val, bool) else raw_val
            if not _valid_utf8_text(val_str, max_len=80, allow_empty=True):
                return None, f"{prefix}.items[{idx}].value must be UTF-8 text (<= 80 chars)."
            clean_item["value"] = val_str.strip()
        if item.get("checked") is not None:
            if not isinstance(item["checked"], bool):
                return None, f"{prefix}.items[{idx}].checked must be a boolean."
            clean_item["checked"] = item["checked"]
        if item.get("action_draft") is not None:
            if not _valid_utf8_text(item["action_draft"], max_len=240):
                return None, f"{prefix}.items[{idx}].action_draft must be non-empty UTF-8 text (<= 240 chars)."
            clean_item["action_draft"] = item["action_draft"].strip()
        clean_items.append(clean_item)
    return clean_items, None


def validate_widget_spec(raw: Any, *, prefix: str = "widget") -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Validate & normalize a declarative home-screen widget spec (shared SSoT for ``upsert_widget`` and ``widget_update``).

    Forgiving for small LLMs: auto-derives ``widget_id`` from ``title`` when omitted, normalizes friendly
    template/accent aliases (e.g. ``tasks`` -> ``checklist``, ``teal`` -> ``#0D6E6E``), and defaults omitted
    ``home_screen_safe`` to ``True`` while strictly rejecting explicit ``home_screen_safe=False``, WCAG-AA
    contrast failures, script/URI injection, and credential/token leaks.
    """
    if not isinstance(raw, dict):
        return None, f"{prefix} must be an object."
    scan_err = _scan_strings_safe(raw)
    if scan_err:
        return None, f"{prefix}: {scan_err}"
    title = raw.get("title")
    if not _valid_utf8_text(title, max_len=80):
        return None, f"{prefix}.title must be non-empty UTF-8 text (<= 80 chars)."
    raw_wid = raw.get("widget_id")
    wid = slugify_id(title, fallback="widget") if (raw_wid is None or raw_wid == "") else raw_wid
    id_err = _validate_safe_id(wid, f"{prefix}.widget_id")
    if id_err:
        return None, id_err
    raw_safe = raw.get("home_screen_safe")
    if raw_safe is False or (raw_safe is not None and not isinstance(raw_safe, bool)):
        return None, f"{prefix}.home_screen_safe must be true (explicitly false blocks home-screen publishing)."
    raw_template = raw.get("template")
    template = (
        WIDGET_TEMPLATE_ALIASES.get(raw_template.strip().lower(), raw_template.strip().lower())
        if isinstance(raw_template, str)
        else raw_template
    )
    if template not in ALLOWED_WIDGET_TEMPLATES:
        return None, f"{prefix}.template must be one of {sorted(ALLOWED_WIDGET_TEMPLATES)}."
    subtitle = raw.get("subtitle")
    if subtitle is not None and not _valid_utf8_text(subtitle, max_len=120, allow_empty=True):
        return None, f"{prefix}.subtitle must be UTF-8 text (<= 120 chars)."

    clean_items, items_err = _normalize_widget_items(raw.get("items", []), prefix)
    if items_err:
        return None, items_err

    spec: dict[str, Any] = {
        "widget_id": wid,
        "title": title.strip(),
        "template": template,
        "items": clean_items,
        "home_screen_safe": True,
    }
    if subtitle is not None and subtitle.strip():
        spec["subtitle"] = subtitle.strip()

    raw_accent = raw.get("accent_hex", raw.get("accent"))
    if raw_accent is not None:
        accent_hex, accent_ink, accent_err = resolve_accent_color(raw_accent, raw.get("accent_ink_hex"))
        if accent_err:
            return None, f"{prefix}: {accent_err}"
        spec["accent_hex"] = accent_hex
        spec["accent_ink_hex"] = accent_ink

    # Reject obvious credential/secret strings before they can ever reach a home screen.
    if _WIDGET_SECRET_RE.search(json.dumps(spec, ensure_ascii=False)):
        return None, f"{prefix} appears to contain a secret or bearer token and cannot be shown on the home screen."

    return spec, None


def validate_suggested_actions(raw: Any) -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:
    """Validate optional contextual 1-tap follow-up chips (1..3 safe declarative chips)."""
    if raw is None:
        return None, None
    if not isinstance(raw, list) or not (1 <= len(raw) <= MAX_SUGGESTED_ACTIONS):
        return None, f"suggested_actions must be an array of 1 to {MAX_SUGGESTED_ACTIONS} chips."
    scan_err = _scan_strings_safe(raw)
    if scan_err:
        return None, f"suggested_actions: {scan_err}"
    cleaned: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            return None, f"suggested_actions[{idx}] must be an object."
        label = item.get("label")
        if not _valid_utf8_text(label, max_len=48):
            return None, f"suggested_actions[{idx}].label must be non-empty UTF-8 text (<= 48 chars)."
        raw_id = item.get("id")
        aid = slugify_id(label, fallback=f"action_{idx + 1}") if (raw_id is None or raw_id == "") else raw_id
        id_err = _validate_safe_id(aid, f"suggested_actions[{idx}].id")
        if id_err:
            return None, id_err
        if aid in seen_ids:
            return None, f"suggested_actions[{idx}].id duplicates '{aid}'."
        seen_ids.add(aid)
        kind = item.get("kind", "prompt")
        if kind not in ALLOWED_SUGGESTED_ACTION_KINDS:
            return None, f"suggested_actions[{idx}].kind must be one of {sorted(ALLOWED_SUGGESTED_ACTION_KINDS)}."
        entry: dict[str, Any] = {"id": aid, "label": label.strip(), "kind": kind}
        if "prompt" in item and item["prompt"] is not None:
            if not _valid_utf8_text(item["prompt"], max_len=MAX_PROMPT_LEN):
                return None, f"suggested_actions[{idx}].prompt must be non-empty UTF-8 text (<= {MAX_PROMPT_LEN} chars)."
            entry["prompt"] = item["prompt"].strip()
        if "target" in item and item["target"] is not None:
            t_err = _validate_safe_id(item["target"], f"suggested_actions[{idx}].target")
            if t_err:
                return None, t_err
            entry["target"] = item["target"]
        if "widget_id" in item and item["widget_id"] is not None:
            w_err = _validate_safe_id(item["widget_id"], f"suggested_actions[{idx}].widget_id")
            if w_err:
                return None, w_err
            entry["widget_id"] = item["widget_id"]
        cleaned.append(entry)
    return cleaned, None


def _validate_upsert_surface(raw_op: dict[str, Any], idx: int) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    title = raw_op.get("title")
    if not _valid_utf8_text(title, max_len=MAX_TITLE_LEN):
        return None, f"ops[{idx}].title must be non-empty UTF-8 text (<= {MAX_TITLE_LEN} chars)."
    raw_sid = raw_op.get("surface_id")
    surface_id = slugify_id(title, fallback=f"surface_{idx + 1}") if (raw_sid is None or raw_sid == "") else raw_sid
    id_err = _validate_safe_id(surface_id, f"ops[{idx}].surface_id")
    if id_err:
        return None, id_err
    raw_kind = raw_op.get("kind")
    kind = (
        SURFACE_KIND_ALIASES.get(raw_kind.strip().lower(), raw_kind.strip().lower())
        if isinstance(raw_kind, str)
        else raw_kind
    )
    if kind not in ALLOWED_SURFACE_KINDS:
        return None, f"ops[{idx}].kind must be one of {sorted(ALLOWED_SURFACE_KINDS)}."
    scope = raw_op.get("scope", "atelier")
    if scope not in ALLOWED_SCOPES:
        return None, f"ops[{idx}].scope must be one of {sorted(ALLOWED_SCOPES)}."
    spec = {} if raw_op.get("spec") is None else raw_op.get("spec")
    if not isinstance(spec, dict):
        return None, f"ops[{idx}].spec must be an object."
    tab = raw_op.get("tab")
    if tab is not None:
        tab_err = _validate_safe_id(tab, f"ops[{idx}].tab")
        if tab_err:
            return None, tab_err
    order = raw_op.get("order", 0)
    if type(order) is not int or not (-1000 <= order <= 1000):
        return None, f"ops[{idx}].order must be an integer between -1000 and 1000."
    return {
        "op": "upsert_surface",
        "surface_id": surface_id,
        "scope": scope,
        "kind": kind,
        "title": title.strip(),
        "spec": spec,
        "order": order,
        **({"tab": tab} if tab is not None else {}),
    }, None


def _validate_remove_surface(raw_op: dict[str, Any], idx: int) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    surface_id = raw_op.get("surface_id")
    id_err = _validate_safe_id(surface_id, f"ops[{idx}].surface_id")
    if id_err:
        return None, id_err
    scope = raw_op.get("scope", "atelier")
    if scope not in ALLOWED_SCOPES:
        return None, f"ops[{idx}].scope must be one of {sorted(ALLOWED_SCOPES)}."
    return {"op": "remove_surface", "surface_id": surface_id, "scope": scope}, None


def _validate_upsert_quick_action(raw_op: dict[str, Any], idx: int) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    label = raw_op.get("label")
    if not _valid_utf8_text(label, max_len=64):
        return None, f"ops[{idx}].label must be non-empty UTF-8 text (<= 64 chars)."
    raw_aid = raw_op.get("action_id")
    action_id = slugify_id(label, fallback=f"action_{idx + 1}") if (raw_aid is None or raw_aid == "") else raw_aid
    id_err = _validate_safe_id(action_id, f"ops[{idx}].action_id")
    if id_err:
        return None, id_err
    action_kind = raw_op.get("action_kind", "prompt")
    if action_kind not in ALLOWED_QUICK_ACTION_KINDS:
        return None, f"ops[{idx}].action_kind must be one of {sorted(ALLOWED_QUICK_ACTION_KINDS)}."
    entry: dict[str, Any] = {
        "op": "upsert_quick_action",
        "action_id": action_id,
        "label": label.strip(),
        "action_kind": action_kind,
    }
    if raw_op.get("icon") is not None:
        icon_err = _validate_safe_id(raw_op["icon"], f"ops[{idx}].icon")
        if icon_err:
            return None, icon_err
        entry["icon"] = raw_op["icon"]
    if action_kind == "prompt":
        prompt = raw_op.get("prompt")
        if not _valid_utf8_text(prompt, max_len=MAX_PROMPT_LEN):
            return None, f"ops[{idx}].prompt must be non-empty UTF-8 text (<= {MAX_PROMPT_LEN} chars)."
        entry["prompt"] = prompt.strip()
    elif action_kind == "open_destination":
        dest = raw_op.get("destination")
        if dest not in ALLOWED_DESTINATIONS:
            return None, f"ops[{idx}].destination must be one of {sorted(ALLOWED_DESTINATIONS)}."
        entry["destination"] = dest
    elif action_kind == "toggle_surface":
        target_surface = raw_op.get("surface_id")
        t_err = _validate_safe_id(target_surface, f"ops[{idx}].surface_id")
        if t_err:
            return None, t_err
        entry["surface_id"] = target_surface
    return entry, None


def _validate_remove_quick_action(raw_op: dict[str, Any], idx: int) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    action_id = raw_op.get("action_id")
    id_err = _validate_safe_id(action_id, f"ops[{idx}].action_id")
    if id_err:
        return None, id_err
    return {"op": "remove_quick_action", "action_id": action_id}, None


def _validate_upsert_widget(raw_op: dict[str, Any], idx: int) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    widget_spec, widget_err = validate_widget_spec(raw_op, prefix=f"ops[{idx}]")
    if widget_err:
        return None, widget_err
    return {"op": "upsert_widget", **widget_spec}, None


def _validate_remove_widget(raw_op: dict[str, Any], idx: int) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    widget_id = raw_op.get("widget_id")
    id_err = _validate_safe_id(widget_id, f"ops[{idx}].widget_id")
    if id_err:
        return None, id_err
    return {"op": "remove_widget", "widget_id": widget_id}, None


def _validate_set_theme_override(raw_op: dict[str, Any], idx: int) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    cleaned_theme, theme_err = validate_theme_override(raw_op.get("theme"))
    if theme_err:
        return None, f"ops[{idx}]: {theme_err}"
    return {"op": "set_theme_override", "theme": cleaned_theme}, None


def _validate_clear_theme_override(_raw_op: dict[str, Any], _idx: int) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    return {"op": "clear_theme_override"}, None


_OP_VALIDATORS: dict[str, Callable[[dict[str, Any], int], tuple[Optional[dict[str, Any]], Optional[str]]]] = {
    "upsert_surface": _validate_upsert_surface,
    "remove_surface": _validate_remove_surface,
    "upsert_quick_action": _validate_upsert_quick_action,
    "remove_quick_action": _validate_remove_quick_action,
    "upsert_widget": _validate_upsert_widget,
    "remove_widget": _validate_remove_widget,
    "set_theme_override": _validate_set_theme_override,
    "clear_theme_override": _validate_clear_theme_override,
}


def _validate_card_item(
    norm_kind: str, item: Any, prefix: str, item_index: int
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    if not isinstance(item, dict):
        return None, f"{prefix} must be an object."
    ilabel = item.get("label")
    if not _valid_utf8_text(ilabel, max_len=MAX_CARD_TEXT_LEN) or _WIDGET_SECRET_RE.search(ilabel):
        return None, f"{prefix}.label must be safe UTF-8 text."
    iid = item.get("id")
    if iid is None:
        item_id = f"item_{item_index + 1}"
    elif isinstance(iid, str) and _SAFE_ID_RE.match(iid.strip()):
        item_id = iid.strip()
    else:
        return None, f"{prefix}.id must be a safe slug."

    row: dict[str, Any] = {"id": item_id, "label": ilabel.strip()}
    if norm_kind == "checklist":
        checked = item.get("checked", False)
        if not isinstance(checked, bool):
            return None, f"{prefix}.checked must be boolean."
        row["checked"] = checked
        if item.get("draft") is not None:
            if not _valid_utf8_text(item["draft"], max_len=MAX_PROMPT_LEN) or _WIDGET_SECRET_RE.search(item["draft"]):
                return None, f"{prefix}.draft must be safe UTF-8 text."
            row["draft"] = item["draft"].strip()
        return row, None

    if norm_kind == "quick_pick":
        draft = item.get("draft")
        if not _valid_utf8_text(draft, max_len=MAX_PROMPT_LEN) or _WIDGET_SECRET_RE.search(draft):
            return None, f"{prefix}.draft is required for quick_pick."
        row["draft"] = draft.strip()
        return row, None

    enabled = item.get("enabled", False)
    if not isinstance(enabled, bool):
        return None, f"{prefix}.enabled must be boolean."
    row["enabled"] = enabled
    if item.get("description") is not None:
        if not _valid_utf8_text(item["description"], max_len=MAX_CARD_TEXT_LEN) or _WIDGET_SECRET_RE.search(item["description"]):
            return None, f"{prefix}.description must be safe text."
        row["description"] = item["description"].strip()
    return row, None


def _validate_optional_buttons(
    raw: dict[str, Any], entry: dict[str, Any], keys: tuple[str, ...], prefix: str
) -> Optional[str]:
    for btn_key in keys:
        if raw.get(btn_key) is not None:
            if not _valid_utf8_text(raw[btn_key], max_len=48):
                return f"{prefix}.{btn_key} must be valid UTF-8 text (<= 48 chars)."
            entry[btn_key] = raw[btn_key].strip()
    return None


def _validate_item_card(
    norm_kind: str, raw: dict[str, Any], entry: dict[str, Any], prefix: str
) -> Optional[str]:
    raw_items = raw.get("items")
    if not isinstance(raw_items, list) or not (1 <= len(raw_items) <= MAX_CARD_ITEMS):
        return f"{prefix}.items must contain 1..{MAX_CARD_ITEMS} entries."
    parsed_items: list[dict[str, Any]] = []
    for j, item in enumerate(raw_items):
        row, err = _validate_card_item(norm_kind, item, f"{prefix}.items[{j}]", j)
        if err:
            return err
        parsed_items.append(row)  # type: ignore[arg-type]
    entry["items"] = parsed_items
    return _validate_optional_buttons(raw, entry, ("confirm_label", "cancel_label"), prefix)


def _validate_customization_preview_card(
    raw: dict[str, Any], entry: dict[str, Any], prefix: str
) -> Optional[str]:
    raw_cust = raw.get("customization")
    if not isinstance(raw_cust, dict) or not raw_cust:
        return f"{prefix}.customization must be a non-empty object."
    theme_allowed = {
        "accent", "accent_hex", "accent_ink_hex", "surface_hex", "ink_hex", "density", "font_scale_step", "surface_style"
    }
    theme_keys = {k: v for k, v in raw_cust.items() if k in theme_allowed}
    cleaned_cust: dict[str, Any] = {}
    if theme_keys:
        validated_theme, theme_err = validate_theme_override(theme_keys)
        if theme_err:
            return f"{prefix}.customization: {theme_err}"
        cleaned_cust.update(validated_theme or {})
    if "theme" in raw_cust:
        t = raw_cust["theme"]
        if not isinstance(t, str) or t.strip().lower() not in {"system", "light", "dark"}:
            return f"{prefix}.customization.theme must be system/light/dark."
        cleaned_cust["theme"] = t.strip().lower()
    if "startup_tab" in raw_cust:
        st = raw_cust["startup_tab"]
        if not isinstance(st, str) or st.strip().lower() not in ALLOWED_DESTINATIONS:
            return f"{prefix}.customization.startup_tab is invalid."
        cleaned_cust["startup_tab"] = st.strip().lower()
    if "greeting" in raw_cust:
        gr = raw_cust["greeting"]
        if not _valid_utf8_text(gr, max_len=120) or _WIDGET_SECRET_RE.search(gr):
            return f"{prefix}.customization.greeting must be safe UTF-8 text."
        cleaned_cust["greeting"] = gr.strip()
    if not cleaned_cust:
        return f"{prefix}.customization has no supported fields."
    entry["customization"] = cleaned_cust
    return _validate_optional_buttons(raw, entry, ("apply_label", "undo_label"), prefix)


def _validate_single_interactive_card(
    raw: Any, idx: int, seen_ids: set[str]
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    prefix = f"interactive_cards[{idx}]"
    if not isinstance(raw, dict):
        return None, f"{prefix} must be an object."
    kind = raw.get("kind")
    if not isinstance(kind, str) or kind.strip().lower() not in ALLOWED_INTERACTIVE_CARD_KINDS:
        return None, f"{prefix}.kind must be one of {sorted(ALLOWED_INTERACTIVE_CARD_KINDS)}."
    norm_kind = kind.strip().lower()

    title = raw.get("title")
    if not _valid_utf8_text(title, max_len=MAX_CARD_TEXT_LEN) or _WIDGET_SECRET_RE.search(title):
        return None, f"{prefix}.title must be safe UTF-8 text (<= {MAX_CARD_TEXT_LEN} chars)."

    raw_id = raw.get("id")
    if raw_id is None:
        card_id = f"card_{idx + 1}"
    elif isinstance(raw_id, str) and _SAFE_ID_RE.match(raw_id.strip()):
        card_id = raw_id.strip()
    else:
        return None, f"{prefix}.id must be a safe slug."
    if card_id in seen_ids:
        return None, f"{prefix}.id '{card_id}' is duplicated."
    seen_ids.add(card_id)

    entry: dict[str, Any] = {"id": card_id, "kind": norm_kind, "title": title.strip()}
    subtitle = raw.get("subtitle")
    if subtitle is not None:
        if not _valid_utf8_text(subtitle, max_len=MAX_CARD_TEXT_LEN) or _WIDGET_SECRET_RE.search(subtitle):
            return None, f"{prefix}.subtitle must be safe UTF-8 text (<= {MAX_CARD_TEXT_LEN} chars)."
        entry["subtitle"] = subtitle.strip()

    if norm_kind == "customization_preview":
        body_err = _validate_customization_preview_card(raw, entry, prefix)
    else:
        body_err = _validate_item_card(norm_kind, raw, entry, prefix)
    if body_err:
        return None, body_err
    return entry, None


def validate_interactive_cards(cards: Any) -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:
    """Validate optional bounded ``interactive_cards`` (strictly additive inline interactive cards)."""
    if cards is None:
        return None, None
    if isinstance(cards, dict):
        cards = [cards]
    if not isinstance(cards, list) or not cards:
        return None, "interactive_cards must be a non-empty array when provided."
    if len(cards) > MAX_INTERACTIVE_CARDS:
        return None, f"interactive_cards supports at most {MAX_INTERACTIVE_CARDS} cards per patch."
    scan_err = _scan_strings_safe(cards)
    if scan_err:
        return None, scan_err

    cleaned: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx, raw in enumerate(cards):
        entry, err = _validate_single_interactive_card(raw, idx, seen_ids)
        if err:
            return None, err
        cleaned.append(entry)  # type: ignore[arg-type]
    return cleaned, None


def validate_ui_patch_ops(ops: Any) -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:
    """Validate and normalize a declarative UI patch op list."""
    if isinstance(ops, dict):
        ops = [ops]
    if not isinstance(ops, list) or not ops:
        return None, "ops must be a non-empty array."
    if len(ops) > MAX_OPS_PER_PATCH:
        return None, f"ops supports at most {MAX_OPS_PER_PATCH} operations per patch."
    err = _scan_strings_safe(ops)
    if err:
        return None, err
    try:
        encoded = json.dumps(ops, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        return None, f"ops must be JSON-serializable: {exc}"
    if len(encoded) > MAX_PATCH_JSON_BYTES:
        return None, f"ops payload exceeds maximum size of {MAX_PATCH_JSON_BYTES} bytes."

    normalized: list[dict[str, Any]] = []
    for idx, raw_op in enumerate(ops):
        if not isinstance(raw_op, dict):
            return None, f"ops[{idx}] must be an object."
        handler = _OP_VALIDATORS.get(raw_op.get("op"))
        if handler is None:
            return None, f"ops[{idx}].op must be one of {sorted(ALLOWED_OP_TYPES)}."
        entry, op_err = handler(raw_op, idx)
        if op_err:
            return None, op_err
        normalized.append(entry)

    return normalized, None


# ── State Application Logic (Pure & Deterministic) ───────────────────────────

def _empty_state() -> dict[str, Any]:
    return {"surfaces": {}, "quick_actions": {}, "widgets": {}, "theme": {}}


def _apply_upsert_surface(state: dict[str, Any], op: dict[str, Any], session_id: Optional[str]) -> None:
    surfaces = state["surfaces"]
    scope = op["scope"]
    prefix = f"{scope}:{session_id or ''}:" if scope == "thread" else "atelier:"
    key = f"{prefix}{op['surface_id']}"
    if key not in surfaces and sum(1 for k in surfaces if k.startswith(prefix)) >= MAX_SURFACES_PER_SCOPE:
        raise ValueError(f"Maximum of {MAX_SURFACES_PER_SCOPE} surfaces reached for scope '{scope}'.")
    surfaces[key] = {
        "surface_id": op["surface_id"],
        "scope": scope,
        "session_id": session_id if scope == "thread" else None,
        "kind": op["kind"],
        "title": op["title"],
        "spec": op["spec"],
        "order": op.get("order", 0),
        **({"tab": op["tab"]} if "tab" in op else {}),
    }


def _apply_remove_surface(state: dict[str, Any], op: dict[str, Any], session_id: Optional[str]) -> None:
    scope = op["scope"]
    key = f"{scope}:{session_id or ''}:{op['surface_id']}" if scope == "thread" else f"atelier:{op['surface_id']}"
    state["surfaces"].pop(key, None)


def _apply_upsert_quick_action(state: dict[str, Any], op: dict[str, Any], _session_id: Optional[str]) -> None:
    quick_actions = state["quick_actions"]
    aid = op["action_id"]
    if aid not in quick_actions and len(quick_actions) >= MAX_QUICK_ACTIONS:
        raise ValueError(f"Maximum of {MAX_QUICK_ACTIONS} quick actions reached.")
    quick_actions[aid] = {k: v for k, v in op.items() if k != "op"}


def _apply_remove_quick_action(state: dict[str, Any], op: dict[str, Any], _session_id: Optional[str]) -> None:
    state["quick_actions"].pop(op["action_id"], None)


def _apply_upsert_widget(state: dict[str, Any], op: dict[str, Any], _session_id: Optional[str]) -> None:
    widgets = state["widgets"]
    wid = op["widget_id"]
    if wid not in widgets and len(widgets) >= MAX_WIDGETS:
        raise ValueError(f"Maximum of {MAX_WIDGETS} widgets reached.")
    widgets[wid] = {k: v for k, v in op.items() if k != "op"}


def _apply_remove_widget(state: dict[str, Any], op: dict[str, Any], _session_id: Optional[str]) -> None:
    state["widgets"].pop(op["widget_id"], None)


def _apply_set_theme_override(state: dict[str, Any], op: dict[str, Any], _session_id: Optional[str]) -> None:
    state["theme"].update(op["theme"])


def _apply_clear_theme_override(state: dict[str, Any], _op: dict[str, Any], _session_id: Optional[str]) -> None:
    state["theme"].clear()


_OP_APPLIERS: dict[str, Callable[[dict[str, Any], dict[str, Any], Optional[str]], None]] = {
    "upsert_surface": _apply_upsert_surface,
    "remove_surface": _apply_remove_surface,
    "upsert_quick_action": _apply_upsert_quick_action,
    "remove_quick_action": _apply_remove_quick_action,
    "upsert_widget": _apply_upsert_widget,
    "remove_widget": _apply_remove_widget,
    "set_theme_override": _apply_set_theme_override,
    "clear_theme_override": _apply_clear_theme_override,
}


def _apply_ops_to_state(base_state: dict[str, Any], ops: list[dict[str, Any]], *, session_id: Optional[str]) -> dict[str, Any]:
    state = json.loads(json.dumps(base_state))
    state.setdefault("surfaces", {})
    state.setdefault("quick_actions", {})
    state.setdefault("widgets", {})
    state.setdefault("theme", {})
    for op in ops:
        applier = _OP_APPLIERS.get(op["op"])
        if applier is not None:
            applier(state, op, session_id)
    return state


# ── Live SSE Emitter Hook ────────────────────────────────────────────────────

@contextmanager
def bind_ui_patch_emitter(callback: Optional[Callable[[dict[str, Any]], None]]):
    """Bind a live run SSE emitter for ``ui.patch`` events during a turn."""
    token = _ui_patch_emitter.set(callback)
    try:
        yield
    finally:
        _ui_patch_emitter.reset(token)


def emit_ui_patch_event(revision: dict[str, Any]) -> None:
    emitter = _ui_patch_emitter.get()
    if emitter is None:
        return
    try:
        emitter(revision)
    except Exception:
        logger.warning("ui.patch SSE emitter failed for revision=%s", revision.get("revision_id"), exc_info=True)


# ── Profile-Scoped SQLite Store ──────────────────────────────────────────────

_UI_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ui_revisions (
    revision_id TEXT PRIMARY KEY,
    parent_revision_id TEXT,
    session_id TEXT,
    scope TEXT NOT NULL,
    summary TEXT NOT NULL,
    ops_json TEXT NOT NULL,
    state_snapshot_json TEXT NOT NULL,
    suggested_actions_json TEXT,
    interactive_cards_json TEXT,
    status TEXT NOT NULL,
    requires_apply INTEGER NOT NULL,
    created_at REAL NOT NULL,
    applied_at REAL
);
CREATE INDEX IF NOT EXISTS idx_ui_revisions_created ON ui_revisions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ui_revisions_session ON ui_revisions (session_id, created_at DESC);
CREATE TABLE IF NOT EXISTS ui_active_state (
    scope_key TEXT PRIMARY KEY,
    active_revision_id TEXT,
    state_json TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""

_GLOBAL_STATE_KEY = "profile"


class RevisionConflictError(ValueError):
    """Raised when attempting to revert an applied revision that is not the active head."""

    def __init__(self, message: str, *, active_revision_id: Optional[str]) -> None:
        super().__init__(message)
        self.active_revision_id = active_revision_id


def _migrate_ui_schema(conn: sqlite3.Connection) -> None:
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(ui_revisions)").fetchall()}
    if "suggested_actions_json" not in cols:
        conn.execute("ALTER TABLE ui_revisions ADD COLUMN suggested_actions_json TEXT")
    if "interactive_cards_json" not in cols:
        conn.execute("ALTER TABLE ui_revisions ADD COLUMN interactive_cards_json TEXT")


class MobileUiStore:
    """Profile-scoped SQLite store for declarative UI revisions and rollback."""

    def __init__(self, profile_home: Optional[Path] = None) -> None:
        self._db_path = mobile_state_db_path(profile_home)

    def _open(self, *, write: bool = False):
        if write:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        return open_mobile_db(
            self._db_path,
            schema_key="mobile_ui",
            schema_sql=_UI_SCHEMA_SQL,
            db_label="mobile_state.db",
            write=write,
            migrate=_migrate_ui_schema,
        )

    def _read_active(self, conn: sqlite3.Connection) -> tuple[Optional[str], dict[str, Any]]:
        row = conn.execute(
            "SELECT active_revision_id, state_json FROM ui_active_state WHERE scope_key = ?",
            (_GLOBAL_STATE_KEY,),
        ).fetchone()
        if row is None:
            return None, _empty_state()
        return row["active_revision_id"], json.loads(row["state_json"])

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        keys = row.keys()
        raw_suggestions = row["suggested_actions_json"] if "suggested_actions_json" in keys else None
        suggestions = json.loads(raw_suggestions) if raw_suggestions else None
        raw_cards = row["interactive_cards_json"] if "interactive_cards_json" in keys else None
        cards = json.loads(raw_cards) if raw_cards else None
        return {
            "revision_id": row["revision_id"],
            "parent_revision_id": row["parent_revision_id"],
            "session_id": row["session_id"],
            "scope": row["scope"],
            "summary": row["summary"],
            "ops": json.loads(row["ops_json"]),
            "status": row["status"],
            "requires_apply": bool(row["requires_apply"]),
            "reversible": True,
            "created_at": row["created_at"],
            "applied_at": row["applied_at"],
            **({"suggested_actions": suggestions} if suggestions else {}),
            **({"interactive_cards": cards} if cards else {}),
        }

    def propose_or_apply(
        self,
        ops: list[dict[str, Any]],
        summary: str,
        *,
        session_id: Optional[str] = None,
        scope: str = "atelier",
        requires_apply: bool = True,
        suggested_actions: Optional[list[dict[str, Any]]] = None,
        interactive_cards: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        normalized, err = validate_ui_patch_ops(ops)
        if err:
            raise ValueError(err)
        clean_suggestions, sugg_err = validate_suggested_actions(suggested_actions)
        if sugg_err:
            raise ValueError(sugg_err)
        clean_cards, cards_err = validate_interactive_cards(interactive_cards)
        if cards_err:
            raise ValueError(cards_err)
        if not _valid_utf8_text(summary, max_len=MAX_SUMMARY_LEN):
            raise ValueError(f"summary must be non-empty UTF-8 text (<= {MAX_SUMMARY_LEN} chars).")
        if scope not in ALLOWED_SCOPES:
            raise ValueError(f"scope must be one of {sorted(ALLOWED_SCOPES)}.")

        now = time.time()
        revision_id = f"uirev_{uuid.uuid4().hex[:16]}"
        with self._open(write=True) as conn:
            parent_id, active_state = self._read_active(conn)
            next_state = _apply_ops_to_state(active_state, normalized, session_id=session_id)
            status = "pending" if requires_apply else "applied"
            applied_at = None if requires_apply else now
            conn.execute(
                """INSERT INTO ui_revisions
                   (revision_id, parent_revision_id, session_id, scope, summary, ops_json,
                    state_snapshot_json, suggested_actions_json, interactive_cards_json,
                    status, requires_apply, created_at, applied_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    revision_id,
                    parent_id,
                    session_id,
                    scope,
                    summary.strip(),
                    json.dumps(normalized, ensure_ascii=False),
                    json.dumps(next_state, ensure_ascii=False),
                    json.dumps(clean_suggestions, ensure_ascii=False) if clean_suggestions else None,
                    json.dumps(clean_cards, ensure_ascii=False) if clean_cards else None,
                    status,
                    1 if requires_apply else 0,
                    now,
                    applied_at,
                ),
            )
            if not requires_apply:
                conn.execute(
                    """INSERT INTO ui_active_state (scope_key, active_revision_id, state_json, updated_at)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(scope_key) DO UPDATE SET
                         active_revision_id=excluded.active_revision_id,
                         state_json=excluded.state_json,
                         updated_at=excluded.updated_at""",
                    (_GLOBAL_STATE_KEY, revision_id, json.dumps(next_state, ensure_ascii=False), now),
                )

        revision = {
            "revision_id": revision_id,
            "parent_revision_id": parent_id,
            "session_id": session_id,
            "scope": scope,
            "summary": summary.strip(),
            "ops": normalized,
            "status": status,
            "requires_apply": bool(requires_apply),
            "reversible": True,
            "created_at": now,
            "applied_at": applied_at,
            **({"suggested_actions": clean_suggestions} if clean_suggestions else {}),
            **({"interactive_cards": clean_cards} if clean_cards else {}),
        }
        logger.info("mobile_ui revision=%s scope=%s status=%s ops=%d", revision_id, scope, status, len(normalized))
        emit_ui_patch_event(revision)
        return revision

    def get_state(self, *, session_id: Optional[str] = None) -> dict[str, Any]:
        with self._open(write=False) as conn:
            active_id, state = self._read_active(conn)
            pending_rows = conn.execute(
                "SELECT * FROM ui_revisions WHERE status = 'pending' ORDER BY created_at DESC LIMIT 20"
            ).fetchall()

        all_surfaces = list(state.get("surfaces", {}).values())
        atelier_surfaces = sorted(
            (s for s in all_surfaces if s.get("scope") == "atelier"),
            key=lambda s: (s.get("order", 0), s.get("surface_id", "")),
        )
        thread_surfaces = sorted(
            (s for s in all_surfaces if s.get("scope") == "thread" and session_id and s.get("session_id") == session_id),
            key=lambda s: (s.get("order", 0), s.get("surface_id", "")),
        )
        pending = [
            self._row_to_dict(r)
            for r in pending_rows
            if r["scope"] == "atelier" or (session_id is not None and r["session_id"] == session_id) or session_id is None
        ]
        return {
            "active_revision_id": active_id,
            "atelier_surfaces": atelier_surfaces,
            "thread_surfaces": thread_surfaces,
            "quick_actions": list(state.get("quick_actions", {}).values()),
            "widgets": list(state.get("widgets", {}).values()),
            "theme": state.get("theme", {}),
            "pending_revisions": pending,
        }

    def list_history(self, *, session_id: Optional[str] = None, limit: int = 20) -> list[dict[str, Any]]:
        capped = max(1, min(int(limit), 100))
        with self._open(write=False) as conn:
            if session_id:
                rows = conn.execute(
                    "SELECT * FROM ui_revisions WHERE session_id = ? OR scope = 'atelier' ORDER BY created_at DESC LIMIT ?",
                    (session_id, capped),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM ui_revisions ORDER BY created_at DESC LIMIT ?",
                    (capped,),
                ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def apply_revision(self, revision_id: str) -> Optional[dict[str, Any]]:
        now = time.time()
        with self._open(write=True) as conn:
            row = conn.execute("SELECT * FROM ui_revisions WHERE revision_id = ?", (revision_id,)).fetchone()
            if row is None:
                return None
            active_id, active_state = self._read_active(conn)
            if row["status"] == "applied" and active_id == revision_id:
                return {"revision": self._row_to_dict(row), "already_applied": True, "active_revision_id": active_id}

            ops = json.loads(row["ops_json"])
            next_state = _apply_ops_to_state(active_state, ops, session_id=row["session_id"])
            conn.execute(
                "UPDATE ui_revisions SET status = 'applied', applied_at = ?, state_snapshot_json = ?, parent_revision_id = ? WHERE revision_id = ?",
                (now, json.dumps(next_state, ensure_ascii=False), active_id, revision_id),
            )
            conn.execute(
                """INSERT INTO ui_active_state (scope_key, active_revision_id, state_json, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(scope_key) DO UPDATE SET
                     active_revision_id=excluded.active_revision_id,
                     state_json=excluded.state_json,
                     updated_at=excluded.updated_at""",
                (_GLOBAL_STATE_KEY, revision_id, json.dumps(next_state, ensure_ascii=False), now),
            )
            updated = conn.execute("SELECT * FROM ui_revisions WHERE revision_id = ?", (revision_id,)).fetchone()
        logger.info("mobile_ui applied revision=%s", revision_id)
        return {"revision": self._row_to_dict(updated), "already_applied": False, "active_revision_id": revision_id}

    def revert_revision(self, revision_id: str) -> Optional[dict[str, Any]]:
        now = time.time()
        with self._open(write=True) as conn:
            row = conn.execute("SELECT * FROM ui_revisions WHERE revision_id = ?", (revision_id,)).fetchone()
            if row is None:
                return None
            if row["status"] == "reverted":
                active_id, _ = self._read_active(conn)
                return {"revision": self._row_to_dict(row), "already_reverted": True, "active_revision_id": active_id}

            parent_id = row["parent_revision_id"]
            if row["status"] == "pending":
                # Dismissing a pending preview leaves active state untouched.
                conn.execute("UPDATE ui_revisions SET status = 'reverted' WHERE revision_id = ?", (revision_id,))
                active_id, _ = self._read_active(conn)
            else:
                current_active_id, _ = self._read_active(conn)
                if current_active_id != revision_id:
                    raise RevisionConflictError(
                        f"Cannot revert non-active revision {revision_id}; active revision is {current_active_id}. "
                        "Revert the active revision first.",
                        active_revision_id=current_active_id,
                    )
                restored_state = _empty_state()
                restored_id = None
                if parent_id:
                    parent_row = conn.execute(
                        "SELECT revision_id, state_snapshot_json FROM ui_revisions WHERE revision_id = ?",
                        (parent_id,),
                    ).fetchone()
                    if parent_row is not None:
                        restored_state = json.loads(parent_row["state_snapshot_json"])
                        restored_id = parent_row["revision_id"]
                conn.execute("UPDATE ui_revisions SET status = 'reverted' WHERE revision_id = ?", (revision_id,))
                conn.execute(
                    """INSERT INTO ui_active_state (scope_key, active_revision_id, state_json, updated_at)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(scope_key) DO UPDATE SET
                         active_revision_id=excluded.active_revision_id,
                         state_json=excluded.state_json,
                         updated_at=excluded.updated_at""",
                    (_GLOBAL_STATE_KEY, restored_id, json.dumps(restored_state, ensure_ascii=False), now),
                )
                active_id = restored_id
            updated = conn.execute("SELECT * FROM ui_revisions WHERE revision_id = ?", (revision_id,)).fetchone()
        logger.info("mobile_ui reverted revision=%s active=%s", revision_id, active_id)
        return {"revision": self._row_to_dict(updated), "already_reverted": False, "active_revision_id": active_id}


# ── Agent Tool Implementation & Registration ─────────────────────────────────

MOBILE_UI_PATCH_SCHEMA = {
    "name": "mobile_ui_patch",
    "description": (
        "Propose or apply a validated, reversible declarative UI patch to the user's Hermes Mobile app "
        "(Atelier tabs, KPI grids, sparklines, bar charts, tables, forms, toggles, chips, sliders, "
        "composer quick-action shortcuts, or WCAG-AA verified theme palettes). Never executes code."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Concise human-readable description of what this UI update changes.",
            },
            "scope": {
                "type": "string",
                "enum": ["atelier", "thread"],
                "description": "Target scope: 'atelier' (persistent mobile workspace) or 'thread' (current discussion).",
            },
            "requires_apply": {
                "type": "boolean",
                "description": "When true (default), stages a live preview banner (Voir / Appliquer / Annuler). Set false only when user asked for immediate application.",
            },
            "ops": {
                "type": "array",
                "description": (
                    "Declarative UI operations: upsert_surface, remove_surface, upsert_quick_action, "
                    "remove_quick_action, upsert_widget, remove_widget, set_theme_override, clear_theme_override."
                ),
                "items": {"type": "object"},
            },
            "suggested_actions": {
                "type": "array",
                "description": (
                    "Optional 1-3 contextual 1-tap follow-up chips (pin_widget, save_memory, create_reminder, "
                    "open_surface, save_skill, prompt) shown below the reply."
                ),
                "items": {"type": "object"},
            },
            "interactive_cards": {
                "type": "array",
                "description": (
                    "Optional 1-4 bounded native inline interactive cards (checklist, quick_pick, "
                    "toggle_confirm, customization_preview) rendered natively inside the discussion."
                ),
                "items": {"type": "object"},
            },
        },
        "required": ["summary", "ops"],
    },
}


_UI_PATCH_EXAMPLE = {
    "summary": "Add morning widget",
    "ops": [{
        "op": "upsert_widget",
        "title": "Mon matin",
        "template": "metric",
        "accent": "teal",
        "items": [{"label": "Tâches", "value": "3"}],
        "home_screen_safe": True,
    }],
}


def mobile_ui_patch_tool(
    ops: Any,
    summary: str = "",
    *,
    scope: str = "atelier",
    requires_apply: bool = True,
    session_id: Optional[str] = None,
    suggested_actions: Any = None,
    interactive_cards: Any = None,
) -> str:
    """Execute ``mobile_ui_patch`` tool call and return JSON result (with self-correcting hints on error)."""
    if not session_id:
        from gateway.session_context import get_session_env
        session_id = get_session_env("HERMES_SESSION_ID") or None
    effective_summary = summary.strip() if isinstance(summary, str) and summary.strip() else ""
    if not effective_summary:
        first_op = ops[0] if isinstance(ops, list) and ops and isinstance(ops[0], dict) else (ops if isinstance(ops, dict) else {})
        label = first_op.get("title") or first_op.get("label") or first_op.get("op") or "UI update"
        effective_summary = f"Update: {str(label)[:120]}"
    store = MobileUiStore()
    try:
        revision = store.propose_or_apply(
            ops,
            effective_summary,
            session_id=session_id,
            scope=scope,
            requires_apply=bool(requires_apply),
            suggested_actions=suggested_actions,
            interactive_cards=interactive_cards,
        )
    except ValueError as exc:
        logger.warning("mobile_ui_patch rejected: %s", exc)
        return json.dumps(
            {
                "ok": False,
                "error": str(exc),
                "hint": (
                    "Pass ops=[{op, title, ...}] using allowed ops/templates/accent presets "
                    "(teal, green, blue, amber, coral, indigo, slate) and safe UTF-8 text."
                ),
                "example": _UI_PATCH_EXAMPLE,
            },
            ensure_ascii=False,
        )
    return json.dumps({"ok": True, "revision": revision}, ensure_ascii=False)


registry.register(
    name="mobile_ui_patch",
    toolset="mobile_ui",
    schema=MOBILE_UI_PATCH_SCHEMA,
    handler=lambda args, **kw: mobile_ui_patch_tool(
        args.get("ops"),
        args.get("summary", ""),
        scope=args.get("scope", "atelier"),
        requires_apply=args.get("requires_apply", True),
        session_id=kw.get("session_id"),
        suggested_actions=args.get("suggested_actions"),
        interactive_cards=args.get("interactive_cards"),
    ),
    check_fn=lambda: True,
)
