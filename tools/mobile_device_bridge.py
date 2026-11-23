"""Approved Bidirectional Android Device Capability Bridge (``request_device_action``).

Provides per-run scoped capability advertisement (mirroring ``agent/run_instructions.py``
expiry semantics so copied execution contexts or later turns never inherit prior capabilities)
and a thread-safe pause/resume broker keyed by ``(profile_key, run_id)``.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Optional

from tools.mobile_ui_patch import (
    ALLOWED_DENSITIES,
    ALLOWED_DESTINATIONS,
    MAX_FONT_SCALE_STEP,
    MIN_FONT_SCALE_STEP,
    _WIDGET_SECRET_RE,
    _scan_strings_safe,
    _valid_utf8_text,
    validate_theme_override,
)
from tools.registry import no_cache_check_fn, registry

logger = logging.getLogger("hermes.mobile_device_bridge")

# ── Single Source of Truth Constants ─────────────────────────────────────────

ALLOWED_DEVICE_CAPABILITIES = frozenset({
    "camera",
    "location_coarse",
    "nearby_osm",
    "calendar_draft",
    "alarm_timer",
    "maps_handoff",
    "map_navigation",
    "share_sheet",
    "clipboard_read",
    "clipboard_write",
    "dial_draft",
    "sms_draft",
    "email_draft",
    "contacts_pick",
    "contact_draft",
    "notifications_digest",
    "notification_reply",
    "media_control",
    "calendar_read",
    "ui_surfaces_v2",
    # Deep Android 16 bidirectional app control & presence actions
    "app_navigate",
    "app_customize",
    "app_read_state",
    "app_create_template",
    "android_set_timer",
    "android_share_sheet",
    "android_open_app",
    "android_sync_shortcuts",
    "android_proactive_nudge",
})

DEEP_ANDROID_ACTIONS = frozenset({
    "app_navigate",
    "app_customize",
    "app_read_state",
    "app_create_template",
    "android_set_timer",
    "android_share_sheet",
    "android_open_app",
    "android_sync_shortcuts",
    "android_proactive_nudge",
})

DEVICE_ACTION_CATALOG: dict[str, dict[str, Any]] = {
    "camera": {"consent_tier": "confirm", "description": "Capture a photo with user confirmation."},
    "location_coarse": {"consent_tier": "confirm", "description": "Read approximate neighborhood location."},
    "nearby_osm": {"consent_tier": "confirm", "description": "Explore nearby places with OpenStreetMap."},
    "calendar_draft": {"consent_tier": "confirm", "description": "Open native calendar event draft."},
    "alarm_timer": {"consent_tier": "confirm", "description": "Set an Android alarm or countdown timer."},
    "maps_handoff": {"consent_tier": "confirm", "description": "Open a location in the native maps app."},
    "map_navigation": {"consent_tier": "confirm", "description": "Launch turn-by-turn maps navigation."},
    "share_sheet": {"consent_tier": "confirm", "description": "Open the Android system share sheet."},
    "clipboard_read": {"consent_tier": "confirm", "description": "Read current clipboard text with consent."},
    "clipboard_write": {"consent_tier": "confirm", "description": "Copy text to clipboard with consent."},
    "dial_draft": {"consent_tier": "confirm", "description": "Open phone dialer with a prefilled number."},
    "sms_draft": {"consent_tier": "confirm", "description": "Open SMS composer with an editable draft."},
    "email_draft": {"consent_tier": "confirm", "description": "Open email composer with an editable draft."},
    "contacts_pick": {"consent_tier": "confirm", "description": "Pick a contact via Android contact picker."},
    "contact_draft": {"consent_tier": "confirm", "description": "Open add-contact form with prefilled fields."},
    "notifications_digest": {"consent_tier": "opt_in_guarded", "description": "Read sanitized allowlisted notification digest."},
    "notification_reply": {"consent_tier": "explicit_approval", "description": "Send a user-reviewed notification quick reply."},
    "media_control": {"consent_tier": "confirm", "description": "Inspect or control active media playback."},
    "calendar_read": {"consent_tier": "opt_in_guarded", "description": "Read upcoming calendar events when opted in."},
    "ui_surfaces_v2": {"consent_tier": "confirm", "description": "Mount or preview declarative native UI surfaces."},
    # 9 ultra-deep Android 16 actions
    "app_navigate": {
        "consent_tier": "low",
        "description": "Navigate within Hermes Mobile (tab, panel, thread, command palette, or workflow template).",
        "required_params": ("target",),
        "allowed_params": ("target", "session_id", "template_id", "route_id"),
    },
    "app_customize": {
        "consent_tier": "confirm",
        "description": "Apply, undo, or reset live app theme, density, font scale, startup tab, greeting, or composer shortcuts.",
        "required_params": ("operation",),
        "allowed_params": (
            "operation",
            "theme",
            "accent",
            "accent_preset",
            "accent_hex",
            "accent_ink_hex",
            "surface_hex",
            "ink_hex",
            "density",
            "font_scale_step",
            "startup_tab",
            "greeting",
            "quick_shortcuts",
        ),
    },
    "app_read_state": {
        "consent_tier": "low",
        "description": "Read sanitized live app presence (screen, active tasks, read-without-reply signals, opted-in location).",
        "required_params": (),
        "allowed_params": ("include",),
    },
    "app_create_template": {
        "consent_tier": "confirm",
        "description": "Save a reusable everyday workflow template (Editable Draft Rule — never auto-sends).",
        "required_params": ("title", "draft_prompt"),
        "allowed_params": ("title", "draft_prompt", "category", "icon"),
    },
    "android_set_timer": {
        "consent_tier": "confirm",
        "description": "Set a native Android AlarmClock countdown timer with explicit confirmation.",
        "required_params": ("duration_seconds",),
        "allowed_params": ("duration_seconds", "label"),
    },
    "android_share_sheet": {
        "consent_tier": "confirm",
        "description": "Open native Android ACTION_CHOOSER share sheet for safe text content.",
        "required_params": ("text",),
        "allowed_params": ("text", "subject", "mime_type"),
    },
    "android_open_app": {
        "consent_tier": "explicit_approval",
        "description": "Launch an installed Android app or safe deep link after explicit user approval.",
        "required_params": (),
        "allowed_params": ("package_name", "deep_link", "label"),
    },
    "android_sync_shortcuts": {
        "consent_tier": "confirm",
        "description": "Sync Android 16 Quick Settings Tile, dynamic App Shortcuts, and Home Screen Widget summary.",
        "required_params": (),
        "allowed_params": ("tile_label", "shortcuts", "widget_summary"),
    },
    "android_proactive_nudge": {
        "consent_tier": "opt_in_guarded",
        "description": "Post an opt-in, rate-limited local helpful notification respecting quiet hours and lock-screen privacy.",
        "required_params": ("title", "body"),
        "allowed_params": ("title", "body", "category", "target_screen", "respect_quiet_hours", "min_interval_minutes"),
    },
}

DEVICE_ACTION_ALIASES: dict[str, str] = {
    "photo": "camera",
    "picture": "camera",
    "take_photo": "camera",
    "location": "location_coarse",
    "gps": "location_coarse",
    "where": "location_coarse",
    "nearby": "nearby_osm",
    "places": "nearby_osm",
    "calendar": "calendar_draft",
    "event": "calendar_draft",
    "alarm": "alarm_timer",
    "timer": "alarm_timer",
    "countdown": "alarm_timer",
    "maps": "maps_handoff",
    "directions": "map_navigation",
    "navigate": "map_navigation",
    "nav": "map_navigation",
    "share": "share_sheet",
    "clipboard": "clipboard_read",
    "copy": "clipboard_write",
    "call": "dial_draft",
    "phone": "dial_draft",
    "dial": "dial_draft",
    "sms": "sms_draft",
    "text_message": "sms_draft",
    "email": "email_draft",
    "mail": "email_draft",
    "contact": "contacts_pick",
    "contacts": "contacts_pick",
    "add_contact": "contact_draft",
    "notifications": "notifications_digest",
    "notifs": "notifications_digest",
    "unread": "notifications_digest",
    "reply_notification": "notification_reply",
    "quick_reply": "notification_reply",
    "media": "media_control",
    "music": "media_control",
    "now_playing": "media_control",
    "agenda": "calendar_read",
    "schedule": "calendar_read",
    "free_busy": "calendar_read",
    # Deep Android 16 ergonomic aliases
    "open_screen": "app_navigate",
    "go_to": "app_navigate",
    "customize_app": "app_customize",
    "theme_app": "app_customize",
    "read_app_state": "app_read_state",
    "app_presence": "app_read_state",
    "create_template": "app_create_template",
    "save_template": "app_create_template",
    "set_timer": "android_set_timer",
    "open_app": "android_open_app",
    "launch_app": "android_open_app",
    "sync_shortcuts": "android_sync_shortcuts",
    "proactive_nudge": "android_proactive_nudge",
    "nudge": "android_proactive_nudge",
}

_NAV_TARGETS = frozenset({
    "discussions",
    "atelier",
    "cerveau",
    "activite",
    "thread",
    "command_palette",
    "template",
    "memory",
    "notes",
    "shared_with_me",
    "search",
    "skills",
    "workspace_files",
    "profile_defaults",
    "ambient",
    "doctor",
    "presence",
})
_READ_STATE_INCLUDES = frozenset({"screen", "active_runs", "read_without_reply", "location", "customization"})
_SHARE_MIME_TYPES = frozenset({"text/plain", "text/markdown", "application/json"})
_NUDGE_CATEGORIES = frozenset({"reminder", "follow_up", "briefing", "suggestion"})
_PACKAGE_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*(\.[a-zA-Z][a-zA-Z0-9_]*)+$")
_SAFE_DEEP_LINK_RE = re.compile(r"^(?:https://|geo:|mailto:|tel:)\S{1,512}$", re.IGNORECASE)
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._:/-]{1,128}$")
_SAFE_SHORTCUT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")


def _contains_secret(text: str) -> bool:
    return bool(_WIDGET_SECRET_RE.search(text))


def _validate_app_navigate(params: dict[str, Any]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    target = params.get("target")
    if not isinstance(target, str) or target.strip().lower() not in _NAV_TARGETS:
        return None, f"app_navigate.target must be one of {sorted(_NAV_TARGETS)}."
    clean: dict[str, Any] = {"target": target.strip().lower()}
    for key in ("session_id", "template_id", "route_id"):
        if key in params and params[key] is not None:
            val = params[key]
            if not isinstance(val, str) or not _SAFE_ID_RE.match(val.strip()):
                return None, f"app_navigate.{key} must be a safe identifier."
            clean[key] = val.strip()
    if clean["target"] == "thread" and "session_id" not in clean:
        return None, "app_navigate with target='thread' requires session_id."
    if clean["target"] == "template" and "template_id" not in clean:
        return None, "app_navigate with target='template' requires template_id."
    return clean, None


def _validate_quick_shortcuts(raw_sc: Any) -> tuple[Optional[list[dict[str, str]]], Optional[str]]:
    if not isinstance(raw_sc, list) or len(raw_sc) > 6:
        return None, "app_customize.quick_shortcuts must be an array of at most 6 items."
    cleaned_sc: list[dict[str, str]] = []
    for idx, item in enumerate(raw_sc):
        if not isinstance(item, dict):
            return None, f"app_customize.quick_shortcuts[{idx}] must be an object."
        label = item.get("label")
        draft = item.get("draft")
        if not _valid_utf8_text(label, max_len=48) or not _valid_utf8_text(draft, max_len=400):
            return None, f"app_customize.quick_shortcuts[{idx}] requires safe label and draft."
        if _contains_secret(label) or _contains_secret(draft):
            return None, f"app_customize.quick_shortcuts[{idx}] contains secret-like material."
        cleaned_sc.append({"label": label.strip(), "draft": draft.strip()})
    return cleaned_sc, None


def _validate_app_customize(params: dict[str, Any]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    op = params.get("operation")
    if not isinstance(op, str) or op.strip().lower() not in {"apply", "undo", "reset"}:
        return None, "app_customize.operation must be one of ['apply', 'reset', 'undo']."
    clean_op = op.strip().lower()
    if clean_op in {"undo", "reset"}:
        return {"operation": clean_op}, None

    clean: dict[str, Any] = {"operation": "apply"}
    theme_input = {
        k: params[k]
        for k in ("accent", "accent_hex", "accent_ink_hex", "surface_hex", "ink_hex", "density", "font_scale_step")
        if k in params and params[k] is not None
    }
    if "accent_preset" in params and params["accent_preset"] is not None and "accent" not in theme_input:
        theme_input["accent"] = params["accent_preset"]
    if theme_input:
        validated_theme, err = validate_theme_override(theme_input)
        if err:
            return None, f"app_customize: {err}"
        clean.update(validated_theme or {})

    if "theme" in params and params["theme"] is not None:
        t = params["theme"]
        if not isinstance(t, str) or t.strip().lower() not in {"system", "light", "dark"}:
            return None, "app_customize.theme must be one of ['dark', 'light', 'system']."
        clean["theme"] = t.strip().lower()

    if "startup_tab" in params and params["startup_tab"] is not None:
        tab = params["startup_tab"]
        if not isinstance(tab, str) or tab.strip().lower() not in ALLOWED_DESTINATIONS:
            return None, f"app_customize.startup_tab must be one of {sorted(ALLOWED_DESTINATIONS)}."
        clean["startup_tab"] = tab.strip().lower()

    if "greeting" in params and params["greeting"] is not None:
        g = params["greeting"]
        if not _valid_utf8_text(g, max_len=120) or _contains_secret(g):
            return None, "app_customize.greeting must be safe UTF-8 text (<= 120 chars, no secrets)."
        clean["greeting"] = g.strip()

    if "quick_shortcuts" in params and params["quick_shortcuts"] is not None:
        cleaned_sc, sc_err = _validate_quick_shortcuts(params["quick_shortcuts"])
        if sc_err:
            return None, sc_err
        clean["quick_shortcuts"] = cleaned_sc

    if len(clean) == 1:
        return None, "app_customize with operation='apply' must specify at least one customization field."
    return clean, None


def _validate_app_read_state(params: dict[str, Any]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    if "include" not in params or params["include"] is None:
        return {"include": sorted(_READ_STATE_INCLUDES)}, None
    inc = params["include"]
    if not isinstance(inc, list) or not inc:
        return None, f"app_read_state.include must be a non-empty list from {sorted(_READ_STATE_INCLUDES)}."
    out: list[str] = []
    for item in inc:
        if not isinstance(item, str) or item.strip().lower() not in _READ_STATE_INCLUDES:
            return None, f"app_read_state.include entries must be in {sorted(_READ_STATE_INCLUDES)}."
        norm = item.strip().lower()
        if norm not in out:
            out.append(norm)
    return {"include": out}, None


def _validate_app_create_template(params: dict[str, Any]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    title = params.get("title")
    draft = params.get("draft_prompt")
    if not _valid_utf8_text(title, max_len=80) or _contains_secret(title):
        return None, "app_create_template.title must be safe UTF-8 text (<= 80 chars)."
    if not _valid_utf8_text(draft, max_len=1000) or _contains_secret(draft):
        return None, "app_create_template.draft_prompt must be safe UTF-8 text (<= 1000 chars)."
    clean: dict[str, Any] = {"title": title.strip(), "draft_prompt": draft.strip()}
    for key, max_len in (("category", 40), ("icon", 32)):
        if key in params and params[key] is not None:
            val = params[key]
            if not _valid_utf8_text(val, max_len=max_len):
                return None, f"app_create_template.{key} must be valid UTF-8 text (<= {max_len} chars)."
            clean[key] = val.strip()
    return clean, None


def _validate_android_set_timer(params: dict[str, Any]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    dur = params.get("duration_seconds")
    if not isinstance(dur, int) or isinstance(dur, bool) or not (1 <= dur <= 86_400):
        return None, "android_set_timer.duration_seconds must be an integer between 1 and 86400."
    clean: dict[str, Any] = {"duration_seconds": dur}
    if "label" in params and params["label"] is not None:
        lbl = params["label"]
        if not _valid_utf8_text(lbl, max_len=80) or _contains_secret(lbl):
            return None, "android_set_timer.label must be safe UTF-8 text (<= 80 chars)."
        clean["label"] = lbl.strip()
    return clean, None


def _validate_android_share_sheet(params: dict[str, Any]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    text = params.get("text")
    if not _valid_utf8_text(text, max_len=8_000):
        return None, "android_share_sheet.text must be non-empty UTF-8 text (<= 8000 chars)."
    clean: dict[str, Any] = {"text": text.strip()}
    if "subject" in params and params["subject"] is not None:
        subj = params["subject"]
        if not _valid_utf8_text(subj, max_len=160):
            return None, "android_share_sheet.subject must be valid UTF-8 text (<= 160 chars)."
        clean["subject"] = subj.strip()
    mime = params.get("mime_type", "text/plain")
    if not isinstance(mime, str) or mime.strip().lower() not in _SHARE_MIME_TYPES:
        return None, f"android_share_sheet.mime_type must be one of {sorted(_SHARE_MIME_TYPES)}."
    clean["mime_type"] = mime.strip().lower()
    return clean, None


def _validate_android_open_app(params: dict[str, Any]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    pkg = params.get("package_name")
    link = params.get("deep_link")
    if pkg is None and link is None:
        return None, "android_open_app requires package_name or deep_link."
    clean: dict[str, Any] = {}
    if pkg is not None:
        if not isinstance(pkg, str) or not _PACKAGE_NAME_RE.match(pkg.strip()) or len(pkg.strip()) > 128:
            return None, "android_open_app.package_name must be a valid Android package name."
        clean["package_name"] = pkg.strip()
    if link is not None:
        if not isinstance(link, str) or not _SAFE_DEEP_LINK_RE.match(link.strip()):
            return None, "android_open_app.deep_link must use https://, geo:, mailto:, or tel:."
        clean["deep_link"] = link.strip()
    if "label" in params and params["label"] is not None:
        lbl = params["label"]
        if not _valid_utf8_text(lbl, max_len=80):
            return None, "android_open_app.label must be valid UTF-8 text (<= 80 chars)."
        clean["label"] = lbl.strip()
    return clean, None


def _validate_android_sync_shortcuts(params: dict[str, Any]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    clean: dict[str, Any] = {}
    if "tile_label" in params and params["tile_label"] is not None:
        tl = params["tile_label"]
        if not _valid_utf8_text(tl, max_len=32) or _contains_secret(tl):
            return None, "android_sync_shortcuts.tile_label must be safe UTF-8 text (<= 32 chars)."
        clean["tile_label"] = tl.strip()
    if "widget_summary" in params and params["widget_summary"] is not None:
        ws = params["widget_summary"]
        if not _valid_utf8_text(ws, max_len=120) or _contains_secret(ws):
            return None, "android_sync_shortcuts.widget_summary must be safe home-screen text (<= 120 chars)."
        clean["widget_summary"] = ws.strip()
    if "shortcuts" in params and params["shortcuts"] is not None:
        raw_sc = params["shortcuts"]
        if not isinstance(raw_sc, list) or len(raw_sc) > 4:
            return None, "android_sync_shortcuts.shortcuts must be an array of at most 4 shortcuts."
        items: list[dict[str, str]] = []
        for idx, item in enumerate(raw_sc):
            if not isinstance(item, dict):
                return None, f"android_sync_shortcuts.shortcuts[{idx}] must be an object."
            sid = item.get("id")
            lbl = item.get("label")
            target = item.get("target", "discussions")
            if not isinstance(sid, str) or not _SAFE_SHORTCUT_ID_RE.match(sid.strip()):
                return None, f"android_sync_shortcuts.shortcuts[{idx}].id must be a safe slug."
            if not _valid_utf8_text(lbl, max_len=40) or _contains_secret(lbl):
                return None, f"android_sync_shortcuts.shortcuts[{idx}].label must be safe text (<= 40 chars)."
            if not isinstance(target, str) or target.strip().lower() not in _NAV_TARGETS:
                return None, f"android_sync_shortcuts.shortcuts[{idx}].target must be a valid destination."
            items.append({"id": sid.strip(), "label": lbl.strip(), "target": target.strip().lower()})
        clean["shortcuts"] = items
    if not clean:
        return None, "android_sync_shortcuts requires at least one of tile_label, shortcuts, or widget_summary."
    return clean, None


def _validate_android_proactive_nudge(params: dict[str, Any]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    title = params.get("title")
    body = params.get("body")
    if not _valid_utf8_text(title, max_len=64) or _contains_secret(title):
        return None, "android_proactive_nudge.title must be safe UTF-8 text (<= 64 chars, no secrets)."
    if not _valid_utf8_text(body, max_len=160) or _contains_secret(body):
        return None, "android_proactive_nudge.body must be safe UTF-8 text (<= 160 chars, no secrets)."
    if params.get("respect_quiet_hours") is False:
        return None, "android_proactive_nudge must respect quiet hours (respect_quiet_hours cannot be false)."
    min_interval = params.get("min_interval_minutes", 30)
    if not isinstance(min_interval, int) or isinstance(min_interval, bool) or not (15 <= min_interval <= 1440):
        return None, "android_proactive_nudge.min_interval_minutes must be an integer between 15 and 1440."
    clean: dict[str, Any] = {
        "title": title.strip(),
        "body": body.strip(),
        "respect_quiet_hours": True,
        "min_interval_minutes": min_interval,
    }
    if "category" in params and params["category"] is not None:
        cat = params["category"]
        if not isinstance(cat, str) or cat.strip().lower() not in _NUDGE_CATEGORIES:
            return None, f"android_proactive_nudge.category must be one of {sorted(_NUDGE_CATEGORIES)}."
        clean["category"] = cat.strip().lower()
    if "target_screen" in params and params["target_screen"] is not None:
        ts = params["target_screen"]
        if not isinstance(ts, str) or ts.strip().lower() not in _NAV_TARGETS:
            return None, f"android_proactive_nudge.target_screen must be one of {sorted(_NAV_TARGETS)}."
        clean["target_screen"] = ts.strip().lower()
    return clean, None


_DEEP_PARAM_VALIDATORS: dict[str, Callable[[dict[str, Any]], tuple[Optional[dict[str, Any]], Optional[str]]]] = {
    "app_navigate": _validate_app_navigate,
    "app_customize": _validate_app_customize,
    "app_read_state": _validate_app_read_state,
    "app_create_template": _validate_app_create_template,
    "android_set_timer": _validate_android_set_timer,
    "android_share_sheet": _validate_android_share_sheet,
    "android_open_app": _validate_android_open_app,
    "android_sync_shortcuts": _validate_android_sync_shortcuts,
    "android_proactive_nudge": _validate_android_proactive_nudge,
}


def validate_device_action_params(action_type: str, params: dict[str, Any]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Validate action-specific parameter schemas for deep Android 16 actions (fail-closed on unknown keys)."""
    validator = _DEEP_PARAM_VALIDATORS.get(action_type)
    if validator is None:
        return params, None
    meta = DEVICE_ACTION_CATALOG.get(action_type, {})
    allowed = set(meta.get("allowed_params") or ())
    unknown = sorted(set(params) - allowed)
    if unknown:
        return None, f"{action_type} received unsupported params: {', '.join(unknown)}."
    return validator(params)

ALLOWED_DEVICE_RESULT_STATUSES = frozenset({"completed", "denied", "unavailable"})
MAX_DEVICE_PARAMS_BYTES = 32_768
MAX_DEVICE_RESULT_BYTES = 131_072
MAX_REASON_LEN = 500
MIN_DEVICE_TIMEOUT_SECONDS = 1.0
MAX_DEVICE_TIMEOUT_SECONDS = 300.0
DEFAULT_DEVICE_TIMEOUT_SECONDS = 120.0


def configured_device_timeout(override: Optional[float] = None) -> float:
    raw = override
    if raw is None:
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
            mobile_cfg = cfg.get("mobile") if isinstance(cfg, dict) else None
            if isinstance(mobile_cfg, dict) and mobile_cfg.get("device_bridge_timeout_seconds") is not None:
                raw = mobile_cfg["device_bridge_timeout_seconds"]
        except (ImportError, OSError, ValueError) as exc:
            logger.debug("Falling back to default device bridge timeout: %s", exc)
            raw = None
    if raw is None:
        raw = DEFAULT_DEVICE_TIMEOUT_SECONDS
    try:
        val = float(raw)
    except (TypeError, ValueError):
        val = DEFAULT_DEVICE_TIMEOUT_SECONDS
    return max(MIN_DEVICE_TIMEOUT_SECONDS, min(val, MAX_DEVICE_TIMEOUT_SECONDS))


def validate_device_capabilities(value: Any) -> tuple[Optional[list[str]], Optional[str]]:
    """Validate optional ``device_capabilities`` array on ``POST /v1/runs``."""
    if value is None:
        return None, None
    if not isinstance(value, list):
        return None, "device_capabilities must be an array of capability strings."
    seen: list[str] = []
    seen_set: set[str] = set()
    for idx, item in enumerate(value):
        if not isinstance(item, str) or item not in ALLOWED_DEVICE_CAPABILITIES:
            return None, (
                f"device_capabilities[{idx}] must be one of {sorted(ALLOWED_DEVICE_CAPABILITIES)}."
            )
        if item not in seen_set:
            seen_set.add(item)
            seen.append(item)
    return seen, None


# ── Per-Run Isolated Scope (Expires in Copied Contexts) ──────────────────────

@dataclass
class _BoundDeviceScope:
    owner: Any
    capabilities: frozenset[str]
    run_id: str
    profile_key: str
    broker: Optional[DeviceBridgeBroker]
    timeout_seconds: Optional[float] = None


_DEVICE_TOOL_NAME = "request_device_action"

_active_device_scope: ContextVar[Optional[_BoundDeviceScope]] = ContextVar(
    "mobile_device_bridge_scope", default=None
)


def _toolsets_include_tool(toolset_names: Any, tool_name: str) -> bool:
    from toolsets import resolve_toolset

    return any(tool_name in resolve_toolset(ts) for ts in toolset_names)


def _owner_allows_device_tool(owner: Any) -> bool:
    """True when ``owner`` carries a mutable tool surface permitted to expose ``request_device_action``."""
    if not isinstance(getattr(owner, "tools", None), list) or not isinstance(
        getattr(owner, "valid_tool_names", None), set
    ):
        return False
    from tools.hosted_tool_policy import hosted_tool_refusal, run_tool_refusal
    from toolsets import agent_tool_drops

    if hosted_tool_refusal(_DEVICE_TOOL_NAME) is not None:
        return False
    if run_tool_refusal(_DEVICE_TOOL_NAME, owner) is not None:
        return False
    if _DEVICE_TOOL_NAME in agent_tool_drops(owner):
        return False
    selection = getattr(owner, "_tool_snapshot_selection", None)
    if isinstance(selection, tuple) and len(selection) == 2:
        enabled, disabled = selection
        if (
            enabled is not None
            and "all" not in enabled
            and "*" not in enabled
            and not _toolsets_include_tool(enabled, _DEVICE_TOOL_NAME)
        ):
            return False
        if disabled and _toolsets_include_tool(disabled, _DEVICE_TOOL_NAME):
            return False
    else:
        disabled_ts = getattr(owner, "disabled_toolsets", None)
        if disabled_ts and _toolsets_include_tool(disabled_ts, _DEVICE_TOOL_NAME):
            return False
    return True


def _attach_device_tool(owner: Any) -> bool:
    if not _owner_allows_device_tool(owner) or _DEVICE_TOOL_NAME in owner.valid_tool_names:
        return False
    owner.tools.append({"type": "function", "function": copy.deepcopy(REQUEST_DEVICE_ACTION_SCHEMA)})
    owner.valid_tool_names.add(_DEVICE_TOOL_NAME)
    return True


def _detach_device_tool(owner: Any) -> None:
    owner.tools[:] = [
        tool for tool in owner.tools
        if not (isinstance(tool, dict) and (tool.get("function") or {}).get("name") == _DEVICE_TOOL_NAME)
    ]
    owner.valid_tool_names.discard(_DEVICE_TOOL_NAME)


@contextmanager
def device_bridge_scope(
    owner: Any,
    *,
    capabilities: Optional[list[str] | frozenset[str]],
    run_id: str,
    profile_key: str,
    broker: Optional[DeviceBridgeBroker],
    timeout_seconds: Optional[float] = None,
):
    """Bind per-run Android device capabilities and dynamically attach ``request_device_action`` to ``owner``."""
    caps = frozenset(c for c in (capabilities or ()) if c in ALLOWED_DEVICE_CAPABILITIES)
    bound = (
        _BoundDeviceScope(
            owner=owner,
            capabilities=caps,
            run_id=run_id,
            profile_key=profile_key,
            broker=broker,
            timeout_seconds=timeout_seconds,
        )
        if caps and broker is not None
        else None
    )
    token = _active_device_scope.set(bound)
    attached = _attach_device_tool(owner) if bound is not None and owner is not None else False
    try:
        yield
    finally:
        if attached:
            _detach_device_tool(owner)
        if bound is not None:
            # Copied execution contexts share this object; scrub it so delayed tasks cannot reuse it.
            bound.capabilities = frozenset()
            bound.broker = None
            bound.owner = None
            bound.run_id = ""
            bound.profile_key = ""
        _active_device_scope.reset(token)


def current_device_capabilities() -> frozenset[str]:
    active = _active_device_scope.get()
    return active.capabilities if active is not None and active.broker is not None else frozenset()


# ── Thread-Safe Pause/Resume Broker ──────────────────────────────────────────

@dataclass
class _PendingDeviceRequest:
    request_id: str
    run_id: str
    profile_key: str
    action_type: str
    params: dict[str, Any]
    reason: str
    created_at: float
    expires_at: float
    event: threading.Event
    result_status: Optional[str] = None
    result_data: Optional[dict[str, Any]] = None
    settled: bool = False

    def public_view(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "run_id": self.run_id,
            "action_type": self.action_type,
            "params": self.params,
            "reason": self.reason,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }


class DeviceBridgeBroker:
    """Thread-safe pause/resume coordinator isolated by ``(profile_key, run_id)``."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._pending: dict[tuple[str, str], dict[str, _PendingDeviceRequest]] = {}
        self._listeners: dict[tuple[str, str], Callable[[str, dict[str, Any]], None]] = {}

    def register_run_listener(
        self,
        profile_key: str,
        run_id: str,
        callback: Callable[[str, dict[str, Any]], None],
    ) -> None:
        with self._lock:
            self._listeners[(profile_key, run_id)] = callback

    def unregister_run(self, profile_key: str, run_id: str) -> None:
        self.cancel_run(profile_key, run_id)
        with self._lock:
            self._listeners.pop((profile_key, run_id), None)
            self._pending.pop((profile_key, run_id), None)

    def list_pending(self, profile_key: str, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            bucket = self._pending.get((profile_key, run_id), {})
            return [entry.public_view() for entry in bucket.values() if not entry.settled]

    def _emit(self, profile_key: str, run_id: str, event_name: str, payload: dict[str, Any]) -> None:
        with self._lock:
            listener = self._listeners.get((profile_key, run_id))
        if listener is None:
            return
        try:
            listener(event_name, payload)
        except Exception:
            logger.warning("device bridge listener failed for run=%s event=%s", run_id, event_name, exc_info=True)

    def request_action(
        self,
        *,
        profile_key: str,
        run_id: str,
        action_type: str,
        params: dict[str, Any],
        reason: str,
        timeout_seconds: Optional[float] = None,
    ) -> dict[str, Any]:
        timeout = configured_device_timeout(timeout_seconds)
        now = time.time()
        request_id = f"dev_{uuid.uuid4().hex[:16]}"
        entry = _PendingDeviceRequest(
            request_id=request_id,
            run_id=run_id,
            profile_key=profile_key,
            action_type=action_type,
            params=params,
            reason=reason,
            created_at=now,
            expires_at=now + timeout,
            event=threading.Event(),
        )
        key = (profile_key, run_id)
        with self._lock:
            self._pending.setdefault(key, {})[request_id] = entry

        logger.info("device_bridge requested run=%s req=%s action=%s", run_id, request_id, action_type)
        self._emit(profile_key, run_id, "device.requested", entry.public_view())

        signaled = entry.event.wait(timeout=timeout)
        with self._lock:
            bucket = self._pending.get(key)
            if bucket is not None:
                bucket.pop(request_id, None)
                if not bucket:
                    self._pending.pop(key, None)
            if not signaled and not entry.settled:
                entry.settled = True
                entry.result_status = "timed_out"
                timed_out = True
            else:
                timed_out = False
            status = entry.result_status or "timed_out"
            data = entry.result_data

        if timed_out:
            logger.info("device_bridge timed_out run=%s req=%s action=%s", run_id, request_id, action_type)
            self._emit(
                profile_key,
                run_id,
                "device.settled",
                {"request_id": request_id, "action_type": action_type, "status": "timed_out"},
            )
            return {
                "ok": False,
                "request_id": request_id,
                "action_type": action_type,
                "status": "timed_out",
                "error": "device_request_timed_out",
            }

        return {
            "ok": status == "completed",
            "request_id": request_id,
            "action_type": action_type,
            "status": status,
            **({"data": data} if data is not None else {}),
            **({"error": f"device_action_{status}"} if status != "completed" else {}),
        }

    def resolve_request(
        self,
        *,
        profile_key: str,
        run_id: str,
        request_id: str,
        status: str,
        data: Optional[dict[str, Any]] = None,
    ) -> tuple[str, Optional[dict[str, Any]]]:
        """Resolve one pending request. Returns ``(outcome, public_summary)`` where outcome is
        ``'resolved'``, ``'not_found'``, or ``'already_settled'``."""
        key = (profile_key, run_id)
        with self._lock:
            bucket = self._pending.get(key)
            if not bucket or request_id not in bucket:
                return "not_found", None
            entry = bucket[request_id]
            if entry.settled:
                return "already_settled", entry.public_view()
            entry.settled = True
            entry.result_status = status
            entry.result_data = data
            bucket.pop(request_id, None)
            if not bucket:
                self._pending.pop(key, None)
            action_type = entry.action_type
            entry.event.set()

        logger.info("device_bridge resolved run=%s req=%s action=%s status=%s", run_id, request_id, action_type, status)
        settled_payload = {"request_id": request_id, "action_type": action_type, "status": status}
        self._emit(profile_key, run_id, "device.settled", settled_payload)
        return "resolved", settled_payload

    def cancel_run(self, profile_key: str, run_id: str) -> int:
        key = (profile_key, run_id)
        cancelled: list[_PendingDeviceRequest] = []
        with self._lock:
            bucket = self._pending.pop(key, None) or {}
            for entry in bucket.values():
                if not entry.settled:
                    entry.settled = True
                    entry.result_status = "cancelled"
                    entry.event.set()
                    cancelled.append(entry)
        for entry in cancelled:
            self._emit(
                profile_key,
                run_id,
                "device.settled",
                {"request_id": entry.request_id, "action_type": entry.action_type, "status": "cancelled"},
            )
        return len(cancelled)


# ── Agent Tool Implementation & Registration ─────────────────────────────────

REQUEST_DEVICE_ACTION_SCHEMA = {
    "name": "request_device_action",
    "description": (
        "Request an explicitly user-approved Android device capability for the current run "
        "(e.g., camera capture, coarse location, calendar draft, alarm/timer, maps handoff, share sheet, "
        "clipboard read, dial/email draft). Pauses the run until the user approves, declines, or times out."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action_type": {
                "type": "string",
                "enum": sorted(ALLOWED_DEVICE_CAPABILITIES),
                "description": "Device capability to invoke (must be advertised by the current run).",
            },
            "reason": {
                "type": "string",
                "description": "Clear user-facing explanation of why this device action is requested.",
            },
            "params": {
                "type": "object",
                "description": "Structured parameters for the requested device action.",
            },
        },
        "required": ["action_type", "reason"],
    },
}


_DEVICE_ACTION_EXAMPLE = {
    "action_type": "alarm_timer",
    "reason": "Régler un minuteur de 10 minutes",
    "params": {"duration_seconds": 600, "label": "Pause"},
}


def request_device_action_tool(
    action_type: Any,
    reason: Any,
    params: Any = None,
) -> str:
    """Execute ``request_device_action`` against the current run's isolated device scope."""
    scope = _active_device_scope.get()
    if scope is None or scope.broker is None:
        return json.dumps({
            "ok": False,
            "error": "device_bridge_unavailable",
            "message": "No active mobile device bridge is bound to this run.",
            "hint": "Only call request_device_action when the client advertised device_capabilities on this run.",
        }, ensure_ascii=False)
    norm_action = (
        DEVICE_ACTION_ALIASES.get(action_type.strip().lower(), action_type.strip().lower())
        if isinstance(action_type, str)
        else action_type
    )
    if not isinstance(norm_action, str) or norm_action not in ALLOWED_DEVICE_CAPABILITIES:
        return json.dumps({
            "ok": False,
            "error": "invalid_action_type",
            "allowed": sorted(ALLOWED_DEVICE_CAPABILITIES),
            "hint": f"Choose action_type from advertised capabilities: {sorted(scope.capabilities)}.",
            "example": _DEVICE_ACTION_EXAMPLE,
        }, ensure_ascii=False)
    if norm_action not in scope.capabilities:
        return json.dumps({
            "ok": False,
            "error": "capability_not_advertised",
            "action_type": norm_action,
            "advertised": sorted(scope.capabilities),
            "hint": f"This phone run only advertised {sorted(scope.capabilities)}; use one of those or answer without device access.",
        }, ensure_ascii=False)
    if not _valid_utf8_text(reason, max_len=MAX_REASON_LEN):
        return json.dumps({
            "ok": False,
            "error": f"reason must be non-empty UTF-8 text (<= {MAX_REASON_LEN} chars).",
            "hint": "Provide a short, plain-language reason shown to the user before they approve.",
            "example": _DEVICE_ACTION_EXAMPLE,
        }, ensure_ascii=False)
    clean_params = {} if params is None else params
    if not isinstance(clean_params, dict):
        return json.dumps({
            "ok": False,
            "error": "params must be a JSON object.",
            "hint": "Pass params as a flat JSON object (or omit it).",
            "example": _DEVICE_ACTION_EXAMPLE,
        }, ensure_ascii=False)
    scan_err = _scan_strings_safe(clean_params)
    if scan_err:
        return json.dumps({
            "ok": False,
            "error": scan_err,
            "hint": "Remove any script tags or unsafe URIs from params and retry.",
            "example": _DEVICE_ACTION_EXAMPLE,
        }, ensure_ascii=False)
    encoded = json.dumps(clean_params, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_DEVICE_PARAMS_BYTES:
        return json.dumps({
            "ok": False,
            "error": f"params exceeds maximum size of {MAX_DEVICE_PARAMS_BYTES} bytes.",
            "hint": "Trim params to only the essential fields for the native Android confirmation screen.",
        }, ensure_ascii=False)
    validated_params, param_err = validate_device_action_params(norm_action, clean_params)
    if param_err:
        return json.dumps({
            "ok": False,
            "error": param_err,
            "action_type": norm_action,
            "hint": "Provide valid allowlisted parameters matching the device action schema.",
        }, ensure_ascii=False)

    outcome = scope.broker.request_action(
        profile_key=scope.profile_key,
        run_id=scope.run_id,
        action_type=norm_action,
        params=validated_params if validated_params is not None else clean_params,
        reason=reason.strip(),
        timeout_seconds=scope.timeout_seconds,
    )
    return json.dumps(outcome, ensure_ascii=False)


@no_cache_check_fn
def _device_bridge_available() -> bool:
    return bool(current_device_capabilities())


registry.register(
    name="request_device_action",
    toolset="mobile_device",
    schema=REQUEST_DEVICE_ACTION_SCHEMA,
    handler=lambda args, **kw: request_device_action_tool(
        args.get("action_type"),
        args.get("reason"),
        args.get("params"),
    ),
    check_fn=_device_bridge_available,
)
