"""Strict fail-closed parser, sanitizer, and LLM context formatter for live mobile app presence.

Tracks client-reported live screen/thread/run state, read-without-reply signals, and strictly
opt-in continuous location snapshots while clamping ranges, scrubbing secrets/injections, and
never retaining or echoing raw unapproved user content.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Optional

from tools.mobile_ui_patch import _FORBIDDEN_PAYLOAD_RE, _WIDGET_SECRET_RE

# ── Single Source of Truth Constants ─────────────────────────────────────────

ALLOWED_SCREENS = frozenset({
    "discussions",
    "atelier",
    "cerveau",
    "activite",
    "thread",
    "memory",
    "notes",
    "shared_with_me",
    "search",
    "skills",
    "workspace_files",
    "profile_defaults",
    "ambient",
    "doctor",
    "templates",
    "palette",
    "presence",
    "settings",
})

ALLOWED_MOVING_STATES = frozenset({"stationary", "walking", "transit", "unknown"})
ALLOWED_LOCATION_MODES = frozenset({"coarse", "precise"})
ALLOWED_THEME_MODES = frozenset({"system", "light", "dark"})
ALLOWED_DENSITIES = frozenset({"compact", "comfortable", "spacious"})

MAX_PRESENCE_BYTES = 16_384
MAX_SESSION_ID_LEN = 128
MAX_PLACE_LABEL_LEN = 80
MAX_UNREPLIED_ITEMS = 8
MAX_SOURCE_LABEL_LEN = 48
MAX_SENDER_HINT_LEN = 64
MAX_ACTIVE_RUN_COUNT = 32
MAX_DRAFT_CHARS = 100_000
MAX_LOCATION_AGE_S = 86_400
MAX_UNREPLIED_AGE_S = 86_400
MAX_ACCURACY_M = 50_000.0
MAX_CONTEXT_CHARS = 1_200

_SAFE_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_SAFE_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
_CONTROL_OR_BIDI_RE = re.compile(r"[\x00-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")
_EXTRA_SECRET_RE = re.compile(
    r"(?i)(?:eyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{6,}|"
    r"\b(?:otp|2fa|passcode|verification\s+code|code\s+de\s+v[ée]rification|cvv|cvc)\b|"
    r"G-\d{5,8})"
)


def _is_real_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def scrub_presence_text(raw: Any, *, max_len: int) -> str:
    """Collapse whitespace, strip control/bidi chars, and redact unsafe payloads or secrets."""
    if not isinstance(raw, str):
        return ""
    cleaned = _CONTROL_OR_BIDI_RE.sub(" ", raw)
    cleaned = " ".join(cleaned.split()).strip()
    if not cleaned:
        return ""
    if _FORBIDDEN_PAYLOAD_RE.search(cleaned) or _WIDGET_SECRET_RE.search(cleaned) or _EXTRA_SECRET_RE.search(cleaned):
        return "[redacted]"
    return cleaned[:max_len].strip()


def _sanitize_read_without_reply(raw: Any) -> Optional[dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    out: dict[str, Any] = {}
    in_app = raw.get("in_app_unread_reply")
    if isinstance(in_app, bool):
        out["in_app_unread_reply"] = in_app
    since_read = raw.get("in_app_seconds_since_read")
    if _is_real_int(since_read) and since_read >= 0:
        out["in_app_seconds_since_read"] = min(since_read, MAX_UNREPLIED_AGE_S)

    raw_notifs = raw.get("notifications")
    if isinstance(raw_notifs, list):
        cleaned_notifs: list[dict[str, Any]] = []
        for entry in raw_notifs[:MAX_UNREPLIED_ITEMS]:
            if not isinstance(entry, dict):
                continue
            source = scrub_presence_text(entry.get("source"), max_len=MAX_SOURCE_LABEL_LEN)
            sender = scrub_presence_text(entry.get("sender_hint"), max_len=MAX_SENDER_HINT_LEN)
            if not source or source == "[redacted]":
                continue
            item: dict[str, Any] = {"source": source}
            if sender and sender != "[redacted]":
                item["sender_hint"] = sender
            age_s = entry.get("age_s")
            if _is_real_int(age_s) and age_s >= 0:
                item["age_s"] = min(age_s, MAX_UNREPLIED_AGE_S)
            if isinstance(entry.get("replyable"), bool):
                item["replyable"] = entry["replyable"]
            cleaned_notifs.append(item)
        if cleaned_notifs:
            out["notifications"] = cleaned_notifs
    return out or None


def _sanitize_continuous_location(raw: Any) -> Optional[dict[str, Any]]:
    """Fail-closed continuous location parser: drops all coordinates unless ``opted_in`` is strictly True."""
    if not isinstance(raw, dict):
        return None
    if raw.get("opted_in") is not True:
        return None
    lat = raw.get("lat")
    lon = raw.get("lon")
    if not (_is_finite_number(lat) and _is_finite_number(lon)):
        return None
    lat_f = float(lat)
    lon_f = float(lon)
    if not (-90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0):
        return None

    raw_mode = raw.get("mode", "coarse")
    mode = raw_mode.strip().lower() if isinstance(raw_mode, str) else "coarse"
    if mode not in ALLOWED_LOCATION_MODES:
        mode = "coarse"
    precision = 2 if mode == "coarse" else 5

    out: dict[str, Any] = {
        "opted_in": True,
        "mode": mode,
        "lat": round(lat_f, precision),
        "lon": round(lon_f, precision),
    }
    acc = raw.get("accuracy_m")
    if _is_finite_number(acc) and float(acc) >= 0.0:
        out["accuracy_m"] = round(min(float(acc), MAX_ACCURACY_M), 1)
    age_s = raw.get("age_s")
    if _is_real_int(age_s) and age_s >= 0:
        out["age_s"] = min(age_s, MAX_LOCATION_AGE_S)
    moving = raw.get("moving_state")
    if isinstance(moving, str) and moving.strip().lower() in ALLOWED_MOVING_STATES:
        out["moving_state"] = moving.strip().lower()
    place = scrub_presence_text(raw.get("place_label"), max_len=MAX_PLACE_LABEL_LEN)
    if place and place != "[redacted]":
        out["place_label"] = place
    return out


def _sanitize_customization_summary(raw: Any) -> Optional[dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    out: dict[str, Any] = {}
    theme = raw.get("theme")
    if isinstance(theme, str) and theme.strip().lower() in ALLOWED_THEME_MODES:
        out["theme"] = theme.strip().lower()
    density = raw.get("density")
    if isinstance(density, str) and density.strip().lower() in ALLOWED_DENSITIES:
        out["density"] = density.strip().lower()
    preset = raw.get("accent_preset")
    if isinstance(preset, str) and _SAFE_SLUG_RE.match(preset.strip().lower()):
        out["accent_preset"] = preset.strip().lower()
    return out or None


def sanitize_app_presence(raw: Any) -> Optional[dict[str, Any]]:
    """Validate and sanitize a client-provided ``app_presence`` snapshot. Fails closed on bad data."""
    if not isinstance(raw, dict) or not raw:
        return None
    try:
        encoded = json.dumps(raw, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError):
        return None
    if len(encoded) > MAX_PRESENCE_BYTES:
        return None

    clean: dict[str, Any] = {}

    screen = raw.get("current_screen")
    if isinstance(screen, str):
        norm_screen = screen.strip().lower()
        if norm_screen in ALLOWED_SCREENS:
            clean["current_screen"] = norm_screen

    sid = raw.get("active_session_id")
    if isinstance(sid, str):
        norm_sid = sid.strip()
        if _SAFE_SESSION_ID_RE.match(norm_sid):
            clean["active_session_id"] = norm_sid

    run_count = raw.get("active_run_count")
    if _is_real_int(run_count) and run_count >= 0:
        clean["active_run_count"] = min(run_count, MAX_ACTIVE_RUN_COUNT)

    draft_chars = raw.get("draft_chars")
    if _is_real_int(draft_chars) and draft_chars >= 0:
        clean["draft_chars"] = min(draft_chars, MAX_DRAFT_CHARS)

    rwr = _sanitize_read_without_reply(raw.get("read_without_reply"))
    if rwr is not None:
        clean["read_without_reply"] = rwr

    loc = _sanitize_continuous_location(raw.get("continuous_location"))
    if loc is not None:
        clean["continuous_location"] = loc

    cust = _sanitize_customization_summary(raw.get("customization_summary"))
    if cust is not None:
        clean["customization_summary"] = cust

    return clean or None


def format_app_presence_context(raw_or_clean: Any) -> str:
    """Format a sanitized ``app_presence`` snapshot as a concise, injection-safe system context block."""
    clean = sanitize_app_presence(raw_or_clean)
    if not clean:
        return ""

    lines: list[str] = ["<mobile_app_presence>"]
    if "current_screen" in clean:
        lines.append(f"screen={clean['current_screen']}")
    if "active_session_id" in clean:
        lines.append(f"active_session_id={clean['active_session_id']}")
    if "active_run_count" in clean:
        lines.append(f"active_run_count={clean['active_run_count']}")
    if "draft_chars" in clean:
        lines.append(f"draft_chars={clean['draft_chars']}")

    rwr = clean.get("read_without_reply")
    if isinstance(rwr, dict):
        parts: list[str] = []
        if rwr.get("in_app_unread_reply") is True:
            age = rwr.get("in_app_seconds_since_read")
            parts.append("in_app_read_without_reply=true" + (f" ({age}s)" if age is not None else ""))
        notifs = rwr.get("notifications") or []
        if notifs:
            summaries = []
            for n in notifs:
                src = n["source"]
                who = f"/{n['sender_hint']}" if "sender_hint" in n else ""
                age = f" {n['age_s']}s" if "age_s" in n else ""
                summaries.append(f"{src}{who}{age}")
            parts.append("unreplied_notifications=[" + "; ".join(summaries) + "]")
        if parts:
            lines.append("read_without_reply: " + ", ".join(parts))

    loc = clean.get("continuous_location")
    if isinstance(loc, dict):
        loc_parts = [f"mode={loc['mode']}", f"lat={loc['lat']}", f"lon={loc['lon']}"]
        if "accuracy_m" in loc:
            loc_parts.append(f"accuracy_m={loc['accuracy_m']}")
        if "moving_state" in loc:
            loc_parts.append(f"moving={loc['moving_state']}")
        if "age_s" in loc:
            loc_parts.append(f"age_s={loc['age_s']}")
        if "place_label" in loc:
            loc_parts.append(f"place={loc['place_label']}")
        lines.append("location(opt_in): " + ", ".join(loc_parts))

    cust = clean.get("customization_summary")
    if isinstance(cust, dict):
        cust_parts = [f"{k}={v}" for k, v in sorted(cust.items())]
        lines.append("customization: " + ", ".join(cust_parts))

    lines.append("</mobile_app_presence>")
    rendered = "\n".join(lines)
    if len(rendered) > MAX_CONTEXT_CHARS:
        suffix = "\n</mobile_app_presence>"
        rendered = rendered[: MAX_CONTEXT_CHARS - len(suffix)].rstrip() + suffix
    return rendered
