"""Single Source of Truth capability-driven prompt builder for mobile-originated runs.

Generates a concise, zero-jargon companion guidance block strictly from the capabilities
advertised on the current run (``device_capabilities``) so Hermes knows every native
mobile feature available on the user's phone, proactively removes friction, and never
expects the user to know commands, tool names, or AI jargon.

Injected via ``run_instruction_scope`` so it inherits full per-run isolation (never
persisted into session defaults, never leaked to background reviewers or resumed Codex
threads, and completely omitted when a run does not advertise mobile capabilities).
"""

from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any, Optional

from tools.mobile_app_presence import format_app_presence_context
from tools.mobile_device_bridge import ALLOWED_DEVICE_CAPABILITIES

ALLOWED_USER_LANGUAGES = frozenset({"fr", "en"})
_LANGUAGE_LABELS = {"fr": "French (fr)", "en": "English (en)"}

# Single source of truth: capability token -> concise 1-line tool-call pattern for any LLM.
_CAPABILITY_GUIDANCE: dict[str, str] = {
    "camera": 'camera -> request_device_action(action_type="camera", reason="...")',
    "location_coarse": 'location_coarse -> request_device_action(action_type="location_coarse", reason="...")',
    "nearby_osm": 'nearby_osm -> request_device_action(action_type="nearby_osm", reason="...", params={"category":"cafe"})',
    "calendar_draft": 'calendar_draft -> request_device_action(action_type="calendar_draft", reason="...", params={"title":"...","start":"..."})',
    "alarm_timer": 'alarm_timer -> request_device_action(action_type="alarm_timer", reason="...", params={"duration_seconds":600})',
    "maps_handoff": 'maps_handoff -> request_device_action(action_type="maps_handoff", reason="...", params={"query":"..."})',
    "map_navigation": 'map_navigation -> request_device_action(action_type="map_navigation", reason="...", params={"destination":"...","mode":"walk|drive|transit"})',
    "share_sheet": 'share_sheet -> request_device_action(action_type="share_sheet", reason="...", params={"text":"..."})',
    "clipboard_read": 'clipboard_read -> request_device_action(action_type="clipboard_read", reason="...")',
    "clipboard_write": 'clipboard_write -> request_device_action(action_type="clipboard_write", reason="...", params={"text":"..."})',
    "dial_draft": 'dial_draft -> request_device_action(action_type="dial_draft", reason="...", params={"phone":"..."})',
    "sms_draft": 'sms_draft -> request_device_action(action_type="sms_draft", reason="...", params={"phone":"...","body":"..."})',
    "email_draft": 'email_draft -> request_device_action(action_type="email_draft", reason="...", params={"to":"...","subject":"...","body":"..."})',
    "contacts_pick": 'contacts_pick -> request_device_action(action_type="contacts_pick", reason="...")',
    "contact_draft": 'contact_draft -> request_device_action(action_type="contact_draft", reason="...", params={"name":"...","phone":"...","email":"..."})',
    "notifications_digest": 'notifications_digest -> request_device_action(action_type="notifications_digest", reason="...", params={"limit":20})',
    "notification_reply": 'notification_reply -> request_device_action(action_type="notification_reply", reason="...", params={"key":"...","reply_text":"..."})',
    "media_control": 'media_control -> request_device_action(action_type="media_control", reason="...", params={"command":"status|play|pause|next|previous"})',
    "calendar_read": 'calendar_read -> request_device_action(action_type="calendar_read", reason="...", params={"days_ahead":7})',
    "ui_surfaces_v2": (
        'ui_surfaces_v2 -> mobile_ui_patch(ops=[{"op":"upsert_widget","title":"...","template":"metric|checklist|briefing|status|quick_actions",'
        '"accent":"teal|green|blue|amber|coral","items":[{"label":"...","value":"..."}],"home_screen_safe":true}]) '
        'or op="upsert_surface" (kpi_grid|sparkline|bar_chart|table|form|toggle_group|chip_filter|slider|markdown_card) / "set_theme_override" / interactive_cards (IDs auto-slug from title; never put secrets on widgets)'
    ),
    "app_navigate": 'app_navigate -> request_device_action(action_type="app_navigate", reason="...", params={"target":"discussions|atelier|cerveau|activite|thread|command_palette|template|memory|notes|skills|doctor|presence|profile_defaults"})',
    "app_customize": 'app_customize -> request_device_action(action_type="app_customize", reason="...", params={"operation":"apply|undo|reset","accent":"teal","theme":"system|light|dark","density":"compact|comfortable|spacious"})',
    "app_read_state": 'app_read_state -> request_device_action(action_type="app_read_state", reason="...", params={"include":["screen","active_runs","read_without_reply","location","customization"]})',
    "app_create_template": 'app_create_template -> request_device_action(action_type="app_create_template", reason="...", params={"title":"...","draft_prompt":"..."})',
    "android_set_timer": 'android_set_timer -> request_device_action(action_type="android_set_timer", reason="...", params={"duration_seconds":600,"label":"..."})',
    "android_share_sheet": 'android_share_sheet -> request_device_action(action_type="android_share_sheet", reason="...", params={"text":"...","subject":"..."})',
    "android_open_app": 'android_open_app -> request_device_action(action_type="android_open_app", reason="...", params={"deep_link":"https://example.org"})',
    "android_sync_shortcuts": 'android_sync_shortcuts -> request_device_action(action_type="android_sync_shortcuts", reason="...", params={"shortcuts":[{"id":"quick_note","label":"...","target":"notes"}]})',
    "android_proactive_nudge": 'android_proactive_nudge -> request_device_action(action_type="android_proactive_nudge", reason="...", params={"title":"...","body":"...","category":"reminder|follow_up|briefing|suggestion","respect_quiet_hours":true,"min_interval_minutes":60})',
}

_HEADER = (
    "[Hermes Mobile Companion]\n"
    "You are running inside the user's native Android companion app. "
    "Remove friction proactively: never expect the user to know commands, feature names, or AI jargon "
    "(use everyday plain language instead of tokens, TTFT, TPS, cron, JSON, SSE, or toolsets unless asked). "
    "Always preview sensitive OS actions and UI edits."
)

_SHARED_MOBILE_TOOLS_LINE = (
    '- Push to Activité: mobile_deliver(kind="briefing|notice|card|widget_update|file|ui_patch|device_action|reminder_result", title="...", payload={...}, urgency="silent|normal|high|urgent")\n'
    '- Follow-up chips: pass suggested_actions=[{"label":"...","kind":"pin_widget|save_memory|create_reminder|open_surface|save_skill|prompt"}] (1-3 chips) on mobile_ui_patch.'
)


def _primary_language_tag(raw: str) -> str:
    tag = raw.split(";", 1)[0].strip().lower().replace("_", "-")
    return tag.split("-", 1)[0]


def parse_accept_language(header: Optional[str]) -> Optional[str]:
    """Extract the highest-priority supported language (``fr`` or ``en``) from ``Accept-Language``."""
    if not isinstance(header, str) or not header.strip():
        return None
    ranked: list[tuple[float, int, str]] = []
    for index, part in enumerate(header.split(",")):
        entry = part.strip()
        if not entry:
            continue
        primary = _primary_language_tag(entry)
        if primary not in ALLOWED_USER_LANGUAGES:
            continue
        weight = 1.0
        for param in entry.split(";")[1:]:
            k, _, v = param.strip().partition("=")
            if k.strip().lower() == "q":
                try:
                    weight = float(v.strip())
                except ValueError:
                    weight = 0.0
        if math.isfinite(weight) and 0.0 < weight <= 1.0:
            ranked.append((-weight, index, primary))
    if not ranked:
        return None
    ranked.sort()
    return ranked[0][2]


def normalize_user_language(
    value: Any = None,
    *,
    accept_language: Optional[str] = None,
    x_hermes_lang: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Validate optional explicit ``user_language`` (``fr`` | ``en``) or fall back to ``X-Hermes-Lang`` / ``Accept-Language``."""
    if value is not None:
        if not isinstance(value, str) or not value.strip():
            return None, "user_language must be 'fr' or 'en'."
        primary = _primary_language_tag(value)
        if primary not in ALLOWED_USER_LANGUAGES:
            return None, f"user_language must be one of {sorted(ALLOWED_USER_LANGUAGES)}."
        return primary, None
    if isinstance(x_hermes_lang, str) and x_hermes_lang.strip():
        header_primary = _primary_language_tag(x_hermes_lang)
        if header_primary in ALLOWED_USER_LANGUAGES:
            return header_primary, None
    return parse_accept_language(accept_language), None


def build_mobile_companion_instructions(
    device_capabilities: Optional[Iterable[str]],
    *,
    user_language: Optional[str] = None,
) -> str:
    """Return concise capability- and language-scoped guidance for mobile runs, or ``''``."""
    lang = user_language if user_language in ALLOWED_USER_LANGUAGES else None
    advertised = (
        [cap for cap in dict.fromkeys(device_capabilities) if cap in ALLOWED_DEVICE_CAPABILITIES]
        if device_capabilities
        else []
    )
    if not advertised:
        if lang is None:
            return ""
        return (
            f"[User Language]\n"
            f"Reply, write Smart Notes, and label any surfaces, widgets, or suggested actions in {_LANGUAGE_LABELS[lang]}."
        )
    lines = [_HEADER, "Available on this phone for this turn:"]
    for cap in advertised:
        guidance = _CAPABILITY_GUIDANCE.get(cap)
        if guidance:
            lines.append(f"- {guidance}")
    lines.append(_SHARED_MOBILE_TOOLS_LINE)
    if lang is not None:
        lines.append(
            f"- User language: reply, write Smart Notes, and label all surfaces, widgets, and suggested actions in {_LANGUAGE_LABELS[lang]}."
        )
    return "\n".join(lines)


def combine_run_instructions(
    run_instructions: Optional[str],
    device_capabilities: Optional[Iterable[str]],
    user_language: Optional[str] = None,
    app_presence: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Combine explicit per-run instructions with mobile companion, presence, and language guidance."""
    companion = build_mobile_companion_instructions(device_capabilities, user_language=user_language)
    presence_block = format_app_presence_context(app_presence) if app_presence else ""
    parts = [
        p.strip()
        for p in (run_instructions, companion, presence_block)
        if isinstance(p, str) and p.strip()
    ]
    return "\n\n".join(parts) if parts else None
