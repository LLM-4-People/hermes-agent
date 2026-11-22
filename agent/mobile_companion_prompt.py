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
from typing import Optional

from tools.mobile_device_bridge import ALLOWED_DEVICE_CAPABILITIES

# Single source of truth: capability token -> concise 1-line tool-call pattern for any LLM.
_CAPABILITY_GUIDANCE: dict[str, str] = {
    "camera": 'camera -> request_device_action(action_type="camera", reason="...")',
    "location_coarse": 'location_coarse -> request_device_action(action_type="location_coarse", reason="...")',
    "nearby_osm": 'nearby_osm -> request_device_action(action_type="nearby_osm", reason="...", params={"category":"cafe"})',
    "calendar_draft": 'calendar_draft -> request_device_action(action_type="calendar_draft", reason="...", params={"title":"...","start":"..."})',
    "alarm_timer": 'alarm_timer -> request_device_action(action_type="alarm_timer", reason="...", params={"duration_seconds":600})',
    "maps_handoff": 'maps_handoff -> request_device_action(action_type="maps_handoff", reason="...", params={"query":"..."})',
    "share_sheet": 'share_sheet -> request_device_action(action_type="share_sheet", reason="...", params={"text":"..."})',
    "clipboard_read": 'clipboard_read -> request_device_action(action_type="clipboard_read", reason="...")',
    "dial_draft": 'dial_draft -> request_device_action(action_type="dial_draft", reason="...", params={"phone":"..."})',
    "email_draft": 'email_draft -> request_device_action(action_type="email_draft", reason="...", params={"to":"...","subject":"...","body":"..."})',
    "contacts_pick": 'contacts_pick -> request_device_action(action_type="contacts_pick", reason="...")',
    "ui_surfaces_v2": (
        'ui_surfaces_v2 -> mobile_ui_patch(ops=[{"op":"upsert_widget","title":"...","template":"metric|checklist|briefing|status|quick_actions",'
        '"accent":"teal|green|blue|amber|coral","items":[{"label":"...","value":"..."}],"home_screen_safe":true}]) '
        'or op="upsert_surface" (kpi_grid|sparkline|bar_chart|table|form|toggle_group|chip_filter|slider|markdown_card) / "set_theme_override" (IDs auto-slug from title; never put secrets on widgets)'
    ),
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


def build_mobile_companion_instructions(device_capabilities: Optional[Iterable[str]]) -> str:
    """Return concise capability-scoped guidance for mobile runs, or ``''`` for non-mobile runs."""
    if not device_capabilities:
        return ""
    advertised = [cap for cap in dict.fromkeys(device_capabilities) if cap in ALLOWED_DEVICE_CAPABILITIES]
    if not advertised:
        return ""
    lines = [_HEADER, "Available on this phone for this turn:"]
    for cap in advertised:
        guidance = _CAPABILITY_GUIDANCE.get(cap)
        if guidance:
            lines.append(f"- {guidance}")
    lines.append(_SHARED_MOBILE_TOOLS_LINE)
    return "\n".join(lines)


def combine_run_instructions(
    run_instructions: Optional[str],
    device_capabilities: Optional[Iterable[str]],
) -> Optional[str]:
    """Combine explicit per-run instructions with mobile companion guidance when present."""
    companion = build_mobile_companion_instructions(device_capabilities)
    parts = [p.strip() for p in (run_instructions, companion) if isinstance(p, str) and p.strip()]
    return "\n\n".join(parts) if parts else None
