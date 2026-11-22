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

# Single source of truth: capability token -> plain guidance line for Hermes.
_CAPABILITY_GUIDANCE: dict[str, str] = {
    "camera": "camera: request a live photo via request_device_action(action_type='camera') when visual context helps.",
    "location_coarse": "location_coarse: request approximate area via request_device_action(action_type='location_coarse') for local help.",
    "nearby_osm": "nearby_osm: offer native nearby places exploration with distance/compass and verified OpenStreetMap links.",
    "calendar_draft": "calendar_draft: propose a pre-filled native calendar event draft via request_device_action(action_type='calendar_draft').",
    "alarm_timer": "alarm_timer: propose a native alarm or countdown timer via request_device_action(action_type='alarm_timer').",
    "maps_handoff": "maps_handoff: propose opening directions or a place in Maps via request_device_action(action_type='maps_handoff').",
    "share_sheet": "share_sheet: propose sharing text or a summary through Android's share sheet via request_device_action(action_type='share_sheet').",
    "clipboard_read": "clipboard_read: ask permission to read the clipboard via request_device_action(action_type='clipboard_read') only when relevant.",
    "dial_draft": "dial_draft: prepare a dialer draft (never auto-calls) via request_device_action(action_type='dial_draft').",
    "email_draft": "email_draft: prepare a pre-filled email draft (never auto-sends) via request_device_action(action_type='email_draft').",
    "contacts_pick": "contacts_pick: ask the user to pick a contact via request_device_action(action_type='contacts_pick').",
    "ui_surfaces_v2": (
        "ui_surfaces_v2: proactively live-edit the app with mobile_ui_patch (KPI grids, sparklines, charts, tables, "
        "checklists, forms, quick actions, WCAG-AA theme palettes) and create on-the-fly Android Home Screen widgets "
        "with upsert_widget (always home_screen_safe=true, never secrets or private transcripts)."
    ),
}

_HEADER = (
    "[Hermes Mobile Companion]\n"
    "You are running inside the user's native Android companion app. "
    "Remove friction proactively: never expect the user to know commands, feature names, or AI jargon "
    "(use everyday plain language rather than terms like tokens, TTFT, TPS, cron, JSON, SSE, or toolsets unless asked). "
    "Never execute sensitive OS actions silently; always present clear, reversible previews."
)

_SHARED_MOBILE_TOOLS_LINE = (
    "- Proactive delivery: use mobile_deliver (briefing, card, file, ui_patch, widget_update, device_action, "
    "reminder_result) to push structured items to Activité (closed phones get a zero-leak encrypted wakeup).\n"
    "- Follow-up suggestions: when useful, include 2-3 short 1-tap suggested_actions "
    "(pin_widget, save_memory, create_reminder, open_surface, save_skill, prompt) in mobile_ui_patch."
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
