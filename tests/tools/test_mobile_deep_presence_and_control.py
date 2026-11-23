"""Comprehensive tests for Track 9A: deep Android 16 device bridge actions, interactive_cards, and app_presence."""

from __future__ import annotations

import json
import threading

import pytest

from tools.mobile_app_presence import (
    MAX_ACTIVE_RUN_COUNT,
    MAX_CONTEXT_CHARS,
    MAX_UNREPLIED_ITEMS,
    format_app_presence_context,
    sanitize_app_presence,
)
from tools.mobile_device_bridge import (
    ALLOWED_DEVICE_CAPABILITIES,
    DEEP_ANDROID_ACTIONS,
    DEVICE_ACTION_CATALOG,
    DeviceBridgeBroker,
    device_bridge_scope,
    request_device_action_tool,
    validate_device_action_params,
    validate_device_capabilities,
)
from tools.mobile_ui_patch import (
    MobileUiStore,
    mobile_ui_patch_tool,
    validate_interactive_cards,
)


# ── 1. Live app_presence Sanitizer & Context Formatter ───────────────────────


def test_app_presence_sanitizes_full_opted_in_snapshot_and_formats_context():
    raw = {
        "current_screen": "Thread",
        "active_session_id": "sess_abc-123:tip",
        "active_run_count": 3,
        "draft_chars": 42,
        "read_without_reply": {
            "in_app_unread_reply": True,
            "in_app_seconds_since_read": 180,
            "notifications": [
                {"source": "Signal", "sender_hint": "Marie", "age_s": 300, "replyable": True},
                {"source": "Gmail", "sender_hint": "Work", "age_s": 900, "replyable": False},
            ],
        },
        "continuous_location": {
            "opted_in": True,
            "mode": "coarse",
            "lat": 45.5016889,
            "lon": -73.567256,
            "accuracy_m": 85.44,
            "moving_state": "walking",
            "age_s": 40,
            "place_label": "Plateau-Mont-Royal, Montréal",
        },
        "customization_summary": {
            "theme": "dark",
            "density": "comfortable",
            "accent_preset": "teal",
        },
        "raw_draft_text": "MUST BE IGNORED",
    }

    clean = sanitize_app_presence(raw)
    assert clean is not None
    assert clean["current_screen"] == "thread"
    assert clean["active_session_id"] == "sess_abc-123:tip"
    assert clean["active_run_count"] == 3
    assert clean["draft_chars"] == 42
    assert "raw_draft_text" not in clean
    # Coarse mode rounds coordinates to 2 decimal places (~1.1 km).
    assert clean["continuous_location"]["lat"] == 45.5
    assert clean["continuous_location"]["lon"] == -73.57
    assert clean["continuous_location"]["moving_state"] == "walking"
    assert clean["continuous_location"]["place_label"] == "Plateau-Mont-Royal, Montréal"

    ctx = format_app_presence_context(raw)
    assert ctx.startswith("<mobile_app_presence>")
    assert ctx.endswith("</mobile_app_presence>")
    assert "screen=thread" in ctx
    assert "active_run_count=3" in ctx
    assert "in_app_read_without_reply=true (180s)" in ctx
    assert "Signal/Marie 300s" in ctx
    assert "location(opt_in): mode=coarse, lat=45.5, lon=-73.57" in ctx
    assert "MUST BE IGNORED" not in ctx
    assert len(ctx) <= MAX_CONTEXT_CHARS


def test_app_presence_fails_closed_when_location_not_opted_in_or_malformed():
    # opted_in missing or False -> coordinates strictly dropped!
    for bad_loc in (
        {"lat": 45.5, "lon": -73.5},
        {"opted_in": False, "lat": 45.5, "lon": -73.5},
        {"opted_in": "true", "lat": 45.5, "lon": -73.5},
        {"opted_in": True, "lat": 95.0, "lon": -73.5},
        {"opted_in": True, "lat": float("nan"), "lon": -73.5},
        {"opted_in": True, "lat": True, "lon": False},
    ):
        clean = sanitize_app_presence({"current_screen": "discussions", "continuous_location": bad_loc})
        assert clean is not None
        assert "continuous_location" not in clean

    # Adversarial injections, secrets, and out-of-range values are scrubbed/clamped.
    adversarial = {
        "current_screen": "javascript:alert(1)",
        "active_session_id": "../etc/passwd",
        "active_run_count": 9999,
        "draft_chars": -5,
        "read_without_reply": {
            "notifications": [
                {"source": "Bank", "sender_hint": "OTP code 123456"},
                {"source": "<script>evil()</script>", "sender_hint": "Bob"},
                {"source": "Signal", "sender_hint": "sk-AbCdEf1234567890"},
            ]
            + [{"source": f"App{i}", "age_s": i} for i in range(20)],
        },
        "continuous_location": {
            "opted_in": True,
            "mode": "precise",
            "lat": 45.5016889,
            "lon": -73.567256,
            "place_label": "Bearer supersecrettoken1234567890",
        },
    }
    clean = sanitize_app_presence(adversarial)
    assert clean is not None
    assert "current_screen" not in clean
    assert "active_session_id" not in clean
    assert clean["active_run_count"] == MAX_ACTIVE_RUN_COUNT
    assert "draft_chars" not in clean
    notifs = clean["read_without_reply"]["notifications"]
    assert len(notifs) <= MAX_UNREPLIED_ITEMS
    assert all("<script>" not in n["source"] for n in notifs)
    assert all("sk-" not in n.get("sender_hint", "") for n in notifs)
    assert all("OTP" not in n.get("sender_hint", "") for n in notifs)
    assert "place_label" not in clean["continuous_location"]
    assert clean["continuous_location"]["lat"] == 45.50169

    # Non-dict / empty / oversized payloads fail closed to None and empty context.
    assert sanitize_app_presence(None) is None
    assert sanitize_app_presence({}) is None
    assert sanitize_app_presence({"current_screen": "x" * 20_000}) is None
    assert format_app_presence_context(None) == ""


# ── 2. Deep Android 16 Device Bridge Catalog & Validation ────────────────────


def test_all_nine_deep_android_actions_registered_and_validatable():
    assert len(DEEP_ANDROID_ACTIONS) == 9
    assert DEEP_ANDROID_ACTIONS.issubset(ALLOWED_DEVICE_CAPABILITIES)
    assert ALLOWED_DEVICE_CAPABILITIES.issubset(set(DEVICE_ACTION_CATALOG))

    caps, err = validate_device_capabilities(sorted(DEEP_ANDROID_ACTIONS))
    assert err is None
    assert set(caps or []) == DEEP_ANDROID_ACTIONS

    valid_cases = [
        ("app_navigate", {"target": "command_palette"}),
        ("app_navigate", {"target": "thread", "session_id": "sess-1"}),
        ("app_customize", {"operation": "apply", "accent_preset": "teal", "density": "compact", "startup_tab": "atelier"}),
        ("app_customize", {"operation": "undo"}),
        ("app_customize", {"operation": "reset"}),
        ("app_read_state", {"include": ["screen", "active_runs", "read_without_reply", "location"]}),
        ("app_create_template", {"title": "Bilan matin", "draft_prompt": "Résume mes priorités du jour."}),
        ("android_set_timer", {"duration_seconds": 600, "label": "Thé"}),
        ("android_share_sheet", {"text": "Résumé prêt", "subject": "Notes", "mime_type": "text/markdown"}),
        ("android_open_app", {"package_name": "com.spotify.music", "label": "Spotify"}),
        ("android_open_app", {"deep_link": "https://maps.google.com/?q=Montreal"}),
        ("android_sync_shortcuts", {"tile_label": "Demander", "shortcuts": [{"id": "morning", "label": "Matin", "target": "atelier"}], "widget_summary": "2 tâches"}),
        ("android_proactive_nudge", {"title": "Rappel", "body": "Votre minuteur se termine bientôt.", "category": "reminder", "min_interval_minutes": 30}),
    ]
    for action, params in valid_cases:
        cleaned, verr = validate_device_action_params(action, params)
        assert verr is None, f"{action} rejected unexpectedly: {verr}"
        assert cleaned is not None


@pytest.mark.parametrize(
    ("action", "bad_params"),
    [
        ("app_navigate", {"target": "unknown_screen"}),
        ("app_navigate", {"target": "thread"}),  # missing session_id
        ("app_navigate", {"target": "discussions", "extra": "boom"}),
        ("app_customize", {"operation": "apply"}),  # empty apply
        ("app_customize", {"operation": "apply", "accent_hex": "#FAFAFA", "surface_hex": "#FFFFFF"}),  # WCAG fail
        ("app_customize", {"operation": "apply", "greeting": "key sk-1234567890abcdef"}),
        ("app_read_state", {"include": ["raw_passwords"]}),
        ("app_create_template", {"title": "", "draft_prompt": "hi"}),
        ("android_set_timer", {"duration_seconds": 0}),
        ("android_set_timer", {"duration_seconds": True}),
        ("android_set_timer", {"duration_seconds": 999_999}),
        ("android_share_sheet", {"text": "hi", "mime_type": "text/html"}),
        ("android_open_app", {}),
        ("android_open_app", {"package_name": "../bad"}),
        ("android_open_app", {"deep_link": "javascript:alert(1)"}),
        ("android_open_app", {"deep_link": "file:///etc/passwd"}),
        ("android_open_app", {"deep_link": "intent:#Intent;end"}),
        ("android_sync_shortcuts", {}),
        ("android_sync_shortcuts", {"widget_summary": "Bearer secret_token_1234567890"}),
        ("android_proactive_nudge", {"title": "Hi", "body": "Msg", "respect_quiet_hours": False}),
        ("android_proactive_nudge", {"title": "Hi", "body": "Msg", "min_interval_minutes": 1}),
        ("android_proactive_nudge", {"title": "Hi", "body": "sk-1234567890abcdef"}),
    ],
)
def test_deep_android_actions_reject_invalid_and_adversarial_params(action, bad_params):
    cleaned, verr = validate_device_action_params(action, bad_params)
    assert cleaned is None
    assert verr is not None


def test_request_device_action_tool_enforces_param_validation_and_broker_roundtrip():
    broker = DeviceBridgeBroker()
    seen_events: list[tuple[str, dict]] = []

    def listener(event_name: str, payload: dict) -> None:
        seen_events.append((event_name, payload))
        if event_name == "device.requested":
            threading.Thread(
                target=lambda: broker.resolve_request(
                    profile_key="alice",
                    run_id="run-deep-1",
                    request_id=payload["request_id"],
                    status="completed",
                    data={"screen": "discussions", "active_run_count": 2},
                ),
                daemon=True,
            ).start()

    broker.register_run_listener("alice", "run-deep-1", listener)
    with device_bridge_scope(
        None,
        capabilities=["app_read_state", "android_open_app"],
        run_id="run-deep-1",
        profile_key="alice",
        broker=broker,
        timeout_seconds=5.0,
    ):
        # Invalid deep_link fails closed before hitting the broker!
        bad_res = json.loads(
            request_device_action_tool("android_open_app", "Open link", {"deep_link": "file:///sdcard/secret"})
        )
        assert bad_res["ok"] is False
        assert not seen_events

        # Valid alias ("read_app_state" -> "app_read_state") round-trips cleanly.
        ok_res = json.loads(
            request_device_action_tool("read_app_state", "Check current screen", {"include": ["screen", "active_runs"]})
        )
        assert ok_res["ok"] is True
        assert ok_res["action_type"] == "app_read_state"
        assert ok_res["data"]["active_run_count"] == 2


# ── 3. Interactive Cards in mobile_ui_patch ──────────────────────────────────


def test_interactive_cards_validation_persistence_and_negative_rejection(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    cards = [
        {
            "id": "morning_check",
            "kind": "checklist",
            "title": "Routine du matin",
            "items": [
                {"id": "coffee", "label": "Préparer le café", "checked": False},
                {"id": "notes", "label": "Relire mes notes", "checked": True, "draft": "Résume mes notes"},
            ],
        },
        {
            "id": "quick_choices",
            "kind": "quick_pick",
            "title": "Prochaine étape",
            "items": [
                {"id": "short", "label": "Version courte", "draft": "Fais une version en 3 points."},
                {"id": "detail", "label": "Version détaillée", "draft": "Développe avec des exemples."},
            ],
        },
        {
            "id": "prefs_confirm",
            "kind": "toggle_confirm",
            "title": "Confirmer les préférences",
            "items": [{"id": "quiet", "label": "Respecter les heures calmes", "enabled": True}],
            "confirm_label": "Appliquer",
            "cancel_label": "Annuler",
        },
        {
            "id": "theme_preview",
            "kind": "customization_preview",
            "title": "Aperçu du thème",
            "customization": {"theme": "dark", "accent": "teal", "density": "spacious", "startup_tab": "cerveau"},
            "apply_label": "Appliquer",
            "undo_label": "Annuler",
        },
    ]

    cleaned, err = validate_interactive_cards(cards)
    assert err is None
    assert cleaned is not None and len(cleaned) == 4

    raw_res = mobile_ui_patch_tool(
        ops=[{"op": "set_theme_override", "theme": {"accent": "teal", "density": "spacious"}}],
        summary="Proposer le thème et la routine",
        session_id="sess-cards",
        requires_apply=True,
        interactive_cards=cards,
    )
    res = json.loads(raw_res)
    assert res["ok"] is True
    rev = res["revision"]
    assert len(rev["interactive_cards"]) == 4

    # Stored in SQLite and returned verbatim via MobileUiStore.list_history().
    store = MobileUiStore(tmp_path)
    history = store.list_history(session_id="sess-cards")
    assert len(history) == 1
    assert len(history[0]["interactive_cards"]) == 4

    # Adversarial / malformed cards fail closed.
    for bad_cards in (
        [{"kind": "unknown_kind", "title": "Bad"}],
        [{"kind": "quick_pick", "title": "Missing draft", "items": [{"label": "Tap"}]}],
        [{"kind": "checklist", "title": "<script>alert(1)</script>", "items": [{"label": "A"}]}],
        [{"kind": "checklist", "title": "Secret sk-1234567890abcdef", "items": [{"label": "A"}]}],
        [{"kind": "customization_preview", "title": "Bad WCAG", "customization": {"accent_hex": "#FEFEFE", "surface_hex": "#FFFFFF"}}],
        [{"id": "dup", "kind": "quick_pick", "title": "A", "items": [{"label": "X", "draft": "Y"}]}] * 2,
        [{"id": f"c{i}", "kind": "quick_pick", "title": "A", "items": [{"label": "X", "draft": "Y"}]} for i in range(5)],
    ):
        c, cerr = validate_interactive_cards(bad_cards)
        assert c is None
        assert cerr is not None
