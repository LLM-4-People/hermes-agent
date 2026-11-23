"""Tests for gateway/i18n.py bilingual SSoT, anti-AI-slop linter, and concurrent user isolation."""

from __future__ import annotations

import asyncio
import json
import string
from unittest.mock import MagicMock

import pytest

from agent import i18n as agent_i18n
from agent import prompt_builder
from gateway.i18n import (
    ANTI_SLOP_PROMPT_RULES,
    BANNED_SLOP_PHRASES_EN,
    BANNED_SLOP_PHRASES_FR,
    BANNED_SLOP_WORDS_EN,
    BANNED_SLOP_WORDS_FR,
    CATALOG,
    active_scoped_lang,
    current_user_lang,
    detect_text_lang,
    find_slop_hits,
    resolve_request_lang,
    resolve_user_lang,
    t,
    user_lang_scope,
)
from tools import approval as approval_mod
from tools import clarify_tool as clarify_mod
from tools import mobile_deliver as deliver_mod


def _extract_placeholders(template: str) -> set[str]:
    formatter = string.Formatter()
    return {field_name for _, field_name, _, _ in formatter.parse(template) if field_name is not None}


def test_catalog_en_fr_exact_key_and_placeholder_parity() -> None:
    en_keys = set(CATALOG["en"].keys())
    fr_keys = set(CATALOG["fr"].keys())
    assert en_keys == fr_keys, f"EN/FR key drift: {en_keys ^ fr_keys}"
    assert len(en_keys) >= 40

    for key in sorted(en_keys):
        en_val = CATALOG["en"][key]
        fr_val = CATALOG["fr"][key]
        assert en_val.strip(), f"Empty EN string for {key}"
        assert fr_val.strip(), f"Empty FR string for {key}"
        assert _extract_placeholders(en_val) == _extract_placeholders(fr_val), (
            f"Placeholder mismatch on {key}: EN={_extract_placeholders(en_val)} vs FR={_extract_placeholders(fr_val)}"
        )


def test_anti_slop_linter_clean_across_catalog_and_prompt_builder() -> None:
    # 1. Every catalog entry in both EN and FR must have 0 AI-slop clichés or filler openers/sign-offs.
    for lang in ("en", "fr"):
        for key, text in CATALOG[lang].items():
            hits = find_slop_hits(text, lang=lang)
            assert not hits, f"Slop detected in CATALOG[{lang!r}][{key!r}]: {hits}"

    # 2. DEFAULT_AGENT_IDENTITY and ANTI_SLOP_STYLE_RULES enforce crisp, anti-slop single-language prose.
    assert prompt_builder.ANTI_SLOP_STYLE_RULES in prompt_builder.DEFAULT_AGENT_IDENTITY
    assert "pure English or pure French" in prompt_builder.ANTI_SLOP_STYLE_RULES
    assert "negative-parallelism" in prompt_builder.ANTI_SLOP_STYLE_RULES
    assert find_slop_hits(prompt_builder.DEFAULT_AGENT_IDENTITY, lang="en") == []
    assert ANTI_SLOP_PROMPT_RULES

    # 3. Positive detector checks: linter catches real EN & FR AI slop reliably.
    assert "great question" in find_slop_hits("Great question! Let's delve into this tapestry.", lang="en")
    assert "delve" in find_slop_hits("Let's delve into this seamlessly.", lang="en")
    assert "excellente question" in find_slop_hits("Excellente question ! Voici une synergie holistique.", lang="fr")
    assert "holistique" in find_slop_hits("Une approche holistique incontournable.", lang="fr")
    assert len(BANNED_SLOP_PHRASES_EN) >= 10
    assert len(BANNED_SLOP_PHRASES_FR) >= 6
    assert len(BANNED_SLOP_WORDS_EN) >= 10
    assert len(BANNED_SLOP_WORDS_FR) >= 5


def test_resolve_user_lang_precedence_and_text_heuristic() -> None:
    # Explicit overrides Accept-Language, profile, and text.
    assert resolve_user_lang(accept_language="en-US,en;q=0.9", explicit_lang="fr-CA", text="Hello world") == "fr"
    assert resolve_user_lang(accept_language="fr-FR,fr;q=0.9", explicit_lang="en", text="Bonjour merci") == "en"

    # Weighted Accept-Language picks highest q-factor.
    assert resolve_user_lang(accept_language="de-DE;q=1.0, fr-CA;q=0.8, en-US;q=0.5") == "fr"
    assert resolve_user_lang(accept_language="fr;q=0.4, en-GB;q=0.9") == "en"

    # Profile fallback when headers absent.
    assert resolve_user_lang(profile_lang="fr") == "fr"
    assert resolve_user_lang(profile_lang="english") == "en"

    # Text heuristic detects natural French vs English without misfiring on short/code strings.
    assert detect_text_lang("Bonjour, peux-tu résumer cette note pour aujourd'hui ?") == "fr"
    assert detect_text_lang("Please explain this error and summarize the fix.") == "en"
    assert detect_text_lang("x=1") is None
    assert resolve_user_lang(text="Explique cette erreur dans mon projet merci") == "fr"
    assert resolve_user_lang() == "en"

    # HTTP header helper supports X-Hermes-Lang > Accept-Language.
    assert resolve_request_lang({"Accept-Language": "en-US", "X-Hermes-Lang": "fr"}) == "fr"
    assert resolve_request_lang({"Accept-Language": "fr-FR,fr;q=0.9"}) == "fr"
    assert resolve_request_lang({"Accept-Language": "en-US,en;q=0.9"}) == "en"


def test_concurrent_user_lang_scope_isolation_and_agent_i18n_bridge() -> None:
    async def exercise() -> None:
        alice_ready = asyncio.Event()
        bob_ready = asyncio.Event()

        async def run_user(
            lang: str,
            key: str,
            my_ready: asyncio.Event,
            peer_ready: asyncio.Event,
        ) -> tuple[str, str, str]:
            with user_lang_scope(lang):
                my_ready.set()
                await peer_ready.wait()
                await asyncio.sleep(0)
                scoped = active_scoped_lang()
                translated = t(key)
                agent_lang = agent_i18n.get_language()
                await asyncio.sleep(0)
                assert current_user_lang() == lang
                return scoped or "", translated, agent_lang

        alice, bob = await asyncio.gather(
            run_user("fr", "gateway.error.timeout", alice_ready, bob_ready),
            run_user("en", "gateway.error.timeout", bob_ready, alice_ready),
        )
        assert alice[0] == "fr"
        assert "délai" in alice[1].lower()
        assert alice[2] == "fr"

        assert bob[0] == "en"
        assert "timed out" in bob[1].lower()
        assert bob[2] == "en"
        assert active_scoped_lang() is None

    asyncio.run(exercise())


def test_tools_localize_user_facing_messages_preserving_machine_codes(tmp_path, monkeypatch) -> None:
    # 1. clarify_tool unavailable error in FR vs EN
    with user_lang_scope("en"):
        res_en = json.loads(clarify_mod.clarify_tool([{"question": "Pick one?"}], callback=None))
    with user_lang_scope("fr"):
        res_fr = json.loads(clarify_mod.clarify_tool([{"question": "Choisir ?"}], callback=None))
    assert "not available" in res_en["error"].lower()
    assert "pas disponible" in res_fr["error"].lower()

    # 2. mobile_deliver example & hint localize without breaking machine fields
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with user_lang_scope("en"):
        deliv_en = json.loads(deliver_mod.mobile_deliver_tool("invalid_kind", "", {}))
    with user_lang_scope("fr"):
        deliv_fr = json.loads(deliver_mod.mobile_deliver_tool("invalid_kind", "", {}))
    assert deliv_en["ok"] is False and deliv_fr["ok"] is False
    assert deliv_en["example"]["title"] == "Morning briefing"
    assert deliv_fr["example"]["title"] == "Point du matin"
    assert "Use kind in" in deliv_en["hint"]
    assert "Utilisez kind dans" in deliv_fr["hint"]

    # 3. approval pending notice localizes cleanly while preserving status/pattern_key
    spec = MagicMock(noun="command", pending_keys=False)
    monkeypatch.setattr("tools.approval_notify_lease.stranded_denial", lambda *a, **kw: None)
    with user_lang_scope("en"):
        p_en = approval_mod._pending_result(
            spec, "s1", command="rm -rf /tmp/x", description="recursive delete",
            pattern_key="rm_rf", pattern_keys=["rm_rf"], body=None, smart_denied=False, gateway=False,
        )
    with user_lang_scope("fr"):
        p_fr = approval_mod._pending_result(
            spec, "s1", command="rm -rf /tmp/x", description="suppression récursive",
            pattern_key="rm_rf", pattern_keys=["rm_rf"], body=None, smart_denied=False, gateway=False,
        )
    assert p_en["status"] == "approval_required" and p_fr["status"] == "approval_required"
    assert "potentially dangerous" in p_en["message"]
    assert "Cette action est sensible" in p_fr["message"]

    spec_pending = MagicMock(noun="command", pending_keys=True)
    with user_lang_scope("en"):
        pp_en = approval_mod._pending_result(
            spec_pending, "s1", command="rm -rf /tmp/x", description="recursive delete",
            pattern_key="rm_rf", pattern_keys=["rm_rf"], body=None, smart_denied=False, gateway=False,
        )
    with user_lang_scope("fr"):
        pp_fr = approval_mod._pending_result(
            spec_pending, "s1", command="rm -rf /tmp/x", description="suppression récursive",
            pattern_key="rm_rf", pattern_keys=["rm_rf"], body=None, smart_denied=False, gateway=False,
        )
    assert pp_en["status"] == "pending_approval" and pp_fr["status"] == "pending_approval"
    assert "STOP: do NOT re-run" in pp_en["message"]
    assert "ARRÊT : ne relancez pas" in pp_fr["message"]


def test_writing_quality_ci_gate_passes_and_catches_slop(tmp_path) -> None:
    import subprocess
    import sys
    from scripts import check_writing_quality as wq

    # Repository prose, i18n catalog, prompt_builder, and mobile tools pass with 0 violations.
    assert wq.run_all_checks() == []

    # Pure-stdlib isolated execution (`python3 -I`) must succeed with zero third-party deps.
    proc = subprocess.run(
        [sys.executable, "-I", str(wq.ROOT / "scripts" / "check_writing_quality.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"python3 -I failed: {proc.stderr}"
    assert "OK (0 violations)" in proc.stdout

    # AST prompt_builder check catches slop in synthetic prompt_builder.py without importing it.
    fake_agent = tmp_path / "agent"
    fake_agent.mkdir()
    (fake_agent / "prompt_builder.py").write_text(
        'ANTI_SLOP_STYLE_RULES = "Plain prose."\n'
        'DEFAULT_AGENT_IDENTITY = "Great question! Let us delve into this." + ANTI_SLOP_STYLE_RULES\n',
        encoding="utf-8",
    )
    pb_hits = wq._check_prompt_builder(tmp_path)
    assert pb_hits and "DEFAULT_AGENT_IDENTITY" in pb_hits[0]

    # Negative checks: catches EN/FR clichés and negative-parallelism slop.
    assert "negative parallelism" in wq.scan_prose_for_slop("It's not just fast, it's magical.", lang="en")
    assert "comprehensive" in wq.scan_prose_for_slop("A comprehensive seamless experience.", lang="en")
    assert "sans couture" in wq.scan_prose_for_slop("Une expérience sans couture.", lang="fr")

    # Code blocks, inline code, and URLs do not trigger false positives.
    sample = "Plain line.\n```python\nword = 'delve tapestry'\n```\nUse `delve` at https://example.com/delve."
    assert wq.scan_prose_for_slop(wq.strip_markdown_code(sample), lang="en") == []
