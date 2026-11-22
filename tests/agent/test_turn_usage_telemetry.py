"""Deterministic positive and negative tests for per-run telemetry in ``agent.turn_usage``."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.turn_usage import (
    build_run_usage_metrics,
    capture_usage_baseline,
    record_response_usage,
    reset_turn_telemetry,
)


def _make_agent(*, model: str = "claude-sonnet-4-6", provider: str = "anthropic", api_mode: str = "anthropic_messages"):
    compressor = SimpleNamespace(
        threshold_tokens=100_000,
        context_length=200_000,
        awaiting_real_usage_after_compression=False,
        _verify_compaction_cleared_threshold=False,
        _context_probed=False,
        _context_probe_persistable=False,
        update_from_response=lambda _u: None,
        note_usage_less_response=lambda: None,
    )
    return SimpleNamespace(
        model=model,
        provider=provider,
        api_mode=api_mode,
        base_url="",
        api_key="",
        platform="api_server",
        session_id="sess-telemetry-1",
        _session_db=None,
        _session_db_created=False,
        context_compressor=compressor,
        session_prompt_tokens=0,
        session_completion_tokens=0,
        session_total_tokens=0,
        session_input_tokens=0,
        session_output_tokens=0,
        session_cache_read_tokens=0,
        session_cache_write_tokens=0,
        session_reasoning_tokens=0,
        session_api_calls=0,
        session_estimated_cost_usd=0.0,
        session_cost_status="unknown",
        session_cost_source="none",
        _api_latency_history=deque(maxlen=10),
        _api_output_history=deque(maxlen=10),
        verbose_logging=False,
        quiet_mode=True,
        log_prefix="",
        _safe_print=lambda *_a, **_kw: None,
        _vprint=lambda *_a, **_kw: None,
        _last_api_first_chunk_at=None,
        client=None,
    )


def _anthropic_response(*, inp: int, out: int, cache_read: int = 0, cache_write: int = 0, thinking: int = 0):
    usage = {
        "input_tokens": inp,
        "output_tokens": out,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_write,
        "output_tokens_details": {"thinking_tokens": thinking},
    }
    return SimpleNamespace(usage=usage, id="msg_test_1")


def test_single_streaming_call_computes_exact_ttft_generation_tps_cache_and_cost():
    agent = _make_agent(model="claude-sonnet-4-6", provider="anthropic", api_mode="anthropic_messages")
    reset_turn_telemetry(agent)
    # Request started at t=1000.0, first token at t=1000.5 (TTFT = 500ms), total API call 2.5s -> 2.0s decode.
    agent._last_api_first_chunk_at = 1000.5
    resp = _anthropic_response(inp=2000, out=200, cache_read=7000, cache_write=1000, thinking=80)
    record_response_usage(
        agent, resp, messages=[], api_call_count=1, api_duration=2.5,
        compression_attempts=0, max_compression_attempts=3, api_start_time=1000.0,
    )

    metrics = build_run_usage_metrics(agent, duration_s=4.0)
    assert metrics["scope"] == "run"
    assert metrics["input_tokens"] == 10000
    assert metrics["output_tokens"] == 200
    assert metrics["total_tokens"] == 10200
    assert metrics["cache_read_tokens"] == 7000
    assert metrics["cache_write_tokens"] == 1000
    assert metrics["cache_hit_ratio"] == pytest.approx(0.7, abs=1e-4)
    assert metrics["reasoning_tokens"] == 80
    assert metrics["api_calls"] == 1
    assert metrics["ttft_ms"] == 500
    assert metrics["duration_ms"] == 4000
    # Generation TPS = 200 / (2.5 - 0.5) = 100.0 tok/s, distinct from end-to-end 200 / 4.0 = 50.0 tok/s.
    assert metrics["tokens_per_sec"] == 100.0
    assert metrics["end_to_end_tokens_per_sec"] == 50.0
    # Claude Sonnet 4.6 official docs snapshot: $3/M in, $15/M out, $0.30/M cache_read, $3.75/M cache_write.
    expected_cost = (2000 * 3.0 + 200 * 15.0 + 7000 * 0.30 + 1000 * 3.75) / 1_000_000.0
    assert metrics["cost_usd"] == pytest.approx(expected_cost, abs=1e-6)
    assert metrics["cost_source"] == "official_docs_snapshot"


def test_multi_call_turn_accumulates_generation_tps_and_keeps_first_ttft():
    agent = _make_agent(model="claude-sonnet-4-6", provider="anthropic", api_mode="anthropic_messages")
    reset_turn_telemetry(agent)

    # Call 1: TTFT 0.4s, total 1.4s -> 1.0s decode, 100 output tokens.
    agent._last_api_first_chunk_at = 2000.4
    record_response_usage(
        agent, _anthropic_response(inp=1000, out=100, cache_read=0, cache_write=0),
        messages=[], api_call_count=1, api_duration=1.4,
        compression_attempts=0, max_compression_attempts=3, api_start_time=2000.0,
    )
    # Call 2 (after 3s of tool execution): TTFT 0.2s, total 3.2s -> 3.0s decode, 300 output tokens.
    agent._last_api_first_chunk_at = 2005.2
    record_response_usage(
        agent, _anthropic_response(inp=500, out=300, cache_read=500, cache_write=0),
        messages=[], api_call_count=2, api_duration=3.2,
        compression_attempts=0, max_compression_attempts=3, api_start_time=2005.0,
    )

    # Total run wall-clock is 8.0s (includes tool execution).
    metrics = build_run_usage_metrics(agent, duration_s=8.0)
    assert metrics["api_calls"] == 2
    assert metrics["ttft_ms"] == 400
    assert metrics["output_tokens"] == 400
    # Combined generation: 400 tok / (1.0s + 3.0s) = 100.0 tok/s; end-to-end: 400 / 8.0 = 50.0 tok/s.
    assert metrics["tokens_per_sec"] == 100.0
    assert metrics["end_to_end_tokens_per_sec"] == 50.0


def test_unknown_pricing_and_missing_ttft_stay_explicit_null_never_zero():
    agent = _make_agent(model="local-mystery-model", provider="custom", api_mode="chat_completions")
    reset_turn_telemetry(agent)
    agent._last_api_first_chunk_at = None  # Non-streaming: TTFT and generation TPS are unknown.
    resp = SimpleNamespace(
        usage={"prompt_tokens": 500, "completion_tokens": 100, "total_tokens": 600},
        id="chatcmpl_unknown",
    )
    record_response_usage(
        agent, resp, messages=[], api_call_count=1, api_duration=2.0,
        compression_attempts=0, max_compression_attempts=3, api_start_time=3000.0,
    )

    metrics = build_run_usage_metrics(agent, duration_s=2.0)
    assert metrics["cost_usd"] is None
    assert metrics["cost_source"] is None
    assert metrics["ttft_ms"] is None
    assert metrics["tokens_per_sec"] is None
    assert metrics["end_to_end_tokens_per_sec"] == 50.0
    assert metrics["api_calls"] == 1


def test_partial_unknown_cost_in_multi_call_turn_reports_null_not_understated_sum():
    agent = _make_agent(model="claude-sonnet-4-6", provider="anthropic", api_mode="anthropic_messages")
    reset_turn_telemetry(agent)
    record_response_usage(
        agent, _anthropic_response(inp=1000, out=50),
        messages=[], api_call_count=1, api_duration=1.0,
        compression_attempts=0, max_compression_attempts=3, api_start_time=100.0,
    )
    # Fallback switches to an unpriced custom model on call 2.
    agent.provider = "custom"
    agent.model = "unpriced-fallback"
    agent.api_mode = "chat_completions"
    record_response_usage(
        agent, SimpleNamespace(usage={"prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050}),
        messages=[], api_call_count=2, api_duration=1.0,
        compression_attempts=0, max_compression_attempts=3, api_start_time=102.0,
    )

    metrics = build_run_usage_metrics(agent, duration_s=3.0)
    assert metrics["api_calls"] == 2
    assert metrics["cost_usd"] is None
    assert metrics["cost_source"] is None


def test_usage_less_response_and_subscription_included_and_mock_double():
    # 1. Usage-less provider response: api_calls increments, ratios/cost/tps remain null.
    agent = _make_agent(model="claude-sonnet-4-6", provider="anthropic")
    reset_turn_telemetry(agent)
    record_response_usage(
        agent, SimpleNamespace(usage=None),
        messages=[], api_call_count=1, api_duration=0.5,
        compression_attempts=0, max_compression_attempts=3, api_start_time=500.0,
    )
    no_usage = build_run_usage_metrics(agent, duration_s=0.5)
    assert no_usage["api_calls"] == 1
    assert no_usage["cache_hit_ratio"] is None
    assert no_usage["cost_usd"] is None
    assert no_usage["tokens_per_sec"] is None

    # 2. Subscription-included route reports known $0.0 with subscription_included source.
    sub_agent = _make_agent(model="gpt-5.6-luna", provider="openai-codex", api_mode="codex_responses")
    reset_turn_telemetry(sub_agent)
    record_response_usage(
        sub_agent,
        SimpleNamespace(usage={"input_tokens": 1000, "output_tokens": 100}),
        messages=[], api_call_count=1, api_duration=1.0,
        compression_attempts=0, max_compression_attempts=3, api_start_time=600.0,
    )
    sub_metrics = build_run_usage_metrics(sub_agent, duration_s=1.0)
    assert sub_metrics["cost_usd"] == 0.0
    assert sub_metrics["cost_source"] == "subscription_included"

    # 3. MagicMock test double never turns mock/unknown attributes into fabricated zeros.
    mock_agent = MagicMock()
    mock_agent.session_prompt_tokens = 100
    mock_agent.session_completion_tokens = 20
    mock_agent.session_total_tokens = 120
    mock_metrics = build_run_usage_metrics(mock_agent)
    assert mock_metrics["input_tokens"] == 100
    assert mock_metrics["output_tokens"] == 20
    assert mock_metrics["cost_usd"] is None
    assert mock_metrics["cost_source"] is None
    assert mock_metrics["ttft_ms"] is None
    assert mock_metrics["tokens_per_sec"] is None
    assert mock_metrics["api_calls"] is None


def test_reused_agent_turn_reset_and_counter_baseline_isolate_turns():
    agent = _make_agent(model="claude-sonnet-4-6", provider="anthropic", api_mode="anthropic_messages")
    reset_turn_telemetry(agent)
    agent._last_api_first_chunk_at = 100.2
    record_response_usage(
        agent, _anthropic_response(inp=4000, out=200, cache_read=1000),
        messages=[], api_call_count=1, api_duration=1.2,
        compression_attempts=0, max_compression_attempts=3, api_start_time=100.0,
    )
    turn1 = build_run_usage_metrics(agent, duration_s=1.2)
    assert turn1["input_tokens"] == 5000
    assert turn1["api_calls"] == 1

    # Turn 2 on the same cached agent resets turn telemetry cleanly.
    baseline = capture_usage_baseline(agent)
    reset_turn_telemetry(agent)
    agent._last_api_first_chunk_at = 200.1
    record_response_usage(
        agent, _anthropic_response(inp=100, out=50, cache_read=400),
        messages=[], api_call_count=1, api_duration=0.6,
        compression_attempts=0, max_compression_attempts=3, api_start_time=200.0,
    )
    turn2 = build_run_usage_metrics(agent, duration_s=0.6, counter_baseline=baseline)
    assert turn2["input_tokens"] == 500
    assert turn2["output_tokens"] == 50
    assert turn2["cache_read_tokens"] == 400
    assert turn2["cache_hit_ratio"] == pytest.approx(0.8, abs=1e-4)
    assert turn2["api_calls"] == 1
    assert turn2["ttft_ms"] == 100
    assert turn2["tokens_per_sec"] == 100.0
