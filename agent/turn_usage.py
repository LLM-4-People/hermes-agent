"""Per-response usage accounting for the conversation turn loop.

After every successful model API call, ``record_response_usage`` folds ``response.usage``
into: the context compressor (``update_from_response`` + the compression-budget rearm
latch), the usage anchor for display/compression math, per-session token/cost counters,
the state.db token-delta queue, and the observability log line. MoA sessions additionally
fold advisor fan-out usage into the reported counts and price the aggregator at its REAL
model/provider. Logger name stays ``agent.conversation_loop`` for caplog parity.
"""

from __future__ import annotations

import logging
import math
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.image_token_cost import calibrate_from_usage
from agent.usage_anchor import capture_usage_anchor, set_usage_anchor
from agent.usage_pricing import estimate_usage_cost, normalize_usage, with_served_service_tier

logger = logging.getLogger("agent.conversation_loop")

_KNOWN_COST_STATUSES = frozenset({"actual", "estimated", "included"})
_BASELINE_INT_ATTRS = (
    ("input_tokens", "session_prompt_tokens"),
    ("output_tokens", "session_completion_tokens"),
    ("total_tokens", "session_total_tokens"),
    ("cache_read_tokens", "session_cache_read_tokens"),
    ("cache_write_tokens", "session_cache_write_tokens"),
    ("reasoning_tokens", "session_reasoning_tokens"),
    ("api_calls", "session_api_calls"),
)


@dataclass
class TurnTelemetry:
    """Per-turn telemetry accumulated at the lowest choking point inside ``record_response_usage``."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    api_calls: int = 0
    usage_calls: int = 0
    cost_usd_sum: float = 0.0
    priced_calls: int = 0
    unknown_cost_calls: int = 0
    cost_source: Optional[str] = None
    cost_status: Optional[str] = None
    first_ttft_s: Optional[float] = None
    api_duration_s: float = 0.0
    generation_duration_s: float = 0.0
    generation_output_tokens: int = 0
    turn_started_monotonic: Optional[float] = None


def _finite_nonneg(value: Any) -> Optional[float]:
    """Finite non-negative float (booleans/mocks rejected), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) and f >= 0.0 else None


def _strict_int(value: Any) -> Optional[int]:
    """Non-negative int (booleans/mocks rejected), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return max(0, int(f)) if math.isfinite(f) else None


def _normalize_cost_source(source: Any, status: Any) -> Optional[str]:
    if status == "included" and (not isinstance(source, str) or source.strip().lower() in ("", "none")):
        return "subscription_included"
    if isinstance(source, str):
        cleaned = source.strip()
        if cleaned and cleaned.lower() != "none":
            return cleaned
    return None


def reset_turn_telemetry(agent: Any) -> TurnTelemetry:
    """Start a clean per-turn telemetry accumulator on *agent*."""
    telemetry = TurnTelemetry(turn_started_monotonic=time.perf_counter())
    agent._turn_telemetry = telemetry
    return telemetry


def capture_usage_baseline(agent: Any) -> dict[str, Any]:
    """Snapshot cumulative session counters on *agent* before a turn on a cached agent."""
    baseline: dict[str, Any] = {
        key: (_strict_int(getattr(agent, attr, None)) or 0) for key, attr in _BASELINE_INT_ATTRS
    }
    cost = _finite_nonneg(getattr(agent, "session_estimated_cost_usd", None))
    baseline["cost_usd"] = cost if cost is not None else 0.0
    baseline["started_monotonic"] = time.perf_counter()
    return baseline


def _record_call_telemetry(
    agent: Any,
    *,
    api_duration: float,
    api_start_time: Optional[float] = None,
    prompt_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    reasoning_tokens: int = 0,
    cost_delta: Optional[float] = None,
    cost_status: Optional[str] = None,
    cost_source: Optional[str] = None,
    has_usage: bool = True,
) -> None:
    """Fold one completed provider API call into ``agent._turn_telemetry``."""
    tel = getattr(agent, "_turn_telemetry", None)
    if not isinstance(tel, TurnTelemetry):
        tel = reset_turn_telemetry(agent)
    tel.api_calls += 1
    dur = _finite_nonneg(api_duration)
    if dur is not None:
        tel.api_duration_s += dur

    ttft_s = _call_ttft_seconds(agent, api_start_time=api_start_time, api_duration=dur)
    if ttft_s is not None and tel.first_ttft_s is None:
        tel.first_ttft_s = ttft_s

    if not has_usage:
        tel.unknown_cost_calls += 1
        return

    tel.usage_calls += 1
    tel.input_tokens += max(0, int(prompt_tokens))
    tel.output_tokens += max(0, int(output_tokens))
    tel.total_tokens += max(0, int(total_tokens))
    tel.cache_read_tokens += max(0, int(cache_read_tokens))
    tel.cache_write_tokens += max(0, int(cache_write_tokens))
    tel.reasoning_tokens += max(0, int(reasoning_tokens))

    if ttft_s is not None and dur is not None and output_tokens > 0:
        gen_s = dur - ttft_s
        if gen_s > 0.0:
            tel.generation_duration_s += gen_s
            tel.generation_output_tokens += int(output_tokens)

    delta = _finite_nonneg(cost_delta)
    if delta is not None and (cost_status in _KNOWN_COST_STATUSES or cost_status is None):
        tel.cost_usd_sum += delta
        tel.priced_calls += 1
        tel.cost_status = cost_status
        tel.cost_source = _normalize_cost_source(cost_source, cost_status) or tel.cost_source
    else:
        tel.unknown_cost_calls += 1
        tel.cost_status = "unknown"
        if tel.cost_source is None:
            tel.cost_source = _normalize_cost_source(cost_source, cost_status)


def _call_ttft_seconds(
    agent: Any, *, api_start_time: Optional[float], api_duration: Optional[float]
) -> Optional[float]:
    start = _finite_nonneg(api_start_time)
    first_chunk = _finite_nonneg(getattr(agent, "_last_api_first_chunk_at", None))
    if start is None or first_chunk is None or first_chunk < start:
        return None
    ttft = first_chunk - start
    if api_duration is not None and api_duration > 0.0:
        ttft = min(ttft, api_duration)
    return ttft


def _counter_fallback_metrics(
    agent: Any, counter_baseline: Optional[dict[str, Any]]
) -> tuple[dict[str, int], Optional[int], Optional[float], Optional[str]]:
    """Read real numeric counters from *agent* when ``TurnTelemetry`` did not record calls."""
    base = counter_baseline or {}
    counts: dict[str, int] = {}
    for key, attr in _BASELINE_INT_ATTRS:
        if key == "api_calls":
            continue
        raw = _strict_int(getattr(agent, attr, None))
        counts[key] = max(0, raw - int(base.get(key, 0))) if raw is not None else 0
    if counts["total_tokens"] == 0 and (counts["input_tokens"] or counts["output_tokens"]):
        counts["total_tokens"] = counts["input_tokens"] + counts["output_tokens"]

    raw_calls = _strict_int(getattr(agent, "session_api_calls", None))
    api_calls = max(0, raw_calls - int(base.get("api_calls", 0))) if raw_calls is not None else None

    cost_status = getattr(agent, "session_cost_status", None)
    raw_source = getattr(agent, "session_cost_source", None)
    cost_source = _normalize_cost_source(raw_source, cost_status)
    cost_usd: Optional[float] = None
    if isinstance(cost_status, str) and cost_status in _KNOWN_COST_STATUSES:
        raw_cost = _finite_nonneg(getattr(agent, "session_estimated_cost_usd", None))
        if raw_cost is not None:
            cost_usd = round(max(0.0, raw_cost - float(base.get("cost_usd", 0.0))), 6)
    return counts, api_calls, cost_usd, cost_source


def build_run_usage_metrics(
    agent: Any,
    *,
    duration_s: Optional[float] = None,
    first_token_s: Optional[float] = None,
    counter_baseline: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Single source of truth for per-run ``usage`` and ``turn.metrics`` payloads.

    Unknown measurements stay explicit ``None`` (serialized as JSON ``null``), never fabricated
    as ``0``. ``tokens_per_sec`` reports generation throughput (post-TTFT decode) when first-token
    timing is known, kept distinct from ``end_to_end_tokens_per_sec``.
    """
    tel = getattr(agent, "_turn_telemetry", None)
    use_tel = isinstance(tel, TurnTelemetry) and (tel.api_calls > 0 or tel.usage_calls > 0)

    if use_tel:
        counts = {
            "input_tokens": tel.input_tokens,
            "output_tokens": tel.output_tokens,
            "total_tokens": tel.total_tokens,
            "cache_read_tokens": tel.cache_read_tokens,
            "cache_write_tokens": tel.cache_write_tokens,
            "reasoning_tokens": tel.reasoning_tokens,
        }
        api_calls: Optional[int] = tel.api_calls
        cost_usd = round(tel.cost_usd_sum, 6) if (tel.priced_calls > 0 and tel.unknown_cost_calls == 0) else None
        cost_source = tel.cost_source if cost_usd is not None else None
        ttft_s = tel.first_ttft_s if tel.first_ttft_s is not None else _finite_nonneg(first_token_s)
    else:
        counts, api_calls, cost_usd, cost_source = _counter_fallback_metrics(agent, counter_baseline)
        ttft_s = _finite_nonneg(first_token_s)

    eff_duration = _finite_nonneg(duration_s)
    if eff_duration is None and use_tel:
        if tel.turn_started_monotonic is not None:
            eff_duration = _finite_nonneg(time.perf_counter() - tel.turn_started_monotonic)
        elif tel.api_duration_s > 0.0:
            eff_duration = tel.api_duration_s
    elif eff_duration is None and counter_baseline and counter_baseline.get("started_monotonic") is not None:
        eff_duration = _finite_nonneg(time.perf_counter() - float(counter_baseline["started_monotonic"]))

    input_tokens = counts["input_tokens"]
    output_tokens = counts["output_tokens"]
    cache_read = counts["cache_read_tokens"]
    cache_hit_ratio = (
        round(min(1.0, max(0.0, cache_read / input_tokens)), 4) if input_tokens > 0 else None
    )

    tokens_per_sec: Optional[float] = None
    if use_tel and tel.generation_duration_s > 0.0 and tel.generation_output_tokens > 0:
        tokens_per_sec = round(tel.generation_output_tokens / tel.generation_duration_s, 2)
    elif ttft_s is not None and eff_duration is not None and output_tokens > 0:
        gen_s = eff_duration - ttft_s
        if gen_s > 0.0:
            tokens_per_sec = round(output_tokens / gen_s, 2)

    end_to_end_tps = (
        round(output_tokens / eff_duration, 2)
        if output_tokens > 0 and eff_duration is not None and eff_duration > 0.0
        else None
    )

    return {
        "scope": "run",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": counts["total_tokens"],
        "cache_read_tokens": cache_read,
        "cache_write_tokens": counts["cache_write_tokens"],
        "cache_hit_ratio": cache_hit_ratio,
        "reasoning_tokens": counts["reasoning_tokens"],
        "cost_usd": cost_usd,
        "cost_source": cost_source,
        "ttft_ms": round(ttft_s * 1000.0) if ttft_s is not None else None,
        "duration_ms": round(eff_duration * 1000.0) if eff_duration is not None else None,
        "tokens_per_sec": tokens_per_sec,
        "end_to_end_tokens_per_sec": end_to_end_tps,
        "api_calls": api_calls,
    }


def add_session_usage(agent: Any, usage: Any, *, prompt_tokens: int, completion_tokens: int,
                      total_tokens: int) -> None:
    """Fold one reply's tokens into the agent's running session totals (the caller holds
    ``session_totals_lock``): a conversation response and a cache keep-alive alike."""
    agent.session_prompt_tokens += prompt_tokens
    agent.session_completion_tokens += completion_tokens
    agent.session_total_tokens += total_tokens
    agent.session_input_tokens += usage.input_tokens
    agent.session_output_tokens += usage.output_tokens
    agent.session_cache_read_tokens += usage.cache_read_tokens
    agent.session_cache_write_tokens += usage.cache_write_tokens
    agent.session_reasoning_tokens += usage.reasoning_tokens


def agent_session_source(agent: Any) -> str:
    """The surface the agent's own row create would stamp (``_ensure_db_session``), so an
    accounting guard that wins the row-creation race never mints an anonymous session."""
    from agent.session_source import session_source_for
    return session_source_for(getattr(agent, "platform", None))


def cache_clock_column(event: Any) -> dict[str, str]:
    """``prompt_cache_clock`` kwarg for the token batch: only when this response moved the clock."""
    if event is None:
        return {}
    from agent.prompt_cache_clock import event_to_json
    return {"prompt_cache_clock": event_to_json(event)}


def queue_usage(agent: Any, session_id: Any = None, source: Optional[str] = None, **counts: Any) -> None:
    """Enqueue one API call's accounting (``queue_token_counts`` keywords) on the agent's store: through its
    storage interface handle when it has one, else on its own store (a worker's ``RuntimeSessionStore``).
    *session_id* / *source* default to the agent's current session and surface (a cache keep-alive passes
    the ones it captured when it was scheduled)."""
    from agent.agent_init_session import agent_interface
    session_id = agent.session_id if session_id is None else session_id
    source = agent_session_source(agent) if source is None else source
    store = agent_interface(agent)
    if store is None:
        agent._session_db.queue_token_counts(session_id, source=source, **counts)
        return
    from persistence.conversation.usage import BillingRoute, TokenCounts, UsageDelta
    tokens = TokenCounts(**{key: counts.pop(key, 0) for key in (
        "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens")})
    route = {key: counts.pop(key, None) for key in ("model", "billing_provider", "billing_base_url", "billing_mode")}
    delta = UsageDelta(tokens=tokens, route=BillingRoute(**route) if route["model"] else None, source=source, **counts)
    store.usage.queue_usage(session_id, delta)


@dataclass
class ResponseUsageOutcome:
    """``compression_attempts`` is the (possibly rearmed-to-zero) budget counter;
    ``rearmed`` tells the loop to also clear its preflight-block latch."""

    compression_attempts: int
    rearmed: bool = False


def note_response_reasoning_tokens(agent: Any, response: Any) -> None:
    """Record *response*'s own reported reasoning count; ``_build_assistant_message`` stamps it on
    every row the response becomes (the replay price of its encrypted reasoning, replayed_reasoning.py).
    Set once per response, before any row is built, so a row never carries an earlier response's count.
    ``response.usage`` is the aggregator's own usage: MoA advisors' thinking never replays."""
    usage = getattr(response, "usage", None)
    agent._last_response_reasoning_tokens = (
        normalize_usage(usage, provider=agent.provider, api_mode=agent.api_mode).reasoning_tokens if usage else 0
    )


def _loop_mod():
    """Lazy ``agent.conversation_loop`` import (avoids an import cycle)."""
    import agent.conversation_loop as _cl

    return _cl


def _fold_moa_usage(agent, canonical_usage):
    """MoA: fold advisor fan-out usage into REPORTED token counts (only aggregator usage is
    returned, so advisor spend would be invisible) and flush the full-turn trace when
    ``moa.save_traces`` is on. Returns ``(client, canonical_usage, advisor_cost)``."""
    _moa_ref_cost = None
    _moa_client = getattr(agent, "client", None)
    if _moa_client is not None and hasattr(_moa_client, "consume_reference_usage"):
        try:
            _ref_usage, _moa_ref_cost = _moa_client.consume_reference_usage()
            if _ref_usage is not None:
                canonical_usage = canonical_usage + _ref_usage
        except Exception as _moa_acct_exc:  # pragma: no cover - defensive
            logger.debug("MoA reference usage accounting failed: %s", _moa_acct_exc)
    if _moa_client is not None and hasattr(_moa_client, "consume_and_save_trace"):
        try:
            # Streaming path: pass the streamed acting text so the trace is self-contained.
            _agg_streamed_text = getattr(agent, "_current_streamed_assistant_text", "") or ""
            _moa_client.consume_and_save_trace(
                agent.session_id, aggregator_output_fallback=_agg_streamed_text or None
            )
        except Exception as _moa_trace_exc:  # pragma: no cover - defensive
            logger.debug("MoA trace flush failed: %s", _moa_trace_exc)
    return _moa_client, canonical_usage, _moa_ref_cost


def _with_declared_write_tier(usage: Any, event: Any) -> Any:
    """A route that honours the tier it was sent but reports no split (OpenRouter Claude) bills its
    writes at the requested tier: name it so pricing charges a 1h write at the 1h rate."""
    if event is None:
        return usage
    from agent.prompt_cache_clock import declared_write_tier
    return with_write_tier(usage, declared_write_tier(event))


def with_write_tier(usage: Any, tier: str | None) -> Any:
    """*usage* with its unsplit cache writes named as *tier* when that is ``1h`` (priced at the 1h rate)."""
    if usage.cache_write_by_ttl or not usage.cache_write_tokens or tier != "1h":
        return usage
    from dataclasses import replace
    return replace(usage, cache_write_by_ttl=(("1h", usage.cache_write_tokens),))


def record_response_usage(
    agent: Any, response: Any, *, messages: list[dict[str, Any]], api_call_count: int,
    api_duration: float, compression_attempts: int, max_compression_attempts: int,
    api_start_time: Optional[float] = None,
) -> ResponseUsageOutcome:
    """Fold ``response.usage`` into compressor, anchors, session counters, state.db
    and the API-call log line (see module docstring). No-usage responses only
    consume a pending compaction verdict. Returns the loop-visible outcome."""
    rearmed = False
    compressor = agent.context_compressor
    # Count every completed provider attempt, including providers that omit usage.
    # Token/cost accounting below stays gated on real usage, but the request itself
    # must remain observable.
    agent.session_api_calls += 1
    if not (hasattr(response, 'usage') and response.usage):
        _record_call_telemetry(
            agent, api_duration=api_duration, api_start_time=api_start_time, has_usage=False,
        )
        if getattr(compressor, "awaiting_real_usage_after_compression", False):
            # No usage -> cannot adjudicate the prior compaction; consume the
            # pending verdict so later readings aren't charged to it and
            # preflight deferral isn't latched indefinitely.
            compressor.update_from_response({})
        _note_usage_less = getattr(compressor, "note_usage_less_response", None)
        if callable(_note_usage_less):
            _note_usage_less()
        logger.info(
            "API call #%d: model=%s provider=%s in=? out=? total=? latency=%.1fs usage=unavailable",
            agent.session_api_calls, agent.model, agent.provider or "unknown", api_duration,
        )
        return ResponseUsageOutcome(compression_attempts=compression_attempts, rearmed=rearmed)

    canonical_usage = with_served_service_tier(
        normalize_usage(response.usage, provider=agent.provider, api_mode=agent.api_mode), response)
    # Aggregator-only usage kept for pricing: advisor tokens are priced at each advisor's
    # OWN model rate and added as dollars below.
    aggregator_usage = canonical_usage
    _moa_client, canonical_usage, _moa_ref_cost = _fold_moa_usage(agent, canonical_usage)
    prompt_tokens = canonical_usage.prompt_tokens
    completion_tokens = canonical_usage.output_tokens
    total_tokens = canonical_usage.total_tokens
    # Gemini long decode sums every hop's re-sent prefix into prompt_tokens (billing, session totals);
    # the context actually in the window is the first hop's prompt: compaction and the meter read that.
    _initial_prompt = getattr(response.usage, "initial_prompt_tokens", None)
    context_prompt_tokens = _initial_prompt if isinstance(_initial_prompt, int) and _initial_prompt > 0 else None
    # The canonical buckets follow the same split, so engines reading them never see cache_read > prompt.
    _initial_cached = getattr(response.usage, "initial_cached_tokens", 0) if context_prompt_tokens else None
    context_cache_read = _initial_cached if isinstance(_initial_cached, int) else None
    # Canonical token + cache buckets for context engines; legacy keys stay for back-compat.
    usage_dict = {
        "prompt_tokens": context_prompt_tokens or prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": (context_prompt_tokens + completion_tokens) if context_prompt_tokens else total_tokens,
        "input_tokens": (context_prompt_tokens - context_cache_read) if context_cache_read is not None else canonical_usage.input_tokens,
        "output_tokens": canonical_usage.output_tokens,
        "cache_read_tokens": canonical_usage.cache_read_tokens if context_cache_read is None else context_cache_read,
        "cache_write_tokens": canonical_usage.cache_write_tokens,
        "reasoning_tokens": canonical_usage.reasoning_tokens,
    }
    # Capture the boundary latch before update_from_response() consumes it: only the real
    # prompt count right after a compaction rearms the budget.
    _completed_compaction_pending = bool(
        getattr(compressor, "_verify_compaction_cleared_threshold", False)
    )
    compressor.update_from_response(usage_dict)
    # Usage-anchored accounting: snapshot exact provider usage against the durable
    # transcript (main-loop ONLY; MoA uses pre-fold aggregator usage). The display meter
    # anchors on the turn's FIRST response: later same-turn responses inflate
    # prompt_tokens with replayed thinking. Display-only; compression math uses real usage.
    # The provider just priced this request exactly: if the delta since the previous anchor
    # introduced images, the residual is their real per-image cost (learned before re-anchoring).
    calibrate_from_usage(agent, messages, context_prompt_tokens or aggregator_usage.prompt_tokens)
    _new_anchor = capture_usage_anchor(
        context_prompt_tokens or aggregator_usage.prompt_tokens, aggregator_usage.output_tokens, messages
    )
    if _new_anchor is not None:
        set_usage_anchor(agent, _new_anchor, turn_base=api_call_count == 1)
    _compression_threshold = int(getattr(compressor, "threshold_tokens", 0) or 0)
    if _loop_mod()._should_rearm_compression_budget(
        compression_attempts, completed_compaction_pending=_completed_compaction_pending,
        prompt_tokens=usage_dict["prompt_tokens"], threshold_tokens=_compression_threshold,
    ):
        logger.info(
            "Compression budget rearmed after provider-confirmed "
            "recovery: prompt=%s < threshold=%s (attempts were %s/%s)",
            f"{usage_dict['prompt_tokens']:,}",
            f"{_compression_threshold:,}",
            compression_attempts,
            max_compression_attempts,
        )
        compression_attempts = 0
        # Confirmed recovery also clears the loop's stale insufficient-progress verdict
        # (``_preflight_compression_blocked``), else a later pressure spike grows unchecked.
        rearmed = True

    # Stash canonical usage for on_turn_complete(); keep the latest call's.
    agent._last_turn_usage = dict(usage_dict)
    # The parent's CURRENT prompt size for headroom math (delegate summary budgets): the
    # aggregator's own prompt, never the MoA-folded total (advisor prompts are not in this context).
    agent._last_prompt_size_tokens = int(context_prompt_tokens or aggregator_usage.prompt_tokens or 0)

    # Persist only provider-confirmed context lengths, not probe tiers.
    if getattr(compressor, "_context_probed", False):
        ctx = compressor.context_length
        if getattr(compressor, "_context_probe_persistable", False):
            from agent.model_metadata import save_provider_context_length

            save_provider_context_length(agent.model, agent.base_url, ctx, agent.provider)
            agent._safe_print(f"{agent.log_prefix}💾 Cached context length: {ctx:,} tokens for {agent.model}")
        compressor._context_probed = False
        compressor._context_probe_persistable = False

    from agent.prompt_cache_lifecycle import session_totals_lock

    with session_totals_lock(agent):  # a keep-alive reply landing now adds to these too
        add_session_usage(agent, canonical_usage, prompt_tokens=prompt_tokens,
                          completion_tokens=completion_tokens, total_tokens=total_tokens)
    # Rolling history for status-bar averages (last 10).
    with suppress(Exception):
        hist = getattr(agent, "_api_latency_history", None)
        if hist is not None:
            hist.append(float(api_duration))
        ohist = getattr(agent, "_api_output_history", None)
        if ohist is not None:
            ohist.append(int(canonical_usage.output_tokens or 0))

    _cache_pct = ""
    if canonical_usage.cache_read_tokens and prompt_tokens:
        _cache_pct = f" cache={canonical_usage.cache_read_tokens}/{prompt_tokens} ({100*canonical_usage.cache_read_tokens/prompt_tokens:.0f}%)"
    # write= is the money (cache writes cost 50x a read); id= is what a provider needs to look the
    # request up; upstream= is who actually served it when the route reports that (OpenRouter's
    # `provider`). Diagnosing the 1,393-agent run's cache misses took a DB join and a live probe
    # because none of the three were on this line.
    if canonical_usage.cache_write_tokens:
        _cache_pct += f" write={canonical_usage.cache_write_tokens}"
    _rid = getattr(response, "id", None)
    _ident = f" id={_rid}" if isinstance(_rid, str) and _rid else ""
    _upstream = getattr(response, "provider", None)
    if isinstance(_upstream, str) and _upstream:
        _ident += f" upstream={_upstream}"
    logger.info(
        "API call #%d: model=%s provider=%s in=%d out=%d total=%d latency=%.1fs%s%s",
        agent.session_api_calls, agent.model, agent.provider or "unknown",
        prompt_tokens, completion_tokens, total_tokens,
        api_duration, _cache_pct, _ident,
    )
    # nous.anthropic_wire=auto: the session's wire is decided once, from this first response.
    if agent.session_api_calls == 1 and (agent.provider or "") == "nous":
        with suppress(Exception):
            from agent.nous_wire import maybe_switch_wire_after_first_response
            maybe_switch_wire_after_first_response(agent, response, agent.session_api_calls)

    # MoA: agent.model/provider are the virtual preset/"moa" with no pricing entry, silently
    # dropping aggregator spend. Price at the REAL model/provider from the aggregator slot.
    _agg_cost_model, _agg_cost_provider, _agg_cost_base_url = agent.model, agent.provider, agent.base_url
    _agg_slot = getattr(_moa_client, "last_aggregator_slot", None) if _moa_client is not None else None
    if _agg_slot and _agg_slot.get("model"):
        _agg_cost_model = _agg_slot["model"]
        _agg_cost_provider = _agg_slot.get("provider") or agent.provider
        _agg_cost_base_url = _agg_slot.get("base_url") or agent.base_url
    from agent.prompt_cache_clock import record_cache_response, response_upstream
    _cache_event = record_cache_response(
        agent, aggregator_usage, model=agent.model, provider=agent.provider, base_url=agent.base_url,
        api_mode=agent.api_mode, upstream=response_upstream(response))
    cost_result = estimate_usage_cost(
        _agg_cost_model, _with_declared_write_tier(aggregator_usage, _cache_event), provider=_agg_cost_provider,
        base_url=_agg_cost_base_url, api_key=getattr(agent, "api_key", ""),
    )
    # Cost delta = aggregator + MoA advisor cost (already priced per-advisor at each
    # advisor's own model rate), so state.db's estimated_cost_usd matches the folded
    # token counts.
    _cost_delta = float(cost_result.amount_usd) if cost_result.amount_usd is not None else None
    if _moa_ref_cost is not None:
        try:
            _moa_cost = float(_moa_ref_cost)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            _moa_cost = None
        if _moa_cost is not None:
            _cost_delta = (_cost_delta or 0.0) + _moa_cost
    if _cost_delta is not None:
        with session_totals_lock(agent):
            agent.session_estimated_cost_usd += _cost_delta
    agent.session_cost_status = cost_result.status
    agent.session_cost_source = cost_result.source
    _record_call_telemetry(
        agent,
        api_duration=api_duration,
        api_start_time=api_start_time,
        prompt_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        total_tokens=total_tokens,
        cache_read_tokens=canonical_usage.cache_read_tokens,
        cache_write_tokens=canonical_usage.cache_write_tokens,
        reasoning_tokens=canonical_usage.reasoning_tokens,
        cost_delta=_cost_delta,
        cost_status=cost_result.status,
        cost_source=cost_result.source,
        has_usage=True,
    )

    # Persist per-call token deltas for any session_id so non-CLI runs can't lose
    # accounting; gateway/session-store writes use absolute totals and safely overwrite
    # these deltas. Enqueued, not written (a cold state.db UPDATE here stalled the tool
    # loop); drained at finalize via _persist_session.
    if agent._session_db and agent.session_id:
        try:
            # Ensure the row exists: under concurrent SQLite load the initial
            # _ensure_db_session() may fail, and UPDATE on a missing row affects 0 rows.
            if not agent._session_db_created:
                agent._ensure_db_session()
            queue_usage(
                agent,
                input_tokens=canonical_usage.input_tokens,
                output_tokens=canonical_usage.output_tokens,
                cache_read_tokens=canonical_usage.cache_read_tokens,
                cache_write_tokens=canonical_usage.cache_write_tokens,
                reasoning_tokens=canonical_usage.reasoning_tokens,
                estimated_cost_usd=_cost_delta,
                cost_status=cost_result.status,
                cost_source=cost_result.source,
                billing_provider=agent.provider,
                billing_base_url=agent.base_url,
                billing_mode="subscription_included"
                if cost_result.status == "included" else None,
                model=agent.model,
                api_call_count=1,
                task=getattr(agent, "_turn_route_task", "") or "",
                **cache_clock_column(_cache_event),
            )
        except Exception as e:  # silent loss here undercounts analytics
            logger.debug(
                "Token persistence failed (session=%s, tokens=%d): %s",
                agent.session_id, total_tokens, e,
            )

    if agent.verbose_logging:
        logging.debug(f"Token usage: prompt={usage_dict['prompt_tokens']:,}, completion={usage_dict['completion_tokens']:,}, total={usage_dict['total_tokens']:,}")

    # Report cache stats for any provider that returns ``prompt_tokens_details.cached_tokens``,
    # not only when we inject cache_control markers.
    cached = canonical_usage.cache_read_tokens
    written = canonical_usage.cache_write_tokens
    prompt = usage_dict["prompt_tokens"]
    if (cached or written) and not agent.quiet_mode:
        hit_pct = (cached / prompt * 100) if prompt > 0 else 0
        agent._vprint(
            f"{agent.log_prefix}   💾 Cache: "
            f"{cached:,}/{prompt:,} tokens "
            f"({hit_pct:.0f}% hit, {written:,} written)"
        )
    return ResponseUsageOutcome(compression_attempts=compression_attempts, rearmed=rearmed)
