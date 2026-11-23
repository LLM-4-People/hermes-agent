#!/usr/bin/env bash
# Downstream verification gate for the LLM-4-People fork.
#
# Single owner for the checks that prove our fork patches stay intact against current upstream
# code. Both .github/workflows/downstream-ci.yml (fast-static + sharded verify) and
# sync-upstream.yml (zero-arg full mode) run this exact script.
#
# Usage:
#   ./scripts/downstream-verify.sh                 # Full run (static + all tests; default for sync-upstream)
#   ./scripts/downstream-verify.sh --fast-static   # Fast static gate + docs/PII/workflow invariant tests
#   ./scripts/downstream-verify.sh --shard 1/2     # Balanced parallel shard K of N (tests only)
#   ./scripts/downstream-verify.sh --tests-only    # All downstream tests without static gate
#   ./scripts/downstream-verify.sh --list-shard K/N  # Print test files assigned to shard K of N
#
# Requires the PM test environment (HERMES_PYTHON set by .github/actions/setup-pm) or an
# activated checkout (`source ./activate`).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="${HERMES_PYTHON:-python3}"

# Ordered heaviest-first so modulo sharding (--shard K/N) balances CPU wall-clock across runners
# and starts the longest test files immediately on each shard.
DOWNSTREAM_TESTS=(
  tests/gateway/test_session_mutation_compress.py
  tests/agent/test_auxiliary_error_policy.py
  tests/persistence_sqlite/conversation/test_photo_previews.py
  tests/hermes_cli/test_model_switch_custom_providers.py
  tests/agent/test_session_affinity_header.py
  tests/gateway/test_native_producer_owner_matrix.py
  tests/gateway/api_server/test_mobile_live_ui_push_bridge.py
  tests/gateway/api_server/test_workspace_profile_analytics.py
  tests/gateway/api_server/test_transcript_admissions.py
  tests/agent/test_auxiliary_named_custom_providers.py
  tests/gateway/test_mirror.py
  tests/gateway/test_signal.py
  tests/gateway/api_server/test_brain_memory_search_fork.py
  tests/gateway/api_server/test_notes_and_habits.py
  tests/gateway/api_server/test_rich_telemetry_and_live_stream.py
  tests/gateway/api_server/test_skill_studio.py
  tests/gateway/api_server/test_skills.py
  tests/gateway/api_server/test_run_instructions.py
  tests/gateway/test_model_command_custom_providers.py
  tests/gateway/test_session_model_override_routing.py
  tests/gateway/test_gateway_route_identity.py
  tests/gateway/test_api_standalone_pending_approvals.py
  tests/gateway/test_runtime_footer.py
  tests/gateway/test_session_info.py
  tests/gateway/test_hygiene_provider_identity.py
  tests/gateway/test_signal_format.py
  tests/gateway/test_signal_inbound_attachments.py
  tests/gateway/test_signal_rate_limit.py
  tests/hermes_cli/test_provider_route_identity.py
  tests/hermes_cli/test_custom_provider_extra_headers.py
  tests/hermes_cli/test_custom_provider_normalize_no_mutate.py
  tests/hermes_cli/test_provider_config_validation.py
  tests/hermes_cli/test_runtime_provider_resolution.py
  tests/hermes_cli/test_keyed_provider_credential_pool.py
  tests/hermes_cli/test_custom_provider_context_length.py
  tests/hermes_cli/test_custom_provider_tls.py
  tests/hermes_cli/test_provider_section3_grouping.py
  tests/hermes_cli/test_status_provider_label.py
  tests/hermes_cli/test_cmd_update_docker.py
  tests/agent/test_custom_provider_extra_headers_client.py
  tests/agent/test_anthropic_adapter_extra_headers.py
  tests/agent/test_anthropic_prompt_cache_policy.py
  tests/agent/test_aux_custom_endpoint_headers.py
  tests/agent/test_switch_model_reapplies_headers.py
  tests/agent/test_switch_model_context.py
  tests/agent/test_prompt_cache_ttl_propagation.py
  tests/agent/test_codex_app_server_persist.py
  tests/agent/test_codex_runtime_prompt_handoff.py
  tests/agent/test_turn_usage_telemetry.py
  tests/persistence_sqlite/conversation/test_session_list_index_choice.py
  tests/contracts/conversation/test_api_transcript_input_contract.py
  tests/tools/test_approval_preview.py
  tests/tools/test_skills_tool_discovery_cache.py
  tests/tools/test_signal_media.py
  tests/scripts/test_mobile_doctor.py
  tests/scripts/test_check_fork_preservation.py
  tests/ci/test_fork_ci_workflows.py
  tests/ci/test_docs_drift.py
  tests/ci/test_check_public_pii.py
)

parse_shard() {
  local spec="$1"
  if [[ ! "$spec" =~ ^([0-9]+)/([0-9]+)$ ]]; then
    echo "error: invalid shard spec '$spec' (expected K/N, e.g. 1/2)" >&2
    exit 2
  fi
  SHARD_INDEX="${BASH_REMATCH[1]}"
  SHARD_TOTAL="${BASH_REMATCH[2]}"
  if [ "$SHARD_TOTAL" -lt 1 ] || [ "$SHARD_INDEX" -lt 1 ] || [ "$SHARD_INDEX" -gt "$SHARD_TOTAL" ]; then
    echo "error: shard index out of range in '$spec' (require 1 <= K <= N)" >&2
    exit 2
  fi
}

select_shard_files() {
  local idx="$1"
  local total="$2"
  SHARD_FILES=()
  local i=0
  for test_file in "${DOWNSTREAM_TESTS[@]}"; do
    if (( (i % total) + 1 == idx )); then
      SHARD_FILES+=("$test_file")
    fi
    i=$((i + 1))
  done
}

run_static_gate() {
  # Static guards + lint. All Python-only, so no Node toolchain is needed.
  "$PYTHON" scripts/check --only \
    footguns,detached-spawns,install-root,config-writers,config-readers,ruff

  # Optional anti-AI-slop & language-purity gate when present on this branch.
  if [ -f scripts/check_writing_quality.py ]; then
    "$PYTHON" -B scripts/check_writing_quality.py
  fi
}

run_targeted_checks() {
  # That file imports the Discord and Slack SDKs at module load, which need the optional
  # `messaging` extra. Select only the Signal class and the delivery-mirror regression so the
  # locked test environment stays minimal. Bytecode was already pre-compiled above.
  HERMES_SKIP_COMPILEALL=1 scripts/run_tests.sh tests/gateway/outbound/test_send_message_tool.py \
    -k "TestSendSignalChunking or test_successful_send_mirrors_as_labelled_user_turn" -q

  # Responses replay: a replayed native message id must not sit next to another assistant message.
  HERMES_SKIP_COMPILEALL=1 scripts/run_tests.sh tests/agent/test_codex_responses_adapter.py \
    -k test_native_message_id_dropped_before_another_assistant_message -q
}

MODE="full"
SHARD_SPEC="1/1"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --fast-static)
      MODE="fast-static"
      shift
      ;;
    --tests-only)
      MODE="tests-only"
      shift
      ;;
    --shard)
      if [ "$#" -lt 2 ]; then
        echo "error: --shard requires a K/N argument" >&2
        exit 2
      fi
      MODE="shard"
      SHARD_SPEC="$2"
      shift 2
      ;;
    --list-shard)
      if [ "$#" -lt 2 ]; then
        echo "error: --list-shard requires a K/N argument" >&2
        exit 2
      fi
      MODE="list-shard"
      SHARD_SPEC="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '1,18p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      exit 2
      ;;
  esac
done

case "$MODE" in
  list-shard)
    parse_shard "$SHARD_SPEC"
    select_shard_files "$SHARD_INDEX" "$SHARD_TOTAL"
    printf '%s\n' "${SHARD_FILES[@]}"
    ;;
  fast-static)
    run_static_gate
    HERMES_SKIP_COMPILEALL=1 scripts/run_tests.sh \
      tests/scripts/test_check_fork_preservation.py \
      tests/ci/test_fork_ci_workflows.py \
      tests/ci/test_docs_drift.py \
      tests/ci/test_check_public_pii.py \
      -q
    ;;
  shard)
    parse_shard "$SHARD_SPEC"
    select_shard_files "$SHARD_INDEX" "$SHARD_TOTAL"
    scripts/run_tests.sh "${SHARD_FILES[@]}" -q
    if [ "$SHARD_INDEX" -eq "$SHARD_TOTAL" ]; then
      run_targeted_checks
    fi
    ;;
  tests-only)
    scripts/run_tests.sh "${DOWNSTREAM_TESTS[@]}" -q
    run_targeted_checks
    ;;
  full)
    run_static_gate
    scripts/run_tests.sh "${DOWNSTREAM_TESTS[@]}" -q
    run_targeted_checks
    ;;
esac
