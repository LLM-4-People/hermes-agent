#!/usr/bin/env bash
# Downstream verification gate for the LLM-4-People fork.
#
# One owner for the checks that prove our provider-routing patch still works against current
# upstream code. Both .github/workflows/downstream-ci.yml and sync-upstream.yml run this exact
# script, so an automatic upstream merge cannot land with a broken patch and CI cannot drift
# from the merge gate.
#
# Requires the PM test environment (HERMES_PYTHON set by .github/actions/setup-pm) or an
# activated checkout (`source ./activate`).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="${HERMES_PYTHON:-python3}"

# Static guards + lint. All Python-only, so no Node toolchain is needed.
"$PYTHON" scripts/check --only \
  footguns,detached-spawns,install-root,config-writers,config-readers,ruff

# Downstream regression suite: provider routing + footer identity + Signal media, plus every
# consumer that resolves route metadata (main client, Anthropic wire, auxiliary, /model,
# gateway overrides).
scripts/run_tests.sh \
  tests/hermes_cli/test_provider_route_identity.py \
  tests/hermes_cli/test_custom_provider_extra_headers.py \
  tests/hermes_cli/test_custom_provider_normalize_no_mutate.py \
  tests/hermes_cli/test_provider_config_validation.py \
  tests/hermes_cli/test_runtime_provider_resolution.py \
  tests/hermes_cli/test_keyed_provider_credential_pool.py \
  tests/hermes_cli/test_custom_provider_context_length.py \
  tests/hermes_cli/test_custom_provider_tls.py \
  tests/hermes_cli/test_model_switch_custom_providers.py \
  tests/hermes_cli/test_provider_section3_grouping.py \
  tests/agent/test_custom_provider_extra_headers_client.py \
  tests/agent/test_anthropic_adapter_extra_headers.py \
  tests/agent/test_anthropic_prompt_cache_policy.py \
  tests/agent/test_aux_custom_endpoint_headers.py \
  tests/agent/test_auxiliary_named_custom_providers.py \
  tests/agent/test_session_affinity_header.py \
  tests/agent/test_switch_model_reapplies_headers.py \
  tests/agent/test_switch_model_context.py \
  tests/agent/test_prompt_cache_ttl_propagation.py \
  tests/agent/test_auxiliary_error_policy.py \
  tests/gateway/test_model_command_custom_providers.py \
  tests/gateway/test_session_model_override_routing.py \
  tests/gateway/test_gateway_route_identity.py \
  tests/gateway/test_api_standalone_pending_approvals.py \
  tests/gateway/test_runtime_footer.py \
  tests/gateway/test_session_info.py \
  tests/persistence_sqlite/conversation/test_photo_previews.py \
  tests/gateway/test_hygiene_provider_identity.py \
  tests/gateway/api_server/test_skills.py \
  tests/tools/test_skills_tool_discovery_cache.py \
  tests/gateway/test_session_mutation_compress.py \
  tests/gateway/test_signal.py \
  tests/gateway/test_signal_inbound_attachments.py \
  tests/gateway/test_signal_rate_limit.py \
  tests/gateway/test_native_producer_owner_matrix.py \
  tests/tools/test_signal_media.py \
  tests/hermes_cli/test_status_provider_label.py \
  tests/hermes_cli/test_cmd_update_docker.py \
  tests/ci/test_docs_drift.py \
  tests/ci/test_check_public_pii.py \
  -q

# That file imports the Discord and Slack SDKs at module load, which need the optional
# `messaging` extra. Select only the Signal class and the delivery-mirror regression so the
# locked test environment stays minimal.
scripts/run_tests.sh tests/gateway/outbound/test_send_message_tool.py \
  -k "TestSendSignalChunking or test_successful_send_mirrors_as_labelled_user_turn" -q

# Delivery mirrors and Responses replay: an external send must not land as an assistant turn, and
# a replayed native message id must not sit next to another assistant message.
scripts/run_tests.sh tests/gateway/test_mirror.py -q
scripts/run_tests.sh tests/agent/test_codex_responses_adapter.py \
  -k test_native_message_id_dropped_before_another_assistant_message -q
