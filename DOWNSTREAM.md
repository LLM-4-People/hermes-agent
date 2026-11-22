# LLM-4-People Hermes downstream

This fork tracks `NousResearch/hermes-agent` and carries a small, isolated patch set. The point
of the layout is that upstream updates arrive automatically and merge cleanly: every downstream
change lives in its own commit and touches as few upstream files as possible.

## Downstream changes

| Change | Owner files | Why |
|---|---|---|
| Provider-aware route metadata | `hermes_cli/config_providers.py`, `hermes_cli/runtime_provider_custom.py`, `agent/agent_init.py`, `agent/client_lifecycle.py`, `agent/anthropic_adapter.py`, `agent/auxiliary_client.py`, `agent/opencode_affinity.py`, `agent/reasoning_promotion.py`, `hermes_cli/model_switch_providers.py`, `hermes_cli/main_provider_setup.py`, `hermes_cli/model_setup_flows_custom.py`, `hermes_cli/models_local.py`, `acp_adapter/model_catalog.py` | `extra_headers`/TLS/api-mode/context/capability lookups matched by base URL, so two providers behind one proxy shared the first entry's metadata. `_entries_for_route` now selects by provider identity (key, display name, `custom:<slug>`) and every consumer passes it. Routing metadata fails closed for a bare identity whose entry was deleted or renamed; capability declarations keep the URL-scoped fallback, because they are a property of the model and endpoint. |
| `providers.<name>.proxy` | `hermes_cli/config_providers.py`, docs | First-class upstream routing for gateways: `proxy.base_url` is sent as `X-Proxy-Base-URL` (configurable via `proxy.header`), with `proxy.headers` and `extra_headers` overrides. Removes hand-written routing headers. |
| Footer `provider` field | `gateway/runtime_footer.py`, `gateway/run_turn_runner.py`, `gateway/run_turn_persistence.py`, `gateway/slash_commands.py`, `hermes_cli/providers.py` | The gateway reply footer now renders `model · provider · ctx% · cwd`; named custom providers show the configured entry name, not the internal slug. |
| Signal media in split deployments | `plugins/platforms/signal/signal_media.py`, `adapter.py`, `standalone.py`, `gateway_env.py`, `plugin.yaml` | signal-cli resolves attachment paths in its own container. Outbound attachments are now self-contained data URIs (`SIGNAL_INLINE_ATTACHMENTS=never` restores paths for shared volumes); inbound documents keep the sender's filename/extension and images with no `contentType` are classified from magic bytes. |
| Delivery mirrors stay out of assistant history | `gateway/outbound/send.py`, `gateway/mirror.py` | `send_message` is never agent-callable, so its mirror is external text and lands as a labelled user turn. An assistant-role mirror after the agent's reply breaks strict alternation. |
| Ambiguous native message ids | `agent/codex_responses_adapter.py` | A replayed Responses `message` id followed directly by another assistant message covers two turns; a strict Responses adapter rejects it (400 "Ambiguous Responses native assistant history"). The id is only a prefix-cache hint, so the earlier message keeps its content/status/phase and loses the id. |
| Image identity | `Dockerfile`, `hermes_cli/config.py` | `HERMES_IMAGE_NAME` build arg is baked into `/etc/hermes/image-provenance.json`, so `hermes update` tells operators to pull this fork's image instead of upstream's. |

Upstream files we do **not** modify: `.github/workflows/*` (upstream's workflows are disabled
at the repository level; ours are additive files), `pyproject.toml`, `uv.lock`, `pm/**`. A
`disable-upstream-workflows` job re-applies that invariant on every push to `main`, so a newly
added upstream workflow cannot run against Blacksmith runners or Docker Hub credentials the fork
does not have.

## Automatic upstream sync

`.github/workflows/sync-upstream.yml` runs every three hours:

1. merge `upstream/main` into the long-lived `automation/upstream-sync` branch (reused so the
   push is a fast-forward);
2. set up the locked toolchain from the trusted `main` checkout, with cache writes disabled, and
   run `scripts/downstream-verify.sh` on the merged tree (static lint + the provider-routing,
   footer and Signal suites);
3. capture the merged workflow files and audit them from a clean `main` checkout with the shared
   `.github/actions/workflow-lint` action;
4. bundle the verified commit; a separate publish job fetches the bundle into a bare repository
   and pushes it with a credential scoped to that step. No merged code runs in the publish job;
5. open or update the `chore: sync upstream main` pull request.

`main` is never written directly by automation. Merging the PR triggers the normal push-triggered
image build, so the published `edge` image always corresponds to a reviewed commit. A repository
ruleset also rejects direct pushes to `main`, so the PR-only rule is mechanically enforced.

The `main-pr-only` ruleset intentionally has no required checks or approvals: the sync PR is
created with `GITHUB_TOKEN`, whose `pull_request`/`pull_request_target` runs GitHub suppresses, so
a required-check rule would leave every automatic sync permanently unmergeable. The public-tree
hygiene job therefore runs and fails visibly on human pull requests but is not a merge gate until
the sync workflow opens its PR with a GitHub App installation token. That app token is the next
step if a second reviewer joins the fork; until then this is an explicit single-maintainer
exception.

A conflict aborts the merge, pushes nothing, and opens/updates the `upstream-sync` issue with the
conflicting files. Resolve on the automation branch:

```bash
git fetch origin
git switch automation/upstream-sync
git merge upstream/main
# resolve, then run the gate
scripts/downstream-verify.sh
git push origin automation/upstream-sync
```

`workflow_dispatch` with `dry_run: true` merges and verifies without pushing the branch or opening
a PR.

## Security scanning

CodeQL runs on every push to `main`, every pull request targeting `main`, and weekly. The first
`main` analysis is the fork's baseline; pull-request scans compare against it and only report
genuinely new alerts. If the baseline is ever deleted, start main-only again until it is recorded.

## Image

- Registry: `ghcr.io/llm-4-people/hermes-agent`
- `edge`, `latest` + `sha-<short>` on every green `main` push
- `X.Y.Z`, `X.Y`, `latest`, `sha-<short>` on a valid `vX.Y.Z` tag (other `v*` tags are rejected)
- Multi-arch: `linux/amd64`, `linux/arm64` (native GitHub arm runners)
- Supply chain: BuildKit provenance (`mode=max`), SPDX SBOM, keyless cosign signature on the
  manifest index.

Verify a digest before deploying:

```bash
cosign verify ghcr.io/llm-4-people/hermes-agent@sha256:<digest> \
  --certificate-identity-regexp '^https://github\.com/LLM-4-People/hermes-agent/\.github/workflows/docker-publish\.yml@refs/(heads/main|tags/v[0-9]+\.[0-9]+\.[0-9]+)$' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

The smoke job also runs the published image and imports
`hermes_cli.config_providers.normalize_proxy_headers`, so a green publish proves the downstream
patch is present in the image.

## Public-tree hygiene

This repository is public, so operator infrastructure is never committed: internal hostnames, IP
addresses, phone numbers, provider slugs and account-specific model ids are all identifying
details. `scripts/ci/check_public_pii.py` (also available locally as
`scripts/check --only public-pii`) rejects any tracked file that contains one.

The identifiers themselves are **not** in the tree — publishing the denylist would defeat it. They
live in the repository secret `PII_DENYLIST` (newline-separated literals or Python regular
expressions prefixed with `re:`; matching is case-insensitive and each rule is matched
independently). `.github/workflows/pii-guard.yml` runs on `pull_request_target`: the workflow and
checker come from the trusted base commit, the candidate is checked out only as data, and the scan
runs with `HERMES_PUBLIC_PII_REQUIRE=1`. No pull-request code executes in that job, so neither a
same-repo PR nor the automation branch (which may contain merged upstream code) can read the
denylist. There is deliberately no `push` or manual trigger, because either would run a
caller-selected checker with the secret; direct pushes to `main` are blocked by the ruleset, so
every merged tree was scanned at its pull request head. During bootstrap, run the checker locally.

Locally, put the same newline-separated list in `~/.config/hermes/pii-denylist` (or point
`HERMES_PUBLIC_PII_DENYLIST` at another file) and run `scripts/check --only public-pii`. Findings
report `path:line` with the matched text redacted; generated base64 lines (inline source maps) are
counted and skipped.

## Local verification

```bash
source ./activate                       # or use the CI-provided PM test environment
scripts/downstream-verify.sh
```

The test/static gate has two shared owners: `scripts/downstream-verify.sh` for the Python checks
and `.github/actions/workflow-lint` for the workflow audit. `downstream-ci.yml` and
`sync-upstream.yml` both invoke them, so CI and the merge gate cannot drift.
