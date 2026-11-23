"""Positive and negative tests for scripts/release_plan.py (code-driven release planner)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "release_plan.py"
PRESERVATION_SCRIPT = ROOT / "scripts" / "check_fork_preservation.py"


@pytest.fixture(scope="module")
def rp():
    spec = importlib.util.spec_from_file_location("release_plan", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["release_plan"] = module
    spec.loader.exec_module(module)
    return module


def test_docs_ci_scripts_tests_evals_only_skip_release(rp):
    non_shippable_batches = [
        ["README.md", "DOWNSTREAM.md", "docs/MOBILE_COMPANION.md", "docs/sources/fork_facts.json"],
        [".github/workflows/auto-release.yml", "scripts/release_plan.py", "scripts/docs_fork.py"],
        ["tests/gateway/test_i18n_and_anti_slop.py", "evals/browser_use/README.md"],
        ["README.es.md", ".github/labeler.yml", "tests/scripts/test_release_plan.py"],
    ]
    for files in non_shippable_batches:
        plan = rp.plan_release(
            event="push",
            existing_tags=["v2026.11.22"],
            changed_files=files,
            title="feat(docs): update generated docs and workflows",
        )
        assert plan.should_release is False
        assert plan.tag == ""
        assert plan.bump == "none"
        assert plan.trigger_source == "non_shippable"
        assert plan.shippable_files == ()


@pytest.mark.parametrize(
    ("changed_file", "title", "body", "expected_bump", "expected_tag"),
    [
        ("gateway/platforms/api_server_ui.py", "fix(api): guard empty presence payload", "", "patch", "v2026.11.23"),
        ("agent/prompt_builder.py", "perf(prompt): trim duplicate guidance", "", "patch", "v2026.11.23"),
        ("tools/mobile_deliver.py", "feat(mobile): add rich briefing cards", "", "minor", "v2026.12.0"),
        ("locales/fr.yaml", "feat!: drop legacy locale keys", "", "major", "v2027.0.0"),
        ("hermes_cli/config_providers.py", "refactor(providers): simplify proxy header", "BREAKING CHANGE: removed old header", "major", "v2027.0.0"),
        ("plugins/platforms/signal/signal_media.py", "fix(signal): classify image bytes", "", "patch", "v2026.11.23"),
        ("persistence/authority/coordination.py", "fix(persistence): localize markers", "", "patch", "v2026.11.23"),
        ("Dockerfile", "build(docker): bump base image", "", "patch", "v2026.11.23"),
        ("pyproject.toml", "chore(deps): bump runtime requirement", "", "patch", "v2026.11.23"),
        ("toolsets.py", "feat(tools): register mobile_profiles toolset", "", "minor", "v2026.12.0"),
        ("hermes_constants.py", "fix(constants): tighten attachment cap", "", "patch", "v2026.11.23"),
    ],
)
def test_shippable_runtime_code_triggers_expected_bump(
    rp,
    changed_file: str,
    title: str,
    body: str,
    expected_bump: str,
    expected_tag: str,
):
    plan = rp.plan_release(
        event="push",
        existing_tags=["v0.23.3", "v2026.8.16.2", "v2026.11.22", "canary.20261122T000948Z-v0.23.3"],
        changed_files=["README.md", changed_file, "tests/gateway/test_x.py"],
        title=title,
        body=body,
    )
    assert plan.should_release is True
    assert plan.bump == expected_bump
    assert plan.tag == expected_tag
    assert plan.trigger_source == "auto_code_change"
    assert plan.shippable_files == (changed_file,)


def test_skip_release_label_overrides_runtime_code_and_release_labels(rp):
    plan = rp.plan_release(
        event="push",
        existing_tags=["v2026.11.22"],
        labels=["skip-release", "release: minor"],
        changed_files=["gateway/i18n.py"],
        title="feat(i18n): add new notices",
    )
    assert plan.should_release is False
    assert plan.trigger_source == "skip_label"


def test_explicit_release_label_overrides_commit_prefix_and_non_runtime_files(rp):
    plan = rp.plan_release(
        event="push",
        existing_tags=["v1.2.3"],
        labels=["release: minor"],
        changed_files=["README.md"],
        title="docs: release notes polish",
    )
    assert plan.should_release is True
    assert plan.bump == "minor"
    assert plan.tag == "v1.3.0"
    assert plan.trigger_source == "label"


def test_conflicting_release_labels_fail_closed(rp):
    with pytest.raises(rp.ReleasePlanError, match="Conflicting release labels"):
        rp.plan_release(
            event="push",
            existing_tags=["v1.0.0"],
            labels=["release: minor", "release: patch"],
            changed_files=["gateway/i18n.py"],
            title="feat: ambiguous",
        )


def test_initial_tag_computation_when_no_semver_tags_exist(rp):
    assert rp.compute_next_tag([], "major") == "v1.0.0"
    assert rp.compute_next_tag([], "minor") == "v0.1.0"
    assert rp.compute_next_tag(["canary.20261122T000948Z-v0.23.3", "v2026.8.16.2"], "patch") == "v0.0.1"


def test_workflow_dispatch_explicit_tag_validation_and_collision(rp):
    ok_plan = rp.plan_release(
        event="workflow_dispatch",
        existing_tags=["v2026.11.22"],
        input_tag="v2026.11.25",
    )
    assert ok_plan.should_release is True
    assert ok_plan.tag == "v2026.11.25"
    assert ok_plan.trigger_source == "manual_tag"

    with pytest.raises(rp.ReleasePlanError, match="Invalid SemVer tag"):
        rp.plan_release(
            event="workflow_dispatch",
            existing_tags=["v2026.11.22"],
            input_tag="v2026.11.22.1",
        )

    with pytest.raises(rp.ReleasePlanError, match="already exists"):
        rp.plan_release(
            event="workflow_dispatch",
            existing_tags=["v2026.11.22"],
            input_tag="v2026.11.22",
        )


def test_isolated_cli_writes_github_output_and_json(tmp_path: Path):
    tags_file = tmp_path / "tags.txt"
    tags_file.write_text("v0.23.3\nv2026.11.22\n", encoding="utf-8")
    gh_out = tmp_path / "github_output.txt"

    proc = subprocess.run(
        [
            sys.executable,
            "-I",
            str(SCRIPT),
            "--event",
            "push",
            "--labels-json",
            "[]",
            "--changed-files-json",
            json.dumps(["gateway/platforms/api_server_ui.py"]),
            "--tags-file",
            str(tags_file),
            "--title",
            "feat(mobile): automatic release trigger",
            "--github-output",
            str(gh_out),
            "--json",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        stdin=subprocess.DEVNULL,
        check=False,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}"
    payload = json.loads(proc.stdout)
    assert payload["should_release"] is True
    assert payload["tag"] == "v2026.12.0"
    assert payload["bump"] == "minor"
    out_text = gh_out.read_text(encoding="utf-8")
    assert "should_release=true\n" in out_text
    assert "tag=v2026.12.0\n" in out_text


def test_fork_preservation_protects_release_plan():
    spec = importlib.util.spec_from_file_location("check_fork_preservation", PRESERVATION_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_fork_preservation"] = mod
    spec.loader.exec_module(mod)

    assert "scripts/release_plan.py" in mod.PROTECTED_EXACT_PATHS
    assert "tests/scripts/test_release_plan.py" in mod.PROTECTED_EXACT_PATHS
