"""Structural & security invariant tests for fork-owned CI workflows and sharded verification."""

from pathlib import Path
import re
import subprocess
import unittest

from ruamel.yaml import YAML


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = ROOT / ".github/workflows"
OWNED_WORKFLOWS_FILE = ROOT / "scripts/fork-owned-workflows.txt"
LABELER_CONFIG = ROOT / ".github/labeler.yml"
RELEASE_CONFIG = ROOT / ".github/release.yml"
VERIFY_SCRIPT = ROOT / "scripts/downstream-verify.sh"

_YAML = YAML(typ="safe")

SHA_PIN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+@[0-9a-f]{40}$")
REQUIRED_LABELS = {
    "area: api-server",
    "area: mobile-bridge",
    "area: security",
    "area: concurrency",
    "area: i18n",
    "area: docs",
    "area: ci",
    "area: tests",
    "area: docker",
    "upstream-sync",
}


def _load_yaml(path: Path) -> dict:
    data = _YAML.load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"Expected YAML mapping in {path}"
    return data


def _trigger_block(wf: dict) -> dict:
    # PyYAML 1.1 parses bare `on:` as boolean True.
    return wf.get("on", wf.get(True, {}))


class ForkCiWorkflowsTest(unittest.TestCase):
    def test_owned_workflows_exist_parse_and_pin_all_actions(self):
        owned = [
            line.strip()
            for line in OWNED_WORKFLOWS_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        self.assertIn("auto-release.yml", owned)
        self.assertIn("pr-labeler.yml", owned)
        self.assertIn("downstream-ci.yml", owned)
        self.assertIn("docker-publish.yml", owned)
        self.assertIn("codeql.yml", owned)

        for name in owned:
            wf_path = WORKFLOWS_DIR / name
            self.assertTrue(wf_path.is_file(), f"Missing owned workflow: {name}")
            wf = _load_yaml(wf_path)
            triggers = _trigger_block(wf)
            if name != "pii-guard.yml":
                self.assertNotIn("pull_request_target", triggers, f"{name} must never use pull_request_target")

            for job_id, job in (wf.get("jobs") or {}).items():
                for step in job.get("steps") or []:
                    uses = step.get("uses")
                    if not uses or uses.startswith("./"):
                        continue
                    self.assertRegex(
                        uses,
                        SHA_PIN,
                        f"{name} job '{job_id}' has non-SHA-pinned action: {uses}",
                    )

    def test_downstream_ci_has_routing_fast_static_sharded_verify_and_fail_closed_gate(self):
        wf = _load_yaml(WORKFLOWS_DIR / "downstream-ci.yml")
        triggers = _trigger_block(wf)
        self.assertIn("merge_group", triggers)
        self.assertIn("checks_requested", triggers["merge_group"].get("types", []))
        self.assertIn("merge_group", str(wf["concurrency"]["cancel-in-progress"]))

        jobs = wf["jobs"]
        self.assertIn("changes", jobs)
        self.assertIn("fast-static", jobs)
        self.assertIn("verify", jobs)
        self.assertIn("workflow-lint", jobs)
        self.assertIn("gate", jobs)

        route_step = jobs["changes"]["steps"][1]
        self.assertIn("merge_group.base_sha", route_step["env"]["BASE_SHA"])
        self.assertIn("merge_group.head_sha", route_step["env"]["HEAD_SHA"])
        self.assertIn('"merge_group"', route_step["run"])

        self.assertEqual(jobs["verify"]["strategy"]["matrix"]["shard"], [1, 2])
        self.assertEqual(jobs["gate"]["if"], "always()")
        gate_needs = set(jobs["gate"]["needs"])
        self.assertTrue({"changes", "fast-static", "verify", "workflow-lint"}.issubset(gate_needs))

    def test_pii_guard_supports_merge_group_without_weakening_trust_boundary(self):
        wf = _load_yaml(WORKFLOWS_DIR / "pii-guard.yml")
        triggers = _trigger_block(wf)
        self.assertIn("pull_request_target", triggers)
        self.assertIn("merge_group", triggers)
        self.assertIn("checks_requested", triggers["merge_group"].get("types", []))
        self.assertIn("merge_group", str(wf["concurrency"]["cancel-in-progress"]))

        steps = wf["jobs"]["scan"]["steps"]
        self.assertIn("merge_group.base_sha", steps[0]["with"]["ref"])
        self.assertIn("merge_group.head_sha", steps[1]["with"]["ref"])
        self.assertIn("github.repository", steps[1]["with"]["repository"])
        self.assertEqual(steps[1]["with"]["path"], "candidate")

    def test_codeql_and_docker_publish_skip_docs_only_work_without_weakening_main_or_tags(self):
        codeql = _load_yaml(WORKFLOWS_DIR / "codeql.yml")
        triggers = _trigger_block(codeql)
        self.assertIn("push", triggers)
        self.assertIn("schedule", triggers)
        pr_ignore = triggers["pull_request"].get("paths-ignore", [])
        self.assertIn("**/*.md", pr_ignore)
        self.assertIn("docs/**", pr_ignore)

        docker = _load_yaml(WORKFLOWS_DIR / "docker-publish.yml")
        d_triggers = _trigger_block(docker)
        self.assertIn("v*", d_triggers["push"]["tags"])
        self.assertIn("workflow_dispatch", d_triggers)
        self.assertIn("should_build", docker["jobs"]["prepare"]["outputs"])
        self.assertEqual(docker["jobs"]["build"]["if"], "needs.prepare.outputs.should_build == 'true'")

    def test_pr_labeler_uses_safe_pull_request_trigger_and_covers_all_required_labels(self):
        wf = _load_yaml(WORKFLOWS_DIR / "pr-labeler.yml")
        triggers = _trigger_block(wf)
        self.assertIn("pull_request", triggers)
        self.assertNotIn("pull_request_target", triggers)
        label_job = wf["jobs"]["label"]
        self.assertEqual(
            label_job["permissions"],
            {"contents": "read", "pull-requests": "write"},
        )
        self.assertIn("github.repository", label_job["if"])

        labeler_cfg = _load_yaml(LABELER_CONFIG)
        self.assertTrue(
            REQUIRED_LABELS.issubset(set(labeler_cfg.keys())),
            f"Missing labels: {REQUIRED_LABELS - set(labeler_cfg.keys())}",
        )

    def test_auto_release_workflow_and_release_notes_config_are_valid(self):
        rel = _load_yaml(RELEASE_CONFIG)
        self.assertIn("skip-release", rel["changelog"]["exclude"]["labels"])
        categories = rel["changelog"]["categories"]
        self.assertGreaterEqual(len(categories), 6)

        wf = _load_yaml(WORKFLOWS_DIR / "auto-release.yml")
        triggers = _trigger_block(wf)
        self.assertIn("workflow_dispatch", triggers)
        self.assertIn("main", triggers["push"]["branches"])
        job = wf["jobs"]["release"]
        self.assertEqual(job["permissions"]["contents"], "write")
        self.assertEqual(job["permissions"]["actions"], "write")
        plan_run = job["steps"][1]["run"]
        self.assertIn("scripts/release_plan.py", plan_run)
        self.assertIn("--changed-files-json", plan_run)
        self.assertIn("--paginate --jq '.[].filename'", plan_run)
        self.assertNotIn("--paginate --jq '[.[].filename]'", plan_run)
        self.assertIn("RELEASE_BOT_TOKEN", job["steps"][2]["env"]["GH_TOKEN"])

    def test_downstream_verify_sharding_partitions_all_files_with_zero_overlap(self):
        shard1 = subprocess.check_output(
            ["bash", str(VERIFY_SCRIPT), "--list-shard", "1/2"], text=True
        ).splitlines()
        shard2 = subprocess.check_output(
            ["bash", str(VERIFY_SCRIPT), "--list-shard", "2/2"], text=True
        ).splitlines()
        all_files = subprocess.check_output(
            ["bash", str(VERIFY_SCRIPT), "--list-shard", "1/1"], text=True
        ).splitlines()

        s1 = [line.strip() for line in shard1 if line.strip()]
        s2 = [line.strip() for line in shard2 if line.strip()]
        full = [line.strip() for line in all_files if line.strip()]

        self.assertGreater(len(s1), 20)
        self.assertGreater(len(s2), 20)
        self.assertEqual(set(s1) & set(s2), set(), "Shards must not duplicate test files")
        self.assertEqual(set(s1) | set(s2), set(full), "Union of shards must equal full suite")
        self.assertLessEqual(abs(len(s1) - len(s2)), 1, "Shards must be balanced")

    def test_downstream_verify_rejects_invalid_shard_spec(self):
        for bad in ("0/2", "3/2", "1/0", "abc", "1/2/3"):
            proc = subprocess.run(
                ["bash", str(VERIFY_SCRIPT), "--list-shard", bad],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0, f"Expected failure for bad shard {bad!r}")


if __name__ == "__main__":
    unittest.main()
