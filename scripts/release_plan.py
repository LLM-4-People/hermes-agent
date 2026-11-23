#!/usr/bin/env python3
"""Deterministic, fail-closed release classifier and SemVer tag planner for LLM-4-People/hermes-agent.

Single source of truth invoked by ``.github/workflows/auto-release.yml``. Zero third-party
dependencies (``python3 -I`` safe).

Precedence for push events on ``main``:
  1. ``skip-release`` label -> never release.
  2. Explicit ``release: major`` / ``release: minor`` / ``release: patch`` label -> honored
     (fails closed when multiple conflicting release labels are present).
  3. Automatic code-change classification -> releases ONLY when at least one shippable runtime
     file changed (skipping docs-only, tests-only, evals-only, and CI/scripts-only changes),
     inferring the bump level from Conventional Commits:
       - ``feat!:`` / ``<type>!:`` / ``BREAKING CHANGE`` -> ``major``
       - ``feat:`` / ``feat(...):`` / ``enhancement`` label -> ``minor``
       - ``fix:`` / ``perf:`` / ``refactor:`` / other runtime changes -> ``patch``

All generated tags strictly follow ``vMAJOR.MINOR.PATCH`` so ``docker-publish.yml`` and Cosign
attestation verification always accept them cleanly without colliding with existing tags.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fnmatch import fnmatch
import json
from pathlib import Path
import re
import sys
from typing import Any

SEMVER_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
_CONVENTIONAL_BREAKING_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*(?:\([^)\n]+\))?!:")
_CONVENTIONAL_FEAT_RE = re.compile(r"^feat(?:\([^)\n]+\))?:", re.IGNORECASE)
_BREAKING_BODY_RE = re.compile(r"(?m)^BREAKING[ -]CHANGE\s*:")

BUMP_LABEL_MAP: dict[str, str] = {
    "release: major": "major",
    "release:major": "major",
    "release: minor": "minor",
    "release:minor": "minor",
    "release: patch": "patch",
    "release:patch": "patch",
}

VALID_BUMPS: frozenset[str] = frozenset({"major", "minor", "patch"})

# Shippable runtime code prefixes & root-level runtime files.
SHIPPABLE_PREFIXES: tuple[str, ...] = (
    "gateway/",
    "agent/",
    "tools/",
    "locales/",
    "hermes_cli/",
    "plugins/",
    "persistence/",
    "providers/",
    "hermes_agent/",
)

SHIPPABLE_EXACT_PATHS: frozenset[str] = frozenset({
    "run_agent.py",
    "model_tools.py",
    "toolsets.py",
    "Dockerfile",
    "pyproject.toml",
    "setup.py",
    "uv.lock",
})

SHIPPABLE_ROOT_GLOBS: tuple[str, ...] = (
    "hermes_*.py",
)


class ReleasePlanError(ValueError):
    """Raised when release inputs are invalid, conflicting, or would collide with an existing tag."""


@dataclass(frozen=True)
class ReleasePlan:
    should_release: bool
    tag: str
    bump: str
    trigger_source: str
    reason: str
    shippable_files: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_release": self.should_release,
            "tag": self.tag,
            "bump": self.bump,
            "trigger_source": self.trigger_source,
            "reason": self.reason,
            "shippable_files": list(self.shippable_files),
        }


def normalize_repo_path(raw: str) -> str:
    cleaned = raw.strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned.lstrip("/")


def is_shippable_runtime_path(path: str) -> bool:
    """Return True iff ``path`` belongs to shippable runtime code, packaging, or locales."""
    norm = normalize_repo_path(path)
    if not norm:
        return False
    if norm in SHIPPABLE_EXACT_PATHS:
        return True
    if any(norm.startswith(prefix) for prefix in SHIPPABLE_PREFIXES):
        return True
    if "/" not in norm and any(fnmatch(norm, pat) for pat in SHIPPABLE_ROOT_GLOBS):
        return True
    return False


def filter_shippable_files(changed_files: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    seen: list[str] = []
    seen_set: set[str] = set()
    for raw in changed_files:
        if not isinstance(raw, str):
            continue
        norm = normalize_repo_path(raw)
        if norm and norm not in seen_set and is_shippable_runtime_path(norm):
            seen.append(norm)
            seen_set.add(norm)
    return tuple(seen)


def infer_bump_from_commit(title: str, body: str = "", labels: set[str] | frozenset[str] = frozenset()) -> str:
    """Infer ``major`` / ``minor`` / ``patch`` from Conventional Commits title/body and labels."""
    clean_title = (title or "").strip()
    clean_body = (body or "").strip()
    lower_labels = {lbl.strip().lower() for lbl in labels if lbl.strip()}

    if _CONVENTIONAL_BREAKING_RE.match(clean_title) or _BREAKING_BODY_RE.search(clean_title) or _BREAKING_BODY_RE.search(clean_body):
        return "major"
    if _CONVENTIONAL_FEAT_RE.match(clean_title) or "enhancement" in lower_labels:
        return "minor"
    return "patch"


def compute_next_tag(existing_tags: list[str] | tuple[str, ...], bump: str) -> str:
    """Compute the next strict ``vMAJOR.MINOR.PATCH`` tag above the highest existing 3-part tag."""
    if bump not in VALID_BUMPS:
        raise ReleasePlanError(f"Unsupported bump level {bump!r}; expected one of {sorted(VALID_BUMPS)}")

    parsed: list[tuple[int, int, int, str]] = []
    all_tags: set[str] = set()
    for raw in existing_tags:
        tag = str(raw).strip()
        if not tag:
            continue
        all_tags.add(tag)
        match = SEMVER_TAG_RE.match(tag)
        if match:
            parsed.append((int(match.group(1)), int(match.group(2)), int(match.group(3)), tag))

    parsed.sort()
    latest = parsed[-1] if parsed else None

    if latest is None:
        major, minor, patch = (1, 0, 0) if bump == "major" else ((0, 1, 0) if bump == "minor" else (0, 0, 1))
    else:
        major, minor, patch = latest[0], latest[1], latest[2]
        if bump == "major":
            major, minor, patch = major + 1, 0, 0
        elif bump == "minor":
            minor, patch = minor + 1, 0
        else:
            patch += 1

    next_tag = f"v{major}.{minor}.{patch}"
    if next_tag in all_tags:
        raise ReleasePlanError(f"Computed tag {next_tag} already exists")
    return next_tag


def plan_release(
    *,
    event: str,
    existing_tags: list[str] | tuple[str, ...],
    labels: list[str] | tuple[str, ...] = (),
    changed_files: list[str] | tuple[str, ...] = (),
    title: str = "",
    body: str = "",
    input_bump: str = "patch",
    input_tag: str = "",
) -> ReleasePlan:
    """Determine whether to publish a release and resolve its exact ``vMAJOR.MINOR.PATCH`` tag."""
    existing_set = {str(t).strip() for t in existing_tags if str(t).strip()}

    if event == "workflow_dispatch":
        explicit_tag = (input_tag or "").strip()
        if explicit_tag:
            if not SEMVER_TAG_RE.match(explicit_tag):
                raise ReleasePlanError(f"Invalid SemVer tag {explicit_tag!r}; expected vMAJOR.MINOR.PATCH")
            if explicit_tag in existing_set:
                raise ReleasePlanError(f"Tag {explicit_tag} already exists")
            return ReleasePlan(
                should_release=True,
                tag=explicit_tag,
                bump="explicit",
                trigger_source="manual_tag",
                reason=f"Manual workflow_dispatch with explicit tag {explicit_tag}.",
                shippable_files=(),
            )
        bump = (input_bump or "patch").strip().lower() or "patch"
        next_tag = compute_next_tag(existing_tags, bump)
        return ReleasePlan(
            should_release=True,
            tag=next_tag,
            bump=bump,
            trigger_source="manual_bump",
            reason=f"Manual workflow_dispatch with {bump} bump.",
            shippable_files=(),
        )

    normalized_labels = {str(lbl).strip().lower() for lbl in labels if str(lbl).strip()}
    if "skip-release" in normalized_labels:
        return ReleasePlan(
            should_release=False,
            tag="",
            bump="none",
            trigger_source="skip_label",
            reason="skip-release label present; skipping release.",
            shippable_files=(),
        )

    matched_bumps = sorted({BUMP_LABEL_MAP[lbl] for lbl in normalized_labels if lbl in BUMP_LABEL_MAP})
    if len(matched_bumps) > 1:
        raise ReleasePlanError(f"Conflicting release labels found on PR: {matched_bumps}")

    shippable = filter_shippable_files(changed_files)
    if len(matched_bumps) == 1:
        bump = matched_bumps[0]
        next_tag = compute_next_tag(existing_tags, bump)
        return ReleasePlan(
            should_release=True,
            tag=next_tag,
            bump=bump,
            trigger_source="label",
            reason=f"Explicit release label selected {bump} bump.",
            shippable_files=shippable,
        )

    if not shippable:
        return ReleasePlan(
            should_release=False,
            tag="",
            bump="none",
            trigger_source="non_shippable",
            reason="No shippable runtime files changed (docs/tests/CI-only change); skipping release.",
            shippable_files=(),
        )

    bump = infer_bump_from_commit(title=title, body=body, labels=normalized_labels)
    next_tag = compute_next_tag(existing_tags, bump)
    return ReleasePlan(
        should_release=True,
        tag=next_tag,
        bump=bump,
        trigger_source="auto_code_change",
        reason=f"Automatic {bump} release triggered by {len(shippable)} shippable runtime file(s).",
        shippable_files=shippable,
    )


def _load_json_list(raw: str, *, flag: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReleasePlanError(f"Invalid JSON for {flag}: {exc}") from exc
    if not isinstance(parsed, list):
        raise ReleasePlanError(f"Expected a JSON array for {flag}")
    return [str(item) for item in parsed]


def _write_github_output(path: Path, plan: ReleasePlan) -> None:
    lines = [
        f"should_release={'true' if plan.should_release else 'false'}",
        f"tag={plan.tag}",
        f"bump={plan.bump}",
        f"trigger_source={plan.trigger_source}",
        f"reason={plan.reason}",
    ]
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="release_plan.py",
        description="Classify merged changes and resolve the next vMAJOR.MINOR.PATCH release tag.",
    )
    parser.add_argument("--event", default="push", choices=("push", "workflow_dispatch"))
    parser.add_argument("--labels-json", default="[]", help="JSON array of PR label strings")
    parser.add_argument("--changed-files-json", default="[]", help="JSON array of changed repo paths")
    parser.add_argument("--tags-json", default="", help="JSON array of existing git tags")
    parser.add_argument("--tags-file", type=Path, help="Optional newline-delimited file of existing git tags")
    parser.add_argument("--title", default="", help="Merged PR title or commit subject")
    parser.add_argument("--body", default="", help="Merged PR body or commit body")
    parser.add_argument("--input-bump", default="patch", help="Manual workflow_dispatch bump level")
    parser.add_argument("--input-tag", default="", help="Manual workflow_dispatch explicit tag")
    parser.add_argument("--github-output", type=Path, help="Optional GITHUB_OUTPUT path to append outputs")
    parser.add_argument("--json", action="store_true", help="Print structured JSON plan to stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        labels = _load_json_list(args.labels_json, flag="--labels-json")
        changed_files = _load_json_list(args.changed_files_json, flag="--changed-files-json")
        if args.tags_file is not None:
            existing_tags = [
                line.strip()
                for line in args.tags_file.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ]
        else:
            existing_tags = _load_json_list(args.tags_json, flag="--tags-json")

        plan = plan_release(
            event=args.event,
            existing_tags=existing_tags,
            labels=labels,
            changed_files=changed_files,
            title=args.title,
            body=args.body,
            input_bump=args.input_bump,
            input_tag=args.input_tag,
        )
    except (ReleasePlanError, OSError) as exc:
        sys.stderr.write(f"::error::{exc}\n")
        return 1

    if args.github_output is not None:
        _write_github_output(args.github_output, plan)

    if args.json:
        sys.stdout.write(json.dumps(plan.to_dict(), indent=2) + "\n")
    else:
        status = f"release {plan.tag} ({plan.bump})" if plan.should_release else "skip"
        sys.stdout.write(f"release_plan: {status} — {plan.reason}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
