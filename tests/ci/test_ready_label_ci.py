# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Exercise the workflow guards against PR lifecycle events without a runner."""

import copy
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import regex as re
import yaml

from tests.helpers.paths import REPO_ROOT

ROOT = REPO_ROOT
_WORKFLOW_COMMAND_TIMEOUT = 10.0


def run_workflow_command(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    input: str | None = None,
) -> subprocess.CompletedProcess[str]:
    # These workflow snippets use mocked tools. Do not load CI shell startup
    # files, exported functions, or Node preload options from the parent.
    child_env = {
        name: os.environ[name]
        for name in ("PATH", "SYSTEMROOT", "WINDIR", "TMPDIR", "TMP", "TEMP", "LANG", "LC_ALL")
        if name in os.environ
    }
    child_env.update(env or {})
    return subprocess.run(
        command,
        env=child_env,
        input=input,
        stdin=None if input is not None else subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=_WORKFLOW_COMMAND_TIMEOUT,
    )


def run_accelerator_detection(
    directory: Path, devices: tuple[str, ...], lane: str
) -> subprocess.CompletedProcess[str]:
    workflow = yaml.safe_load((ROOT / ".github/workflows/test-levels.yml").read_text())
    detect = workflow["jobs"]["level"]["steps"][0]
    for name in devices:
        executable = directory / name
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
    return run_workflow_command(
        [shutil.which("bash") or "bash", "-eu", "-o", "pipefail", "-c", detect["run"]],
        env={
            "PATH": str(directory),
            "GITHUB_OUTPUT": str(directory / "output"),
            "GITHUB_STEP_SUMMARY": str(directory / "summary"),
            "TEST_LEVEL": "l3",
            "LANE_DEVICE": lane,
        },
    )


def namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: namespace(item) for key, item in value.items()})
    return value


def context(
    action: str = "synchronize",
    *,
    ready: bool = True,
    draft: bool = False,
    label: str = "",
    event: str = "pull_request",
    number: int = 481,
    head_ref: str = "feature",
    head_repo: str = "owner/vllm-gr",
    base_ref: str = "main",
) -> Any:
    return namespace(
        {
            "workflow": "test",
            "repository": "JiusiServe/vllm-gr",
            "event_name": event,
            "run_id": 123,
            "ref": "refs/pull/481/merge",
            "ref_name": "481/merge",
            "sha": "merge-sha",
            "event": {
                "action": action,
                "label": {"name": label},
                "pull_request": {
                    "number": number,
                    "draft": draft,
                    "labels": [{"name": "ready"}] if ready else [],
                    "base": {"ref": base_ref, "sha": "base-sha"},
                    "head": {
                        "ref": head_ref,
                        "repo": {"full_name": head_repo},
                    },
                },
            },
        }
    )


def evaluate(
    expression: str,
    github: Any,
    results: dict[str, str] | None = None,
    *,
    cancelled: bool = False,
    benchmark_device: str = "npu",
    docs_only: bool = False,
) -> Any:
    # Evaluate only the small expression subset used by these tracked workflow
    # guards. Keep the expressions in YAML as the source of truth.
    expression = expression.strip().removeprefix("${{").removesuffix("}}").strip()
    expression = expression.replace("github.event.pull_request.labels.*.name", "labels")
    expression = expression.replace("needs.benchmark-policy.outputs.policy-json", "policy_json")
    expression = expression.replace("needs.docs-scope.outputs.docs-only", "docs_only")
    expression = re.sub(r"needs\.([\w-]+)\.result", r"results['\1']", expression)
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?!=)", " not ", expression)
    expression = re.sub(r"\btrue\b", "True", expression)
    expression = re.sub(r"\bfalse\b", "False", expression)
    return eval(  # noqa: S307 - only repository-owned workflow expressions
        " ".join(expression.split()),
        {"__builtins__": {}},
        {
            "github": github,
            "labels": [item["name"] for item in github.event.pull_request.labels],
            "results": results or {},
            "policy_json": json.dumps({"BENCHMARK_DEVICE": benchmark_device}),
            "fromJSON": lambda value: namespace(json.loads(value)),
            "docs_only": "True" if docs_only else "False",
            "always": lambda: True,
            "cancelled": lambda: cancelled,
            "contains": lambda items, value: value in items,
            "format": lambda template, *args: template.format(*args),
        },
    )


def render(template: str, github: Any) -> str:
    rendered = re.sub(
        r"\$\{\{(.*?)\}\}",
        lambda match: str(evaluate(match.group(1), github)),
        template,
        flags=re.DOTALL,
    )
    assert isinstance(rendered, str)
    return rendered


class TestReadyLabelCI(unittest.TestCase):
    def setUp(self) -> None:
        self.workflows = {
            path.name: yaml.safe_load(path.read_text())
            for path in (ROOT / ".github/workflows").glob("*.yml")
            if path.name in {"ci.yml", "build_wheel.yml"}
        }

    def scheduled(
        self,
        workflow: dict[str, Any],
        github: Any,
        failed: str | None = None,
        *,
        cancelled: bool = False,
        benchmark_device: str = "npu",
        docs_only: bool = False,
    ) -> list[str]:
        results: dict[str, str] = {}
        running = []
        pending = dict(workflow["jobs"])
        while pending:
            progressed = False
            for name, job in list(pending.items()):
                needs = job.get("needs", [])
                if isinstance(needs, str):
                    needs = [needs]
                if any(need not in results for need in needs):
                    continue
                condition = job.get("if", "true")
                dependencies_ok = all(results[need] == "success" for need in needs)
                explicit_status = any(status in condition for status in ("always()", "cancelled()"))
                enabled = (explicit_status or (dependencies_ok and not cancelled)) and evaluate(
                    condition,
                    github,
                    results,
                    cancelled=cancelled,
                    benchmark_device=benchmark_device,
                    docs_only=docs_only and results.get("docs-scope") == "success",
                )
                results[name] = (
                    ("failure" if name == failed else "success") if enabled else "skipped"
                )
                if enabled:
                    running.append(name)
                del pending[name]
                progressed = True
            if not progressed:
                raise AssertionError(f"unresolved job dependencies: {set(pending)}")
        return running

    def test_pr_lifecycle(self) -> None:
        events = [
            ("opened", False, False, "", False),
            ("synchronize", False, False, "", False),
            ("reopened", False, False, "", False),
            ("ready_for_review", False, False, "", False),
            ("labeled", True, False, "ready", True),
            ("synchronize", True, False, "", True),
            ("reopened", True, False, "", True),
            ("ready_for_review", True, False, "", True),
            ("labeled", True, True, "ready", False),
            ("synchronize", True, True, "", False),
            ("converted_to_draft", True, True, "", False),
            ("unlabeled", False, False, "ready", False),
            ("labeled", True, False, "bug", False),
            ("unlabeled", True, False, "bug", False),
            ("labeled", False, False, "bug", False),
        ]
        for filename, workflow in self.workflows.items():
            triggers = workflow.get("on", workflow.get(True))["pull_request"]["types"]
            for action, ready, draft, label, expected in events:
                with self.subTest(
                    workflow=filename, action=action, ready=ready, draft=draft, label=label
                ):
                    self.assertIn(action, triggers)
                    github = context(action, ready=ready, draft=draft, label=label)
                    running = self.scheduled(workflow, github)
                    validation = [job for job in running if job != "benchmark-blocking"]
                    self.assertEqual(bool(validation), expected)
                    if expected:
                        self.assertEqual(set(running), set(workflow["jobs"]) - {"docs-check"})
                    if "benchmark-blocking" in workflow["jobs"]:
                        ignored_label = action in {"labeled", "unlabeled"} and label != "ready"
                        self.assertEqual("benchmark-blocking" in running, not ignored_label)
                        gate_name = render(workflow["jobs"]["benchmark-blocking"]["name"], github)
                        self.assertEqual(gate_name, "Benchmark blocking gate")
                    if "run-name" in workflow:
                        self.assertEqual(
                            render(workflow["run-name"], github).startswith("ci-meta:v1;"),
                            expected,
                        )

    @pytest.mark.l1  # type: ignore[misc]
    def test_reviews_do_not_trigger_validation(self) -> None:
        for filename, workflow in self.workflows.items():
            with self.subTest(workflow=filename):
                triggers = workflow.get("on", workflow.get(True))
                self.assertNotIn("pull_request_review", triggers)

    def test_manual_dispatch_remains_enabled_without_push(self) -> None:
        for name, workflow in self.workflows.items():
            triggers = workflow.get("on", workflow.get(True))
            branches = ["main", "release/v*"] if name == "build_wheel.yml" else ["main"]
            self.assertEqual(triggers["pull_request"]["branches"], branches)
            self.assertIn("workflow_dispatch", triggers)
            self.assertNotIn("push", triggers)
            github = context(event="workflow_dispatch", ready=False, draft=True)
            self.assertEqual(
                set(self.scheduled(workflow, github)),
                set(workflow["jobs"]) - {"docs-scope", "docs-check"},
            )

    def test_label_edits_do_not_cancel_unrelated_runs(self) -> None:
        for workflow in self.workflows.values():
            group = workflow["concurrency"]["group"]
            normal = render(group, context())
            for action, ready, draft, label in (
                ("labeled", True, False, "ready"),
                ("unlabeled", False, False, "ready"),
                ("converted_to_draft", True, True, ""),
            ):
                github = context(action, ready=ready, draft=draft, label=label)
                self.assertEqual(render(group, github), normal)
                self.assertTrue(evaluate(workflow["concurrency"]["cancel-in-progress"], github))
            for action in ("labeled", "unlabeled"):
                github = context(action, label="bug")
                self.assertNotEqual(render(group, github), normal)
                second = copy.deepcopy(github)
                second.run_id += 1
                self.assertNotEqual(render(group, second), render(group, github))

    def test_failed_validation_does_not_skip_blocking_gate(self) -> None:
        workflow = self.workflows["ci.yml"]
        if "benchmark-blocking" not in workflow["jobs"]:
            self.skipTest("This branch uses a single validation job")
        for failed in workflow["jobs"]["benchmark-blocking"]["needs"]:
            with self.subTest(failed=failed):
                self.assertIn("benchmark-blocking", self.scheduled(workflow, context(), failed))

    def test_blocking_gate_still_rejects_unsuccessful_checks(self) -> None:
        job = self.workflows["ci.yml"]["jobs"]["benchmark-blocking"]
        success = dict.fromkeys(job["needs"], "success")
        scenarios = [(success, 0)]
        for variable in success.keys() - {
            "docs-scope",
            "docs-check",
        }:
            for status in ("failure", "cancelled", "skipped"):
                scenarios.append(({**success, variable: status}, 1))
        for values, expected_code in scenarios:
            with self.subTest(values=values):
                result = self.run_gate(values)
                self.assertEqual(result.returncode, expected_code, result.stdout + result.stderr)

    @pytest.mark.l1  # type: ignore[misc]
    def test_blocking_gate_names_the_root_cause_not_every_blocked_check(self) -> None:
        # One failing prerequisite skips the rest. The gate must point at the
        # job that actually failed, and describe the others as blocked by it,
        # otherwise a single root cause looks like several failures and
        # sends reviewers after the wrong thing.
        job = self.workflows["ci.yml"]["jobs"]["benchmark-blocking"]
        success = dict.fromkeys(job["needs"], "success")
        broken = {
            **success,
            "benchmark-policy": "failure",
            "pre-commit": "skipped",
            "decode-benchmark": "skipped",
        }
        result = self.run_gate(broken)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        errors = [line for line in result.stdout.splitlines() if "::error::" in line]
        self.assertTrue(errors, result.stdout)
        # The first annotation must be the root cause, not a blocked check. The
        # gate reports the result variables it evaluates, not the job identifiers.
        self.assertIn("POLICY_RESULT", errors[0])
        for blocked in (
            "PRE_COMMIT_RESULT",
            "DECODE_RESULT",
        ):
            self.assertNotIn(
                f"{blocked} did not pass",
                " ".join(errors),
                "blocked checks must not be reported as failures",
            )
        # Exactly one annotation per group: the failure and the blocked set.
        self.assertLessEqual(len(errors), 2, "\n".join(errors))

    @pytest.mark.l1  # type: ignore[misc]
    def test_blocking_gate_reports_pure_skips_as_unrun_not_failed(self) -> None:
        # Nothing failed, but required checks did not run. That must still fail
        # the gate, and must say so plainly rather than blaming a nonexistent
        # failure.
        job = self.workflows["ci.yml"]["jobs"]["benchmark-blocking"]
        success = dict.fromkeys(job["needs"], "success")
        skipped = {**success, "benchmark-policy": "skipped", "pre-commit": "skipped"}
        result = self.run_gate(skipped)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        errors = [line for line in result.stdout.splitlines() if "::error::" in line]
        self.assertTrue(errors, result.stdout)
        self.assertIn("did not run", errors[0])
        self.assertNotIn("failed and blocked", " ".join(errors))

    def test_docs_only_runs_the_docs_gate(self) -> None:
        workflow = self.workflows["ci.yml"]
        for event in (
            context("synchronize", ready=True),
            context("labeled", ready=True, label="ready"),
        ):
            with self.subTest(action=event.event.action):
                self.assertEqual(
                    self.scheduled(workflow, event, docs_only=True),
                    ["docs-scope", "docs-check", "benchmark-blocking"],
                )

    def test_docs_gate_requires_docs_checks(self) -> None:
        gate = self.workflows["ci.yml"]["jobs"]["benchmark-blocking"]
        results = dict.fromkeys(gate["needs"], "skipped")
        results.update({"docs-scope": "success", "docs-check": "success"})
        self.assertEqual(self.run_gate(results, docs_only=True).returncode, 0)
        for status in ("failure", "cancelled", "skipped"):
            result = self.run_gate({**results, "docs-check": status}, docs_only=True)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)

    def run_gate(
        self,
        results: dict[str, str],
        github: Any | None = None,
        *,
        benchmark_device: str = "npu",
        docs_only: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        steps = self.workflows["ci.yml"]["jobs"]["benchmark-blocking"]["steps"]
        step = next(
            item for item in steps if item.get("name") == "Require every blocking check to pass"
        )
        github = github or context()
        env = {
            key: str(
                evaluate(
                    value,
                    github,
                    results,
                    benchmark_device=benchmark_device,
                    docs_only=docs_only,
                )
            ).lower()
            for key, value in step["env"].items()
        }
        return run_workflow_command(
            [shutil.which("bash") or "bash", "-eu", "-o", "pipefail", "-c", step["run"]],
            env=env,
        )

    def test_pr_decode_routing(self) -> None:
        workflow = self.workflows["ci.yml"]
        gate = workflow["jobs"]["benchmark-blocking"]
        event = context("synchronize", ready=True)
        for action in ("labeled", "unlabeled"):
            self.assertEqual(
                self.scheduled(workflow, context(action, ready=True, label="abtest")), []
            )
        for device in ("gpu", "npu"):
            with self.subTest(device=device):
                running = self.scheduled(workflow, event, benchmark_device=device)
                self.assertEqual("decode-benchmark" in running, device == "npu")
                results = {job: "success" if job in running else "skipped" for job in gate["needs"]}
                result = self.run_gate(results, event, benchmark_device=device)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_recall_and_decode_benchmark_cover_both_target_branches_and_devices(self) -> None:
        steps = self.workflows["ci.yml"]["jobs"]["decode-benchmark"]["steps"]
        by_name = {step.get("name"): step for step in steps}
        recall_steps = (
            "Cache OneRec Recall constraint tables",
            "Build OneRec Recall constraint tables",
            "Gate Beam V1 B=4 OneRec Recall on fixed samples",
            "Upload Beam V1 Recall measurements",
        )
        for branch in ("main", "decode_graph", "feature"):
            for device in ("gpu", "npu"):
                for event in ("pull_request", "workflow_dispatch"):
                    with self.subTest(branch=branch, device=device, event=event):
                        github = context(event=event, base_ref=branch)
                        if event == "workflow_dispatch":
                            github.ref_name = branch
                            github.event.pull_request.base.ref = ""
                        else:
                            github.ref_name = "481/merge"
                        github.base_ref = github.event.pull_request.base.ref
                        target = render(
                            self.workflows["ci.yml"]["env"]["POLICY_TARGET_BRANCH"], github
                        )
                        expect_recall = device == "gpu" and (
                            branch == "main"
                            or (event == "workflow_dispatch" and branch == "decode_graph")
                        )
                        for name in recall_steps:
                            condition = (
                                by_name[name]["if"]
                                .replace("env.POLICY_TARGET_BRANCH", repr(target))
                                .replace("env.BENCHMARK_DEVICE", repr(device))
                                .replace(
                                    "steps.cache-recall-constraints.outputs.cache-hit", "'false'"
                                )
                            )
                            self.assertEqual(bool(evaluate(condition, github)), expect_recall, name)
                        benchmark = by_name["Execute decode benchmark gates"]["if"].replace(
                            "env.BENCHMARK_DEVICE", repr(device)
                        )
                        benchmark = benchmark.replace("env.POLICY_TARGET_BRANCH", repr(target))
                        self.assertEqual(bool(evaluate(benchmark, github)), not expect_recall)

    def test_unready_and_draft_events_fail_required_gate(self) -> None:
        workflow = self.workflows["ci.yml"]
        gate = workflow["jobs"]["benchmark-blocking"]
        for github in (
            context("opened", ready=False),
            context("synchronize", ready=False),
            context("unlabeled", ready=False, label="ready"),
            context("converted_to_draft", draft=True),
            context("labeled", draft=True, label="ready"),
        ):
            with self.subTest(event=github.event):
                running = self.scheduled(workflow, github)
                self.assertEqual(running, ["benchmark-blocking"])
                self.assertEqual(render(gate["name"], github), "Benchmark blocking gate")
                results = {
                    need: "success" if need in running else "skipped" for need in gate["needs"]
                }
                result = self.run_gate(results)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)

    @pytest.mark.l1  # type: ignore[misc]
    def test_benchmark_jobs_wait_for_successful_precommit_and_policy(self) -> None:
        workflow = self.workflows["ci.yml"]
        job = workflow["jobs"]["decode-benchmark"]
        self.assertEqual(set(job["needs"]), {"benchmark-policy", "pre-commit"})
        for policy in ("success", "failure", "cancelled", "skipped"):
            for precommit in ("success", "failure", "cancelled", "skipped"):
                for cancelled in (False, True):
                    with self.subTest(policy=policy, precommit=precommit, cancelled=cancelled):
                        results = {"benchmark-policy": policy, "pre-commit": precommit}
                        expected = policy == precommit == "success" and not cancelled
                        self.assertEqual(
                            bool(evaluate(job["if"], context(), results, cancelled=cancelled)),
                            expected,
                        )

        for failed in ("benchmark-policy", "pre-commit"):
            with self.subTest(failed=failed):
                running = self.scheduled(workflow, context(), failed=failed)
                self.assertNotIn("decode-benchmark", running)
                self.assertIn("benchmark-blocking", running)

    def test_level_workflow_schedule_and_resource_safety(self) -> None:
        workflow = yaml.safe_load((ROOT / ".github/workflows/test-levels.yml").read_text())
        triggers = workflow.get("on", workflow.get(True))
        self.assertEqual(triggers["push"]["branches"], ["main"])
        self.assertNotIn("pull_request", triggers)  # PR levels run in the existing gate.
        self.assertEqual(
            [item["cron"] for item in triggers["schedule"]],
            ["0 18 * * *", "0 19 * * 0"],
        )
        self.assertEqual(
            triggers["workflow_dispatch"]["inputs"]["level"]["options"],
            ["l1", "l2", "l3", "l4", "l5"],
        )
        self.assertEqual(workflow["concurrency"]["queue"], "max")
        self.assertFalse(workflow["concurrency"]["cancel-in-progress"])

        pr_workflow = self.workflows["ci.yml"]
        gpu_lock = pr_workflow["jobs"]["pre-commit"]["concurrency"]["group"]
        self.assertEqual(gpu_lock, "vllm-gr-accelerator-${{ github.repository }}")
        self.assertEqual(workflow["concurrency"]["group"], gpu_lock)
        # A workflow must not wait on a job that requests the same lock it owns.
        self.assertNotIn("concurrency", workflow["jobs"]["level"])
        self.assertNotEqual(
            render(pr_workflow["concurrency"]["group"], context()), render(gpu_lock, context())
        )

        job = workflow["jobs"]["level"]
        self.assertEqual(job["runs-on"], "self-hosted")
        expression = job["env"]["TEST_LEVEL"].strip().removeprefix("${{").removesuffix("}}")
        expression = expression.replace("&&", " and ").replace("||", " or ")
        for event, schedule, chosen, expected in (
            ("push", "", "", "l3"),
            ("schedule", "0 18 * * *", "", "l4"),
            ("schedule", "0 19 * * 0", "", "l5"),
            ("workflow_dispatch", "", "l2", "l2"),
        ):
            with self.subTest(event=event, schedule=schedule, chosen=chosen):
                selected = eval(  # noqa: S307 - only repository-owned workflow expressions
                    expression.strip(),
                    {"__builtins__": {}},
                    {
                        "github": namespace({"event_name": event, "event": {"schedule": schedule}}),
                        "inputs": namespace({"level": chosen}),
                    },
                )
                self.assertEqual(selected, expected)

        steps = {step.get("name"): step for step in job["steps"] if step.get("name")}
        self.assertIn("vars.VLLM_GR_ISOLATED_RUNNER", job["env"]["VLLM_GR_ISOLATED_RUNNER"])
        resource = steps["Run L2 resource probe if isolated"]
        self.assertIn("env.TEST_LEVEL == 'l2'", resource["if"])
        self.assertEqual(resource["run"], "bash tools/ci/run_resource_level.sh")
        self.assertIn("torch.cuda.is_available", steps["Verify accelerator prerequisites"]["run"])
        self.assertNotIn("kill_vllm_on_device", str(job))
        self.assertNotIn("check_cuda_device_mapping", str(job))
        self.assertIn("HF_TOKEN", steps["Download OpenOneRec dataset"]["env"])
        self.assertIn("run_beam_v1_recall.sh", steps["Run full OneRec recall gate"]["run"])
        self.assertIn(
            "run_beam_v1_performance.sh", steps["Run OneRec performance scenarios"]["run"]
        )
        self.assertIn("VLLM_GR_TARGET_DEVICE == 'cuda'", steps["Run full OneRec recall gate"]["if"])

    @pytest.mark.l1  # type: ignore[misc]
    def test_level_workflow_skips_npu_before_setup(self) -> None:
        workflow = yaml.safe_load((ROOT / ".github/workflows/test-levels.yml").read_text())
        steps = workflow["jobs"]["level"]["steps"]
        detect = steps[0]
        self.assertEqual(detect["id"], "accelerator")
        self.assertEqual(detect["name"], "Detect runner accelerator")

        def enabled(step: dict[str, Any], runner: str, level: str, hit: str) -> bool:
            condition = step["if"]
            condition = condition.replace("steps.accelerator.outputs.run", repr(runner))
            condition = condition.replace("steps.cache-venv.outputs.cache-hit", repr(hit))
            condition = condition.replace("steps.cache-dataset.outputs.cache-hit", repr(hit))
            condition = condition.replace("env.TEST_LEVEL", repr(level))
            condition = condition.replace("env.VLLM_GR_TARGET_DEVICE", "'cuda'")
            return bool(evaluate(condition, context()))

        for step in steps[1:]:
            self.assertIn("steps.accelerator.outputs.run", step["if"])
            for level in ("l1", "l2", "l3", "l4", "l5"):
                for hit in ("true", "false"):
                    with self.subTest(
                        step=step.get("name", step.get("uses")), level=level, hit=hit
                    ):
                        self.assertFalse(enabled(step, "false", level, hit))

        by_name = {step.get("name"): step for step in steps if step.get("name")}
        for level in ("l1", "l2", "l3", "l4", "l5"):
            self.assertTrue(enabled(by_name["Run selected level"], "true", level, "true"))
        self.assertTrue(enabled(by_name["Run L2 resource probe if isolated"], "true", "l2", "true"))
        self.assertFalse(
            enabled(by_name["Run L2 resource probe if isolated"], "true", "l3", "true")
        )
        self.assertTrue(enabled(by_name["Create virtual environment"], "true", "l3", "false"))
        self.assertFalse(enabled(by_name["Create virtual environment"], "true", "l3", "true"))
        for name in (
            "Run full OneRec recall gate",
            "Run OneRec performance scenarios",
            "Upload L4 system results",
        ):
            self.assertTrue(enabled(by_name[name], "true", "l4", "false"))
            self.assertFalse(enabled(by_name[name], "true", "l3", "false"))

    def test_level_ci_preserves_existing_benchmark_gates(self) -> None:
        workflow = self.workflows["ci.yml"]
        jobs = workflow["jobs"]
        self.assertEqual(jobs["pre-commit"]["env"]["VLLM_GR_TEST_LEVEL"], "pr-device")
        self.assertIn(
            "vars.VLLM_GR_ISOLATED_RUNNER",
            jobs["pre-commit"]["env"]["VLLM_GR_ISOLATED_RUNNER"],
        )
        probe = next(
            step
            for step in jobs["pre-commit"]["steps"]
            if step.get("name") == "Run L2 resource probe if isolated"
        )
        self.assertEqual(probe["run"], "bash tools/ci/run_resource_level.sh")
        self.assertTrue(
            {"pre-commit", "decode-benchmark"} <= set(jobs["benchmark-blocking"]["needs"])
        )
        self.assertNotIn("beam-performance-ab-only", jobs)
        ready = yaml.safe_load((ROOT / ".buildkite/test-ready.yml").read_text())
        self.assertIn("beam-v1-performance-ab", {step["key"] for step in ready["steps"]})
        local_runner = (ROOT / "tools/pre_commit/run_tests.sh").read_text()
        self.assertIn("VLLM_GR_TEST_LEVEL", local_runner)
        self.assertIn(
            "python3 -m pytest -x --run-slow --strict-test-levels --durations=10 -s tests",
            local_runner,
        )
        level_runner = (ROOT / "tools/ci/run_test_level.sh").read_text()
        self.assertIn("--strict-test-levels", level_runner)
        self.assertIn("--run-slow", level_runner)
        self.assertIn("$status -eq 5", level_runner)
        self.assertIn('"$level" == l5', level_runner)
        resource_runner = (ROOT / "tools/ci/run_resource_level.sh").read_text()
        self.assertIn('"${VLLM_GR_ISOLATED_RUNNER:-false}" != true', resource_runner)
        self.assertIn("bash tests/development/test_resource.sh", resource_runner)
        self.assertIn("grep -q 'SKIP'", resource_runner)
        self.assertIn("status=${PIPESTATUS[0]}", resource_runner)

    @pytest.mark.l1  # type: ignore[misc]
    def test_parallel_benchmark_jobs_provision_on_cache_miss(self) -> None:
        jobs = self.workflows["ci.yml"]["jobs"]
        steps = jobs["decode-benchmark"]["steps"]
        names = [step.get("name") for step in steps]
        create = steps[names.index("Create virtual environment")]
        install = steps[names.index("Install CUDA base environment")]
        self.assertLess(names.index("Restore virtual environment"), names.index(create["name"]))
        self.assertLess(names.index(create["name"]), names.index("Set up virtual environment"))
        self.assertLess(names.index("Set up virtual environment"), names.index(install["name"]))
        self.assertLess(names.index(install["name"]), names.index("Refresh editable installation"))
        for hit in ("true", "false"):
            for device in ("cuda", "npu"):
                with self.subTest(cache_hit=hit, device=device):
                    commands = []
                    for step in (create, install):
                        guard = (
                            step["if"]
                            .replace("steps.cache-venv.outputs.cache-hit", repr(hit))
                            .replace("env.VLLM_GR_TARGET_DEVICE", repr(device))
                        )
                        if evaluate(guard, context()):
                            result = run_workflow_command(
                                [
                                    shutil.which("bash") or "bash",
                                    "-eu",
                                    "-o",
                                    "pipefail",
                                    "-c",
                                    "uv() { printf '%s\\n' \"$*\"; }\n" + step["run"],
                                ],
                                env={
                                    "VLLM_GR_TARGET_DEVICE": device,
                                    "CI_VLLM_REQUIREMENT": "vllm==test",
                                },
                            )
                            self.assertEqual(result.returncode, 0, result.stderr)
                            commands.extend(result.stdout.splitlines())
                    expected = []
                    if hit == "false":
                        expected.append(
                            "venv --seed --system-site-packages .venv"
                            if device == "npu"
                            else "venv --seed .venv"
                        )
                        if device == "cuda":
                            expected.append("pip install vllm==test")
                    self.assertEqual(commands, expected)


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "devices,lane,expected,exit_code",
    [
        ((), "gpu", None, 1),
        ((), "npu", None, 1),
        (("npu-smi",), "gpu", "false", 0),
        (("npu-smi",), "npu", "true", 0),
        (("nvidia-smi",), "gpu", "true", 0),
        (("nvidia-smi",), "npu", "false", 0),
        (("npu-smi", "nvidia-smi"), "gpu", "false", 0),
        (("npu-smi", "nvidia-smi"), "npu", "true", 0),
    ],
)
def test_level_workflow_detects_accelerator(
    tmp_path: Path, devices: tuple[str, ...], lane: str, expected: str | None, exit_code: int
) -> None:
    # Separate pytest items give each command its own timeout and let -x stop
    # immediately, instead of recording a timeout inside unittest.subTest.
    result = run_accelerator_detection(tmp_path, devices, lane)
    assert result.returncode == exit_code, result.stderr
    if expected is None:
        assert "No supported GPU or NPU" in result.stderr
    else:
        assert f"run={expected}" in (tmp_path / "output").read_text()
        if expected == "false":
            assert "not run" in (tmp_path / "summary").read_text()
            assert "::warning::" in result.stdout


@pytest.mark.l1  # type: ignore[misc]
def test_accelerator_detection_ignores_shell_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    startup = tmp_path / "startup.sh"
    startup.write_text("echo unexpected-shell-startup >&2\nexit 99\n", encoding="utf-8")
    monkeypatch.setenv("BASH_ENV", str(startup))
    monkeypatch.setenv("BASH_FUNC_command%%", "() { return 0; }")
    result = run_accelerator_detection(tmp_path, (), "gpu")
    assert result.returncode == 1
    assert "No supported GPU or NPU" in result.stderr
    assert "unexpected-shell-startup" not in result.stderr


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.timeout(5, func_only=True)  # type: ignore[misc]
def test_accelerator_detection_times_out_and_reaps_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Replace only the Bash executable. Exercise the real detection path with
    # a process that never reaches the workflow snippet.
    sleep = shutil.which("sleep")
    assert sleep is not None
    bash = tmp_path / "bash"
    bash.write_text(f"#!/bin/sh\nexec '{sleep}' 30\n", encoding="utf-8")
    bash.chmod(0o755)
    which = shutil.which
    monkeypatch.setattr(shutil, "which", lambda name: str(bash) if name == "bash" else which(name))
    monkeypatch.setattr(sys.modules[__name__], "_WORKFLOW_COMMAND_TIMEOUT", 0.2)
    processes: list[subprocess.Popen[str]] = []
    popen = subprocess.Popen

    def track_process(*args: Any, **kwargs: Any) -> subprocess.Popen[str]:
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", track_process)
    with pytest.raises(subprocess.TimeoutExpired):
        run_accelerator_detection(tmp_path, (), "gpu")
    assert len(processes) == 1
    assert processes[0].poll() is not None


if __name__ == "__main__":
    unittest.main()


pytestmark = pytest.mark.cpu_test
