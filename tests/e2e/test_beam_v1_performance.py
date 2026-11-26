# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Measure and compare Beam V1 latency against a local baseline."""

import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from benchmarks.beam_search.compare_latency_ab import compare, verify_workload
from tests.helpers.datasets import download_datasets
from tests.helpers.paths import REPO_ROOT

pytestmark = [pytest.mark.gpu_test, pytest.mark.l4, pytest.mark.slow, pytest.mark.timeout(0)]


def _git_revision(ref: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--verify", ref], cwd=REPO_ROOT, text=True
    ).strip()


def _prepare_triples(work_dir: Path) -> Path:
    dataset_dir = download_datasets(work_dir)
    catalog = work_dir / "video_catalog.json"
    triples = work_dir / "video_constraint_triples.json"
    subprocess.run(
        [
            sys.executable,
            "benchmarks/open_one_rec/tools/create_task_catalog.py",
            "-t",
            "video",
            "-d",
            str(dataset_dir),
            "-o",
            str(catalog),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "tools/constraints/build_constraint_triples.py",
            "--catalog",
            str(catalog),
            "--output",
            str(triples),
            "--begin",
            "<|sid_begin|>",
            "--end",
            "<|sid_end|>",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    return triples


_BASELINE_FILES = (
    "commit.txt",
    "gpu.csv",
    "setup.json",
    "offline.json",
    "prompt_token_ids.json",
    "constraint_table.safetensors",
)


def _pack_baseline(runs_dir: Path, path: Path) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for run in ("1", "2"):
            for name in _BASELINE_FILES:
                relative = f"{run}/{name}"
                archive.add(runs_dir / relative, arcname=relative, recursive=False)


def _unpack_baseline(path: Path, destination: Path, base_sha: str) -> None:
    if not path.is_file():
        pytest.fail(f"Missing Beam V1 baseline file: {path}")

    with tarfile.open(path, "r:gz") as archive:
        for run in ("1", "2"):
            run_dir = destination / run
            run_dir.mkdir(parents=True)
            for name in _BASELINE_FILES:
                relative = f"{run}/{name}"
                try:
                    member = archive.getmember(relative)
                except KeyError:
                    pytest.fail(f"Missing Beam V1 baseline member: {relative}")
                if not member.isfile():
                    pytest.fail(f"Expected a regular baseline file: {relative}")
                stream = archive.extractfile(member)
                assert stream is not None
                with stream, (run_dir / name).open("wb") as output:
                    shutil.copyfileobj(stream, output)
            if (run_dir / "commit.txt").read_text(encoding="utf-8").strip() != base_sha:
                pytest.fail(f"Beam V1 baseline run {run} in {path} does not match {base_sha}")


@pytest.mark.parametrize("mode", ["baseline", "compare"])  # type: ignore[misc]
def test_beam_v1_performance(mode: str, tmp_path: Path) -> None:
    if not (output := os.environ.get("V1_PERF_OUTPUT_DIR", "")):
        pytest.skip("Set V1_PERF_OUTPUT_DIR to opt in to the Beam V1 benchmark")

    output_dir = Path(output).resolve()
    candidate_sha = _git_revision("HEAD")
    if mode == "compare":
        baseline = os.environ.get("V1_PERF_BASELINE_DIR", "")
        if not baseline:
            pytest.fail("Set V1_PERF_BASELINE_DIR to the downloaded benchmark baseline")
        baseline_dir = tmp_path / "baseline"
        base_sha = _git_revision("HEAD^1")
        _unpack_baseline(Path(baseline) / "baseline.tar.gz", baseline_dir, base_sha)
        verify_workload(REPO_ROOT, base_sha, candidate_sha)

    triples = _prepare_triples(tmp_path)
    runs_dir = output_dir if mode == "baseline" else output_dir / "pr"
    for run in ("1", "2"):
        subprocess.run(
            ["bash", "tests/system/run_beam_v1_performance.sh", "--ab-prefill-grid"],
            cwd=REPO_ROOT,
            env={
                **os.environ,
                "VLLM_GR_TARGET_DEVICE": "cuda",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "TRIPLES_PATH": str(triples),
                "V1_PERF_OUTPUT_DIR": str(runs_dir / run),
                "V1_PERF_BEAM_WIDTH": "128",
                "V1_PERF_WARMUP_RUNS": "10",
                "V1_PERF_MEASURED_RUNS": "20",
                "V1_PERF_MEASURE_ONLY": "1",
                "V1_PERF_OFFLINE_ONLY": "1",
            },
            check=True,
        )

    if mode == "baseline":
        publication = output_dir / "public"
        publication.mkdir(exist_ok=True)
        _pack_baseline(runs_dir, publication / "baseline.tar.gz")
    else:
        result, _ = compare(baseline_dir, runs_dir, base_sha, candidate_sha)
        (output_dir / "report.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        assert not result["failures"], "\n".join(result["failures"])
