# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Gate Beam V1 Recall against fixed OneRec samples."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.helpers.datasets import download_datasets
from tests.helpers.paths import REPO_ROOT

pytestmark = [pytest.mark.gpu_test, pytest.mark.l4, pytest.mark.slow, pytest.mark.timeout(0)]


def test_beam_v1_recall(tmp_path: Path) -> None:
    if not (output := os.environ.get("V1_RECALL_OUTPUT_DIR", "")):
        pytest.skip("Set V1_RECALL_OUTPUT_DIR to opt in to the Beam V1 Recall benchmark")

    dataset = download_datasets(tmp_path)
    for mapping in ("sid2pid", "sid2iid"):
        catalog = tmp_path / f"{mapping}_catalog.json"
        subprocess.run(
            [
                sys.executable,
                "benchmarks/open_one_rec/tools/convert_catalog.py",
                "--dataset-path",
                str(dataset),
                "--mapping-file",
                f"{mapping}.json",
                "--output-path",
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
                str(tmp_path / f"{mapping}_triples.json"),
                "--begin",
                "<|sid_begin|>",
                "--end",
                "<|sid_end|>",
            ],
            cwd=REPO_ROOT,
            check=True,
        )

    subprocess.run(
        ["bash", "tests/system/run_beam_v1_recall.sh"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "VLLM_GR_TARGET_DEVICE": "cuda",
            "V1_RECALL_BASELINE_FILE": str(
                REPO_ROOT / "benchmarks/beam_search/recall_baseline_batch4.json"
            ),
            "V1_RECALL_DATASET_PATH": str(dataset),
            "V1_RECALL_DIAGNOSTICS": "1",
            "V1_RECALL_OUTPUT_DIR": str(Path(output).resolve()),
            "V1_RECALL_TRIPLES_PATH": str(tmp_path / "sid2pid_triples.json"),
            "V1_RECALL_TRIPLES_IID_PATH": str(tmp_path / "sid2iid_triples.json"),
        },
        check=True,
    )
