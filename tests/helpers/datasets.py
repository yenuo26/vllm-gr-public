# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Dataset helpers for tests."""

import os
from pathlib import Path


def download_datasets(work_dir: Path) -> Path:
    if dataset := os.environ.get("VLLM_GR_ONEREC_DATASET_DIR"):
        return Path(dataset).resolve() / "benchmark_data"

    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        "OpenOneRec/OpenOneRec-RecIF", repo_type="dataset", local_dir=work_dir / "OpenOneRec"
    )
    return Path(snapshot) / "benchmark_data"
