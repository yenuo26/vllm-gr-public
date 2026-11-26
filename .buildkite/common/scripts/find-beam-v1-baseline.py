# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Download the Beam V1 baseline for the current PR test merge commit."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def baseline_commit() -> str:
    commit = subprocess.check_output(["git", "rev-parse", "--verify", "HEAD^1"], text=True).strip()
    if len(commit) != 40:
        raise SystemExit("Expected a full baseline commit SHA")
    return commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        commit = baseline_commit()
    except subprocess.CalledProcessError as exc:
        raise SystemExit("Could not resolve the first parent of HEAD for the baseline.") from exc

    repository = "public.ecr.aws/q9t5s3a7/vllm-ci-test-repo"
    reference = f"{repository}:vllm-gr-beam-v1-baseline-{commit}"
    destination = Path(tempfile.mkdtemp(dir=args.destination.resolve()))
    try:
        subprocess.run(
            ["oras", "pull", "--output", str(destination), reference],
            stdout=sys.stderr,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Could not download Beam V1 baseline {reference}. "
            "Check that the main Buildkite build published the baseline for this commit."
        ) from exc
    print(destination)


if __name__ == "__main__":
    main()
