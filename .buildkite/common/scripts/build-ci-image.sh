#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

set -euo pipefail

: "${BUILDKITE_COMMIT:?Buildkite must provide the commit SHA}"

# Keep the default in sync with ci_mirror_hardwares.yml.
repository="${VLLM_GR_CI_IMAGE_REPOSITORY:-public.ecr.aws/q9t5s3a7/vllm-ci-test-repo}"
image="${repository}:vllm-gr-${BUILDKITE_COMMIT}"
dockerfile="docker/Dockerfile.ci"
builder="vllm-gr-builder"

if [[ "$repository" == public.ecr.aws/* ]]; then
  aws ecr-public get-login-password --region us-east-1 \
    | docker login --username AWS --password-stdin public.ecr.aws
fi

cache_key="$(sha256sum "$dockerfile" "${dockerfile}.dockerignore" pyproject.toml setup.py \
  tools/packaging/git_version.py requirements/*.txt | sha256sum | cut -c1-16)"
cache_ref="${repository}:vllm-gr-deps-cache-${cache_key}"
echo "Building ${image} with cache ${cache_ref}"

docker buildx inspect "$builder" >/dev/null 2>&1 \
  || docker buildx create --name "$builder" --driver docker-container
docker buildx build --builder "$builder" --push --progress=plain \
  --cache-from "type=registry,ref=${cache_ref}" \
  --cache-to "type=registry,ref=${cache_ref},mode=max,compression=zstd" \
  --file "$dockerfile" \
  --tag "$image" \
  .
