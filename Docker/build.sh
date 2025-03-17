#!/bin/bash
#
# This script builds a docker image for Rapthor.
# It is inspired by the way docker images are built for LINC
# (see Docker/build_docker.sh in the LINC repository).

set -euxo pipefail

SCRIPT_DIR=$(cd "$(dirname "${0}")" && pwd)
DOCKERFILE=${SCRIPT_DIR}/Dockerfile
REPO_ROOT="$(git rev-parse --show-toplevel)"

echo "Determining version information ..."
eval $(${SCRIPT_DIR}/fetch_commit_hashes.sh | tee commits.txt)

COMMUNITY="astronrd"
REPOSITORY=$(basename -s .git "$(git remote get-url origin)")
VERSION=$("${SCRIPT_DIR}/get_scm_version.sh")

# Docker tags may only contain lowercase and uppercase letters, digits,
# underscores, periods, and hyphens. Replace every invalid character with
# a hyphen.
TAG=${VERSION//[^[:alnum:]_.-]/-}

docker build \
  --build-arg AOFLAGGER_COMMIT=${AOFLAGGER_COMMIT} \
  --build-arg CASACORE_COMMIT=${CASACORE_COMMIT} \
  --build-arg DP3_COMMIT=${DP3_COMMIT} \
  --build-arg EVERYBEAM_COMMIT=${EVERYBEAM_COMMIT} \
  --build-arg IDG_COMMIT=${IDG_COMMIT} \
  --build-arg PYTHONCASACORE_COMMIT=${PYTHONCASACORE_COMMIT} \
  --build-arg SAGECAL_COMMIT=${SAGECAL_COMMIT} \
  --build-arg WSCLEAN_COMMIT=${WSCLEAN_COMMIT} \
  --build-arg VERSION=${VERSION} \
  --progress plain \
  --file "${DOCKERFILE}" \
  --tag "${COMMUNITY}/${REPOSITORY}:${TAG}" \
  "${REPO_ROOT}"

docker push "${COMMUNITY}/${REPOSITORY}:${TAG}"
