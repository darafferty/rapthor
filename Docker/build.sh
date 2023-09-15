#!/bin/bash
set -ex

SCRIPT_DIR=$(cd "$(dirname "${0}")" && pwd)
DOCKERFILE=${SCRIPT_DIR}/Dockerfile

# Change directory to the top-level directory of the Git repository
cd "$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

# Get info from the Git repository
COMMUNITY="astronrd"
REPOSITORY=$(basename -s .git "$(git remote get-url origin)")
VERSION=$("${SCRIPT_DIR}/get_scm_version.sh")

# Docker tags may only contain lowercase and uppercase letters, digits,
# underscores, periods, and hyphens. Replace every invalid character with
# a hyphen.
TAG=${VERSION//[^[:alnum:]_.-]/-}

docker build \
    --build-arg VERSION="${VERSION}" \
    --file "${DOCKERFILE}" \
    --tag "${COMMUNITY}/${REPOSITORY}:${TAG}" \
    .

docker push "${COMMUNITY}/${REPOSITORY}:${TAG}"
