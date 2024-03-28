#!/bin/bash

# Helper script to aid debugging the Docker image.
# This enters the CI's docker image with the git checkout mounted,
# so modifications can be tested locally without the CI.

set -e

ROOT="$(git rev-parse --show-toplevel)"
if [[ ! -f "${ROOT}/ci/debug.sh" ]]; then
    echo "The script needs to be executed in the repository."
    exit 1
else
    cd ${ROOT}
fi

docker build --tag rapthor/ubuntu_22_04-base  -f "${ROOT}/ci/ubuntu_22_04-base" .
docker run -it --volume "${ROOT}:/rapthor" --workdir "/rapthor" rapthor/ubuntu_22_04-base bash
