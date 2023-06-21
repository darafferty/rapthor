#!/bin/bash
set -e

DOCKER_TAG=debug

cd $(dirname ${0})

docker build --no-cache ${PWD}/.. -f Dockerfile -t astronrd/rapthor:${DOCKER_TAG}
docker push astronrd/rapthor:${DOCKER_TAG}
