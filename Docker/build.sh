#!/bin/bash
set -e

DOCKER_TAG=latest

cd $(dirname ${0})

docker build ${PWD}/.. -f Dockerfile -t astronrd/rapthor:${DOCKER_TAG}
docker push astronrd/rapthor:${DOCKER_TAG}
