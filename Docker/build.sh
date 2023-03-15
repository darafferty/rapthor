#!/bin/bash
set -e

DOCKER_TAG=latest

cd $(dirname ${0})

docker build ${PWD}/.. -f Dockerfile -t loose/rapthor:${DOCKER_TAG}
docker push loose/rapthor:${DOCKER_TAG}
