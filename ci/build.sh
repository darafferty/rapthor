#!/bin/bash
set -e

DOCKER_TAG=latest

cd $(dirname ${0})

docker build ${PWD}/.. -f ubuntu_20_04-base -t rapthor-base:${DOCKER_TAG}
docker build ${PWD}/.. -f ubuntu_20_04-rapthor -t rapthor:${DOCKER_TAG}
docker tag rapthor:${DOCKER_TAG} loose/rapthor:${DOCKER_TAG}
docker push loose/rapthor:${DOCKER_TAG}
