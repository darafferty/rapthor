#! /bin/bash
set -e

DOCKER_TAG=latest

cd $(dirname ${0} && pwd)

docker build ${PWD}/.. -f ubuntu_20_04-base -t rapthor-base:${DOCKER_TAG} && \
docker build ${PWD}/.. -f ubuntu_20_04-rapthor -t rapthor:${DOCKER_TAG}
#docker push lofareosc/rapthor3-cwl:${DOCKER_TAG}
