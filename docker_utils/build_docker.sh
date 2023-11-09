#!/usr/bin/env bash

if [ ! $# -eq 0]; then
    CONTAINER_NAME=$1
else
    CONTAINER_NAME=dory_docker
    echo "No container name supplied; building with name 'dory_docker'"
fi

docker build -t $CONTAINER_NAME - < Dockerfile

echo "Done - built container with name $CONTAINER_NAME"
