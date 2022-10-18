#! /usr/bin/env bash

if [ ! $# -eq 0]; then
    CONTAINER_NAME=$1
else
    CONTAINER_NAME=dory_docker
    echo "No container name supplied; running container 'dory_docker'"
fi
SCRIPT_PATH=`realpath $BASH_SOURCE`
SCRIPT_DIR=`dirname $SCRIPT_PATH`
DORY_HOME=`dirname $SCRIPT_DIR`
docker run -v $DORY_HOME:/dory_checkout --workdir /dory_checkout/ -it $CONTAINER_NAME bash
