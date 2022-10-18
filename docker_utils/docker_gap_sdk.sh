#! /usr/bin/env bash

if [ ! $# -eq 0 ]; then
    CONTAINER_NAME=$1
else
    CONTAINER_NAME=dory_docker
    echo "No container name supplied; running container 'dory_docker'"
fi
SCRIPT_PATH=`realpath $BASH_SOURCE`
SCRIPT_DIR=`dirname $SCRIPT_PATH`
DORY_HOME=`dirname $SCRIPT_DIR`
unset PULP_RISCV_GCC_TOOLCHAIN

source /dory_env/bin/activate
source /gap_riscv_toolchain_ubuntu_18/gap_sdk/configs/gapuino_v3.sh

find $DORY_HOME  -path "*GAP8*HW_description.json" -exec sed -E -i 's/("name" *: *)"pulp-sdk"/\1"gap_sdk"/' {} +
