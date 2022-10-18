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

source /dory_env/bin/activate
source /pulp-sdk/configs/pulp-open-nn.sh
export PULP_RISCV_GCC_TOOLCHAIN=/riscv-nn-toolchain/

find $DORY_HOME -path "*GAP8*HW_description.json" -exec sed -E -i 's/("name" *: *)"gap_sdk"/\1"pulp-sdk"/' {} +

