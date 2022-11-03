#! /usr/bin/env bash

SCRIPT_PATH=`realpath $BASH_SOURCE`
SCRIPT_DIR=`dirname $SCRIPT_PATH`
DORY_HOME=`dirname $SCRIPT_DIR`

source /dory_env/bin/activate
source /pulp-sdk/configs/pulp-open-nn.sh
export PULP_RISCV_GCC_TOOLCHAIN=/riscv-nn-toolchain/

find $DORY_HOME -path "*Hardware_targets/PULP*HW_description.json" -exec sed -E -i 's/("name" *: *)"gap_sdk"/\1"pulp-sdk"/' {} +

