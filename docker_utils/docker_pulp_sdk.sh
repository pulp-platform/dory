#! /usr/bin/env bash

source /dory_env/bin/activate
source /pulp-sdk/configs/pulp-open-nn.sh
export PULP_RISCV_GCC_TOOLCHAIN=/riscv-nn-toolchain/

find /dory_checkout -path "*GAP8*HW_description.json" -exec sed -E -i 's/("name" *: *)"gap_sdk"/\1"pulp-sdk"/' {} +

