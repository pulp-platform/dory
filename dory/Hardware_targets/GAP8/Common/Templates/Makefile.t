# Makefile
# Alessio Burrello <alessio.burrello@unibo.it>
#
# Copyright (C) 2019-2020 University of Bologna
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


CORE ?= 1

APP = main
APP_SRCS := $(wildcard src/*.c)
APP_CFLAGS += -DNUM_CORES=$(CORE) -Iinc -O3 -w

% if sdk == 'pulp-sdk':
APP_CFLAGS += -DPULP_SDK=1
APP_CFLAGS += -fno-tree-loop-distribute-patterns -flto
APP_LDFLAGS += -lm -flto -Wl,--print-memory-usage
FLASH_TYPE ?= HYPERFLASH
RAM_TYPE ?= HYPERRAM
CONFIG_HYPERRAM = 1
CONFIG_HYPERFLASH = 1
% elif sdk == 'gap_sdk':
APP_CFLAGS += -DGAP_SDK=1
FLASH_TYPE ?= SPIFLASH
RAM_TYPE ?= DEFAULT_RAM
FS_TYPE ?= read_fs
USE_PMSIS_BSP=1
% endif

ifeq '$(FLASH_TYPE)' 'MRAM'
READFS_FLASH = target/chip/soc/mram
endif

APP_CFLAGS += -DFLASH_TYPE=$(FLASH_TYPE) -DUSE_$(FLASH_TYPE) -DUSE_$(RAM_TYPE)

% for layer in layers_w:
FLASH_FILES += hex/${layer}
% endfor
FLASH_FILES += hex/inputs.hex

READFS_FILES := $(FLASH_FILES)
% if sdk == 'gap_sdk':
APP_CFLAGS += -DFS_READ_FS
% endif
#PLPBRIDGE_FLAGS += -f

include $(RULES_DIR)/pmsis_rules.mk
