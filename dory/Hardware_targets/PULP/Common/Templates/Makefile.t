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
# -O2 with -fno-indirect-inlining is just as fast as -O3 and reduces code size considerably
# by not inlining of small functions in the management code
APP_CFLAGS += -DNUM_CORES=$(CORE) -Iinc -O2 -fno-indirect-inlining -flto -w
APP_LDFLAGS += -lm -Wl,--print-memory-usage -flto
FLASH_TYPE ?= HYPERFLASH
RAM_TYPE ?= HYPERRAM

% if sdk == 'pulp-sdk':
APP_CFLAGS += -DPULP_SDK=1

CONFIG_HYPERRAM = 1
CONFIG_HYPERFLASH = 1
% elif sdk == 'gap_sdk':
APP_CFLAGS += -DGAP_SDK=1
% endif

ifeq '$(FLASH_TYPE)' 'MRAM'
READFS_FLASH = target/chip/soc/mram
endif


APP_CFLAGS += -DFLASH_TYPE=$(FLASH_TYPE) -DUSE_$(FLASH_TYPE) -DUSE_$(RAM_TYPE)
% if blocking_dma:
APP_CFLAGS += -DALWAYS_BLOCK_DMA_TRANSFERS
% endif
% if do_flash:
% for layer in layers_w:
FLASH_FILES += hex/${layer}
% endfor
% if n_inputs > 1:
% for n_in in range(n_inputs):
FLASH_FILES += hex/inputs_${n_in}.hex
% endfor
% else:
FLASH_FILES += hex/inputs.hex
% endif

READFS_FILES := $(FLASH_FILES)
% endif
% if sdk == 'gap_sdk':
APP_CFLAGS += -DFS_READ_FS
% endif
#PLPBRIDGE_FLAGS += -f

include $(RULES_DIR)/pmsis_rules.mk
