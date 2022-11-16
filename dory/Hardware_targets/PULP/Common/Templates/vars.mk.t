% if do_flash:
% for layer in layers_w:
FLASH_FILES += hex/${layer}
% endfor
% if n_inputs > 1:
% for n_in in range(n_inputs):
FLASH_FILES += hex/${prefix}inputs_${n_in}.hex
% endfor
% else:
FLASH_FILES += hex/${prefix}inputs.hex
% endif

READFS_FILES := $(FLASH_FILES)
% endif
% if sdk == 'gap_sdk':
APP_CFLAGS += -DFS_READ_FS
% endif
#PLPBRIDGE_FLAGS += -f