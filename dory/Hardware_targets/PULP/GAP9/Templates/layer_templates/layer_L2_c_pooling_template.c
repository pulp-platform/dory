/*
 * pooling_layer_template.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 *
 * Copyright (C) 2019-2020 University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "${func_name}.h"
#include "pmsis.h"
#include "dory_get_tile.h"
#include "dory_dma.h"
#include "pulp_nn_kernels.h"
#include "tile_index.h"
#include "layer.h"
#include "net_utils.h"

% if ULTRA_VERBOSE:
#define VERBOSE_PRINT(...) printf(__VA_ARGS__)
% endif


static const TileIndex index_end = {
  .height = ${tile_dim_h},
  .width = ${tile_dim_w},
  .input_channel = ${tile_dim_nif},
  .output_channel = ${tile_dim_nof}
};

static const Layer body = {
  .input = {
    .height = ${x_tile_size_h},
    .width = ${x_tile_size_w},
    .channel = ${x_tile_size_nif},
    .channel_size = ${x_tile_size_nif_byte}
  },
  .output = {
    .height = ${y_tile_size_h},
    .width = ${y_tile_size_w},
    .channel = ${y_tile_size_nof},
    .channel_size = ${y_tile_size_nof_byte}
  }
};

static const Layer border = {
  .input = {
    .height = ${x_tile_size_h_last},
    .width = ${x_tile_size_w_last},
    .channel = ${x_tile_size_nif_last},
    .channel_size = ${x_tile_size_nif_byte_last}
  },
  .output = {
    .height = ${y_tile_size_h_last},
    .width = ${y_tile_size_w_last},
    .channel = ${y_tile_size_nof_last},
    .channel_size = ${y_length_nof_byte_last}
  }
};


static void load_input_async(Layer tile, Layer body, Layer layer, TileIndex index) {
  // additionally overlap by padding for the first tile after a border one
  // this because in the first tile we use less pixels from x_buffer, since we have the ones of padding
  const int x_offset_h = index.height > 0 ? layer.padding.top : 0;
  const int x_offset_w = index.width > 0 ? layer.padding.left : 0;

  dma_transfer_async((DmaTransferConf) {
    .ext = dory_get_tile_3d(layer.addr.input,
                            index.height, index.width, index.input_channel,
                            body.input.height, body.input.width, body.input.channel,
                            layer.input.width, layer.input.channel,
                            ${conv_overlap1}, ${conv_overlap2}, 0,
                            x_offset_h, x_offset_w, 0,
                            ${x_data_size_byte}),
    .loc = tile.addr.input,
    .number_of_2d_copies = tile.input.height,
    .number_of_1d_copies = tile.input.width,
    .length_1d_copy = tile.input.channel_size,
    .hwc_to_chw = 0,
    .stride_2d = ${x_stride_w_byte},
    .stride_1d = ${x_stride_c_byte},
    .dir = 1
  });
}

static void store_output_async(Layer tile, Layer body, Layer layer, TileIndex index) {
  dma_transfer_async((DmaTransferConf) {
    .ext = dory_get_tile_3d(layer.addr.output,
                            index.height, index.width, index.output_channel,
                            body.output.height, body.output.width, body.output.channel,
                            layer.output.width, layer.output.channel,
                            0, 0, 0,
                            0, 0, 0,
                            ${y_data_size_byte}),
    .loc = tile.addr.output,
    .number_of_2d_copies = tile.output.height,
    .number_of_1d_copies = tile.output.width,
    .length_1d_copy = tile.output.channel_size,
    .hwc_to_chw = 0,
    .stride_2d = ${y_stride_w_byte},
    .stride_1d = ${y_stride_c_byte},
    .dir = 0
  }); 
}


static void kernel(Layer tile) {
  % if 'Max' in optional:
  % if optional_type == 'mixed-sw':
  pulp_nn_maxpool_${data_type_y[0]}${y_data_size_byte}(
  % elif optional_type == 'mixed-hw':
  xpulp_nn_maxpool_${data_type_y[0]}${y_data_size_byte}(
  % else:
  pulp_nn_maxpool(
  % endif
  % else:
  % if 'mixed' in optional_type:
  ${"x" if "hw" in optional_type else ""}pulp_nn_avgpool_${data_type_x[0]}${x_data_size_byte}_${data_type_y[0]}${y_data_size_byte}(
  % else:
  pulp_nn_avgpool(
  % endif
  % endif
    tile.addr.input, tile.addr.output,
    % if 'Max' not in optional:
    ${out_mul},
    ${out_shift},
    ${out_add},
  % endif
    tile.input.width,
    tile.input.height,
    tile.input.channel,
    tile.output.width,
    tile.output.height,
    ${fs2},
    ${fs1},
    tile.padding.top,
    tile.padding.bottom,
    tile.padding.left,
    tile.padding.right,
    ${stride},
    ${stride}${"," if "Max" not in optional else ""}
% if 'Max' not in optional:
    ${FLAG_RELU}
% endif
  );
}

typedef struct PoolingArgs {
  Layer tile;
} PoolingArgs;

static void pooling(void * args) {
  PoolingArgs * poolingArgs = (PoolingArgs *)args;
  kernel(poolingArgs->tile);
}


void __attribute__ ((noinline)) ${func_name}(
  void *args
) {
  layer_args_t *layer_args = (layer_args_t *)args;
  unsigned int l1_buffer = layer_args->L1_buffer;

  Layer layer = {
    .addr = {
      .input = layer_args->L2_input,
      .output = layer_args->L2_output
    },
    .input = {
      .width = ${x_w},
      .channel = ${nif}
    },
    .output = {
      .width = ${y_w},
      .channel = ${nof}
    },
    .padding = {
        .top    = ${padding_top},
        .right  = ${padding_right},
        .bottom = ${padding_bottom},
        .left   = ${padding_left}
    }
  };

  pi_team_config_offload(NUM_CORES);

  DmaTransfer transfer = dma_transfer_create();

  TileIndex index = { .height = 0, .width = 0, .input_channel = 0, .output_channel = 0 };

  const int total_tiles = index_end.output_channel * index_end.height * index_end.width * index_end.input_channel;

  // tile loop nest
  for(int iter=0; iter<total_tiles; iter++) {
    Address addr = {
      .input = l1_buffer + ${l1_x_offset},
      .output = l1_buffer + ${l1_y_offset}
    };
    Layer tile = tile_create(index, index_end, body, border, layer, addr);

    load_input_async(tile, body, layer, index);

    dma_transfer_wait(transfer);
    PoolingArgs poolingArgs = { .tile = tile };
    pi_team_offload_preset(pooling, &poolingArgs);
    pi_team_offload_wait();

    store_output_async(tile, body, layer, index);

    index = tile_index_get_next(index, index_end);
  }
}
