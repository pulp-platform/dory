/*
 * add_layer_template.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 * Thorir Mar Ingolfsson <thoriri@iis.ee.ethz.ch>
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

  dma_transfer_async((DmaTransferConf) {
    .ext = dory_get_tile_3d(layer.addr.input,
                            index.height, index.width, index.input_channel,
                            body.input.height, body.input.width, body.input.channel,
                            layer.input.width, layer.input.channel,
                            ${conv_overlap1}, ${conv_overlap2}, 0,
                            0, 0, 0,
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

static void kernel(Layer tile, Layer tile2) {
  % if optional_type == '8bit':
  pulp_nn_add(
    tile.addr.input,
    tile2.addr.input,
    tile.addr.output,
    ${inmul1},
    ${inmul2},
    ${outshift},
    tile.input.width,
    tile.input.height,
    tile.input.channel
    );
  % else:
  ${"x" if 'hw' in optional_type else ""}pulp_nn_add_${data_type_x[0]}${x_data_size_byte}_${data_type_x2[0]}${x_data_size_byte2}_${data_type_y[0]}${y_data_size_byte}(
    tile.addr.input,
    tile2.addr.input,
    tile.addr.output,
    ${inmul2},
    ${inadd2},
    ${inshift2},
    ${inmul1},
    ${inadd1},
    ${inshift1},
    ${outmul},
    ${outadd},
    ${outshift},
    tile.input.width,
    tile.input.height,
    tile.input.channel,
    1
    );
  % endif
}

typedef struct AdditionArgs {
  Layer tile;
  Layer tile2;
} AdditionArgs;

static void addition(void * args) {
  AdditionArgs * additionArgs = (AdditionArgs *)args;
  kernel(additionArgs->tile, additionArgs->tile2);
}


void __attribute__ ((noinline)) ${func_name}(
  void *args
) {
  layer_args_t *layer_args = (layer_args_t *)args;
  unsigned int l1_buffer = layer_args->L1_buffer;

  const Layer layer = {
    .addr = {
      .input = layer_args->L2_input,
      .output = layer_args->L2_output
    },
    .output = {
      .width = ${y_w},
      .channel = ${nof}
    },
    .input = {
      .width = ${x_w},
      .channel = ${nif}
    }
  };

  const Layer layer2 = {
    .addr = {
      .input = layer_args->bypass,
      .output = layer_args->L2_output
    },
    .input = {
      .width = ${x_w},
      .channel = ${nif}
    }
  };

  pi_team_config_offload(NUM_CORES);

  DmaTransfer transfer = dma_transfer_create();

  TileIndex index = { .height = 0, .width = 0, .input_channel = 0, .output_channel = 0 };

  const int total_tiles = index_end.output_channel * index_end.height * index_end.width * index_end.input_channel;

  for(int iter=0; iter<total_tiles; iter++) {
    Layer tile = tile_create(index, index_end, body, border, layer,
                              (Address) { .input = l1_buffer + ${l1_x_offset}, .output = l1_buffer + ${l1_y_offset} });
    Layer tile2 = tile_create(index, index_end, body, border, layer,
                              (Address) { .input = l1_buffer + ${l1_x2_offset}, .output = l1_buffer + ${l1_y_offset} });

    load_input_async(tile, body, layer, index);
    load_input_async(tile2, body, layer2, index);

    dma_transfer_wait(transfer);
    AdditionArgs additionArgs = { .tile = tile, .tile2 = tile2 };
    pi_team_offload_preset(addition, &additionArgs);
    pi_team_offload_wait();

    store_output_async(tile, body, layer, index);

    index = tile_index_get_next(index, index_end);
  }
}
