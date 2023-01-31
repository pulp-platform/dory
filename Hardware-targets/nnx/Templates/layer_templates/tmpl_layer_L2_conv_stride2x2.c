/*
 * layer_template_nnx.c
 * Francesco Conti <f.conti@unibo.it>
 * Alessio Burrello <alessio.burrello@unibo.it>
 * Luka Macan <luka.macan@unibo.it>
 *
 * Copyright (C) 2018-2022 University of Bologna
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
#include "pulp_nnx.h"
#include "pulp_nnx_util.h"
#include "network.h"
#include "dory_dma_v2.h"
#include "dory_get_tile.h"
#include "layer_debug.h"
#include "tile_descriptor.h"

/* Defines
 *
 * 1. Debugging
 *    - DEBUG_GVSOC - enable gvsoc logging
 *    - DEBUG_TILE_CHECKSUM - calculate and print per-tile-checksum
 */

/////////////////
// Total tiles //
/////////////////

const int total_tiles = ${tile_dim_nof} /*tile_dim_nof*/ * \
% if not flag_DW:
${tile_dim_nif} /*tile_dim_nif*/ * \
% endif
${tile_dim_h} /*tile_dim_h*/ * ${tile_dim_w} /*tile_dim_w*/;

typedef struct BuffIndex {
    int input, weights, output;
} BuffIndex;

static BuffIndex buff_index_get_next(BuffIndex current, TileDescriptor tile_next) {
    return (BuffIndex) {
        .input = tile_next.is_change_input ? !current.input : current.input,
        .weights = tile_next.is_change_weights ? !current.weights : current.weights,
        .output = !current.output
    };
}

typedef struct Address {
    uint32_t input, weights, scale, bias, output;
} Address;

static Address address_get(BuffIndex index, Address *local_addresses) {
    return (Address) {
        .input = local_addresses[index.input].input,
        .weights = local_addresses[index.weights].weights,
        .scale = local_addresses[index.weights].scale,
        .bias = local_addresses[index.weights].bias,
        .output = local_addresses[index.output].output
    };
}

typedef struct Tile {
    TileDescriptor desc;
    Address addr;
    struct {
        DmaTransfer input, weights, scale, bias, output;
    } transfer;
} Tile;

static uint32_t padded_input_ptr(uint32_t input,
                                 const int padding_top, const int padding_left,
                                 const int width, const int channel, const int bits) {
    return input - (padding_top * width + padding_left) * channel * bits / 8;
}

typedef DmaTransfer LoadInput;

static LoadInput load_input_async(Tile tile, TileDescriptor body_desc, Padding layer_padding, DmaTransferConf conf, uint32_t ext) {
      // additionally overlap by padding for the first tile after a border one
      // this because in the first tile we use less pixels from x_buffer, since we have the ones of padding
      const int x_offset_h = tile.desc.index.height > 0 ? layer_padding.top : 0;
      const int x_offset_w = tile.desc.index.width > 0 ? layer_padding.left : 0;

      conf.ext = dory_get_tile_3d(ext,
            tile.desc.index.height, tile.desc.index.width, 0,
            body_desc.input.height, body_desc.input.width, body_desc.input.channel,
            ${x_w}, ${nif*g},
            ${conv_overlap1}, ${conv_overlap2}, 0,
            x_offset_h, x_offset_w, 0,
            ${x_data_size_byte});
      conf.loc = tile.addr.input;
      conf.number_of_2d_copies = tile.desc.input.height;
      conf.number_of_1d_copies = tile.desc.input.width;
      conf.length_1d_copy = tile.desc.input.channel;

      return dma_transfer_async(conf);
}

static void execute_stride2x2_blocking(nnx_task_t task, Tile tile,
        int weights_height, int weights_width
) {
    const int stride = 2;

    nnx_conv_set_strides(&task, tile.desc.input.channel, tile.desc.input.width, tile.desc.input.channel,
                         tile.desc.output.width, tile.desc.output.channel);
    nnx_conv_set_counters(&task, tile.desc.input.channel, 3, 3, tile.desc.output.channel);

    const int n_h = DIVNCEIL(tile.desc.output.height, stride);
    const int n_w = DIVNCEIL(tile.desc.output.width, stride);
    const int input_height_offset = tile.desc.output.height % stride == 1 ? stride : 0;
    const int input_width_offset  = tile.desc.output.width  % stride == 1 ? stride : 0;
    const int output_height_offset = tile.desc.output.height % stride == 1 ? 1 : 0;
    const int output_width_offset  = tile.desc.output.width  % stride == 1 ? 1 : 0;

    task.weights_ptr = tile.addr.weights;
    task.scale_ptr = tile.addr.scale;
    task.scale_bias_ptr = tile.addr.bias;

    const uint32_t input_base = padded_input_ptr(tile.addr.input, tile.desc.padding.top,
                                                 tile.desc.padding.left, tile.desc.input.width,
                                                 tile.desc.input.channel, 8);
    const uint32_t output_base = tile.addr.output;

    tile.desc.padding.bottom = (tile.desc.input.height + tile.desc.padding.top - weights_height) % stride == 0 ? 0 : tile.desc.padding.bottom;
    tile.desc.padding.right = (tile.desc.input.width + tile.desc.padding.left - weights_width) % stride == 0 ? 0 : tile.desc.padding.right;

    for (int i = 0; i < n_h; i++) {
        for (int j = 0; j < n_w; j++) {
            task.infeat_ptr = dory_get_tile_3d(input_base,
                    i, j, 0,                    /* index */
                    3 + weights_height - 1, 3 + weights_width - 1, tile.desc.input.channel,        /* size */
                    tile.desc.input.width, tile.desc.input.channel, /* stride */
                    weights_height - stride, weights_width - stride, 0, /* overlap */
                    i == 0 ? 0 : input_height_offset, j == 0 ? 0 : input_width_offset, 0, /* offset */
                    8 /* data size */
                );
            task.outfeat_ptr = dory_get_tile_3d(output_base,
                    i, j, 0,                      /* index */
                    2, 2, tile.desc.output.channel,         /* size */
                    tile.desc.output.width, tile.desc.output.channel, /* stride */
                    0, 0, 0,                      /* overlap */
                    i == 0 ? 0 : output_height_offset, j == 0 ? 0 : output_width_offset, 0, /* offset */
                    8 /* data size */
                );
            nnx_pad_input(&task.cfg, (nnx_padding_t) {
                              .top = i == 0 ? tile.desc.padding.top : 0,
                              .bottom = i == n_h - 1 ? tile.desc.padding.bottom : 0,
                              .left = j == 0 ? tile.desc.padding.left : 0,
                              .right = j == n_w - 1 ? tile.desc.padding.right : 0,
                              .value = 0
                          });

            nnx_acquire_blocking();
            nnx_offload(&task);
            nnx_run_async();
        }
    }
    nnx_wait_empty();
}

void ${func_name}(
  void *args
) {
  /////////////
  // Logging //
  /////////////

  #ifdef DEBUG_GVSOC
  nnx_activate_gvsoc_logging(GVSOC_LOGGING_FORMAT_DECIMAL);
  #endif

  //////////////////////////
  // Arguments assignment //
  //////////////////////////

  // Keep the same interface between L2 and L3 memory
  layer_args_t *layer_args = (layer_args_t *) args;
  const unsigned int l2_x = layer_args->L2_input;
  const unsigned int l2_y = layer_args->L2_output;
  const unsigned int l2_W = layer_args->L2_weights;
% if FLAG_BATCHNORM == 1:
  const unsigned int l2_scale = l2_W + ${l2_k_offset};
  const unsigned int l2_bias  = l2_W + ${l2_lambda_offset};
% endif
  const unsigned int l1_buffer = layer_args->L1_buffer;
  const unsigned int out_shift = layer_args->out_shift;
  Padding layer_padding = {
    .top    = layer_args->padding & PAD_TOP ? ${padding_top} : DONT_PAD,
    .right  = ${padding_right},
    .bottom = layer_args->padding & PAD_BOTTOM ? ${padding_bottom} : DONT_PAD,
    .left   = ${padding_left}
  };

  ////////////////////////
  // Double buffer init //
  ////////////////////////

  const int l1_buffer_x = l1_buffer + ${l1_x_offset};
  const int l1_buffer_y = l1_buffer + ${l1_y_offset};
  const int l1_buffer_w = l1_buffer + ${l1_W_offset};
  const int l1_buffer_scale = l1_buffer + ${l1_k_offset};
  const int l1_buffer_bias = l1_buffer + ${l1_lambda_offset};

  Address double_buffer_addresses[2] = {
    {
      .input = l1_buffer_x,
      .weights = l1_buffer_w,
      .scale = l1_buffer_scale,
      .bias = l1_buffer_bias,
      .output = l1_buffer_y
    },
    {
      .input = l1_buffer_x + ${l1_x_tile_size},
      .weights = l1_buffer_w + ${l1_W_tile_size},
      .scale = l1_buffer_scale + ${l1_k_tile_size},
      .bias = l1_buffer_bias + ${l1_lambda_tile_size},
      .output = l1_buffer_y + ${l1_y_tile_size}
    }
  };

  //////////////////////////
  // Variable declaration //
  //////////////////////////

  TileDescriptor body_desc = {
    .input = {
      .height = ${x_tile_size_h},
      .width = ${x_tile_size_w},
      .channel = ${x_tile_size_nif}
    },
    .output = {
      .height = ${y_tile_size_h},
      .width = ${y_tile_size_w},
      .channel = ${y_tile_size_nof}
    }
  };

  TileDescriptor border_desc = {
    .input = {
      .height = ${x_tile_size_h_last},
      .width = ${x_tile_size_w_last},
      .channel = ${x_tile_size_nif_last}
    },
    .output = {
      .height = ${y_tile_size_h_last},
      .width = ${y_tile_size_w_last},
      .channel = ${y_tile_size_nof_last}
    }
  };

  TileIndex end = {
    .height = ${tile_dim_h},
    .width = ${tile_dim_w},
    .output_channel = ${tile_dim_nof}
  };

  BuffIndex buff_index = {
    .input = 0,
    .weights = 0,
    .output = 0
  };

  Tile tile = {
    .addr = address_get(buff_index, double_buffer_addresses),
    .desc = {
      .index = {
        .height = 0,
        .width = 0,
        .output_channel = 0
      },
      .input = {
        .height = ${x_tile_size_h},
        .width = ${x_tile_size_w},
        .channel = ${x_tile_size_nif}
      },
      .output = {
        .height = ${y_tile_size_h},
        .width = ${y_tile_size_w},
        .channel = ${y_tile_size_nof}
      },
      .padding = padding_get((TileIndex) {0, 0, 0}, end, layer_padding),
      .is_change_input = 1,
      .is_change_weights = 1
    }
  };

  /////////////////////
  // DMA declaration //
  /////////////////////

  DmaTransferConf transfer_conf_input = {
    .ext = l2_x,
    .loc = tile.addr.input,
    .number_of_2d_copies = tile.desc.input.height,
    .stride_2d = ${l1_x_dma_stride_2d},
    .number_of_1d_copies = tile.desc.input.width,
    .stride_1d = ${l1_x_dma_stride_1d},
    .length_1d_copy = tile.desc.input.channel,
    .dir = 1
  };
  tile.transfer.input = dma_transfer_async(transfer_conf_input);

  DmaTransferConf transfer_conf_weights = {
    .ext = l2_W,
    .loc = tile.addr.weights,
    .number_of_2d_copies = 1,
    .stride_2d = 0,
    .number_of_1d_copies = 1,
    .stride_1d = 0,
    .length_1d_copy = ${l1_W_tile_ko_len} * ${l1_W_tile_ki_size},
    .dir = 1
  };
  tile.transfer.weights = dma_transfer_1d_async(transfer_conf_weights);

% if FLAG_BATCHNORM == 1:
  DmaTransferConf transfer_conf_scale = {
    .ext = l2_scale,
    .loc = tile.addr.scale,
    .stride_2d = 0,
    .number_of_2d_copies = 1,
    .stride_1d = 0,
    .number_of_1d_copies = 1,
    .length_1d_copy = tile.desc.output.channel * ${int(act_dim_bit/8)},
    .dir = 1
  };
  tile.transfer.scale = dma_transfer_1d_async(transfer_conf_scale);

  DmaTransferConf transfer_conf_bias = {
    .ext = l2_bias,
    .loc = tile.addr.bias,
    .stride_2d = 0,
    .number_of_2d_copies = 1,
    .stride_1d = 0,
    .number_of_1d_copies = 1,
    .length_1d_copy = tile.desc.output.channel * ${int(bias_bits/8)},
    .dir = 1
  };
  tile.transfer.bias = dma_transfer_1d_async(transfer_conf_bias);

% endif
  
  DmaTransferConf transfer_conf_output = {
    .stride_2d = ${l1_y_dma_stride_2d},
    .stride_1d = ${l1_y_dma_stride_1d},
    .dir = 0
  };

  //////////////////////
  // Accelerator init //
  //////////////////////

  nnx_init();

  ///////////////////
  // NNX task init //
  ///////////////////

  nnx_task_t task = nnx_task_create(
        ${fs1}, ${int(flag_DW)},
        8, 8, ${W_data_size},
        weightOffsetModeLayerWise, ${-(2**(W_data_size-1))},
        (nnx_quant_t) {
            .shift_amount = out_shift,
            .mode = quantMode8Bit,
            .function = quantFunctionRelu,
            .flag_rounding = FLAG_UNUSED
        }, (nnx_norm_t) {
            .mode  = ${"normMode32Bit" if act_dim_bit == 32 else "normMode8Bit" if act_dim_bit == 8 else "FLAG_UNUSED"},
            .flag_bias  = FLAG_USED,
            .flag_shift = FLAG_UNUSED
        }, ${stride});


//  /$$$$$$$$ /$$$$$$ /$$       /$$$$$$$$       /$$        /$$$$$$   /$$$$$$  /$$$$$$$ 
// |__  $$__/|_  $$_/| $$      | $$_____/      | $$       /$$__  $$ /$$__  $$| $$__  $$
//    | $$     | $$  | $$      | $$            | $$      | $$  \ $$| $$  \ $$| $$  \ $$
//    | $$     | $$  | $$      | $$$$$         | $$      | $$  | $$| $$  | $$| $$$$$$$/
//    | $$     | $$  | $$      | $$__/         | $$      | $$  | $$| $$  | $$| $$____/ 
//    | $$     | $$  | $$      | $$            | $$      | $$  | $$| $$  | $$| $$      
//    | $$    /$$$$$$| $$$$$$$$| $$$$$$$$      | $$$$$$$$|  $$$$$$/|  $$$$$$/| $$      
//    |__/   |______/|________/|________/      |________/ \______/  \______/ |__/      

  for (int i_tile = 0; i_tile < total_tiles; i_tile++)
  {
                                                                                        
    TileDescriptor tile_desc_next =
                    tile_descriptor_get_next(tile.desc, body_desc, border_desc, end, layer_padding);

    BuffIndex buff_index_next = buff_index_get_next(buff_index, tile_desc_next);

    Tile tile_next = {
        .desc = tile_desc_next,
        .addr = address_get(buff_index_next, double_buffer_addresses)
    };

    // LOAD Next Tile

    if (tile_next.desc.is_change_input) {
        tile_next.transfer.input = load_input_async(tile_next, body_desc, layer_padding, transfer_conf_input, l2_x);
    }

    if (tile_next.desc.is_change_weights) {
      // Special because of accelerators special memory layout
      const int W_tile_size_nof_k_major = tile_next.desc.index.output_channel + 1 == end.output_channel ? ${l1_W_tile_ko_len_last} : ${l1_W_tile_ko_len};

      transfer_conf_weights.ext += transfer_conf_weights.length_1d_copy;
      transfer_conf_weights.loc = tile_next.addr.weights;
      transfer_conf_weights.length_1d_copy = W_tile_size_nof_k_major * ${l1_W_tile_ki_size};
      tile_next.transfer.weights = dma_transfer_1d_async(transfer_conf_weights);

% if FLAG_BATCHNORM == 1:
      transfer_conf_scale.ext += transfer_conf_scale.length_1d_copy;
      transfer_conf_scale.loc = tile_next.addr.scale;
      transfer_conf_scale.length_1d_copy = tile_next.desc.output.channel * ${int(act_dim_bit/8)};
      tile_next.transfer.scale = dma_transfer_1d_async(transfer_conf_scale);

      transfer_conf_bias.ext += transfer_conf_bias.length_1d_copy;
      transfer_conf_bias.loc = tile_next.addr.bias;
      transfer_conf_bias.length_1d_copy = tile_next.desc.output.channel * ${int(bias_bits/8)};
      tile_next.transfer.bias = dma_transfer_1d_async(transfer_conf_bias);
% endif
    }


    // WAIT LOAD Current Tile

    if (tile.desc.is_change_input) {
      dma_transfer_wait(tile.transfer.input);
    }

    if (tile.desc.is_change_weights) {
      dma_transfer_wait(tile.transfer.weights);
      dma_transfer_wait(tile.transfer.scale);
      dma_transfer_wait(tile.transfer.bias);
    }


    // EXECUTE Current Tile

    execute_stride2x2_blocking(task, tile, ${fs1}, ${fs2});


    // STORE Current Tile

    transfer_conf_output.ext = dory_get_tile_3d(l2_y,
        tile.desc.index.height, tile.desc.index.width, tile.desc.index.output_channel,
        body_desc.output.height, body_desc.output.width, body_desc.output.channel,
        ${y_w}, ${int(nof*factor)},
        0, 0, 0,
        0, 0, 0,
        ${y_data_size_byte});
    transfer_conf_output.loc = tile.addr.output;
    transfer_conf_output.number_of_2d_copies = tile.desc.output.height;
    transfer_conf_output.number_of_1d_copies = tile.desc.output.width;
    transfer_conf_output.length_1d_copy = tile.desc.output.channel;

    TILE_CHECKSUM_PRINT(transfer_conf_output, i_tile);

    tile_next.transfer.output = dma_transfer_async(transfer_conf_output);

    tile = tile_next;
    buff_index = buff_index_next;
  }

  dma_transfer_wait(tile.transfer.output);
  nnx_term();
}
