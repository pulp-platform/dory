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

/* Defines
 *
 * 1. Debugging
 *    - DEBUG_GVSOC - enable gvsoc logging
 *    - DEBUG_TILE_CHECKSUM - calculate and print per-tile-checksum
 */

#ifdef DEBUG_TILE_CHECKSUM
#define TILE_CHECKSUM_PRINT(dma, i_tile)                    ${"\\"} 
    do {                                                    ${"\\"}
        uint8_t *ptr = (uint8_t *)dma.loc;                  ${"\\"}
        int sum = 0;                                        ${"\\"}
        for (int i = 0; i < dma.number_of_2d_copies; i++)   ${"\\"}
          for (int j = 0; j < dma.number_of_1d_copies; j++) ${"\\"}
            for (int k = 0; k < dma.length_1d_copy; k++)    ${"\\"}
              sum += *ptr++;                                ${"\\"}
        printf("[%d] Checksum: %d\n", i_tile, sum);         ${"\\"}
    } while (0)
#else  // DEBUG_TILE_CHECKSUM
#define TILE_CHECKSUM_PRINT(dma, i_tile)
#endif // DEBUG_TILE_CHECKSUM

#define MIN(a,b) (a < b ? a : b)

/////////////////
// Total tiles //
/////////////////

const int total_tiles = ${tile_dim_nof} /*tile_dim_nof*/ * \
% if not flag_DW:
${tile_dim_nif} /*tile_dim_nif*/ * \
% endif
${tile_dim_h} /*tile_dim_h*/ * ${tile_dim_w} /*tile_dim_w*/;

typedef struct TileIndex {
    int height, width, output_channel;
} TileIndex;

typedef struct TileDescriptor {
    TileIndex index;
    struct {
        int height, width, channel;
    } input, output;
    int is_load_input, is_load_weights;
    int i_buff_input, i_buff_weights, i_buff_output;
    struct {
        DmaTransfer input, weights, scale, bias, output;
    } transfer;
} TileDescriptor;

static TileIndex tile_index_get_next(TileIndex index, TileIndex end) {
    index.width += 1;
    if (index.width >= end.width) {
        index.width = 0;
        index.height += 1;
        if (index.height >= end.height) {
            index.height = 0;
            index.output_channel += 1;
            if (index.output_channel >= end.output_channel) {
                index.output_channel = 0;
            }
        }
    }
    return index;
}

static TileDescriptor tile_descriptor_get_next(TileDescriptor current, TileDescriptor body, TileDescriptor border, TileIndex end) {
    TileDescriptor next = {
        .index = tile_index_get_next(current.index, end)
    };

    next.input.height = next.index.height + 1 == end.height ? border.input.height : body.input.height;
    next.input.width = next.index.width + 1 == end.width ? border.input.width : body.input.width;
    next.input.channel = current.input.channel;

    next.output.height = next.index.height + 1 == end.height ? border.output.height : body.output.height;
    next.output.width = next.index.width + 1 == end.width ? border.output.width : body.output.width;
    next.output.channel = next.index.output_channel + 1 == end.output_channel ? border.output.channel : body.output.channel;

    int is_first_input = next.index.height == 0 && next.index.width == 0 && next.index.output_channel == 0;
    int is_only_input_tile = end.height == 1 && end.width == 1;
    int is_change_input = next.index.height != current.index.height || next.index.width != current.index.width;
    next.is_load_input = !(is_first_input || is_only_input_tile) && is_change_input;

    int is_first_weights = current.index.output_channel + 1 == end.output_channel;
    int is_only_weights_tile = end.output_channel == 1;
    int is_change_weights = next.index.output_channel != current.index.output_channel;
    next.is_load_weights = !(is_first_weights || is_only_weights_tile) && is_change_weights;

    if (next.is_load_input) {
        next.i_buff_input = !current.i_buff_input;
    } else {
        next.i_buff_input =  current.i_buff_input;
    }

    if (next.is_load_weights) {
        next.i_buff_weights = !current.i_buff_weights;
    } else {
        next.i_buff_weights =  current.i_buff_weights;
    }

    next.i_buff_output = !current.i_buff_output;

    return next;
}

static uint32_t padded_input_ptr(uint32_t input,
                                 const int padding_top, const int padding_left,
                                 const int width, const int channel, const int bits) {
    return input - (padding_top * width + padding_left) * channel * bits / 8;
}

static void execute_stride2x2_blocking(nnx_task_t task,
        int input_height, int input_width, int input_channel,
        int output_height, int output_width, int output_channel,
        int weights_height, int weights_width,
        nnx_padding_t padding
) {
    const int stride = 2;

    nnx_conv_set_strides(&task, input_channel, input_width, input_channel,
                         output_width, output_channel);
    nnx_conv_set_counters(&task, input_channel, 3, 3, output_channel);

    const int n_h = DIVNCEIL(output_height, stride);
    const int n_w = DIVNCEIL(output_width, stride);
    const int input_height_offset = output_height % stride == 1 ? stride : 0;
    const int input_width_offset  = output_width  % stride == 1 ? stride : 0;
    const int output_height_offset = output_height % stride == 1 ? 1 : 0;
    const int output_width_offset  = output_width  % stride == 1 ? 1 : 0;

    const uint32_t input_base = padded_input_ptr(task.infeat_ptr, padding.top,
                                                 padding.left, input_width,
                                                 input_channel, 8);
    const uint32_t output_base = task.outfeat_ptr;

    padding.bottom = (input_height + padding.top - weights_height) % stride == 0 ? 0 : padding.bottom;
    padding.right = (input_width + padding.left - weights_width) % stride == 0 ? 0 : padding.right;

    for (int i = 0; i < n_h; i++) {
        for (int j = 0; j < n_w; j++) {
            task.infeat_ptr = dory_get_tile_3d(input_base,
                    i, j, 0,                    /* index */
                    3 + weights_height - 1, 3 + weights_width - 1, input_channel,        /* size */
                    input_width, input_channel, /* stride */
                    weights_height - stride, weights_width - stride, 0, /* overlap */
                    i == 0 ? 0 : input_height_offset, j == 0 ? 0 : input_width_offset, 0, /* offset */
                    8 /* data size */
                );
            task.outfeat_ptr = dory_get_tile_3d(output_base,
                    i, j, 0,                      /* index */
                    2, 2, output_channel,         /* size */
                    output_width, output_channel, /* stride */
                    0, 0, 0,                      /* overlap */
                    i == 0 ? 0 : output_height_offset, j == 0 ? 0 : output_width_offset, 0, /* offset */
                    8 /* data size */
                );
            nnx_pad_input(&task.cfg, (nnx_padding_t) {
                              .top = i == 0 ? padding.top : 0,
                              .bottom = i == n_h - 1 ? padding.bottom : 0,
                              .left = j == 0 ? padding.left : 0,
                              .right = j == n_w - 1 ? padding.right : 0
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
  nnx_padding_t padding = {
    .top    = layer_args->padding & PAD_TOP ? ${padding_top} : DONT_PAD,
    .right  = ${padding_right},
    .bottom = layer_args->padding & PAD_BOTTOM ? ${padding_bottom} : DONT_PAD,
    .left   = ${padding_left},
    .value = 0
  };

  ////////////////////////
  // Double buffer init //
  ////////////////////////

  const int l1_buffer_x = l1_buffer + ${l1_x_offset};
  const int l1_buffer_y = l1_buffer + ${l1_y_offset};
  const int l1_buffer_w = l1_buffer + ${l1_W_offset};
% if FLAG_BATCHNORM:
  const int l1_buffer_scale = l1_buffer + ${l1_k_offset};
  const int l1_buffer_bias = l1_buffer + ${l1_lambda_offset};
% endif

  const struct {
% if FLAG_BATCHNORM == 1:
    int scale;
    int bias;
% endif
    int input;
    int output;
    int weights;
  } db[2] = {
    {
% if FLAG_BATCHNORM == 1:
      .scale = l1_buffer_scale,
      .bias = l1_buffer_bias,
% endif
      .input = l1_buffer_x,
      .output = l1_buffer_y,
      .weights = l1_buffer_w
    },
    {
% if FLAG_BATCHNORM == 1:
      .scale = l1_buffer_scale + ${l1_k_tile_size},
      .bias = l1_buffer_bias + ${l1_lambda_tile_size},
% endif
      .input = l1_buffer_x + ${l1_x_tile_size},
      .output = l1_buffer_y + ${l1_y_tile_size},
      .weights = l1_buffer_w + ${l1_W_tile_size}
    }
  };

  //////////////////////////
  // Variable declaration //
  //////////////////////////

  TileDescriptor tile = {
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
    .is_load_input = 1,
    .is_load_weights = 1,
    .i_buff_input = 0,
    .i_buff_weights = 0,
    .i_buff_output = 0
  };

  TileDescriptor body = {
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

  TileDescriptor border = {
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

  /////////////////////
  // DMA declaration //
  /////////////////////

  DmaTransferConf transfer_conf_input = {
    .ext = l2_x,
    .loc = db[tile.i_buff_input].input,
    .number_of_2d_copies = tile.input.height,
    .stride_2d = ${l1_x_dma_stride_2d},
    .number_of_1d_copies = tile.input.width,
    .stride_1d = ${l1_x_dma_stride_1d},
    .length_1d_copy = tile.input.channel,
    .dir = 1
  };
  tile.transfer.input = dma_transfer_async(transfer_conf_input);

  DmaTransferConf transfer_conf_weights = {
    .ext = l2_W,
    .loc = db[tile.i_buff_weights].weights,
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
    .loc = db[tile.i_buff_weights].scale,
    .stride_2d = 0,
    .number_of_2d_copies = 1,
    .stride_1d = 0,
    .number_of_1d_copies = 1,
    .length_1d_copy = tile.output.channel * ${int(act_dim_bit/8)},
    .dir = 1
  };
  tile.transfer.scale = dma_transfer_1d_async(transfer_conf_scale);

  DmaTransferConf transfer_conf_bias = {
    .ext = l2_bias,
    .loc = db[tile.i_buff_weights].bias,
    .stride_2d = 0,
    .number_of_2d_copies = 1,
    .stride_1d = 0,
    .number_of_1d_copies = 1,
    .length_1d_copy = tile.output.channel * ${int(bias_bits/8)},
    .dir = 1
  };
  tile.transfer.bias = dma_transfer_1d_async(transfer_conf_bias);

% endif
  
  DmaTransferConf transfer_conf_output = {
    .stride_2d = ${l1_y_dma_stride_2d},
    .stride_1d = ${l1_y_dma_stride_1d},
    .dir = 0
  };

  ////////////////
  // First Load //
  ////////////////
  
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

  int i_spatial_tile = 0;

  for (int i_tile = 0; i_tile < total_tiles; i_tile++)
  {
                                                                                        
    TileDescriptor tile_next = tile_descriptor_get_next(tile, body, border, end);


    // LOAD Next Tile

    if (tile_next.is_load_input) {
      // additionally overlap by padding for the first tile after a border one
      // this because in the first tile we use less pixels from x_buffer, since we have the ones of padding
      const int x_offset_h = tile_next.index.height > 0 ? padding.top : 0;
      const int x_offset_w = tile_next.index.width > 0 ? padding.left : 0;

      transfer_conf_input.ext = dory_get_tile_3d(l2_x,
            tile_next.index.height, tile_next.index.width, 0,
            body.input.height, body.input.width, body.input.channel,
            ${x_w}, ${nif*g},
            ${conv_overlap1}, ${conv_overlap2}, 0,
            x_offset_h, x_offset_w, 0,
            ${x_data_size_byte});
      transfer_conf_input.loc = db[tile_next.i_buff_input].input;
      transfer_conf_input.number_of_2d_copies = tile_next.input.height;
      transfer_conf_input.number_of_1d_copies = tile_next.input.width;
      transfer_conf_input.length_1d_copy = tile_next.input.channel;

      tile_next.transfer.input = dma_transfer_async(transfer_conf_input);
    }

    if (tile_next.is_load_weights) {
      // Special because of accelerators special memory layout
      const int W_tile_size_nof_k_major = tile_next.index.output_channel + 1 == end.output_channel ? ${l1_W_tile_ko_len_last} : ${l1_W_tile_ko_len};

      transfer_conf_weights.ext += transfer_conf_weights.length_1d_copy;
      transfer_conf_weights.loc = db[tile_next.i_buff_weights].weights;
      transfer_conf_weights.length_1d_copy = W_tile_size_nof_k_major * ${l1_W_tile_ki_size};
      tile_next.transfer.weights = dma_transfer_1d_async(transfer_conf_weights);

% if FLAG_BATCHNORM == 1:
      transfer_conf_scale.ext += transfer_conf_scale.length_1d_copy;
      transfer_conf_scale.loc = db[tile_next.i_buff_weights].scale;
      transfer_conf_scale.length_1d_copy = tile_next.output.channel * ${int(act_dim_bit/8)};
      tile_next.transfer.scale = dma_transfer_1d_async(transfer_conf_scale);

      transfer_conf_bias.ext += transfer_conf_bias.length_1d_copy;
      transfer_conf_bias.loc = db[tile_next.i_buff_weights].bias;
      transfer_conf_bias.length_1d_copy = tile_next.output.channel * ${int(bias_bits/8)};
      tile_next.transfer.bias = dma_transfer_1d_async(transfer_conf_bias);
% endif
    }


    // WAIT LOAD Current Tile

    if (tile.is_load_input) {
      dma_transfer_wait(tile.transfer.input);
    }

    if (tile.is_load_weights) {
      dma_transfer_wait(tile.transfer.weights);
      dma_transfer_wait(tile.transfer.scale);
      dma_transfer_wait(tile.transfer.bias);
    }


    // EXECUTE Current Tile

    task.infeat_ptr = db[tile.i_buff_input].input;
    task.outfeat_ptr = db[tile.i_buff_output].output;
    task.weights_ptr = db[tile.i_buff_weights].weights;
    % if FLAG_BATCHNORM == 1:
    task.scale_ptr = db[tile.i_buff_weights].scale;
    task.scale_bias_ptr = db[tile.i_buff_weights].bias;
    % endif

    execute_stride2x2_blocking(task,
                tile.input.height, tile.input.width, tile.input.channel,
                tile.output.height, tile.output.width, tile.output.channel,
                ${fs1}, ${fs2},
                (nnx_padding_t) {
                    .top    = tile.index.height == 0 ? padding.top : DONT_PAD,
                    .right  = tile.index.width == end.width - 1 ? padding.right : DONT_PAD,
                    .bottom = tile.index.height == end.height - 1 ? padding.bottom : DONT_PAD,
                    .left   = tile.index.width == 0 ? padding.left : DONT_PAD
                });


    // STORE Current Tile

    transfer_conf_output.ext = dory_get_tile_3d(l2_y,
        tile.index.height, tile.index.width, tile.index.output_channel,
        body.output.height, body.output.width, body.output.channel,
        ${y_w}, ${int(nof*factor)},
        0, 0, 0,
        0, 0, 0,
        ${y_data_size_byte});
    transfer_conf_output.loc = db[tile.i_buff_output].output;
    transfer_conf_output.number_of_2d_copies = tile.output.height;
    transfer_conf_output.number_of_1d_copies = tile.output.width;
    transfer_conf_output.length_1d_copy = tile.output.channel;

    TILE_CHECKSUM_PRINT(transfer_conf_output, i_tile);

    tile_next.transfer.output = dma_transfer_async(transfer_conf_output);

    tile = tile_next;
  }

  dma_transfer_wait(tile.transfer.output);
  nnx_term();
}
