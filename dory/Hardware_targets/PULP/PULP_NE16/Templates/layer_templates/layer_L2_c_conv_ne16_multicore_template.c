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
#include "net_utils.h"
#include "dory_dma_single_core.h"
#include "dory_get_tile.h"
#include "tile_status.h"
#include "execute.h"

static const Kernel kernel = {
    .shape = {
        .height = ${fs1},
        .width = ${fs2}
    },
    .stride = {
        .height = ${stride},
        .width = ${stride}
    },
    .groups = ${g}
};

static const TileIndex end_index = {
    .height = ${tile_dim_h},
    .width = ${tile_dim_w},
    .output_channel = ${tile_dim_nof}
};

static const Layer body = {
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

static const Layer border = {
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

static int dma_tid;

static inline void load_input_prepare(Layer tile, Layer body, Layer layer, TileIndex index, DMA_copy * const conf, Kernel kernel) {
    // additionally overlap by padding for the first tile after a border one
    // this because in the first tile we use less pixels from x_buffer, since we have the ones of padding
    const int x_offset_h = index.height > 0 ? layer.padding.top : 0;
    const int x_offset_w = index.width > 0 ? layer.padding.left : 0;

    conf->ext = dory_get_tile_3d(layer.addr.input,
                                 index.height, index.width, kernel.groups > 1 ? index.output_channel : 0,
                                 body.input.height, body.input.width, body.input.channel,
                                 ${x_w}, ${nif*g},
                                 ${conv_overlap1}, ${conv_overlap2}, 0,
                                 x_offset_h, x_offset_w, 0,
                                 ${x_data_size_byte});
    conf->loc = tile.addr.input;
    conf->number_of_2d_copies = tile.input.height;
    conf->number_of_1d_copies = tile.input.width;
    conf->length_1d_copy = tile.input.channel;
    conf->stride_2d = ${l1_x_dma_stride_2d};
    conf->stride_1d = ${l1_x_dma_stride_1d};
    conf->dir = 1;
    conf->tid = dma_tid;
}

static inline void load_weights_prepare(Layer tile, Kernel kernel,
                                        MemoryStatus status_weights,
                                        MemoryStatus status_scale,
                                        MemoryStatus status_bias,
                                        DMA_copy * const conf_weights,
                                        DMA_copy * const conf_scale,
                                        DMA_copy * const conf_bias
                                        ) {
    // Special because of accelerators special memory layout
    const int size_weights = (kernel.groups > 1 ? divnceil(tile.output.channel, 16) : tile.output.channel) * ${l1_W_tile_ki_size};
    const int size_scale = tile.output.channel * ${int(act_dim_bit/8)};
    const int size_bias = tile.output.channel * ${int(b_data_size_byte/8)};

    #define CONF_SET(name)                             ${"\\"}
        conf_ ## name->number_of_2d_copies = 1;        ${"\\"}
        conf_ ## name->number_of_1d_copies = 1;        ${"\\"}
        conf_ ## name->ext = status_ ## name.addr_ext; ${"\\"}
        conf_ ## name->loc = tile.addr.name;           ${"\\"}
        conf_ ## name->length_1d_copy = size_ ## name; ${"\\"}
        conf_ ## name->dir = 1;                        ${"\\"}
        conf_ ## name->tid = dma_tid;

    CONF_SET(weights);
    CONF_SET(scale);
    CONF_SET(bias);
}

static inline void store_prepare(Layer tile, Layer body, Layer layer, TileIndex index, DMA_copy * const conf) {
    conf->ext = dory_get_tile_3d(layer.addr.output,
                                 index.height, index.width, index.output_channel,
                                 body.output.height, body.output.width, body.output.channel,
                                 ${y_w}, ${int(nof*factor)},
                                 0, 0, 0,
                                 0, 0, 0,
                                 ${y_data_size_byte});
    conf->loc = tile.addr.output;
    conf->number_of_2d_copies = tile.output.height;
    conf->number_of_1d_copies = tile.output.width;
    conf->length_1d_copy = tile.output.channel;
    conf->stride_2d = ${l1_y_dma_stride_2d};
    conf->stride_1d = ${l1_y_dma_stride_1d};
    conf->dir = 0;
    conf->tid = dma_tid;
}

static void load_async(Layer tile, TileStatus * const status, Layer body, Layer layer, Kernel kernel) {

    DMA_copy conf_input, conf_weights, conf_scale, conf_bias;

    if (status->input.is_transfer) {
        load_input_prepare(tile, body, layer, status->index, &conf_input, kernel);
        dory_dma_memcpy_single_core_async(&conf_input);
    }

    if (status->weights.is_transfer) {
        load_weights_prepare(tile, kernel,
                             status->weights,
                             status->scale,
                             status->bias,
                             &conf_weights,
                             &conf_scale,
                             &conf_bias);
        dory_dma_memcpy_1d_single_core_async(&conf_weights);
        dory_dma_memcpy_1d_single_core_async(&conf_scale);
        dory_dma_memcpy_1d_single_core_async(&conf_bias);

        status->weights.addr_ext += conf_weights.length_1d_copy;
        status->scale.addr_ext += conf_scale.length_1d_copy;
        status->bias.addr_ext += conf_bias.length_1d_copy;
    }
}


static void store_async(Layer tile, TileStatus * const status, Layer body, Layer layer) {
    DMA_copy conf;
    store_prepare(tile, body, layer, status->index, &conf);
    dory_dma_memcpy_single_core_async(&conf);
}


void ${func_name}(void *args) {
    // Only core 0 executes this code, all the rest exit immediately
    if (pi_core_id() != 0) {
        return;
    }

    nnx_task_t nnx_task;

    nnx_task_init(&nnx_task,
            ${fs1}, ${int(flag_DW)},
            ${x_data_size_byte}, ${y_data_size_byte}, ${W_data_size_byte},
            weightOffsetModeLayerWise, ${-(2**(W_data_size_byte-1))},
            (nnx_quant_t) {
                .shift_amount = ${out_shift},
                .mode = quantMode8Bit,
                .function = quantFunctionRelu,
                .flag_rounding = NE16_FLAG_UNUSED
            }, (nnx_norm_t) {
                .mode  = ${"normMode32Bit" if act_dim_bit == 32 else "normMode8Bit" if act_dim_bit == 8 else "NE16_FLAG_UNUSED"},
                .flag_bias  = NE16_FLAG_USED,
                .flag_shift = NE16_FLAG_UNUSED
            }, ${stride});

    const int max_stall = 8;
    nnx_init(max_stall);

    dma_tid = dory_dma_allocate_single_core();
    DMA_copy dummy_conf_for_waitin = { .tid = dma_tid };

    layer_args_t *layer_args = (layer_args_t *)args;

    Layer layer = {
        .addr = {
            .input = layer_args->L2_input,
            .weights = layer_args->L2_weights,
            .scale = layer_args->L2_weights + ${l2_k_offset},
            .bias = layer_args->L2_weights + ${l2_lambda_offset},
            .output = layer_args->L2_output
        },
        .padding = {
            .top    = layer_args->padding & NET_UTILS_PAD_TOP ? ${padding_top} : NE16_DONT_PAD,
            .right  = ${padding_right},
            .bottom = layer_args->padding & NET_UTILS_PAD_BOTTOM ? ${padding_bottom} : NE16_DONT_PAD,
            .left   = ${padding_left}
        }
    };

    // Double buffer address init

    const unsigned int l1_buffer = layer_args->L1_buffer;
    const int l1_buffer_input = l1_buffer + ${l1_x_offset};
    const int l1_buffer_output = l1_buffer + ${l1_y_offset};
    const int l1_buffer_weights = l1_buffer + ${l1_W_offset};
    const int l1_buffer_scale = l1_buffer + ${l1_k_offset};
    const int l1_buffer_bias = l1_buffer + ${l1_lambda_offset};

    Address buffer_addresses[2] = {
        {
            .input = l1_buffer_input,
            .weights = l1_buffer_weights,
            .scale = l1_buffer_scale,
            .bias = l1_buffer_bias,
            .output = l1_buffer_output
        },
        {
            .input = l1_buffer_input + ${l1_x_tile_size},
            .weights = l1_buffer_weights + ${l1_W_tile_size},
            .scale = l1_buffer_scale + ${l1_k_tile_size},
            .bias = l1_buffer_bias + ${l1_lambda_tile_size},
            .output = l1_buffer_output + ${l1_y_tile_size}
        }
    };

    #define MEMORY_STATUS_INIT(name)     ${"\\"}
        .name = {                        ${"\\"}
            .addr_ext = layer.addr.name, ${"\\"}
            .is_transfer = 1,            ${"\\"}
            .buffer_index = 0            ${"\\"}
        }

    TileStatus tile_status = {
        .index = { 0 },
        MEMORY_STATUS_INIT(input),
        MEMORY_STATUS_INIT(weights),
        MEMORY_STATUS_INIT(scale),
        MEMORY_STATUS_INIT(bias),
        MEMORY_STATUS_INIT(output)
    };

    const int total_tiles = end_index.height * end_index.width * end_index.output_channel;

    Layer tile = tile_create(tile_status.index, end_index, body, border, layer,
                             tile_status_get_addr(tile_status, buffer_addresses));

    load_async(tile, &tile_status, body, layer, kernel);

    for (int i_tile = 0; i_tile < total_tiles; i_tile++) {
        TileStatus tile_status_next = tile_status_get_next(tile_status, end_index, layer, 0, kernel);
        Layer tile_next = tile_create(tile_status_next.index, end_index, body, border, layer,
                                      tile_status_get_addr(tile_status_next, buffer_addresses));

        % if stride == 1:
        execute_prepare(tile, &nnx_task);
        % elif stride == 2:
        execute_stride2x2_prepare(tile, kernel, &nnx_task);
        % endif

        dory_dma_barrier_single_core(&dummy_conf_for_waitin);

        load_async(tile_next, &tile_status_next, body, layer, kernel);

        % if stride == 1:
        execute_async(&nnx_task);
        % elif stride == 2:
        execute_stride2x2_blocking(&nnx_task, tile, kernel, tile.output.channel);
        % endif
        execute_wait(&nnx_task);

        store_async(tile, &tile_status, body, layer);

        tile_status = tile_status_next;
        tile = tile_next;
    }

    dory_dma_barrier_single_core(&dummy_conf_for_waitin);
    nnx_term();
}
