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

// ne16 includes
#include "pulp_nnx_ne16.h"
#include "pulp_nnx_util.h"
#include "ne16_pulp_bsp.h"
#include "ne16_task.h"
#include "ne16.h"
#include "hwpe.h"

#include "network.h"
#include "net_utils.h"
#include "dory_dma.h"
#include "dory_get_tile.h"
#include "tile_status.h"
#include "execute.h"
#include "monitor.h"

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

static inline void load_input_prepare(Layer tile, Layer body, Layer layer, TileIndex index, DmaTransferConf * const conf, Kernel kernel) {
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
}

static inline void load_weights_prepare(Layer tile, Kernel kernel,
                                        MemoryStatus status_weights,
                                        MemoryStatus status_scale,
                                        MemoryStatus status_bias,
                                        DmaTransferConf * const conf_weights,
                                        DmaTransferConf * const conf_scale,
                                        DmaTransferConf * const conf_bias
                                        ) {
    // Special because of accelerators special memory layout
    const int size_weights = (kernel.groups > 1 ? nnx_calculate_number_of_tiles(tile.output.channel, 16) : tile.output.channel) * ${l1_W_tile_ki_size};
    const int size_scale = tile.output.channel * ${int(act_dim_bit/8)};
    const int size_bias = tile.output.channel * ${int(b_data_size_byte/8)};

    #define CONF_SET(name)                             ${"\\"}
        conf_ ## name->ext = status_ ## name.addr_ext; ${"\\"}
        conf_ ## name->loc = tile.addr.name;           ${"\\"}
        conf_ ## name->length_1d_copy = size_ ## name; ${"\\"}
        conf_ ## name->dir = 1;

    CONF_SET(weights);
    CONF_SET(scale);
    CONF_SET(bias);
}

static inline void store_prepare(Layer tile, Layer body, Layer layer, TileIndex index, DmaTransferConf * const conf) {
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
}

static void load_async(Layer tile, TileStatus * const status, Layer body, Layer layer, Kernel kernel) {

    DmaTransferConf conf_input, conf_weights, conf_scale, conf_bias;

    if (status->input.is_transfer) {
        load_input_prepare(tile, body, layer, status->index, &conf_input, kernel);
        dma_transfer_async(conf_input);
    }

    if (status->weights.is_transfer) {
        load_weights_prepare(tile, kernel,
                             status->weights,
                             status->scale,
                             status->bias,
                             &conf_weights,
                             &conf_scale,
                             &conf_bias);
        dma_transfer_1d_async(conf_weights);
        dma_transfer_1d_async(conf_scale);
        dma_transfer_1d_async(conf_bias);

        status->weights.addr_ext += conf_weights.length_1d_copy;
        status->scale.addr_ext += conf_scale.length_1d_copy;
        status->bias.addr_ext += conf_bias.length_1d_copy;
    }
}

static int inc(int index, int end) {
    return index + 1 < end ? index + 1 : 0;
}

#define LOADER_ID (0)
#define EXECUTER_ID (1)
#define STORER_ID (2)
#define CORES (3)

#define BUFFER_SIZE (2)

struct layer_task_fork_args_t {
    uint32_t L2_input;
    uint32_t L2_weights;
    uint32_t L2_output;
    uint32_t L1_buffer;
    uint32_t padding;
    ne16_task_t *ne16_tasks;
    Layer *tiles;
    DmaTransferConf *store_conf;
    TaskMonitors *monitor;
};


static void layer_task_fork(void *void_args) {
    const int total_tiles = end_index.height * end_index.width * end_index.output_channel;

    struct layer_task_fork_args_t *args = (struct layer_task_fork_args_t *)void_args;
    ne16_task_t *ne16_tasks = args->ne16_tasks;
    Layer *tiles = args->tiles;
    DmaTransferConf *store_conf = args->store_conf;
    TaskMonitors *monitor = args->monitor;

    // Loader

    if (pi_core_id() == LOADER_ID) {
        Layer layer = {
            .addr = {
                .input = args->L2_input,
                .weights = args->L2_weights,
                .scale = args->L2_weights + ${l2_k_offset},
                .bias = args->L2_weights + ${l2_lambda_offset},
                .output = args->L2_output
            },
            .padding = {
                .top    = args->padding & NET_UTILS_PAD_TOP ? ${padding_top} : NE16_DONT_PAD,
                .right  = ${padding_right},
                .bottom = args->padding & NET_UTILS_PAD_BOTTOM ? ${padding_bottom} : NE16_DONT_PAD,
                .left   = ${padding_left}
            }
        };

        // Double buffer address init

        const unsigned int l1_buffer = args->L1_buffer;
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

        int i_buff = 0;

        for (int i_tile = 0; i_tile < total_tiles; i_tile++) {
            const Address local_addr = tile_status_get_addr(tile_status, buffer_addresses);
            Layer tile = tile_create(tile_status.index, end_index, body, border, layer, local_addr);

            monitor_produce_begin(monitor->input);
            tiles[i_buff] = tile;
            dma_mutex_lock();
            DmaTransfer transfer = dma_transfer_create();
            load_async(tiles[i_buff], &tile_status, body, layer, kernel);
            dma_mutex_unlock();
            % if stride == 1:
            execute_prepare(tile, &ne16_tasks[i_buff]);
            % elif stride == 2:
            execute_stride2x2_prepare(tile, kernel, &ne16_tasks[i_buff]);
            % endif
            dma_mutex_lock();
            dma_transfer_wait(transfer);
            dma_mutex_unlock();
            monitor_produce_end(monitor->input);

            monitor_produce_begin(monitor->store_conf);
            store_prepare(tiles[i_buff], body, layer, tile_status.index, &store_conf[i_buff]);
            monitor_produce_end(monitor->store_conf);

            i_buff = inc(i_buff, BUFFER_SIZE);
            tile_status = tile_status_get_next(tile_status, end_index, layer, 0, kernel);
        }
    }


    // Executer

    if (pi_core_id() == EXECUTER_ID) {
        int i_buff = 0;
        for (int i_tile = 0; i_tile < total_tiles; i_tile++) {
            monitor_consume_begin(monitor->input);
            monitor_produce_begin(monitor->output);

            % if stride == 1:
            execute_async(&ne16_tasks[i_buff]);
            % elif stride == 2:
            execute_stride2x2_blocking(&ne16_tasks[i_buff], tiles[i_buff], kernel);
            % endif

            monitor_produce_end(monitor->output);

            i_buff = inc(i_buff, BUFFER_SIZE);
        }
    }


    // Storer

    if (pi_core_id() == STORER_ID) {
        int i_buff = 0;
        for (int i_tile = 0; i_tile < total_tiles; i_tile++) {
            monitor_consume_begin(monitor->store_conf);
            monitor_consume_begin(monitor->output);

            execute_wait(&ne16_tasks[i_buff]);
            monitor_consume_end(monitor->input);

            dma_mutex_lock();
            DmaTransfer transfer = dma_transfer_create();
            dma_transfer_async(store_conf[i_buff]);
            dma_mutex_unlock();

            dma_mutex_lock();
            dma_transfer_wait(transfer);
            dma_mutex_unlock();

            monitor_consume_end(monitor->store_conf);
            monitor_consume_end(monitor->output);

            i_buff = inc(i_buff, BUFFER_SIZE);
        }
    }
}

void ${func_name}(void *args) {
    #ifdef DEBUG_GVSOC
    nnx_activate_gvsoc_logging(GVSOC_LOG_LEVEL_CONFIG, GVSOC_LOGGING_FORMAT_DECIMAL);
    #endif

    ne16_dev_t *ne16_dev = ne16_pulp_get_dev();
    hwpe_soft_clear(&ne16_dev->hwpe_dev);

    layer_args_t *layer_args = (layer_args_t *)args;

    // Init nnx tasks
    ne16_task_t ne16_tasks[BUFFER_SIZE];
    for (int i = 0; i < BUFFER_SIZE; i++) {
        ne16_task_init(&ne16_tasks[i]);
        ne16_task_set_op_to_conv(&ne16_tasks[i], ${fs1}, ${int(flag_DW)}, ${stride});
        ne16_task_set_bits(&ne16_tasks[i], ${x_data_size_byte}, ${y_data_size_byte}, ${W_data_size_byte});
        ne16_task_set_norm_quant(
            &ne16_tasks[i],
            (ne16_quant_t) {
                .shift_amount = ${out_shift},
                .function = ${"quantFunctionIdentity" if node.min < 0 else "quantFunctionRelu"},
                .flag_rounding = ne16TaskFlagFalse
            }, (ne16_norm_t) {
                .mode  = ${"normMode32Bit" if act_dim_bit == 32 else "normMode8Bit" if act_dim_bit == 8 else ""},
                .flag_bias  = ne16TaskFlagTrue,
                .flag_shift = ne16TaskFlagFalse
            });
        ne16_task_set_weight_offset(&ne16_tasks[i], weightOffsetModeLayerWise, ${weight_offset});
    }

    Layer tiles[BUFFER_SIZE];
    DmaTransferConf store_conf[BUFFER_SIZE];

    // Fork

    struct layer_task_fork_args_t layer_task_fork_args = {
        .L2_input = layer_args->L2_input,
        .L2_weights = layer_args->L2_weights,
        .L2_output = layer_args->L2_output,
        .L1_buffer = layer_args->L1_buffer,
        .padding = layer_args->padding,
        .ne16_tasks = ne16_tasks,
        .tiles = tiles,
        .store_conf = store_conf,
        .monitor = layer_args->monitor,
    };
    pi_cl_team_fork(CORES, layer_task_fork, (void *)&layer_task_fork_args);
    // Terminate

}
