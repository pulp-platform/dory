#ifndef __EXECUTE_STRIDE2x2_H__
#define __EXECUTE_STRIDE2x2_H__

#include <stddef.h>
#include "pulp_nnx.h"
#include "pulp_nnx_defs.h"
#include "pulp_nnx_hal.h"
#include "tile_status.h"
#include "dory_get_tile.h"


static uint32_t padded_input_ptr(uint32_t input,
                                 const int padding_top, const int padding_left,
                                 const int width, const int channel, const int bits) {
    return input - (padding_top * width + padding_left) * channel * bits / 8;
}

static uint32_t get_padding(int i, int j, int n_h, int n_w, uint32_t tile_padding) {
    uint32_t padding = tile_padding;
    if (i > 0)
        padding &= ~(0xf << 28);
    if (j < n_w - 1)
        padding &= ~(0xf << 24);
    if (i < n_h - 1)
        padding &= ~(0xf << 20);
    if (j > 0)
        padding &= ~(0xf << 16);
    return padding;
}

static void execute_stride2x2_prepare(Layer tile, Kernel kernel, nnx_task_t * const task) {
    const int stride = 2;

    nnx_conv_set_strides(task, tile.input.channel, tile.input.width, tile.input.channel,
                         tile.output.width, tile.output.channel);
    nnx_conv_set_counters(task, tile.input.channel, 3, 3, tile.output.channel);

    tile.padding.bottom = (tile.input.height + tile.padding.top - kernel.shape.height) % stride == 0 ? 0 : tile.padding.bottom;
    tile.padding.right = (tile.input.width + tile.padding.left - kernel.shape.width) % stride == 0 ? 0 : tile.padding.right;

    task->cfg.padding = (tile.padding.top << 28)
        | (tile.padding.right << 24)
        | (tile.padding.bottom << 20)
        | (tile.padding.left << 16)
        | 0;

    task->infeat_ptr = padded_input_ptr(tile.addr.input, tile.padding.top,
                                        tile.padding.left, tile.input.width,
                                        tile.input.channel, 8);
    task->weights_ptr = tile.addr.weights;
    task->scale_ptr = tile.addr.scale;
    task->scale_bias_ptr = tile.addr.bias;
    task->outfeat_ptr = tile.addr.output;
}

static int execute_stride2x2_blocking(nnx_task_t task, Layer tile, Kernel kernel) {
    const int stride = 2;
    int last_job_id = -1;

    const int n_h = DIVNCEIL(tile.output.height, stride);
    const int n_w = DIVNCEIL(tile.output.width, stride);
    const int input_height_offset = tile.output.height % stride == 1 ? stride : 0;
    const int input_width_offset  = tile.output.width  % stride == 1 ? stride : 0;
    const int output_height_offset = tile.output.height % stride == 1 ? 1 : 0;
    const int output_width_offset  = tile.output.width  % stride == 1 ? 1 : 0;

    const uint32_t input_base = task.infeat_ptr;
    const uint32_t output_base = task.outfeat_ptr;
    const uint32_t tile_padding = task.cfg.padding;

    int i_job = 0;
    for (int i = 0; i < n_h; i++) {
        for (int j = 0; j < n_w; j++) {
            task.infeat_ptr = dory_get_tile_3d(input_base,
                                               i, j, 0,                    /* index */
                                               3 + kernel.shape.height - 1, 3 + kernel.shape.width - 1, tile.input.channel,        /* size */
                                               tile.input.width, tile.input.channel, /* stride */
                                               kernel.shape.height - stride, kernel.shape.width - stride, 0, /* overlap */
                                               i == 0 ? 0 : input_height_offset, j == 0 ? 0 : input_width_offset, 0, /* offset */
                                               8 /* data size */
                                               );
            task.outfeat_ptr = dory_get_tile_3d(output_base,
                                                i, j, 0,                      /* index */
                                                2, 2, tile.output.channel,         /* size */
                                                tile.output.width, tile.output.channel, /* stride */
                                                0, 0, 0,                      /* overlap */
                                                i == 0 ? 0 : output_height_offset, j == 0 ? 0 : output_width_offset, 0, /* offset */
                                                8 /* data size */
                                                );

            task.cfg.padding = get_padding(i, j, n_h, n_w, tile_padding);

            last_job_id = nnx_acquire_blocking();
            if (i_job < 2) {
                nnx_offload(&task);
            } else {
                NE16_WRITE_IO_REG(NE16_REG_INFEAT_PTR, task.infeat_ptr);
                NE16_WRITE_IO_REG(NE16_REG_OUTFEAT_PTR, task.outfeat_ptr);
                NE16_WRITE_IO_REG(NE16_REG_PADDING, task.cfg.padding);
            }
            nnx_run_async();

            i_job++;
        }
    }

    return last_job_id;
}

static void execute_stride2x2_wait() {
    nnx_wait_empty();
}


#endif  // __EXECUTE_STRIDE2x2_H__
