#ifndef __EXECUTE_H__
#define __EXECUTE_H__

#include "dory_get_tile.h"
#include "hwpe.h"
#include "ne16.h"
#include "ne16_pulp_bsp.h"
#include "ne16_task.h"
#include "pulp_nnx_ne16.h"
#include "pulp_nnx_util.h"
#include "tile_status.h"
#include <stdint.h>

static inline void execute_prepare(Layer tile, ne16_task_t *const task) {
  ne16_task_set_dims(task, tile.input.width, tile.input.channel,
                     tile.input.width * tile.input.channel, tile.input.channel,
                     tile.output.height, tile.output.width, tile.output.channel,
                     tile.output.width * tile.output.channel,
                     tile.output.channel, tile.padding.top, tile.padding.bottom,
                     tile.padding.left, tile.padding.right);
  ne16_task_set_addr_conv(task, tile.addr.input, tile.input.width,
                          tile.input.channel, tile.padding.top, tile.padding.left,
                          tile.addr.output, tile.addr.weights);
  ne16_task_set_addr_norm_quant(task, tile.addr.scale, 0 /*shift_addr*/, tile.addr.bias);
}

static inline void execute_async(ne16_task_t *task) {
  ne16_nnx_dispatch_wait(ne16_pulp_get_dev());
  ne16_nnx_dispatch(ne16_pulp_get_dev(), task);
}

static inline void execute_stride2x2_prepare(Layer tile, Kernel kernel,
                                             ne16_task_t *const task) {
  ne16_task_set_dims_stride2x2(
      task, tile.input.height, tile.input.width, tile.input.channel,
      tile.input.width * tile.input.channel, tile.input.channel,
      tile.output.height, tile.output.width, tile.output.channel,
      tile.output.width * tile.output.channel, tile.output.channel,
      kernel.shape.height, kernel.shape.width, tile.padding.top,
      tile.padding.bottom, tile.padding.left, tile.padding.right);
  ne16_task_set_addr_conv(task, tile.addr.input, tile.input.width,
                          tile.input.channel, tile.padding.top, tile.padding.left,
                          tile.addr.output, tile.addr.weights);
  ne16_task_set_addr_norm_quant(task, tile.addr.scale, 0 /*shift_addr*/, tile.addr.bias);
}

static inline void execute_stride2x2_blocking(ne16_task_t *task, Layer tile,
                                              Kernel kernel) {
  ne16_nnx_dispatch_stride2x2(ne16_pulp_get_dev(), task, tile.input.width, tile.input.channel,
                              tile.output.height, tile.output.width,
                              tile.output.channel, kernel.shape.height,
                              kernel.shape.width);
}

static inline void execute_wait(ne16_task_t *task) {
  while (!ne16_nnx_resolve_check(ne16_pulp_get_dev(), task))
    ;
}

#endif // __EXECUTE_H__
