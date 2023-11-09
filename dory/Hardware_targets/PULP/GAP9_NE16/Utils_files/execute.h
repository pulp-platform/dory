#ifndef __EXECUTE_H__
#define __EXECUTE_H__

#include "dory_get_tile.h"
#include "ne16_hal.h"
#include "pulp_nnx.h"
#include "pulp_nnx_util.h"
#include "tile_status.h"
#include <stdint.h>

static inline void execute_prepare(Layer tile, nnx_task_t *const task) {
  nnx_task_set_dims(task, tile.input.width, tile.input.channel,
                    tile.input.width, tile.input.channel, tile.output.height,
                    tile.output.width, tile.output.channel, tile.output.width,
                    tile.output.channel, tile.padding.top, tile.padding.bottom,
                    tile.padding.right, tile.padding.left);
  nnx_task_set_ptrs(task, tile.addr.input, tile.input.width, tile.input.channel,
                    8 /*bits_in*/, tile.padding.top, tile.padding.left,
                    tile.addr.output, tile.addr.weights, tile.addr.scale,
                    0 /*shift_ptr*/, tile.addr.bias);
}

static inline void execute_async(nnx_task_t *task) {
  nnx_dispatch_check_blocking();
  nnx_dispatch_task(task);
}

static inline void execute_stride2x2_prepare(Layer tile, Kernel kernel,
                                             nnx_task_t *const task) {
  nnx_task_set_dims_stride2x2(
      task, tile.input.height, tile.input.width, tile.input.channel,
      tile.input.width, tile.input.channel, tile.output.height,
      tile.output.width, tile.output.channel, tile.output.width,
      tile.output.channel, kernel.shape.height, kernel.shape.width,
      tile.padding.top, tile.padding.bottom, tile.padding.right,
      tile.padding.left);
  nnx_task_set_ptrs(task, tile.addr.input, tile.input.width, tile.input.channel,
                    8 /*bits_in*/, tile.padding.top, tile.padding.left,
                    tile.addr.output, tile.addr.weights, tile.addr.scale,
                    0 /*shift_ptr*/, tile.addr.bias);
}

static inline void execute_stride2x2_blocking(nnx_task_t *task, Layer tile,
                                              Kernel kernel,
                                              uint32_t output_channel_stride) {
  nnx_dispatch_task_stride2x2(
      task, tile.input.width, tile.input.channel, tile.input.width,
      tile.input.channel, tile.output.height, tile.output.width,
      tile.output.channel, tile.output.width, output_channel_stride,
      kernel.shape.height, kernel.shape.width);
}

static inline void execute_wait(nnx_task_t *task) {
#if __PLATFORM__ == ARCHI_PLATFORM_GVSOC && defined GAP_SDK
  // Temporary hack because the gvsoc model of ne16 in gap_sdk
  // has a broken running_id.
  while(!ne16_empty())
    ;
#else
  while(!nnx_resolve_check(task))
    ;
#endif
}

#endif // __EXECUTE_H__
