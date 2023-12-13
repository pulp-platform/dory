#include "dory_dma.h"

#include "pmsis.h"

#ifndef MCHAN_BASE_ADDR
// FIXME: For GAP9, this must point to ARCHI_MCHAN_EXT_ADDR!!!
// In PULP-SDK for Kraken, this is fixed.
// GAP8 hardware to be tested...
#define MCHAN_BASE_ADDR (ARCHI_MCHAN_DEMUX_ADDR)  // CLUSTER_MCHAN_ADDR
#endif
#define MCHAN_EVENT
//#define MCHAN_POLLED
#ifdef MCHAN_EVENT
#define MCHAN_EVENT_BIT (ARCHI_CL_EVT_DMA0)  // 8
#endif
#include "mchan.h"


#if   defined(MCHAN_POLLED)
#define MCHAN_FLAGS    (MCHAN_CMD_FLAG_INCREMENTAL)
#elif defined(MCHAN_EVENT)
#define MCHAN_FLAGS    (MCHAN_CMD_FLAG_EVENT_ENABLE | MCHAN_CMD_FLAG_INCREMENTAL)
#elif defined(MCHAN_INTERRUPT)
#define MCHAN_FLAGS    (MCHAN_CMD_FLAG_INTERRUPT_ENABLE | MCHAN_CMD_FLAG_INCREMENTAL)
#endif

#define MCHAN_FLAGS_1D (MCHAN_FLAGS)
#define MCHAN_FLAGS_2D (MCHAN_FLAGS | MCHAN_CMD_FLAG_2D_TRANSFER_EXTERNAL)

#define MIN(a,b) ((a)<(b)?(a):(b))

void dory_dma_memcpy_hwc_to_chw_single_core(DMA_copy *copy){
  int start_pixel, stop_pixel; // "pixel" is a misnomer; the CHANNELS are divided between the cores
  // this function assumes that a DW tile is always as wide as the complete feature map (this is enforced by DORY's tiler)
  // if there is only 1 DMA control unit for the cluster (e.g., Kraken), we can't execute DMA calls on multiple clusters.
  start_pixel = 0;
  stop_pixel = copy->length_1d_copy;
  void * loc = copy->loc + copy->number_of_1d_copies*copy->number_of_2d_copies*start_pixel;
  void * ext = copy->ext + start_pixel;
  const int size_2d = copy->number_of_1d_copies * copy->number_of_2d_copies;

  for (int i=start_pixel; i<stop_pixel; i++) {
    mchan_transfer_t trans = {
      .cmd = size_2d | copy->dir << MCHAN_CMD_SHIFT_DIRECTION | MCHAN_FLAGS_2D,
      .size = size_2d,
      .ext = ext,
      .loc = loc,
      .ext_size_1d = 1, // one byte at a time...
      .ext_stride_1d = copy->stride_1d
    };
    mchan_transfer_push_2d(trans);
#ifdef ALWAYS_BLOCK_DMA_TRANSFERS // needed on GAP8 board
    dory_dma_barrier(copy);
#endif
    ext += 1; // next channel
    loc += copy->number_of_1d_copies * copy->number_of_2d_copies;
  }
}

void dory_dma_memcpy_1d_single_core_async(DMA_copy *copy) {
  mchan_transfer_t trans = {
    .cmd = copy->length_1d_copy * copy->number_of_1d_copies * copy->number_of_2d_copies | (copy->dir << MCHAN_CMD_SHIFT_DIRECTION) | MCHAN_FLAGS_1D,
    .size = copy->length_1d_copy * copy->number_of_1d_copies * copy->number_of_2d_copies,
    .ext = copy->ext,
    .loc = copy->loc
  };
  mchan_transfer_push_1d(trans);
}

void dory_dma_memcpy_2d_single_core_async(DMA_copy *copy) {
  const int size_2d = copy->number_of_1d_copies * copy->length_1d_copy * copy->number_of_2d_copies;
  const int stride = (copy->number_of_2d_copies == 1) ? copy->stride_1d : copy->stride_2d;
  const int size_1d = (copy->number_of_2d_copies == 1) ? copy->length_1d_copy : copy->length_1d_copy * copy->number_of_1d_copies;

  mchan_transfer_t trans = {
    .cmd = size_2d | copy->dir << MCHAN_CMD_SHIFT_DIRECTION | MCHAN_FLAGS_2D,
    .size = size_2d,
    .ext = copy->ext,
    .loc = copy->loc,
    .ext_size_1d = size_1d,
    .ext_stride_1d = stride
  };
  mchan_transfer_push_2d(trans);
}

void dory_dma_memcpy_3d_single_core_async(DMA_copy *copy) {
  int start_pixel, stop_pixel;
  start_pixel = 0;
  stop_pixel = copy->number_of_2d_copies;
  void *ext = copy->ext + copy->stride_2d*start_pixel;
  void *loc = copy->loc + copy->length_1d_copy*copy->number_of_1d_copies*start_pixel;
  const int size_2d = copy->number_of_1d_copies * copy->length_1d_copy;
  for (int i = start_pixel; i < stop_pixel; i++) {
    mchan_transfer_t trans = {
      .cmd = size_2d | copy->dir << MCHAN_CMD_SHIFT_DIRECTION | MCHAN_FLAGS_2D,
      .size = size_2d,
      .ext = ext,
      .loc = loc,
      .ext_size_1d = copy->length_1d_copy,
      .ext_stride_1d = copy->stride_1d
    };
    mchan_transfer_push_2d(trans);
#ifdef ALWAYS_BLOCK_DMA_TRANSFERS // needed on GAP8 board
    dory_dma_barrier(copy);
#endif
    loc += size_2d;
    ext += copy->stride_2d;
  }
}

void dory_dma_memcpy_single_core_async(DMA_copy *copy) {
  if (copy->hwc_to_chw == 1) {
    dory_dma_memcpy_hwc_to_chw(copy);
  }
  else if ((copy->number_of_2d_copies == 1 && copy->number_of_1d_copies == 1) || (copy->stride_1d == copy->length_1d_copy &&  copy->number_of_1d_copies * copy->length_1d_copy == copy->stride_2d) || (copy->number_of_2d_copies == 1 && copy->length_1d_copy == copy->stride_1d)) {
    dory_dma_memcpy_1d_single_core_async(copy);
  } else if ((copy->number_of_2d_copies == 1) || (copy->length_1d_copy == copy->stride_1d)) {// wrong!
    dory_dma_memcpy_2d_single_core_async(copy);
  } else {
    dory_dma_memcpy_3d_single_core_async(copy);
  }
}

void dory_dma_free_single_core(DMA_copy *copy) {
  mchan_transfer_free(copy->tid);
}

void dory_dma_barrier_single_core(DMA_copy *copy) {
  mchan_transfer_wait(copy->tid);
}

int dory_dma_allocate_single_core() {
  return mchan_transfer_get_id();
}
