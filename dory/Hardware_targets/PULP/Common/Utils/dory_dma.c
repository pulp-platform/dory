#include "dory_dma.h"

#include "pmsis.h"

#ifndef MCHAN_BASE_ADDR
// FIXME: For GAP9, this must point to ARCHI_MCHAN_EXT_ADDR!!!
// In PULP-SDK for Kraken, this is fixed.
// GAP8 hardware to be tested...
#define MCHAN_BASE_ADDR (ARCHI_MCHAN_DEMUX_ADDR)  // CLUSTER_MCHAN_ADDR
#endif
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

void dory_dma_memcpy_hwc_to_chw(DMA_copy *copy){
#ifdef SINGLE_CORE_DMA
  if (pi_core_id() == 0) {
#endif
  int start_pixel, stop_pixel; // "pixel" is a misnomer; the CHANNELS are divided between the cores
  // this function assumes that a DW tile is always as wide as the complete feature map (this is enforced by DORY's tiler)
  // if there is only 1 DMA control unit for the cluster (e.g., Kraken), we can't execute DMA calls on multiple clusters.
#ifndef SINGLE_CORE_DMA
  int core_id = pi_core_id();
  int Log2Core = log2(NUM_CORES);
  int number_of_copies_per_core = (copy->length_1d_copy >> Log2Core) + ((copy->length_1d_copy & (NUM_CORES-1))!=0);
  start_pixel = MIN(number_of_copies_per_core * core_id, copy->length_1d_copy);
  stop_pixel = MIN(start_pixel + number_of_copies_per_core, copy->length_1d_copy);
#else
  start_pixel = 0;
  stop_pixel = copy->length_1d_copy;
#endif
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
#ifdef SINGLE_CORE_DMA
  }
#endif
}

void dory_dma_memcpy_1d_async(DMA_copy *copy) {
  if (pi_core_id() == 0) {
    mchan_transfer_t trans = {
      .cmd = copy->length_1d_copy * copy->number_of_1d_copies * copy->number_of_2d_copies | (copy->dir << MCHAN_CMD_SHIFT_DIRECTION) | MCHAN_FLAGS_1D,
      .size = copy->length_1d_copy * copy->number_of_1d_copies * copy->number_of_2d_copies,
      .ext = copy->ext,
      .loc = copy->loc
    };
    mchan_transfer_push_1d(trans);
  }
}

void dory_dma_memcpy_2d_async(DMA_copy *copy) {
  if (pi_core_id() == 0) {
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
}

void dory_dma_memcpy_3d_async(DMA_copy *copy) {
#ifdef SINGLE_CORE_DMA
  if (pi_core_id() == 0) {
#endif
  int start_pixel, stop_pixel;
#ifndef SINGLE_CORE_DMA
  int core_id = pi_core_id();
  int Log2Core = log2(NUM_CORES);
  int number_of_2d_copies_per_core = (copy->number_of_2d_copies >> Log2Core) + ((copy->number_of_2d_copies & (NUM_CORES-1))!=0);
  start_pixel = MIN(number_of_2d_copies_per_core * core_id, copy->number_of_2d_copies);
  stop_pixel = MIN(start_pixel + number_of_2d_copies_per_core, copy->number_of_2d_copies);
#else
  start_pixel = 0;
  stop_pixel = copy->number_of_2d_copies;
#endif
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
#ifdef SINGLE_CORE_DMA
  }
#endif
}

void dory_dma_memcpy_async(DMA_copy *copy) {
  if (copy->hwc_to_chw == 1) {
    dory_dma_memcpy_hwc_to_chw(copy);
  }
  else if ((copy->number_of_2d_copies == 1 && copy->number_of_1d_copies == 1) || (copy->stride_1d == copy->length_1d_copy &&  copy->number_of_1d_copies * copy->length_1d_copy == copy->stride_2d) || (copy->number_of_2d_copies == 1 && copy->length_1d_copy == copy->stride_1d)) {
    dory_dma_memcpy_1d_async(copy);
  } else if ((copy->number_of_2d_copies == 1) || (copy->length_1d_copy == copy->stride_1d)) {// wrong!
    dory_dma_memcpy_2d_async(copy);
  } else {
    dory_dma_memcpy_3d_async(copy);
  }
}

void dory_dma_free(DMA_copy *copy) {
  mchan_transfer_free(copy->tid);
}

void dory_dma_barrier(DMA_copy *copy) {
#ifdef SINGLE_CORE_DMA
  // if DMA is only used by a single core (only 1 ctrl interface), other cores must not access its register file. Instead, they should all wait for core 0 to confirm the transfer is over.
  if (pi_core_id() == 0)
    mchan_transfer_wait(copy->tid);
  pi_cl_team_barrier(0);
#else
  mchan_transfer_wait(copy->tid);
#endif
}

int dory_dma_allocate() {
  return mchan_transfer_get_id();
}
