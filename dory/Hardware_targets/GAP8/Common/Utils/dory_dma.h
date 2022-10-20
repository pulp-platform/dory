/*
 * dory.h
 * Alessio Burrello <alessio.burrello@unibo.it>
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

#ifndef _DORY_DMA_H
#define _DORY_DMA_H

#include "pmsis.h"

#ifdef GAP_SDK
#define ARCHI_MCHAN_DEMUX_ADDR (0x00201800)
#endif
#ifndef MCHAN_BASE_ADDR
#define MCHAN_BASE_ADDR (ARCHI_MCHAN_DEMUX_ADDR)  // CLUSTER_MCHAN_ADDR
#endif
#define MCHAN_EVENT
//#define MCHAN_POLLED
#ifdef MCHAN_EVENT
#ifdef PULP_SDK
#define CL_IRQ_DMA0 (8)
#endif
#define MCHAN_EVENT_BIT (CL_IRQ_DMA0)  // 8
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

typedef struct
{
  void *ext;
  void *loc;
  unsigned short hwc_to_chw;
  unsigned short stride_2d;
  unsigned short number_of_2d_copies;
  unsigned short stride_1d;
  unsigned short number_of_1d_copies;
  unsigned short length_1d_copy;
  int dir; // 0 l1->l2, 1 l2->l1
  int tid;
} DMA_copy;

static void dory_dma_memcpy_hwc_to_chw(DMA_copy *copy){
  copy->tid = mchan_transfer_get_id();
  int core_id = pi_core_id();
  int Log2Core = log2(NUM_CORES);
  int number_of_copies_per_core = (copy->length_1d_copy >> Log2Core) + ((copy->length_1d_copy & (NUM_CORES-1))!=0);
  int start_pixel, stop_pixel; // "pixel" is a misnomer; the CHANNELS are divided between the cores
  // this function assumes that a DW tile is always as wide as the complete feature map (this is enforced by DORY's tiler)
  start_pixel = MIN(number_of_copies_per_core * core_id, copy->length_1d_copy);
  stop_pixel = MIN(start_pixel + number_of_copies_per_core, copy->length_1d_copy);
  void * ext = copy->ext + start_pixel;
  void * loc = copy->loc + copy->number_of_1d_copies*copy->number_of_2d_copies*start_pixel;
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
    ext += 1; // next channel
    loc += copy->number_of_1d_copies * copy->number_of_2d_copies;
  }
  mchan_transfer_wait(copy->tid);
}

static void dory_dma_memcpy_1d_async(DMA_copy *copy) {
  if (pi_core_id() == 0) {
    copy->tid = mchan_transfer_get_id();
    mchan_transfer_t trans = {
      .cmd = copy->length_1d_copy * copy->number_of_1d_copies * copy->number_of_2d_copies | (copy->dir << MCHAN_CMD_SHIFT_DIRECTION) | MCHAN_FLAGS_1D,
      .size = copy->length_1d_copy * copy->number_of_1d_copies * copy->number_of_2d_copies,
      .ext = copy->ext,
      .loc = copy->loc
    };
    mchan_transfer_push_1d(trans);
    mchan_transfer_wait(copy->tid);
  }
}

static void dory_dma_memcpy_2d_async(DMA_copy *copy) {
  if (pi_core_id() == 0) {
    const int size_2d = copy->number_of_1d_copies * copy->length_1d_copy * copy->number_of_2d_copies;
    const int stride = (copy->number_of_2d_copies == 1) ? copy->stride_1d : copy->stride_2d;
    const int size_1d = (copy->number_of_2d_copies == 1) ? copy->length_1d_copy : copy->length_1d_copy * copy->number_of_1d_copies;
    copy->tid = mchan_transfer_get_id();

    mchan_transfer_t trans = {
      .cmd = size_2d | copy->dir << MCHAN_CMD_SHIFT_DIRECTION | MCHAN_FLAGS_2D,
      .size = size_2d,
      .ext = copy->ext,
      .loc = copy->loc,
      .ext_size_1d = size_1d,
      .ext_stride_1d = stride
    };
    mchan_transfer_push_2d(trans);
    mchan_transfer_wait(copy->tid);
  }
}

static void dory_dma_memcpy_3d_async(DMA_copy *copy) {
  int core_id = pi_core_id();
  int Log2Core = log2(NUM_CORES);
  int number_of_2d_copies_per_core = (copy->number_of_2d_copies >> Log2Core) + ((copy->number_of_2d_copies & (NUM_CORES-1))!=0);
  int start_pixel, stop_pixel;
  start_pixel = MIN(number_of_2d_copies_per_core * core_id, copy->number_of_2d_copies);
  stop_pixel = MIN(start_pixel + number_of_2d_copies_per_core, copy->number_of_2d_copies);
  void *ext = copy->ext + copy->stride_2d*start_pixel;
  void *loc = copy->loc + copy->length_1d_copy*copy->number_of_1d_copies*start_pixel;
  const int size_2d = copy->number_of_1d_copies * copy->length_1d_copy;

  copy->tid = mchan_transfer_get_id();

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

    loc += size_2d;
    ext += copy->stride_2d;
  }
  mchan_transfer_wait(copy->tid);
}

static void dory_dma_memcpy_async(DMA_copy *copy) {
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

static void dory_dma_free(DMA_copy *copy) {
  mchan_transfer_free(copy->tid);
}

static void dory_dma_barrier(DMA_copy *copy) {
  mchan_transfer_wait(copy->tid);
}


#endif
