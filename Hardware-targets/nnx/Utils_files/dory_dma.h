#include "pmsis.h"

#ifdef GAP_SDK
#define ARCHI_MCHAN_DEMUX_ADDR (0x00201800)
#endif
#define MCHAN_BASE_ADDR (ARCHI_MCHAN_DEMUX_ADDR)  // CLUSTER_MCHAN_ADDR
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

#define DORY_DMA_DIR_LOC2EXT 0
#define DORY_DMA_DIR_EXT2LOC 1

typedef struct 
{
  void *ext;
  void *loc;
  unsigned short stride_2d;
  unsigned short number_of_2d_copies;
  unsigned short stride_1d;
  unsigned short number_of_1d_copies;
  unsigned short length_1d_copy;
  int dir; // 0 l1->l2, 1 l2->l1
  int tid;
} DMA_copy;

static void dory_dma_memcpy_1d_async(DMA_copy *copy) {
  copy->tid = mchan_transfer_get_id();
  mchan_transfer_t trans = {
    .cmd = copy->length_1d_copy | (copy->dir << MCHAN_CMD_SHIFT_DIRECTION) | MCHAN_FLAGS_1D,
    .size = copy->length_1d_copy,
    .ext = copy->ext,
    .loc = copy->loc,
  };
  mchan_transfer_push_1d(trans);
}

static void dory_dma_memcpy_2d_async(DMA_copy *copy) {
  const int size_2d = copy->number_of_1d_copies * copy->length_1d_copy;

  copy->tid = mchan_transfer_get_id();

  mchan_transfer_t trans = {
    .cmd = size_2d | copy->dir << MCHAN_CMD_SHIFT_DIRECTION | MCHAN_FLAGS_2D,
    .size = size_2d,
    .ext = copy->ext,
    .loc = copy->loc,
    .ext_size_1d = copy->length_1d_copy,
    .ext_stride_1d = copy->stride_1d
  };
  mchan_transfer_push_2d(trans);
}

static void dory_dma_memcpy_3d_async(DMA_copy *copy) {
  void *ext = copy->ext;
  void *loc = copy->loc;
  const int size_2d = copy->number_of_1d_copies * copy->length_1d_copy;

  copy->tid = mchan_transfer_get_id();

  for (int i = 0; i < copy->number_of_2d_copies; i++) {
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
}

static void dory_dma_memcpy_async(DMA_copy *copy) {
  if (copy->number_of_2d_copies == 1 && copy->number_of_1d_copies == 1) {
    dory_dma_memcpy_1d_async(copy);
  } else if (copy->number_of_2d_copies == 1) {
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
