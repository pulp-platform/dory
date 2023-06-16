#include "dory_dma.h"
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

void dma_transfer_hwc_to_chw(DmaTransferConf conf){
  int start_pixel=0, stop_pixel=conf.length_1d_copy; // "pixel" is a misnomer; the CHANNELS are divided between the cores
  void * loc = conf.loc + conf.number_of_1d_copies*conf.number_of_2d_copies*start_pixel;
  void * ext = conf.ext + start_pixel;
  const int size_2d = conf.number_of_1d_copies * conf.number_of_2d_copies;

  for (int i=start_pixel; i<stop_pixel; i++) {
    mchan_transfer_t trans = {
      .cmd = size_2d | conf.dir << MCHAN_CMD_SHIFT_DIRECTION | MCHAN_FLAGS_2D,
      .size = size_2d,
      .ext = ext,
      .loc = loc,
      .ext_size_1d = 1, // one byte at a time...
      .ext_stride_1d = conf.stride_1d
    };
    mchan_transfer_push_2d(trans);

    ext += 1; // next channel
    loc += conf.number_of_1d_copies * conf.number_of_2d_copies;
  }
}

void dma_transfer_1d_async(DmaTransferConf conf) {
  mchan_transfer_push_1d((mchan_transfer_t) {
    .cmd = conf.length_1d_copy | (conf.dir << MCHAN_CMD_SHIFT_DIRECTION) | MCHAN_FLAGS_1D,
    .size = conf.length_1d_copy,
    .loc = conf.loc,
    .ext = conf.ext,
  });
}

void dma_transfer_2d_async(DmaTransferConf conf) {
  const int size_2d = conf.number_of_1d_copies * conf.length_1d_copy;

  mchan_transfer_push_2d((mchan_transfer_t) {
    .cmd = size_2d | conf.dir << MCHAN_CMD_SHIFT_DIRECTION | MCHAN_FLAGS_2D,
    .size = size_2d,
    .loc = conf.loc,
    .ext = conf.ext,
    .ext_size_1d = conf.length_1d_copy,
    .ext_stride_1d = conf.stride_1d
  });
}

void dma_transfer_3d_async(DmaTransferConf conf) {
  const int size_2d = conf.number_of_1d_copies * conf.length_1d_copy;

  for (int i = 0; i < conf.number_of_2d_copies; i++) {
    dma_transfer_2d_async(conf);
    conf.loc += size_2d;
    conf.ext += conf.stride_2d;
  }
}

void dma_transfer_async(DmaTransferConf conf) {
  if (conf.hwc_to_chw == 1) {
    dma_transfer_hwc_to_chw(conf);
  } else if (conf.number_of_2d_copies == 1 && conf.number_of_1d_copies == 1) {
    dma_transfer_1d_async(conf);
  } else if (conf.number_of_2d_copies == 1) {
    dma_transfer_2d_async(conf);
  } else {
    dma_transfer_3d_async(conf);
  }
}

DmaTransfer dma_transfer_create() {
  return (DmaTransfer) { .id = mchan_transfer_get_id() };
}

void dma_transfer_free(DmaTransfer transfer) {
  mchan_transfer_free(transfer.id);
}

void dma_transfer_wait(DmaTransfer transfer) {
  mchan_transfer_wait(transfer.id);
  mchan_transfer_free(transfer.id);
}

static uint32_t dma_mutex;

void dma_mutex_init() {
  dma_mutex = eu_mutex_addr(0);
}

void dma_mutex_lock() {
  eu_mutex_lock(dma_mutex);
}

void dma_mutex_unlock() {
  eu_mutex_unlock(dma_mutex);
}
