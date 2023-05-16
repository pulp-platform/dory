#ifndef __DORY_DMA_H__
#define __DORY_DMA_H__

#include "pmsis.h"

#define DORY_DMA_DIR_LOC2EXT 0
#define DORY_DMA_DIR_EXT2LOC 1

#ifdef TARGET_CHIP_FAMILY_GAP9
#define MCHAN_VERSION (7)
#endif

typedef struct DmaTransferConf {
  uint32_t ext;
  uint32_t loc;
  int stride_2d;
  int number_of_2d_copies;
  int stride_1d;
  int number_of_1d_copies;
  int length_1d_copy;
  int hwc_to_chw;
  int dir; // 0 l1->l2, 1 l2->l1
} DmaTransferConf;

typedef struct DmaTransfer {
  int id;
} DmaTransfer;

void dma_transfer_1d_async(DmaTransferConf conf);
void dma_transfer_2d_async(DmaTransferConf conf);
void dma_transfer_3d_async(DmaTransferConf conf);
void dma_transfer_async(DmaTransferConf conf);

DmaTransfer dma_transfer_create();
void dma_transfer_free(DmaTransfer transfer);
void dma_transfer_wait(DmaTransfer transfer);

void dma_mutex_init();
void dma_mutex_lock();
void dma_mutex_unlock();

#endif  // __DORY_DMA_H__
