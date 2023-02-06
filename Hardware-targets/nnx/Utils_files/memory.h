#ifndef __MEMORY_H__
#define __MEMORY_H__

#include <stddef.h>
#include "dory_dma_v2.h"

typedef struct MemoryStatus {
    uint32_t addr_ext;
    int is_wait, is_transfer;
    int buffer_index;
    DmaTransfer transfer;
} MemoryStatus;

static void memory_transfer_async(DmaTransferConf conf, MemoryStatus * const status) {
    if (status->is_transfer) {
        status->transfer = dma_transfer_async(conf);
        status->is_wait = 1;
        status->is_transfer = 0;
    }
}

static void memory_transfer_1d_async(DmaTransferConf dma_conf, MemoryStatus * const status) {
    if (status->is_transfer) {
        status->transfer = dma_transfer_1d_async(dma_conf);
        status->addr_ext += dma_conf.length_1d_copy;
        status->is_wait = 1;
        status->is_transfer = 0;
    }
}

static void memory_wait(MemoryStatus * const status) {
    if (status->is_wait) {
        dma_transfer_wait(status->transfer);
        status->is_wait = 0;
    }
}

#endif
