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

void dory_dma_memcpy_hwc_to_chw(DMA_copy *copy);

void dory_dma_memcpy_1d_async(DMA_copy *copy);

void dory_dma_memcpy_2d_async(DMA_copy *copy);

void dory_dma_memcpy_3d_async(DMA_copy *copy);

void dory_dma_memcpy_async(DMA_copy *copy);

void dory_dma_free(DMA_copy *copy);

void dory_dma_barrier(DMA_copy *copy);

int dory_dma_allocate();
#endif
