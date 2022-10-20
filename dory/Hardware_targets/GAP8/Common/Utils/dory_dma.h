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

#include "mchan_test.h"

#if (GAP_SDK == 1)
#include "pulp.h"
#endif 

#pragma once
typedef enum
{
  TRANSFER_1D,
  TRANSFER_2D,
  TRANSFER_3D,
  TRANSFER_HWC_TO_CHW
} Transfer_Type;


typedef struct 
{
  unsigned int ext;
  unsigned int loc;
  unsigned int hwc_to_chw;
  unsigned short stride_2d;
  unsigned short number_of_2d_copies;
  unsigned short stride_1d;
  unsigned short number_of_1d_copies;
  unsigned short length_1d_copy;
  unsigned int dir;
  unsigned int dma_channel;
} DMA_copy;

void dory_dma_memcpy_async(DMA_copy DMA_copy_current);

void __attribute__ ((noinline)) dory_dma_barrier(DMA_copy DMA_copy_current);

uint32_t __attribute__ ((noinline)) dory_dma_allocate();

void __attribute__ ((noinline)) dory_dma_deallocate(uint32_t dma_channel);

void __attribute__ ((noinline)) dory_cores_barrier();

#endif
