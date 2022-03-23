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

#include "printf.h"
#include "stdint.h"

#pragma once

typedef struct 
{
  unsigned int L2_input;
  unsigned int L2_input_add;
  unsigned int L2_output;
  unsigned int L2_weights;
  unsigned int l2_zeros;
  unsigned int out_mult;
  unsigned int inmul1;
  unsigned int inmul2;
  unsigned int out_shift;
} layer;

typedef struct 
{
  float *   pInBuffer;
  uint16_t  dim_in_x;
  uint16_t  dim_in_y;
  uint16_t  ch_in;
  float *   pWeight;
  uint16_t  ch_out;
  uint16_t  dim_kernel_x;
  uint16_t  dim_kernel_y;
  uint16_t  padding_y_top;
  uint16_t  padding_y_bottom;
  uint16_t  padding_x_left;
  uint16_t  padding_x_right;
  uint16_t  stride_x;
  uint16_t  stride_y;
  int8_t *  bias;
  uint16_t  bias_shift;
  uint16_t  out_shift;
  uint16_t  out_mult;
  float *   pOutBuffer;
  uint16_t  dim_out_x;
  uint16_t  dim_out_y;
  float *   kappa;
  float *   lambda;
  uint8_t * pIm2ColBuffer;
  int       flag_relu;
  int       flag_batch_norm;
  int       flag_y_accumulate_start;
  int       flag_y_accumulate_end;
  unsigned int * memory_chan;
} kernel;

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
  unsigned short stride_L1_1d;
  unsigned short stride_L1_2d;
  unsigned int dir;
  unsigned int dma_channel;
} DMA_copy;


unsigned int dory_get_tile_1d(
  unsigned x,
  int tile_ii,
  int tile_size_i,
  int data_size
);
unsigned int dory_get_tile_2d(
  unsigned int x,
  int tile_ii,
  int tile_jj,
  int tile_size_i,
  int tile_size_j,
  int tile_stride_j,
  int data_size
);
unsigned int dory_get_tile_3d(
  unsigned int x,
  int tile_ii,
  int tile_jj,
  int tile_kk,
  int tile_size_i,
  int tile_size_j,
  int tile_size_k,
  int tile_stride_j,
  int tile_stride_k,
  int tile_overlap_i,
  int tile_overlap_j,
  int tile_overlap_k,
  int tile_offset_i,
  int tile_offset_j,
  int tile_offset_k,
  int data_size
);

void dory_dma_memcpy_async(DMA_copy DMA_copy_current);

void __attribute__ ((noinline)) dory_dma_barrier(DMA_copy DMA_copy_current);

uint32_t __attribute__ ((noinline)) dory_dma_allocate();

void __attribute__ ((noinline)) dory_dma_deallocate(uint32_t dma_channel);