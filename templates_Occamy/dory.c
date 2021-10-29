/*
 * dory.c
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

#include "dory.h"

Transfer_Type current_transfer;

/**
 *  @brief Gets a tile over a one-dimensional tiling grid.
 *
 *  Computes a pointer to the base of a particular tile in a one-dimensional
 *  tiling grid indexed by a (ii) index; in pseudo-Python
 *      ccn_get_tile_1d(x,ii) = x[ii*si:(ii+1)*si-1]
 *  where (si) os defining the pitch of the tiling grid in the (i) dimension.
 *
 *  @param x
 *      a pointer to the base of the 2d tiling grid.
 *  @param tile_ii
 *      the tiling index.
 *  @param tile_size_i
 *      the pitch of the tiling grid in the outer dimension, i.e. the distance
 *      between two "ticks" in the i dimension.
 *  @param data_size
 *      size of data in bytes
 */
unsigned int dory_get_tile_1d(
  unsigned x,
  int tile_ii,
  int tile_size_i,
  int data_size
) {
  unsigned int y = x + tile_ii*tile_size_i * data_size;
  return y;
}

/**
 *  @brief Gets a tile over a two-dimensional tiling grid.
 *
 *  Computes a pointer to the base of a particular tile in a two-dimensional
 *  tiling grid indexed by a (ii,jj) couple of indeces; in pseudo-Python
 *      ccn_get_tile_2d(x,ii,jj) = x[ii*si:(ii+1)*si-1,jj*sj:(jj+1)*sj-1]
 *  where (si,sj) is the couple defining the pitch of the tiling grid in the
 *  (i,j) dimensions.
 *
 *  @param *x
 *      a pointer to the base of the 2d tiling grid.
 *  @param tile_ii
 *      the tiling index in the outer dimension.
 *  @param tile_jj
 *      the tiling index in the inner dimension.
 *  @param tile_size_i
 *      the pitch of the tiling grid in the outer dimension, i.e. the distance
 *      between two "ticks" in the i dimension.
 *  @param tile_size_j
 *      the pitch of the tiling grid in the inner dimension, i.e. the distance
 *      between two "ticks" in the j dimension.
 *  @param tile_stride_j
 *      the total size of the tiling grid in the inner dimension, i.e. the 
 *      number of ticks in the j dimension.
 *  @param data_size
 *      size of data in bytes
 */
unsigned int  dory_get_tile_2d(
  unsigned int x,
  int tile_ii,
  int tile_jj,
  int tile_size_i,
  int tile_size_j,
  int tile_stride_j,
  int data_size
) {
  unsigned int y = x + tile_ii*tile_size_i * tile_stride_j * data_size
                     + tile_jj*tile_size_j * data_size;
  return y;
}

/**
 *  @brief Gets a tile over a three-dimensional tiling grid.
 *
 *  Computes a pointer to the base of a particular tile in a three-dimensional
 *  tiling grid indexed by a (ii,jj,kk) triple of indeces; in pseudo-Python
 *      ccn_get_tile_3d(x,ii,jj,kk) =
 *        x[ii*si:(ii+1)*si-1, jj*sj:(jj+1)*sj-1, kk*sk:(kk+1)*sk-1]
 *  where (si,sj,sk) is the triple defining the pitch of the tiling grid in the
 *  (i,j,k) dimensions.
 *
 *  @param *x
 *      a pointer to the base of the 2d tiling grid.
 *  @param tile_ii
 *      the tiling index in the outer dimension.
 *  @param tile_jj
 *      the tiling index in the middle dimension.
 *  @param tile_kk
 *      the tiling index in the inner dimension.
 *  @param tile_size_i
 *      the pitch of the tiling grid in the outer dimension, i.e. the distance
 *      between two "ticks" in the i dimension.
 *  @param tile_size_j
 *      the pitch of the tiling grid in the middle dimension, i.e. the distance
 *      between two "ticks" in the j dimension.
 *  @param tile_size_k
 *      the pitch of the tiling grid in the inner dimension, i.e. the distance
 *      between two "ticks" in the k dimension.
 *  @param tile_stride_j
 *      the total size of the tiling grid in the middle dimension, i.e. the 
 *      total number of ticks in the j dimension.
 *  @param tile_stride_k
 *      the total size of the tiling grid in the inner dimension, i.e. the 
 *      total number of ticks in the k dimension. 
 *  @param data_size
 *      size of data in bytes
 */
unsigned int  dory_get_tile_3d(
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
) {
  unsigned int y = x + (tile_ii*(tile_size_i - tile_overlap_i) - tile_offset_i) * tile_stride_j * tile_stride_k * data_size / 8
                     + (tile_jj*(tile_size_j - tile_overlap_j) - tile_offset_j) * tile_stride_k * data_size / 8
                     + (tile_kk*(tile_size_k - tile_overlap_k) - tile_offset_k) * data_size / 8;
  return y;
}


#define MIN(a,b) ((a)<(b)?(a):(b))

void __attribute__ ((noinline)) dory_dma_memcpy_async(DMA_copy DMA_copy_current) 
{

  if ((DMA_copy_current.number_of_1d_copies == 1) && (DMA_copy_current.number_of_2d_copies == 1))
    current_transfer = TRANSFER_1D;
  else if ((DMA_copy_current.number_of_1d_copies != 1) && (DMA_copy_current.number_of_2d_copies == 1))
    current_transfer = TRANSFER_2D;
  else
    current_transfer = TRANSFER_3D;
  if (DMA_copy_current.hwc_to_chw == 1)
    current_transfer = TRANSFER_HWC_TO_CHW;

  switch (current_transfer)
  {

    case TRANSFER_1D:
      printf("Transfer 1D to be implemented\n");
      break;
    case TRANSFER_2D:
      printf("Transfer 2D to be implemented\n");
      break;

    case TRANSFER_3D:
      printf("Transfer 3D to be implemented\n");
      break;

    case TRANSFER_HWC_TO_CHW:
      printf("Transfer 3D-HWC to be implemented\n");
      break;

  }
}

void __attribute__ ((noinline)) dory_dma_barrier(DMA_copy DMA_copy_current) 
{

  if ((DMA_copy_current.number_of_1d_copies == 1) && (DMA_copy_current.number_of_2d_copies == 1))
    current_transfer = TRANSFER_1D;
  else if ((DMA_copy_current.number_of_1d_copies != 1) && (DMA_copy_current.number_of_2d_copies == 1))
    current_transfer = TRANSFER_2D;
  else
    current_transfer = TRANSFER_3D;
  if (DMA_copy_current.hwc_to_chw == 1)
    current_transfer = TRANSFER_HWC_TO_CHW;

  switch (current_transfer)
  {
    case TRANSFER_1D:
      printf("Barrier 1D to be implemented\n");
      break;
    case TRANSFER_2D:
      printf("Barrier 2D to be implemented\n");
      break;
    case TRANSFER_3D:
      printf("Barrier 3D to be implemented\n");
      break;
    case TRANSFER_HWC_TO_CHW:
      printf("Barrier 3D-HWC to be implemented\n");
      break;
  }
}

uint32_t __attribute__ ((noinline)) dory_dma_allocate() 
{
  uint32_t dma_channel = 1;
  printf("DMA allocation to be implemented\n");
  return dma_channel;
}

void __attribute__ ((noinline)) dory_dma_deallocate(uint32_t dma_channel) 
{
  printf("DMA free to be implemented\n");
}