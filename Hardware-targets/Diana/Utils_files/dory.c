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

void __attribute__ ((noinline)) dory_dma_memcpy_async_digital(DMA_copy DMA_copy_current)
{

  if      ( ( DMA_copy_current.number_of_1d_copies * DMA_copy_current.length_1d_copy ) == DMA_copy_current.stride_2d)
    current_transfer = TRANSFER_1D;
  else if ( DMA_copy_current.length_1d_copy == DMA_copy_current.stride_1d )
    current_transfer = TRANSFER_2D;
  else
    current_transfer = TRANSFER_3D;
  if (DMA_copy_current.hwc_to_chw == 1)
    current_transfer = TRANSFER_HWC_TO_CHW;

  switch (current_transfer)
  {

    case TRANSFER_1D:
      memcpy_dig((unsigned int *)(DMA_copy_current.ext), (unsigned int)(DMA_copy_current.loc), DMA_copy_current.length_1d_copy * DMA_copy_current.number_of_1d_copies * DMA_copy_current.number_of_2d_copies, DMA_copy_current.dir, 1);
      global_sync_digital();
      break;

    case TRANSFER_2D:
      for (int i = 0; i < DMA_copy_current.number_of_2d_copies; i++)
      {
        memcpy_dig((unsigned int *)(DMA_copy_current.ext), (unsigned int)(DMA_copy_current.loc), DMA_copy_current.length_1d_copy * DMA_copy_current.number_of_1d_copies, DMA_copy_current.dir, 1);
        global_sync_digital();
        DMA_copy_current.loc += DMA_copy_current.length_1d_copy * DMA_copy_current.number_of_1d_copies;
        DMA_copy_current.ext += DMA_copy_current.stride_2d;
      }
      break;

    case TRANSFER_3D:
      for ( int j = 0; j < DMA_copy_current.number_of_2d_copies; j++)
      {
        for (int i = 0; i < DMA_copy_current.number_of_1d_copies; i++)
        {
          memcpy_dig((unsigned int *)(DMA_copy_current.ext), (unsigned int)(DMA_copy_current.loc), DMA_copy_current.length_1d_copy, DMA_copy_current.dir, 1);
          global_sync_digital();
          DMA_copy_current.loc += DMA_copy_current.length_1d_copy;
          DMA_copy_current.ext += DMA_copy_current.stride_1d;
        }
        DMA_copy_current.ext = DMA_copy_current.ext - DMA_copy_current.stride_1d * DMA_copy_current.number_of_1d_copies + DMA_copy_current.stride_2d;
        
      }
      break;
  }
}


void __attribute__ ((noinline)) dory_dma_memcpy_async_analog(DMA_copy DMA_copy_current)
{

  if      ( ( DMA_copy_current.number_of_1d_copies * DMA_copy_current.length_1d_copy ) == DMA_copy_current.stride_2d)
    current_transfer = TRANSFER_1D;
  else if ( DMA_copy_current.length_1d_copy == DMA_copy_current.stride_1d )
    current_transfer = TRANSFER_2D;
  else
    current_transfer = TRANSFER_3D;
  if (DMA_copy_current.hwc_to_chw == 1)
    current_transfer = TRANSFER_HWC_TO_CHW;

  switch (current_transfer)
  {

    case TRANSFER_1D:
      memcpy_analog((unsigned int *)(DMA_copy_current.ext), (unsigned int)(DMA_copy_current.loc), DMA_copy_current.length_1d_copy * DMA_copy_current.number_of_1d_copies * DMA_copy_current.number_of_2d_copies, DMA_copy_current.dir, 1);
      global_sync_analog();
      break;

    case TRANSFER_2D:
      for (int i = 0; i < DMA_copy_current.number_of_2d_copies; i++)
      {
        memcpy_analog((unsigned int *)(DMA_copy_current.ext), (unsigned int)(DMA_copy_current.loc), DMA_copy_current.length_1d_copy * DMA_copy_current.number_of_1d_copies, DMA_copy_current.dir, 1);
        global_sync_analog();
        DMA_copy_current.loc += DMA_copy_current.length_1d_copy * DMA_copy_current.number_of_1d_copies;
        DMA_copy_current.ext += DMA_copy_current.stride_2d;
      }
      break;

    case TRANSFER_3D:
      for ( int j = 0; j < DMA_copy_current.number_of_2d_copies; j++)
      {
        for (int i = 0; i < DMA_copy_current.number_of_1d_copies; i++)
        {
          memcpy_analog((unsigned int *)(DMA_copy_current.ext), (unsigned int)(DMA_copy_current.loc), DMA_copy_current.length_1d_copy, DMA_copy_current.dir, 1);
          global_sync_analog();
          DMA_copy_current.loc += DMA_copy_current.length_1d_copy;
          DMA_copy_current.ext += DMA_copy_current.stride_1d;
        }
        DMA_copy_current.ext = DMA_copy_current.ext - DMA_copy_current.stride_1d * DMA_copy_current.number_of_1d_copies + DMA_copy_current.stride_2d;
        
      }
      break;
  }
}

void __attribute__ ((noinline)) dory_dma_barrier_digital(DMA_copy DMA_copy_current)
{
  global_sync_digital();
}

void __attribute__ ((noinline)) dory_dma_barrier_analog(DMA_copy DMA_copy_current)
{
  global_sync_analog();
}

uint32_t __attribute__ ((noinline)) dory_dma_allocate()
{
  plp_hwme_enable();
  return 1;
}

void __attribute__ ((noinline)) dory_dma_deallocate(uint32_t dma_channel)
{
  plp_hwme_disable();
}

void __attribute__ ((noinline)) dory_cores_barrier_digital()
{
  global_sync_digital();
}

void __attribute__ ((noinline)) dory_cores_barrier_analog()
{
  global_sync_analog();
}

void memcpy_dig(unsigned int* L2_Addr_Byte,
                unsigned int L1_Addr,
                unsigned int Length,
                unsigned int direction_L1_L2, // 0 to L1 and 1 to L2
                unsigned int BankNum
                ){
    unsigned int L1_Addr_16Byte = (int) (L1_Addr / 16);
    unsigned int Lenth_4Byte = (int) (padding(Length, DMA_PARALLELISM) / 4);
    unsigned int row_size = 4*Lenth_4Byte/(BankNum*16);
    /* memcpy  (DIGITAL)*/
    switch(direction_L1_L2)
    {
        case 0: //L2_to_L1
            if (BankNum==1)
                hwme_memcpy_op((unsigned int) 2);
            else 
                hwme_memcpy_op((unsigned int) 4);
            break;
        case 1: //L1_to_L2
            if (BankNum == 1)
                hwme_memcpy_op((unsigned int) 1);
            else
                hwme_memcpy_op((unsigned int) 3);
            break;
    }
    if (BankNum==1)
    {
        hwme_memcpy_addr_set(L2_Addr_Byte);
        hwme_l1addr_set(L1_Addr_16Byte); // absolute address of L1 (128b / address)
        hwme_memcpy_n_set(Lenth_4Byte);
    }
    else 
    {
        hwme_memcpy_addr_set(L2_Addr_Byte);
        hwme_l1addr_set(L1_Addr_16Byte); // absolute address of L1 (128b / address)
        hwme_memcpy_bank_length_set(BankNum); // number of bank
        hwme_memcpy_row_length_set(row_size);
    }
    // start HWME operation
    hwme_trigger_job();
}


void memcpy_analog(unsigned int* L2_Addr_Byte,
                  unsigned int L1_Addr,
                  unsigned int Length,
                  unsigned int direction_L1_L2, // 0 to L1 and 1 to L2
                  unsigned int BankNum
                  ){
    unsigned int L1_Addr_16Byte = (int) (L1_Addr / 16);
    unsigned int Lenth_4Byte = (int) (padding(Length, DMA_PARALLELISM) / 4);
    unsigned int row_size = 4*Lenth_4Byte/(BankNum*16);
    /* memcpy  (DIGITAL)*/
    switch(direction_L1_L2)
    {
        case 0: //L2_to_L1
            if (BankNum==1)
                hwme_ana_memcpy_op((unsigned int) 2);
            else 
                hwme_ana_memcpy_op((unsigned int) 4);
            break;
        case 1: //L1_to_L2
            if (BankNum == 1)
                hwme_ana_memcpy_op((unsigned int) 1);
            else
                hwme_ana_memcpy_op((unsigned int) 3);
            break;
    }
    if (BankNum==1)
    {
        hwme_ana_memcpy_addr_set(L2_Addr_Byte);
        hwme_ana_l1addr_set(L1_Addr_16Byte); // absolute address of L1 (128b / address)
        hwme_ana_memcpy_n_set(Lenth_4Byte);
    }
    else 
    {
        hwme_ana_memcpy_addr_set(L2_Addr_Byte);
        hwme_ana_l1addr_set(L1_Addr_16Byte); // absolute address of L1 (128b / address)
        hwme_ana_memcpy_bank_length_set(BankNum); // number of bank
        hwme_ana_memcpy_row_length_set(row_size);
    }
    // start HWME operation
    hwme_ana_trigger_job();
}


