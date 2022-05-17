/*
 * xpulp_nn_depthwise_u8_u4_i8.c
 * Nazareno Bruschi <nazareno.bruschi@unibo.it>
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

#include "pmsis.h"
#include "pulp_nn_utils.h"


void xpulp_nn_depthwise_u8_u4_i8(
                        uint8_t *pIn,
                        uint8_t *pIm2ColBuffer,
                        int8_t *pBias,
                        uint8_t *pOut,
                        int8_t *pWeight,
                        int8_t *pWtBuffer,
                        int64_t *pKappa,
                        int64_t *pLambda,
                        uint16_t out_mult,
                        uint16_t out_shift,
                        uint16_t dim_in_x,
                        uint16_t dim_in_y,
                        uint16_t ch_in,
                        uint16_t dim_out_x,
                        uint16_t dim_out_y,
                        uint16_t ch_out,
                        uint16_t dim_kernel_x,
                        uint16_t dim_kernel_y,
                        uint16_t padding_y_top,
                        uint16_t padding_y_bottom,
                        uint16_t padding_x_left,
                        uint16_t padding_x_right,
                        uint16_t stride_x,
                        uint16_t stride_y,
                        uint8_t flag_relu,
                        uint8_t flag_batch_norm)
{
  uint8_t core_id = pi_core_id();
  uint8_t Log2Core = log2(NUM_CORES);

  uint16_t ch_out_r = ch_out >> 1;
  uint16_t ch_in_r = ch_out;
  uint16_t ch_wt_r = ch_out;

  uint16_t ch_min = ch_out >> 1;

  int chunk = (ch_min >> Log2Core) + ((ch_min & (NUM_CORES - 1)) != 0);

  int start_channel = min(chunk * core_id, ch_min);
  int stop_channel = min(start_channel + chunk, ch_min);

  uint16_t dim_kernel_x_size_rem = dim_kernel_x & 0x3;
  uint16_t dim_kernel_x_size_padded = (dim_kernel_x >> 2) + (dim_kernel_x_size_rem != 0);
  uint16_t dim_incr = (dim_kernel_x_size_padded << 2) - dim_kernel_x;
  uint16_t dim_incr_pad_left = (dim_kernel_x_size_padded << 2) - dim_kernel_x + padding_x_left;
  uint16_t dim_incr_pad_right = (dim_kernel_x_size_padded << 2) - dim_kernel_x + 1;
  uint16_t kernel_size = dim_kernel_x * dim_kernel_y;
  uint16_t im2col_size = ((dim_kernel_x * (dim_in_y + padding_y_top + padding_y_bottom)) + dim_kernel_x);
  uint16_t in_image_size = dim_in_x * dim_in_y;

  uint8_t * pIm2ColBase = pIm2ColBuffer + (core_id * im2col_size << 1);
  int8_t * pWtBase = pWtBuffer + (core_id * (kernel_size << 1));

  int i_out_x, i_buff_y;
  uint16_t colCnt = kernel_size >> 2;
  uint16_t leftCnt = kernel_size & 0x3;

  int8_t mask = 0xf0;
  int8_t n_mask = ~ mask;
  int8_t off = 0x04;

  int i_out_ch = start_channel;
  int i_in_ch = (start_channel << 1) * in_image_size;
  int i_wt_ch = (start_channel << 1) * kernel_size;


  int64_t * k1 = pKappa + core_id * (chunk << 1);
  int64_t * lambda1 = pLambda + core_id * (chunk << 1);

  for(int i_ch = start_channel; i_ch < stop_channel; i_ch++)
  {
    i_out_x = 0;
    int8_t * pWt = pWeight + i_wt_ch;
    int8_t * pWt2 = pWt + kernel_size;
    if(padding_x_left > 0)
    {
      do
      {
        uint8_t *pOutBuffer = pOut + i_out_ch + (i_out_x * ch_out_r);
        uint8_t *pIm2Col = pIm2ColBase;
        uint8_t *pIm2Col2 = pIm2Col + im2col_size;
        i_buff_y = - padding_y_top;
        if(padding_y_top > 0)
        {
          do
          {
            int i=0;
            do
            {
              *(v4u *) pIm2Col = (v4u) {0, 0, 0, 0};
              pIm2Col+=4;
              *(v4u *) pIm2Col2 = (v4u) {0, 0, 0, 0};
              pIm2Col2+=4;
              i++;
            }while(i<dim_kernel_x_size_padded);
            pIm2Col-=dim_incr;
            pIm2Col2-=dim_incr;
            i_buff_y++;
          }while(i_buff_y < 0);
        }
        int const1 = (i_out_x * stride_x);
        int base_ptr = pIn + i_in_ch;
        do
        {
          for(int j=0; j< (padding_x_left - const1); j++)
          {
            *(uint8_t *) pIm2Col = 0;
            pIm2Col++;
            *(uint8_t *) pIm2Col2 = 0;
            pIm2Col2++;
          }
          int idx = 0;
          int i = 0;
          do
          {
            *((v4u*) pIm2Col) = *((v4u*) (base_ptr + idx));
            pIm2Col+=4;
            *((v4u*) pIm2Col2) = *((v4u*) (base_ptr + idx + in_image_size));
            pIm2Col2+=4;
            idx+=4;
            i++;
          }while(i<dim_kernel_x_size_padded);
          pIm2Col-=(dim_incr_pad_left - const1);
          pIm2Col2-=(dim_incr_pad_left - const1);
          base_ptr+=dim_in_x;
          i_buff_y++;
        }while(i_buff_y < dim_in_y);
        for (i_buff_y; i_buff_y < dim_in_y + padding_y_bottom; i_buff_y++)
        {
          int i=0;
          do
          {
            *(v4u *) pIm2Col = (v4u) {0, 0, 0, 0};
            pIm2Col+=4;
            *(v4u *) pIm2Col2 = (v4u) {0, 0, 0, 0};
            pIm2Col2+=4;
            i++;
          }while(i<dim_kernel_x_size_padded);
          pIm2Col-=dim_incr;
          pIm2Col2-=dim_incr;
        }

        int l=0;
        do
        {
          pWt = pWeight + i_wt_ch;
          pWt2 = pWt + kernel_size;
          int sum = 0;
          int sum2 = 0;
          if (pBias != NULL)
          {
            sum = ((int) (pBias[i_ch]));
            sum2 = ((int) (pBias[i_ch + 1]));
          }
          pIm2Col = (pIm2ColBase + ((l * stride_y) * dim_kernel_x));
          pIm2Col2 = pIm2Col + im2col_size;
          int32_t  ptrA  = (int32_t *) pWt;
          int32_t  ptrA2 = (int32_t *) pWt2;
          uint32_t ptrB  = (uint32_t *) pIm2Col;
          uint32_t ptrB2 = (uint32_t *) pIm2Col2;

          ptrA  = MacLoadInit(1, 0, 0, 0, ptrA);
          ptrA2 = MacLoadInit(1, 0, 1, 0, ptrA2);
          ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);
          ptrB2 = MacLoadInit(0, 1, 0, 1, ptrB2);
          int j=0;
          do
          {
            sum  = MacLoad4(1, 0, 0, 0, ptrA, sum);
            ptrA = MacLoadUpdate(ptrA);
            ptrB = MacLoadInit(0, 1, 0, 0, ptrB);
            sum2  = MacLoad4(1, 0, 1, 1, ptrA2, sum2);
            ptrA2 = MacLoadUpdate(ptrA2);
            ptrB2 = MacLoadInit(0, 1, 1, 1, ptrB2);
            j++;
          }while(j<colCnt);
          if(leftCnt)
          {
            pWt+=(j << 2);
            pWt2+=(j << 2);
            pIm2Col+=(j << 2);
            pIm2Col2+=(j << 2);
            do
            {
              int8_t w = *(int8_t *) pWt++;
              uint8_t x = *(uint8_t *) pIm2Col++;
              sum += x * w;
              int8_t w2 = *(int8_t *) pWt2++;
              uint8_t x2 = *(uint8_t *) pIm2Col2++;
              sum2 += x2 * w2;
              j++;
            }while(j<leftCnt);
          }
          if (flag_batch_norm && flag_relu)
          {
            sum = pulp_nn_bn_quant_u4(sum, *k1, *lambda1, out_shift);
            sum2 = pulp_nn_bn_quant_u4(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
            *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
          }
          else
          {
            if(flag_relu == 1)
            {
              sum = pulp_nn_quant_u4(sum, out_mult, out_shift);
              sum2 = pulp_nn_quant_u4(sum2, out_mult, out_shift);
              *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
            }
            else
            {
              sum = (uint8_t) clip4(sum >> out_shift);
              sum2 = (uint8_t) clip4(sum2 >> out_shift);
              *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
            }
          }
          pOutBuffer+=(dim_out_x * ch_out_r);
          l++;
        }while(l<dim_out_y);
        i_out_x++;
      }while((i_out_x * stride_x) < padding_x_left);
    }
    do
    {
      uint8_t *pOutBuffer = pOut + i_out_ch + (i_out_x * ch_out_r);
      uint8_t *pIm2Col = pIm2ColBase;
      uint8_t *pIm2Col2 = pIm2Col + im2col_size;
      i_buff_y = - padding_y_top;
      if(padding_y_top > 0)
      {
        do
        {
          int i=0;
          do
          {
            *(v4u *) pIm2Col = (v4u) {0, 0, 0, 0};
            pIm2Col+=4;
            *(v4u *) pIm2Col2 = (v4u) {0, 0, 0, 0};
            pIm2Col2+=4;
            i++;
          }while(i<dim_kernel_x_size_padded);
          pIm2Col-=dim_incr;
          pIm2Col2-=dim_incr;
          i_buff_y++;
        }while(i_buff_y < 0);
      }
      int base_ptr = pIn + i_in_ch + (i_out_x * stride_x) - padding_x_left;
      do
      {
        int idx = 0;
        for (int i=0; i<dim_kernel_x_size_padded; i++)
        {
          *((v4u*) pIm2Col) = *((v4u*) (base_ptr + idx));
          pIm2Col+=4;
          *((v4u*) pIm2Col2) = *((v4u*) (base_ptr + idx + in_image_size));
          pIm2Col2+=4;
          idx+=4;
        }
        pIm2Col-=dim_incr;
        pIm2Col2-=dim_incr;
        base_ptr+=dim_in_x;
        i_buff_y++;
      }while(i_buff_y < dim_in_y);
      for (i_buff_y; i_buff_y < dim_in_y + padding_y_bottom; i_buff_y++)
      {
        int i=0;
        do
        {
          *(v4u *) pIm2Col = (v4u) {0, 0, 0, 0};
          pIm2Col+=4;
          *(v4u *) pIm2Col2 = (v4u) {0, 0, 0, 0};
          pIm2Col2+=4;
          i++;
        }while(i<dim_kernel_x_size_padded);
        pIm2Col-=dim_incr;
        pIm2Col2-=dim_incr;
      }
      int l=0;
      do
      {
        pWt = pWeight + i_wt_ch;
        pWt2 = pWt + kernel_size;
        int sum = 0;
        int sum2 = 0;
        if (pBias != NULL)
        {
          sum = ((int) (pBias[i_ch]));
          sum2 = ((int) (pBias[i_ch + 1]));
        }
        pIm2Col = (pIm2ColBase + ((l * stride_y) * dim_kernel_x));
        pIm2Col2 = pIm2Col + im2col_size;
        int32_t  ptrA  = (int32_t *) pWt;
        int32_t  ptrA2 = (int32_t *) pWt2;
        uint32_t ptrB  = (uint32_t *) pIm2Col;
        uint32_t ptrB2 = (uint32_t *) pIm2Col2;

        ptrA  = MacLoadInit(1, 0, 0, 0, ptrA);
        ptrA2 = MacLoadInit(1, 0, 1, 0, ptrA2);
        ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);
        ptrB2 = MacLoadInit(0, 1, 0, 1, ptrB2);
        int j=0;
        do
        {
          sum  = MacLoad4(1, 0, 0, 0, ptrA, sum);
          ptrA = MacLoadUpdate(ptrA);
          ptrB = MacLoadInit(0, 1, 0, 0, ptrB);
          sum2  = MacLoad4(1, 0, 1, 1, ptrA2, sum2);
          ptrA2 = MacLoadUpdate(ptrA2);
          ptrB2 = MacLoadInit(0, 1, 1, 1, ptrB2);
          j++;
        }while(j<colCnt);
        if(leftCnt)
        {
          pWt+=(j << 2);
          pWt2+=(j << 2);
          pIm2Col+=(j << 2);
          pIm2Col2+=(j << 2);
          do
          {
            int8_t w = *(int8_t *) pWt++;
            uint8_t x = *(uint8_t *) pIm2Col++;
            sum += x * w;
            int8_t w2 = *(int8_t *) pWt2++;
            uint8_t x2 = *(uint8_t *) pIm2Col2++;
            sum2 += x2 * w2;
            j++;
          }while(j<leftCnt);
        }
        if (flag_batch_norm && flag_relu)
        {
          sum = pulp_nn_bn_quant_u4(sum, *k1, *lambda1, out_shift);
          sum2 = pulp_nn_bn_quant_u4(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
          *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
        }
        else
        {
          if(flag_relu == 1)
          {
            sum = pulp_nn_quant_u4(sum, out_mult, out_shift);
            sum2 = pulp_nn_quant_u4(sum2, out_mult, out_shift);
            *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
          }
          else
          {
            sum = (uint8_t) clip4(sum >> out_shift);
            sum2 = (uint8_t) clip4(sum2 >> out_shift);
            *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
          }
        }
        pOutBuffer+=(dim_out_x * ch_out_r);
        l++;
      }while(l<dim_out_y);
      i_out_x++;
    }while((i_out_x * stride_x) < ((dim_out_x * stride_x) - padding_x_right));
    for (i_out_x; i_out_x < dim_out_x; i_out_x++)
    {
      uint8_t *pOutBuffer = pOut + i_out_ch + (i_out_x * ch_out_r);
      uint8_t *pIm2Col = pIm2ColBase;
      uint8_t *pIm2Col2 = pIm2Col + im2col_size;
      asm volatile ("":::"memory");
      i_buff_y = - padding_y_top;
      if(padding_y_top > 0)
      {
        do
        {
          int i=0;
          do
          {
            *(v4u *) pIm2Col = (v4u) {0, 0, 0, 0};
            pIm2Col+=4;
            *(v4u *) pIm2Col2 = (v4u) {0, 0, 0, 0};
            pIm2Col2+=4;
            i++;
          }while(i<dim_kernel_x_size_padded);
          pIm2Col-=dim_incr;
          pIm2Col2-=dim_incr;
          i_buff_y++;
        }while(i_buff_y < 0);
      }
      int base_ptr = pIn + i_in_ch + (i_out_x * stride_x) - padding_x_left;
      do
      {
        int i = 0;
        int idx = 0;
        do
        {
          *((v4u*) pIm2Col) = *((v4u*) (base_ptr + idx));
          pIm2Col+=4;
          *((v4u*) pIm2Col2) = *((v4u*) (base_ptr + idx + in_image_size));
          pIm2Col2+=4;
          idx+=4;
          i++;
        }while(i<dim_kernel_x_size_padded);

        pIm2Col-=(dim_incr_pad_right + (i_out_x * stride_x) - (dim_out_x * stride_x) + padding_x_right);
        pIm2Col2-=(dim_incr_pad_right + (i_out_x * stride_x) - (dim_out_x * stride_x) + padding_x_right);
        base_ptr+=dim_in_x;
        for(int j=0; j<(1 + (i_out_x * stride_x) - (dim_out_x * stride_x) + padding_x_right); j++)
        {
          *(uint8_t *) pIm2Col = 0;
          pIm2Col++;
          *(uint8_t *) pIm2Col2 = 0;
          pIm2Col2++;
        }
        i_buff_y++;
      }while(i_buff_y < dim_in_y);
      for (i_buff_y; i_buff_y < dim_in_y + padding_y_bottom; i_buff_y++)
      {
        int i=0;
        do
        {
          *(v4u *) pIm2Col = (v4u) {0, 0, 0, 0};
          pIm2Col+=4;
          *(v4u *) pIm2Col2 = (v4u) {0, 0, 0, 0};
          pIm2Col2+=4;
          i++;
        }while(i<dim_kernel_x_size_padded);
        pIm2Col-=dim_incr;
        pIm2Col2-=dim_incr;
      }

      int l=0;
      do
      {
        pWt = pWeight + i_wt_ch;
        pWt2 = pWt + kernel_size;
        int sum = 0;
        int sum2 = 0;
        if (pBias != NULL)
        {
          sum = ((int) (pBias[i_ch]));
          sum2 = ((int) (pBias[i_ch + 1]));
        }
        pIm2Col = (pIm2ColBase + ((l * stride_y) * dim_kernel_x));
        pIm2Col2 = pIm2Col + im2col_size;
        int32_t  ptrA  = (int32_t *) pWt;
        int32_t  ptrA2 = (int32_t *) pWt2;
        uint32_t ptrB  = (uint32_t *) pIm2Col;
        uint32_t ptrB2 = (uint32_t *) pIm2Col2;

        ptrA  = MacLoadInit(1, 0, 0, 0, ptrA);
        ptrA2 = MacLoadInit(1, 0, 1, 0, ptrA2);
        ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);
        ptrB2 = MacLoadInit(0, 1, 0, 1, ptrB2);
        int j=0;
        do
        {
          sum  = MacLoad4(1, 0, 0, 0, ptrA, sum);
          ptrA = MacLoadUpdate(ptrA);
          ptrB = MacLoadInit(0, 1, 0, 0, ptrB);
          sum2  = MacLoad4(1, 0, 1, 1, ptrA2, sum2);
          ptrA2 = MacLoadUpdate(ptrA2);
          ptrB2 = MacLoadInit(0, 1, 1, 1, ptrB2);
          j++;
        }while(j<colCnt);
        if(leftCnt)
        {
          pWt+=(j << 2);
          pWt2+=(j << 2);
          pIm2Col+=(j << 2);
          pIm2Col2+=(j << 2);
          do
          {
            int8_t w = *(int8_t *) pWt++;
            uint8_t x = *(uint8_t *) pIm2Col++;
            sum += x * w;
            int8_t w2 = *(int8_t *) pWt2++;
            uint8_t x2 = *(uint8_t *) pIm2Col2++;
            sum2 += x2 * w2;
            j++;
          }while(j<leftCnt);
        }
        if (flag_batch_norm && flag_relu)
        {
          sum = pulp_nn_bn_quant_u4(sum, *k1, *lambda1, out_shift);
          sum2 = pulp_nn_bn_quant_u4(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
          *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
        }
        else
        {
          if(flag_relu == 1)
          {
            sum = pulp_nn_quant_u4(sum, out_mult, out_shift);
            sum2 = pulp_nn_quant_u4(sum2, out_mult, out_shift);
            *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
          }
          else
          {
            sum = (uint8_t) clip4(sum >> out_shift);
            sum2 = (uint8_t) clip4(sum2 >> out_shift);
            *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
          }
        }
        pOutBuffer+=(dim_out_x * ch_out_r);
        l++;
      }while(l<dim_out_y);
    }
    i_in_ch+=(in_image_size << 1);
    i_wt_ch+=(kernel_size << 1);
    k1+=2;
    lambda1+=2;
    i_out_ch++;
  }
  pi_cl_team_barrier(0);
}
