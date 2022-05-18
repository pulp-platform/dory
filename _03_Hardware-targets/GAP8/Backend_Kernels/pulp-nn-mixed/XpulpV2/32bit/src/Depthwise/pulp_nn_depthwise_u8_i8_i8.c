/*
 * pulp_nn_depthwise_u8_i8_i8.c
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



void pulp_nn_depthwise_u8_i8_i8(
                        uint8_t *pIn,
                        uint8_t *pIm2ColBuffer,
                        int8_t *pBias,
                        int8_t *pOut,
                        int8_t *pWeight,
                        int8_t *pWtBuffer,
                        int32_t *pKappa,
                        int32_t *pLambda,
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

  uint16_t ch_out_r = ch_out;
  uint16_t ch_in_r = ch_out;
  uint16_t ch_wt_r = ch_out;

  uint16_t ch_min = ch_out;

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

  uint8_t * pIm2ColBase = pIm2ColBuffer + (core_id * im2col_size);
  int8_t * pWtBase = pWtBuffer + kernel_size;

  int i_out_x, i_buff_y;
  uint16_t colCnt = kernel_size >> 2;
  uint16_t leftCnt = kernel_size & 0x3;


  int i_out_ch = start_channel;
  int i_in_ch = start_channel * in_image_size;
  int i_wt_ch = start_channel * kernel_size;



  int32_t * k1 = pKappa + core_id * (chunk << 0);
  int32_t * lambda1 = pLambda + core_id * (chunk << 0);

  for(int i_ch = start_channel; i_ch < stop_channel; i_ch++)
  {
    i_out_x = 0;
    int8_t * pWt = pWeight + i_wt_ch;
    if(padding_x_left > 0)
    {
      do
      {
        int8_t *pOutBuffer = pOut + i_out_ch + (i_out_x * ch_out_r);
        uint8_t *pIm2Col = pIm2ColBase;
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
              i++;
            }while(i<dim_kernel_x_size_padded);
            pIm2Col-=dim_incr;
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
          }
          int idx = 0;
          int i = 0;
          do
          {
            *((v4u*) pIm2Col) = *((v4u*) (base_ptr + idx));
            pIm2Col+=4;
            idx+=4;
            i++;
          }while(i<dim_kernel_x_size_padded);
          pIm2Col-=(dim_incr_pad_left - const1);
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
            i++;
          }while(i<dim_kernel_x_size_padded);
          pIm2Col-=dim_incr;
        }

        int l=0;
        do
        {
          pWt = pWeight + i_wt_ch;
          int sum = 0;
          if (pBias != NULL)
          {
            sum = ((int) (pBias[i_ch]));
          }
          pIm2Col = (pIm2ColBase + ((l * stride_y) * dim_kernel_x));
          int j=0;
          do
          {
            v4s w = *(v4s *) pWt;
            v4u x = *(v4u *) pIm2Col;
            sum = SumDotp4(x, w, sum);
            pWt += 4;
            pIm2Col += 4;
            j++;
          }while(j<colCnt);
          for(int j=0; j<leftCnt; j++)
          {
            int8_t w = *(int8_t *) pWt++;
            uint8_t x = *(uint8_t *) pIm2Col++;
            sum += x * w;
          }
          if (flag_batch_norm && flag_relu)
          {
            *pOutBuffer = pulp_nn_bn_quant_i8(sum, *k1, *lambda1, out_shift);
          }
          else
          {
            if(flag_relu == 1)
            {
              *pOutBuffer = pulp_nn_quant_i8(sum, out_mult, out_shift);
            }
            else
            {
              *pOutBuffer = (int8_t) clips8(sum >> out_shift);
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
      int8_t *pOutBuffer = pOut + i_out_ch + (i_out_x * ch_out_r);
      uint8_t *pIm2Col = pIm2ColBase;
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
            i++;
          }while(i<dim_kernel_x_size_padded);
          pIm2Col-=dim_incr;
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
          idx+=4;
        }
        pIm2Col-=dim_incr;
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
          i++;
        }while(i<dim_kernel_x_size_padded);
        pIm2Col-=dim_incr;
      }
      int l=0;
      do
      {
        pWt = pWeight + i_wt_ch;
        int sum = 0;
        if (pBias != NULL)
        {
          sum = ((int) (pBias[i_ch]));
        }
        pIm2Col = (pIm2ColBase + ((l * stride_y) * dim_kernel_x));
        int j=0;
        do
        {
          v4s w = *(v4s *) pWt;
          v4u x = *(v4u *) pIm2Col;
          sum = SumDotp4(x, w, sum);
          pWt += 4;
          pIm2Col += 4;
          j++;
        }while(j<colCnt);
        for(int j=0; j<leftCnt; j++)
        {
          int8_t w = *(int8_t *) pWt++;
          uint8_t x = *(uint8_t *) pIm2Col++;
          sum += x * w;
        }
        if (flag_batch_norm && flag_relu)
        {
          *pOutBuffer = pulp_nn_bn_quant_i8(sum, *k1, *lambda1, out_shift);
        }
        else
        {
          if(flag_relu == 1)
          {
            *pOutBuffer = pulp_nn_quant_i8(sum, out_mult, out_shift);
          }
          else
          {
            *pOutBuffer = (int8_t) clips8(sum >> out_shift);
          }
        }
        pOutBuffer+=(dim_out_x * ch_out_r);
        l++;
      }while(l<dim_out_y);
      i_out_x++;
    }while((i_out_x * stride_x) < ((dim_out_x * stride_x) - padding_x_right));
    for (i_out_x; i_out_x < dim_out_x; i_out_x++)
    {
      int8_t *pOutBuffer = pOut + i_out_ch + (i_out_x * ch_out_r);
      uint8_t *pIm2Col = pIm2ColBase;
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
            i++;
          }while(i<dim_kernel_x_size_padded);
          pIm2Col-=dim_incr;
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
          idx+=4;
          i++;
        }while(i<dim_kernel_x_size_padded);

        pIm2Col-=(dim_incr_pad_right + (i_out_x * stride_x) - (dim_out_x * stride_x) + padding_x_right);
        base_ptr+=dim_in_x;
        for(int j=0; j<(1 + (i_out_x * stride_x) - (dim_out_x * stride_x) + padding_x_right); j++)
        {
          *(uint8_t *) pIm2Col = 0;
          pIm2Col++;
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
          i++;
        }while(i<dim_kernel_x_size_padded);
        pIm2Col-=dim_incr;
      }

      int l=0;
      do
      {
        pWt = pWeight + i_wt_ch;
        int sum = 0;
        if (pBias != NULL)
        {
          sum = ((int) (pBias[i_ch]));
        }
        pIm2Col = (pIm2ColBase + ((l * stride_y) * dim_kernel_x));
        int j=0;
        do
        {
          v4s w = *(v4s *) pWt;
          v4u x = *(v4u *) pIm2Col;
          sum = SumDotp4(x, w, sum);
          pWt += 4;
          pIm2Col += 4;
          j++;
        }while(j<colCnt);
        for(int j=0; j<leftCnt; j++)
        {
          int8_t w = *(int8_t *) pWt++;
          uint8_t x = *(uint8_t *) pIm2Col++;
          sum += x * w;
        }
        if (flag_batch_norm && flag_relu)
        {
          *pOutBuffer = pulp_nn_bn_quant_i8(sum, *k1, *lambda1, out_shift);
        }
        else
        {
          if(flag_relu == 1)
          {
            *pOutBuffer = pulp_nn_quant_i8(sum, out_mult, out_shift);
          }
          else
          {
            *pOutBuffer = (int8_t) clips8(sum >> out_shift);
          }
        }
        pOutBuffer+=(dim_out_x * ch_out_r);
        l++;
      }while(l<dim_out_y);
    }
    i_in_ch+=in_image_size;
    i_wt_ch+=kernel_size;
    k1++;
    lambda1++;
    i_out_ch++;
  }
  pi_cl_team_barrier(0);
}
