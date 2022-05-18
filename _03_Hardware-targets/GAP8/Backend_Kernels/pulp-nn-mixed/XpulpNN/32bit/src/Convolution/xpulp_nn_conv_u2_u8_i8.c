/*
 * xpulp_nn_conv_u2_u8_i8.c
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
#include "pulp_nn_kernels.h"




void __attribute__((noinline)) xpulp_nn_conv_u2_u8_i8(
                        uint8_t *pIn,
                        uint8_t *pIm2ColBuffer,
                        int8_t *pBias,
                        uint8_t *pOut,
                        int8_t *pWeight,
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
  uint16_t ch_in_r = PACK_INT2_SIZE(ch_in);
  uint16_t ch_out_r = PACK_INT8_SIZE(ch_out);

  int core_id = pi_core_id();
  uint8_t * pIm2ColBase = pIm2ColBuffer + (2 * core_id * PACK_INT8_SIZE(ch_in) * dim_kernel_x * dim_kernel_y);
  int i_out_y, i_out_x, i_ker_y, i_ker_x;
  int Log2Core;

  uint8_t extra_chunk = ((dim_out_y & (NUM_CORES-1)) != 0);
  uint8_t extra_chunk_r;
  uint16_t dim_out_x_r;
  uint8_t section;
  int core_id_r;

  if(extra_chunk && dim_out_x > 1)
  {
    Log2Core = log2(NUM_CORES >> 1);
    core_id_r = (core_id >> 1);
    dim_out_x_r = (dim_out_x >> 1);
    section = (core_id & 0x1);
    extra_chunk_r = ((dim_out_y & ((NUM_CORES >> 1) - 1)) != 0);
  }
  else
  {
    Log2Core = log2(NUM_CORES);
    core_id_r = core_id;
    dim_out_x_r = dim_out_x;
    section = 0;
    extra_chunk_r = extra_chunk;
    extra_chunk = 0;
  }

  uint8_t flag_dim_out_x_odd = dim_out_x & 0x01;

  int chunk = (dim_out_y >> Log2Core) + extra_chunk_r;

  int start_pixel = min((chunk * core_id_r), dim_out_y);
  int stop_pixel = min(start_pixel + chunk, dim_out_y);

  uint8_t *pIm2Col = pIm2ColBase;
  uint8_t *pOutBuffer = pOut + (start_pixel * ch_out_r * dim_out_x) + (section * ch_out_r * dim_out_x_r);

  for (i_out_y = start_pixel; i_out_y < stop_pixel; i_out_y++)
  {
    for(i_out_x=(section * dim_out_x_r); i_out_x<(dim_out_x_r + (section * (dim_out_x_r + flag_dim_out_x_odd))); i_out_x++)
    {
      if(i_out_y < padding_y_top)
      {
        for(i_ker_y=((i_out_y * stride_y) - padding_y_top); i_ker_y<((i_out_y * stride_y) - padding_y_top + dim_kernel_y); i_ker_y++)
        {
          for(i_ker_x=((i_out_x * stride_x) - padding_x_left); i_ker_x<((i_out_x * stride_x) - padding_x_left + dim_kernel_x); i_ker_x++)
          {
            if((i_ker_y < 0) || (i_ker_y >= dim_in_y) || (i_ker_x < 0) || (i_ker_x >= dim_in_x))
            {
              xpulp_nn_zero_mem_u8(pIm2Col, ch_in);
            }
            else
            {
              xpulp_nn_im2col_u2_to_u8((uint8_t*) (pIn + ((i_ker_y * dim_in_x + i_ker_x) * ch_in_r)), pIm2Col, ch_in);
            }
            pIm2Col+=PACK_INT8_SIZE(ch_in);
          }
        }
      }
      else if(i_out_y < dim_out_y - padding_y_bottom)
      {
        if(i_out_x < padding_x_left)
        {
          for(i_ker_y=((i_out_y * stride_y) - padding_y_top); i_ker_y<((i_out_y * stride_y) - padding_y_top + dim_kernel_y); i_ker_y++)
          {
            for(i_ker_x=((i_out_x * stride_x) - padding_x_left); i_ker_x<((i_out_x * stride_x) - padding_x_left + dim_kernel_x); i_ker_x++)
            {
              if((i_ker_x < 0) || (i_ker_x >= dim_in_x))
              {
                xpulp_nn_zero_mem_u8(pIm2Col, ch_in);
              }
              else
              {
                xpulp_nn_im2col_u2_to_u8((uint8_t*) (pIn + ((i_ker_y * dim_in_x + i_ker_x) * ch_in_r)), pIm2Col, ch_in);
              }
              pIm2Col+=PACK_INT8_SIZE(ch_in);
            }
          }
        }
        else if(i_out_x < (dim_out_x - padding_x_right))
        {
          for(i_ker_y=((i_out_y * stride_y) - padding_y_top); i_ker_y<((i_out_y * stride_y) - padding_y_top + dim_kernel_y); i_ker_y++)
          {
            xpulp_nn_im2col_u2_to_u8((uint8_t*) pIn + (i_ker_y * dim_in_x + i_out_x * stride_x - padding_x_left)*ch_in_r,pIm2Col,ch_in * dim_kernel_x);
            pIm2Col+=PACK_INT8_SIZE(ch_in * dim_kernel_x);
          }
        }
        else
        {
          for(i_ker_y=((i_out_y * stride_y) - padding_y_top); i_ker_y<((i_out_y * stride_y) - padding_y_top + dim_kernel_y); i_ker_y++)
          {
            for(i_ker_x = i_out_x * stride_x - padding_x_left; i_ker_x < i_out_x * stride_x - padding_x_left + dim_kernel_x; i_ker_x++)
            {
              if((i_ker_x < 0) || (i_ker_x >= dim_in_x))
              {
                xpulp_nn_zero_mem_u8(pIm2Col, ch_in);
              }
              else
              {
                xpulp_nn_im2col_u2_to_u8((uint8_t *)pIn + (i_ker_y*dim_in_x+i_ker_x)* ch_in_r, pIm2Col, ch_in);
              }
              pIm2Col+=PACK_INT8_SIZE(ch_in);
            }
          }
        }
      }
      else
      {
        for(i_ker_y=((i_out_y * stride_y) - padding_y_top); i_ker_y<((i_out_y * stride_y) - padding_y_top + dim_kernel_y); i_ker_y++)
        {
          for(i_ker_x = i_out_x * stride_x - padding_x_left; i_ker_x < i_out_x * stride_x - padding_x_left + dim_kernel_x; i_ker_x++)
          {
            if(i_ker_y < 0 || (i_ker_y >= dim_in_y) || i_ker_x < 0 || i_ker_x >= dim_in_x)
            {
              xpulp_nn_zero_mem_u8(pIm2Col, ch_in);
            }
            else
            {
              xpulp_nn_im2col_u2_to_u8((uint8_t *) pIn + (i_ker_y * dim_in_x + i_ker_x) * ch_in_r, pIm2Col, ch_in);
            }
            pIm2Col+=PACK_INT8_SIZE(ch_in);
          }
        }
      }
      if(pIm2Col == (pIm2ColBase + ((PACK_INT8_SIZE(ch_in) * dim_kernel_x * dim_kernel_y) << 1)))
      {
        pOutBuffer = xpulp_nn_matmul_u2_u8_i8(
          pIm2ColBase,
          pBias,
          pOutBuffer,
          pOutBuffer + ch_out_r,
          pWeight,
          pKappa,
          pLambda,
          out_mult,
          out_shift,
          (ch_in * dim_kernel_x * dim_kernel_y),
          ch_out,
          flag_relu,
          flag_batch_norm
          );

        pIm2Col = pIm2ColBase;
      }
    }

    if(pIm2Col != pIm2ColBase)
    {
      const int8_t *pA = pWeight;
      int i;
      int32_t * k1 = pKappa;
      int32_t * lambda1 = pLambda;

      uint8_t out[1];
      uint16_t num_col_im2col = ch_in * dim_kernel_x * dim_kernel_y;
      uint16_t num_col_im2col_w = PACK_INT8_SIZE(ch_in) * dim_kernel_x * dim_kernel_y;

      for(i = 0; i < ch_out; i++)
      {
        int sum = 0;
        if (pBias != NULL)
        {
          sum = ((int) (*pBias++));
        }

        uint8_t *pB = pIm2ColBase;

        int32_t *ptrA  = (int32_t *)pA;
        uint32_t *ptrB = (uint32_t *)pB;

        for(int j=0; j < (num_col_im2col >> 2); j++)
        {
          sum = SumDotp4(*(v4u *)ptrB, *(v4s *)ptrA, sum);
          ptrA++;
          ptrB++;
        }

        int col_cnt_im2col = num_col_im2col & 0x3;

        if(col_cnt_im2col)
        {
          uint16_t loop_cnt_im2col_w = (num_col_im2col >> 2) << 2;
          pA+=loop_cnt_im2col_w;

          uint16_t loop_cnt_im2col_a = (num_col_im2col >> 2) << 2;
          pB+=loop_cnt_im2col_a;

          do
          {
            int8_t inA1 = *pA++;
            uint8_t inB1 = *pB++;
            asm volatile("": : :"memory");
            sum += inA1 * inB1;

            col_cnt_im2col--;
          } while(col_cnt_im2col);
          pA-=num_col_im2col_w;
        }
        if (flag_batch_norm && flag_relu)
        {
          *pOutBuffer = pulp_nn_bn_quant_u8(sum, *k1, *lambda1, out_shift);
          k1++;
          lambda1++;
          pOutBuffer++;
        }
        else
        {
          if(flag_relu == 1)
          {
            *pOutBuffer = pulp_nn_quant_u8(sum, out_mult, out_shift);
            pOutBuffer++;
          }
          else
          {
            *pOutBuffer = (uint8_t) clip8(sum >> out_shift);
            pOutBuffer++;
          }
        }
        pA+=num_col_im2col_w;
      }
    }
    pOutBuffer+=(extra_chunk * ((dim_out_x_r + ((1 - section) * flag_dim_out_x_odd)) * ch_out_r));
    pIm2Col = pIm2ColBase;
  }
  pi_cl_team_barrier(0);
}
