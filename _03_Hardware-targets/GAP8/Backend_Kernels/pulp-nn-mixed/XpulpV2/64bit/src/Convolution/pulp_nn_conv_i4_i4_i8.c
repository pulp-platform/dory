/*
 * pulp_nn_conv_i4_i4_i8.c
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



void pulp_nn_conv_i4_i4_i8(
                        int8_t *pIn,
                        int8_t *pIm2ColBuffer,
                        int8_t *pBias,
                        int8_t *pOut,
                        int8_t *pWeight,
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
  uint16_t ch_in_r = ch_in >> 1;
  uint16_t ch_out_r = ch_out >> 1;

  int core_id = pi_core_id();
  uint8_t * pIm2ColBase = pIm2ColBuffer + (2 * core_id * ch_in * dim_kernel_x * dim_kernel_y);
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
              pulp_zero_mem(pIm2Col, ch_in);
            }
            else
            {
              pulp_nn_im2col_i4_to_i8((int8_t*) (pIn + ((i_ker_y * dim_in_x + i_ker_x) * ch_in_r)), pIm2Col, ch_in);
            }
            pIm2Col+=ch_in;
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
                pulp_zero_mem(pIm2Col, ch_in);
              }
              else
              {
                pulp_nn_im2col_i4_to_i8((int8_t*) (pIn + ((i_ker_y * dim_in_x + i_ker_x) * ch_in_r)), pIm2Col, ch_in);
              }
              pIm2Col+=ch_in;
            }
          }
        }
        else if(i_out_x < (dim_out_x - padding_x_right))
        {
          for(i_ker_y=((i_out_y * stride_y) - padding_y_top); i_ker_y<((i_out_y * stride_y) - padding_y_top + dim_kernel_y); i_ker_y++)
          {
            pulp_nn_im2col_i4_to_i8((int8_t*) pIn + (i_ker_y * dim_in_x + i_out_x * stride_x - padding_x_left)*ch_in_r,pIm2Col,ch_in * dim_kernel_x);
            pIm2Col+=(ch_in * dim_kernel_x);
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
                pulp_zero_mem(pIm2Col, ch_in);
              }
              else
              {
                pulp_nn_im2col_i4_to_i8((int8_t *)pIn + (i_ker_y*dim_in_x+i_ker_x)* ch_in_r, pIm2Col, ch_in);
              }
              pIm2Col+=ch_in;
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
              pulp_zero_mem(pIm2Col, ch_in);
            }
            else
            {
              pulp_nn_im2col_i4_to_i8((int8_t *) pIn + (i_ker_y * dim_in_x + i_ker_x) * ch_in_r, pIm2Col, ch_in);
            }
            pIm2Col+=ch_in;
          }
        }
      }
      if(pIm2Col == (pIm2ColBase + ((ch_in * dim_kernel_x * dim_kernel_y) << 1)))
      {
        pOutBuffer = pulp_nn_matmul_i8_i4_i8(
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
  //   pOut+=(extra_chunk * ((dim_out_x_r + ((1 - section) * flag_dim_out_x_odd)) * ch_out_r));
  // }

    if(pIm2Col != pIm2ColBase)
    {
      int8_t mask = 0xf0;
      int8_t n_mask = ~ mask;
      int8_t off = 0x04;
      const int8_t *pA = pWeight;
      int i;
      int64_t * k1 = pKappa;
      int64_t * lambda1 = pLambda;
      v4s inA[1];
      v4s inB;
      int8_t out[2];
      for(i = 0; i < ch_out; i++)
      {
        int sum = 0;
        if (pBias != NULL)
        {
          sum = ((int) (*pBias++));
        }

        uint8_t *pB = pIm2ColBase;
        uint16_t col_cnt_im2col = ch_in * dim_kernel_x * dim_kernel_y >> 2;
        for(int j=0; j < col_cnt_im2col; j++)
        {
          v4s inA = *((v4s*) pA);
          v4s inB = *((v4s*) pB);

          sum = SumDotps4(inB, inA, sum);
          pA+=4;
          pB+=4;
        }
        col_cnt_im2col = (ch_in * dim_kernel_y * dim_kernel_x) & 0x3;
        while (col_cnt_im2col)
        {
          int8_t inA1 = *pA++;
          int8_t inB1 = *pB++;
          asm volatile("": : :"memory");
          sum += inA1 * inB1;

          col_cnt_im2col--;
        }
        if (flag_batch_norm && flag_relu)
        {
          uint8_t i_o = i & 0x01;
          out[i_o] = pulp_nn_bn_quant_i4(sum, *k1, *lambda1, out_shift);
          k1++;
          lambda1++;
          if(i_o == 0x01)
          {
            *pOutBuffer = bitins(out[0], n_mask, out[1], mask, off);
            pOutBuffer++;
          }
        }
        else
        {
          if(flag_relu == 1)
          {
            uint8_t i_o = i & 0x01;
            out[i_o] = pulp_nn_quant_i4(sum, out_mult, out_shift);
            if(i_o == 0x01)
            {
              *pOutBuffer = bitins(out[0], n_mask, out[1], mask, off);
              pOutBuffer++;
            }
          }
          else
          {
            uint8_t i_o = i & 0x01;
            out[i_o] = (int8_t) clips4(sum >> out_shift);
            if(i_o == 0x01)
            {
              *pOutBuffer = bitins(out[0], n_mask, out[1], mask, off);
              pOutBuffer++;
            }
          }
        }
      }
    }
    pOutBuffer+=(extra_chunk * ((dim_out_x_r + ((1 - section) * flag_dim_out_x_odd)) * ch_out_r));
    pIm2Col = pIm2ColBase;
  }
  pi_cl_team_barrier(0);
}
