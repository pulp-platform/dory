/*
 * xpulp_nn_pointwise_i8_u2_i8.c
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


void xpulp_nn_pointwise_i8_u2_i8(
                        int8_t *pIn,
                        int8_t *pIm2ColBuffer,
                        int8_t *pBias,
                        uint8_t *pOut,
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
  uint16_t ch_in_r = PACK_INT8_SIZE(ch_in);
  uint16_t ch_out_r = PACK_INT2_SIZE(ch_out);

  int core_id = pi_core_id();
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

  uint8_t *pOutBuffer = pOut + (start_pixel * ch_out_r * dim_out_x) + (section * ch_out_r * dim_out_x_r);

  for (i_out_y = start_pixel; i_out_y < stop_pixel; i_out_y++)
  {
    i_out_x= (section * dim_out_x_r);

    for(int n = 0; n<((dim_out_x_r + (section * flag_dim_out_x_odd)) >> 1); n++)
    {
      int8_t *pIm2Col = (pIn + (i_out_x * ch_in_r) + (i_out_y * dim_in_x * ch_in_r));
      pOutBuffer = xpulp_nn_matmul_i8_u2_i8(
          pIm2Col,
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
      i_out_x+=2;
    }

    if(((dim_out_x_r + (section * flag_dim_out_x_odd)) & 0x0001))
    {
      int8_t mask2 = 0x0c;
      int8_t n_mask2 = ~ mask2;
      int8_t mask4 = 0x30;
      int8_t n_mask4 = ~ mask4;
      int8_t mask6 = 0xc0;
      int8_t n_mask6 = ~ mask6;
      int8_t off2 = 2;
      int8_t off4 = 4;
      int8_t off6 = 6;
      const int8_t *pA = pWeight;
      int i;
      int64_t * k1 = pKappa;
      int64_t * lambda1 = pLambda;

      v4s inA;
      v4s inB;
      uint8_t out[4];
      for(i = 0; i < ch_out; i++)
      {
        int sum = 0;
        if (pBias != NULL)
        {
          sum = ((int) (*pBias++));
        }

        int8_t *pB = (pIn + (i_out_x * ch_in) + (i_out_y * dim_in_x * ch_in));

        uint16_t col_cnt_im2col = ch_in * dim_kernel_x * dim_kernel_y;

        for(int j=0; j < (col_cnt_im2col >> 2); j++)
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
          uint8_t i_o = i & 0x03;
          out[i_o] = pulp_nn_bn_quant_u2(sum, *k1, *lambda1, out_shift);
          k1++;
          lambda1++;
          if(i_o == 0x03)
          {
            out[0] = bitins(out[0], n_mask2, out[1], mask2, off2);
            out[0] = bitins(out[0], n_mask4, out[2], mask4, off4);
            *pOutBuffer = bitins(out[0], n_mask6, out[3], mask6, off6);
            pOutBuffer++;
          }
        }
        else
        {
          if(flag_relu == 1)
          {
            uint8_t i_o = i & 0x03;
            out[i_o] = pulp_nn_quant_u2(sum, out_mult, out_shift);
            if(i_o == 0x03)
            {
              out[0] = bitins(out[0], n_mask2, out[1], mask2, off2);
              out[0] = bitins(out[0], n_mask4, out[2], mask4, off4);
              *pOutBuffer = bitins(out[0], n_mask6, out[3], mask6, off6);
              pOutBuffer++;
            }
          }
          else
          {
            uint8_t i_o = i & 0x03;
            out[i_o] = (uint8_t) clip2(sum >> out_shift);
            if(i_o == 0x03)
            {
              out[0] = bitins(out[0], n_mask2, out[1], mask2, off2);
              out[0] = bitins(out[0], n_mask4, out[2], mask4, off4);
              *pOutBuffer = bitins(out[0], n_mask6, out[3], mask6, off6);
              pOutBuffer++;
            }
          }
        }
      }
    }
    pOutBuffer+=(extra_chunk * ((dim_out_x_r + ((1 - section) * flag_dim_out_x_odd)) * ch_out));
  }
  pi_cl_team_barrier(0);
}
