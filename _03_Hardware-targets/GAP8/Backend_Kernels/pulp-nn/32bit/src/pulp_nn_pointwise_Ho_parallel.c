/*
 * pulp_nn_pointwise_Ho_parallel.c
 * Nazareno Bruschi <nazareno.bruschi@unibo.it>
 * Angelo Garofalo <angelo.garofalo@unibo.it>
 *
 * Copyright (C) 2018-2020 University of Bologna
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

#define log2(x) __builtin_pulp_fl1(x)
#define min(a,b) ((a)<(b)?(a):(b))
#define SumDotp(a, b, c)        __builtin_pulp_sdotusp4(a, b, c)
#define clip8(x)                __builtin_pulp_clipu_r(x, 255)

void __attribute__ ((noinline)) pulp_nn_pointwise_Ho_parallel(
  const uint8_t * pInBuffer,
  uint8_t *       pIm2ColBuffer,
  const int8_t *  bias,
  uint8_t *       pOutBuffer,
  const int8_t *  pWeight,
  int32_t *       k,
  int32_t *       lambda,
  const uint16_t  out_mult,
  const uint16_t  out_shift,
  const uint16_t  dim_in_x,
  const uint16_t  dim_in_y,
  const uint16_t  ch_in,
  const uint16_t  dim_out_x,
  const uint16_t  dim_out_y,
  const uint16_t  ch_out,
  const uint16_t  dim_kernel_x,
  const uint16_t  dim_kernel_y,
  const uint16_t  padding_y_top,
  const uint16_t  padding_y_bottom,
  const uint16_t  padding_x_left,
  const uint16_t  padding_x_right,
  const uint16_t  stride_x,
  const uint16_t  stride_y,
  int             flag_relu,
  int             flag_batch_norm
) {
  int core_id = pi_core_id();

  // local vars
  int i_out_y, i_out_x, i_ker_y, i_ker_x;
  int Log2Core = log2(NUM_CORES);

  /*chunks are built along the spatial dimension of the OFM */
  int chunk = (dim_out_y >> Log2Core) + ((dim_out_y & (NUM_CORES-1))!=0);

  /* defining the specific pixels computed by each core */
  int start_pixel, stop_pixel;
  start_pixel = min(chunk *  core_id, dim_out_y);
  stop_pixel = min(start_pixel+chunk, dim_out_y);

  uint8_t *pOut = pOutBuffer + start_pixel * ch_out * dim_out_x;

  for (i_out_y = start_pixel; i_out_y < stop_pixel; i_out_y++)
  {
    i_out_x = 0;

    for (int n = 0; n < (dim_out_x >> 1); n++)
    {
      uint8_t *pB = (pInBuffer + (i_out_x * ch_in) + (i_out_y * dim_in_x * ch_in));
      pOut = pulp_nn_matmul(
        pWeight,
        pB,
        ch_out,
        ch_in,
        out_shift,
        out_mult,
        k,
        lambda,
        bias,
        pOut,
        pOut + ch_out,
        flag_relu,
        flag_batch_norm
      );
      i_out_x+=2;
    }
    /* check if there is left-over for compute */
    if (i_out_x != dim_out_x)
    {
      const int8_t *pA = pWeight;
      int32_t *k1 = k;
      int32_t *lambda1 = lambda;
      for (int i = 0; i < ch_out; i++)
      {
        int sum = 0;

        if (bias != NULL)
        {
          sum = ((int)(bias[i]));
        }

        uint8_t *pB = (pInBuffer + (i_out_x * ch_in) + (i_out_y * dim_in_x * ch_in));;
        /* basically each time it process 4 entries */
        uint16_t  col_cnt_im2col = ch_in >> 2;

        for (int j=0 ; j < col_cnt_im2col; j++)
        {
          v4s inA = *((v4s*) pA);
          v4u inB = *((v4u*) pB);

          sum = SumDotp(inB, inA, sum);
          pA+=4;
          pB+=4;
        }
        col_cnt_im2col = ch_in & 0x3;
        while (col_cnt_im2col)
        {
          int8_t      inA1 = *pA++;
          uint8_t     inB1 = *pB++;
          asm volatile("": : :"memory");
          sum += inA1 * inB1;

          col_cnt_im2col--;
        }
        /* if activation layer follows batch normalization */
        if (flag_batch_norm && flag_relu)
        {
          *pOut = pulp_nn_bn_quant_u8(sum, *k1, *lambda1, out_shift);
          k1++;
          lambda1++;
          pOut++;
        }
        else
        {
          /* if there isn't batch normalization but there is activation layer */
          if(flag_relu == 1)
          {
            *pOut = pulp_nn_quant_u8(sum, out_mult, out_shift);
          }
          else
          {
            *pOut = (uint8_t) clip8(sum >> out_shift);
          }
          pOut++;
        }
      }
    }
  }
  pi_cl_team_barrier(0);
}
