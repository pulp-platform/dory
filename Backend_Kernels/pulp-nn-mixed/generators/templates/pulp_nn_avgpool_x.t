/*
 * ${config.filename}
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


void __attribute__ ((noinline)) ${config.fn_name}(
  uint8_t * pIn,
  uint8_t * pOut,
  uint16_t  dim_im_in_x,
  uint16_t  dim_im_in_y,
  uint16_t  ch_im_in,
  uint16_t  dim_im_out_x,
  uint16_t  dim_im_out_y,
  uint16_t  dim_kernel_x,
  uint16_t  dim_kernel_y,
  uint16_t  padding_t,
  uint16_t  padding_b,
  uint16_t  padding_l,
  uint16_t  padding_r,
  uint16_t  stride_x,
  uint16_t  stride_y)
{
  /* parallelization */
  int core_id = pi_core_id();
  int n_cores = NUM_CORES;
  int Log2Core = log2(n_cores);
  int chunck = (dim_im_in_y >> Log2Core) + (dim_im_in_y & (n_cores -1)!=0);
  int start = chunck * core_id;
  int stop = min(start + chunck, dim_im_in_y);
  int   i_x, i_y;
%if config.kernel.in_data_t == 8:
  int ch_im_in_r = ch_im_in;
%elif config.kernel.in_data_t == 4:
  int ch_im_in_r = ch_im_in >> 1;
%elif config.kernel.in_data_t == 2:
  int ch_im_in_r = ch_im_in >> 2;
%endif

  /* start kernel: pooling along the x axis */
  for (i_y = start; i_y < stop; i_y++)
  {
    for (i_x = 0; i_x < dim_im_out_x; i_x++)
    {
      /* define target and the kernel windows */
      uint8_t     *target = pIn + (i_y * dim_im_in_x + i_x) * ch_im_in_r;
      uint8_t     *win_start;
      uint8_t     *win_stop;
      if (i_x * stride_x - padding_l < 0)
      {
        win_start = target;
      }
      else
      {
        win_start = pIn + (i_y * dim_im_in_x + i_x * stride_x - padding_l) * ch_im_in_r;
      }

      if (i_x * stride_x - padding_l + dim_kernel_x >= dim_im_in_x)
      {
        win_stop = pIn + (i_y * dim_im_in_x + dim_im_in_x) * ch_im_in_r;
      }
      else
      {
        win_stop = pIn + (i_y * dim_im_in_x + i_x * stride_x - padding_l + dim_kernel_x) * ch_im_in_r;
      }

      /* copy the data into target */
      for (int i = 0; i< ch_im_in_r; i++) target[i] = win_start[i];

      /* start the avg operation (comparison) */
        win_start += ch_im_in_r;
      for (; win_start < win_stop; win_start += ch_im_in_r)
      {
        ${config.comp_and_avg_fn}(target, win_start, ch_im_in_r);
      }
    }
  }

  /* synch barrier + parallelization for the second pooling phase */
  pi_cl_team_barrier(0);
  if (dim_im_out_y < NUM_CORES)
  {
    n_cores = dim_im_out_y;
  }
  Log2Core = log2(n_cores);
  int chunck2 = (dim_im_out_y >> Log2Core) + (dim_im_out_y & (n_cores -1)!=0);
  int start2 = chunck2 * core_id;
  int stop2 = min(start2 + chunck2, dim_im_out_y);

  /* pooling along y axis */
  for (i_y = start2; i_y < stop2; i_y++)
  {

    uint8_t     *target = pOut + i_y * dim_im_out_x * ch_im_in_r;
    uint8_t     *row_start;
    uint8_t     *row_end;

    /* define the starting row */
    if (i_y * stride_y - padding_t < 0)
    {
      row_start = pIn;
    }
    else
    {
      row_start = pIn + (i_y * stride_y - padding_t) * dim_im_in_x * ch_im_in_r;
    }

    /* define the stopping row */
    if (i_y * stride_y - padding_t + dim_kernel_y >= dim_im_in_y)
    {
      row_end = pIn + dim_im_in_x * dim_im_in_x * ch_im_in_r;
    }
    else
    {
      row_end = pIn + (i_y * stride_y - padding_t + dim_kernel_y) * dim_im_in_x * ch_im_in_r;
    }

    /* copy data of the first row*/
    for (int i = 0; i< dim_im_out_x * ch_im_in_r; i++) target[i] = row_start[i];

    /* move over to next row */
      row_start += ch_im_in_r * dim_im_in_x;

    for (; row_start < row_end; row_start += dim_im_in_x * ch_im_in_r)
    {
      ${config.comp_and_avg_fn}(target, row_start, dim_im_out_x * ch_im_in_r);
    }
  }

  pi_cl_team_barrier(0);
}