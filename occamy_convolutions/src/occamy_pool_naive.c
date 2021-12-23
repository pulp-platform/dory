/*
 * pulp_nn_conv_Ho_parallel.c
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
#include "occamy_nn_utils.h"
#include "occamy_nn_kernels.h"
#include "printf.h"

void __attribute__ ((noinline)) occamy_pool_naive(
  const float * pInBuffer,
  const uint16_t  dim_in_x,
  const uint16_t  dim_in_y,
  const uint16_t  ch_in,
  const uint16_t  ch_out,
  const uint16_t  dim_kernel_x,
  const uint16_t  dim_kernel_y,
  const uint16_t  padding_y_top,
  const uint16_t  padding_y_bottom,
  const uint16_t  padding_x_left,
  const uint16_t  padding_x_right,
  const uint16_t  stride_x,
  const uint16_t  stride_y,
  float *       pOutBuffer,
  const uint16_t  dim_out_x,
  const uint16_t  dim_out_y
) {
  ////// NAIVE CONVOLUTION ////////////
  int input_x_index, input_y_index, input_index, output_index;
  float max;
  //// Extension of input image with padding
  for (int z = 0; z < dim_out_y; z++)
  {
    for (int j = 0; j < dim_out_x; j++)
    {
      for (int i = 0; i < ch_out; i++)
      {
        max = 0;
        for (int n = 0; n < dim_kernel_x; n++)
        {
          for (int t = 0; t < dim_kernel_y; t++)
          {
            input_x_index = j * stride_x - padding_x_left + n;
            input_y_index = z * stride_y - padding_y_top + t;
            input_index = i + input_x_index * ch_in + input_y_index * dim_in_x * ch_in;
            if (input_x_index >= 0 && input_y_index >= 0 && input_x_index < dim_in_x && input_y_index < dim_in_y)
              if (pInBuffer[input_index] > max)
                max = pInBuffer[input_index];
          }
        }
        output_index = i + j * ch_out + z * ch_out * dim_out_x;
        pOutBuffer[output_index] = max;
      }
    }
  }
}