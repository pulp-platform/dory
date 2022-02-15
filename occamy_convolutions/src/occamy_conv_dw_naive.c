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

void __attribute__ ((noinline)) occamy_conv_dw_naive(
  const float * pInBuffer,
  const uint16_t  dim_in_x,
  const uint16_t  dim_in_y,
  const uint16_t  ch_in,
  const float *  pWeight,
  const uint16_t  ch_out,
  const uint16_t  dim_kernel_x,
  const uint16_t  dim_kernel_y,
  const uint16_t  padding_y_top,
  const uint16_t  padding_y_bottom,
  const uint16_t  padding_x_left,
  const uint16_t  padding_x_right,
  const uint16_t  stride_x,
  const uint16_t  stride_y,
  const int8_t *  bias,
  const uint16_t  bias_shift,
  const uint16_t  out_shift,
  const uint16_t  out_mult,
  float *       pOutBuffer,
  const uint16_t  dim_out_x,
  const uint16_t  dim_out_y,
  float *       k,
  float *       lambda,
  uint8_t *       pIm2ColBuffer,
  int             flag_relu,
  int             flag_batch_norm,
  int             flag_y_accumulate_start,
  int             flag_y_accumulate_end,
  unsigned int * memory_chan
) 
{
  ////// NAIVE CONVOLUTION ////////////
  if (snrt_cluster_compute_core_idx() == 0)
  {
	int input_x_index, input_y_index, input_index, kernel_index, output_index;
	float sum;
	float * k_start;
	float * lambda_start;
	k_start = k;
	lambda_start = lambda;
	float out_shit_2 = 2;
	for (int z = 0; z < (out_shift-1); z++)
		out_shit_2 *= 2;
	//// Extension of input image with padding
	for (int z = 0; z < (dim_out_y); z++)
	{
	  for (int j = 0; j < (dim_out_x); j++)
	  {
	    k = k_start;
	    lambda = lambda_start;
	    for (int i = 0; i < ch_out; i++)
	    {
	      sum = 0;
	      for (int n = 0; n < dim_kernel_x; n++)
	      {
	        for (int t = 0; t < dim_kernel_y; t++)
	        {
	          input_x_index = j * stride_x + n;
	          input_y_index = z * stride_y + t;
	          input_index = i + input_x_index * ch_in + input_y_index * (dim_in_x + padding_x_left + padding_x_right) * ch_in;
	          kernel_index = n+ t * dim_kernel_x + i * dim_kernel_x * dim_kernel_y;
	          sum += pInBuffer[input_index] * pWeight[kernel_index];
	        }
	      }
	      output_index = i + j * ch_out + z * ch_out * dim_out_x;
	      pOutBuffer[output_index] = sum;
	      if (flag_relu == 1 && flag_batch_norm == 1)
	      {
	        pOutBuffer[output_index] = pulp_nn_bn_quant(pOutBuffer[output_index], *k, *lambda, out_shit_2);
	        k++; lambda++;
	      }
	    }
	  }
	}
  }
  snrt_cluster_hw_barrier();
}