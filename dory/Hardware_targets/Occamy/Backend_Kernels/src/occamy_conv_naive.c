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

void __attribute__ ((noinline)) occamy_conv_naive(
  kernel* kernel_i
) {
  ////// NAIVE CONVOLUTION ////////////
  if (snrt_cluster_compute_core_idx() == 0)
  {
	int input_x_index, input_y_index, input_index, kernel_index, output_index;
	float sum;
	float * k_start;
	float * lambda_start;
	k_start = kernel_i->kappa;
	lambda_start = kernel_i->lambda;
	float out_shit_2 = 2;
	for (int z = 0; z < (kernel_i->out_shift-1); z++)
		out_shit_2 *= 2;
	//// Extension of input image with padding
	for (int z = 0; z < (kernel_i->dim_out_y); z++)
	{
	  for (int j = 0; j < (kernel_i->dim_out_x); j++)
	  {
	    kernel_i->kappa = k_start;
	    kernel_i->lambda = lambda_start;
	    for (int i = 0; i < kernel_i->ch_out; i++)
	    {
	      sum = 0;
	      for (int m = 0; m < kernel_i->ch_in; m++)
	      {
	        for (int n = 0; n < kernel_i->dim_kernel_x; n++)
	        {
	          for (int t = 0; t < kernel_i->dim_kernel_y; t++)
	          {
	            input_x_index = j * kernel_i->stride_x + n;
	            input_y_index = z * kernel_i->stride_y + t;
	            input_index = m + input_x_index * kernel_i->ch_in + input_y_index * (kernel_i->dim_in_x + kernel_i->padding_x_left + kernel_i->padding_x_right) * kernel_i->ch_in;
	            kernel_index = m + n * kernel_i->ch_in + t * kernel_i->dim_kernel_x * kernel_i->ch_in + i * kernel_i->dim_kernel_x * kernel_i->dim_kernel_y * kernel_i->ch_in;
	            sum += kernel_i->pInBuffer[input_index] * kernel_i->pWeight[kernel_index];
	          }
	        }
	      }
	      output_index = i + j * kernel_i->ch_out + z * kernel_i->ch_out * kernel_i->dim_out_x;
	      if (kernel_i->flag_y_accumulate_start == 1)
	        kernel_i->pOutBuffer[output_index] = sum;
	      else
	        kernel_i->pOutBuffer[output_index] += sum;

	      if (kernel_i->flag_relu == 1 && kernel_i->flag_batch_norm == 1 && kernel_i->flag_y_accumulate_end == 1)
	        kernel_i->pOutBuffer[output_index] = pulp_nn_bn_quant(kernel_i->pOutBuffer[output_index], *kernel_i->kappa, *kernel_i->lambda, out_shit_2);
	      else if (kernel_i->flag_y_accumulate_end == 1)
	      {
	      	double x = (kernel_i->pOutBuffer[output_index] / out_shit_2);
			if (x < 0)
			  kernel_i->pOutBuffer[output_index] = 0;
			else if (x > 255)
			  kernel_i->pOutBuffer[output_index] = 255;
			else
			  kernel_i->pOutBuffer[output_index] = (int) x;
	      }
	      if (kernel_i->flag_relu == 1 && kernel_i->flag_batch_norm == 1)
	      {
	        kernel_i->kappa++; kernel_i->lambda++;
	      }
	    }
	  }
	}
  }
  snrt_cluster_hw_barrier();
}

