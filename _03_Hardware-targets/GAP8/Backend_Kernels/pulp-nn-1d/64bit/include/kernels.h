/*
 * kernels.h
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

#include "pulp.h"

void __attribute__ ((noinline))  pulp_nn_add (
  uint8_t * Im_in_1,             // pointer to the input feature map1
  uint8_t * Im_in_2,             // pointer to the input feature map2
  uint16_t  ch_im_in,          // number of channels of the IFM
  uint16_t  dim_im_in_h,
  uint16_t  dim_im_in_w,
  uint8_t * Im_out,            // pointer to the output
  uint16_t out_mult1,            // paramter to requantize
  uint16_t out_mult2,            // paramter to requantize
  uint16_t out_mult3,            // paramter to requantize
  uint16_t out_shift,            // paramter to requantize
  uint16_t out_shift2            // paramter to requantize
);

void __attribute__ ((noinline))  pulp_nn_1D_avgpool (
  uint8_t *  Im_in,            // pointer to the input feature map
  uint16_t  dim_im_in_x,      // spatial dimension of the input feature map
  uint16_t  ch_im_in,         // number of channels of the IFM
  uint16_t  dim_kernel_x,       // spatial dimension of the pooling filter
  uint16_t  padding,          // amount of padding
  uint16_t  stride,           // amount of stride
  uint16_t  dim_im_out_x,     // reduced spatial dimension of output
  int8_t *  bufferA,          // actually not used in this fx
  uint8_t *  Im_out,           // pointer to the output
  int32_t * pOutBufferAcc,
  int8_t    flag_acc_buff_out,
  int8_t    flag_first_ch_out,
  int       flag_relu,
  const uint16_t  out_shift,
  const uint16_t  out_mult
);

void pulp_nn_linear_out_32(
               uint8_t *pInBuffer,
               int8_t *pWeights,
               uint16_t dim_vec,
               uint16_t num_o_neurons,
               int8_t *bias,
               uint16_t bias_shift,
               int8_t out_shift,
               uint16_t out_mult,
               int64_t *k,
               int64_t *lambda,
               int32_t *pOutBuffer,
               int flag_relu,
               int flag_batch_norm,
               unsigned int * memory_chan
               );

uint8_t * pulp_nn_matmul(
  const int8_t *  pWeight,
  uint8_t *       pInBuffer,
  uint16_t        ch_out,
  uint16_t        num_col_im2col,
  uint16_t        bias_shift,
  uint16_t        out_shift,
  uint16_t        out_mult,
  int64_t *       k,
  int64_t *       lambda,
  const int8_t *  bias,
  uint8_t *       pOut,
  int             flag_relu,
  int             flag_batch_norm
);



void pulp_nn_convolution_1D_uint8_nopad_nodilation(
	const uint8_t * pInBuffer,
  	const uint16_t  dim_in_y,
	const uint16_t  ch_in,
  	const int8_t *  pWeight,
	const uint16_t  ch_out,
  	const uint16_t  dim_kernel_y,
  	const uint16_t  padding_y_top,
  	const uint16_t  padding_y_bottom,
  	const uint16_t  stride_y,
  	const int8_t *  bias,
  	const uint16_t  bias_shift,
  	const uint16_t  out_shift,
	const int16_t   out_mult,
	uint8_t *       pOutBuffer,
  	const uint16_t  dim_out_y,
	int64_t *       k,
  	int64_t *       lambda,
	uint8_t *       pIm2ColBuffer, 
	const uint16_t	dil,
  	uint8_t *       pReserved,
	int             flag_relu,
  	int             flag_batch_norm
);

uint8_t *pulp_nn_matmul_4x2_uint8_nopad_nodilation(
	int8_t * pWeight,
	uint8_t * pInBuffer,
	uint16_t ch_in,
	uint16_t ch_out,
	uint16_t  stride_y,
	uint16_t dim_ker,
	uint16_t bias_shift,
	uint16_t out_shift,
	int16_t out_mult,
  	int64_t * k,
  	int64_t * lambda,
	int8_t * bias,
	uint8_t * pOut,
	int flag_relu,
  	int flag_batch_norm
);

uint8_t *pulp_nn_matmul_4x1_uint8(
	int8_t * pWeight,
	uint8_t * pInBuffer,
	uint16_t ch_out,
	uint16_t dim_ker,
	uint16_t bias_shift,
	uint16_t out_shift,
	int16_t out_mult,
  	int64_t * k,
  	int64_t * lambda,
	int8_t * bias,
	uint8_t * pOut,
	int flag_relu,
  	int flag_batch_norm
);

uint8_t *pulp_nn_matmul_4x2_uint8(
	int8_t * pWeight,
	uint8_t * pInBuffer,
	uint16_t ch_out,
	uint16_t dim_ker,
	uint16_t bias_shift,
	uint16_t out_shift,
	int16_t out_mult,
  	int64_t * k,
  	int64_t * lambda,
	int8_t * bias,
	uint8_t * pOut,
	int flag_relu,
  	int flag_batch_norm
);

void pulp_nn_convolution_1D_uint8(
	const uint8_t * pInBuffer,
  	const uint16_t  dim_in_y,
	const uint16_t  ch_in,
  	const int8_t *  pWeight,
	const uint16_t  ch_out,
  	const uint16_t  dim_kernel_y,
  	const uint16_t  padding_y_top,
  	const uint16_t  padding_y_bottom,
  	const uint16_t  stride_y,
  	const int8_t *  bias,
  	const uint16_t  bias_shift,
  	const uint16_t  out_shift,
	const int16_t  out_mult,
	uint8_t *       pOutBuffer,
  	const uint16_t  dim_out_y,
	int64_t *       k,
  	int64_t *       lambda,
	uint8_t *       pIm2ColBuffer, 
	const uint16_t	dil,
  	uint8_t *        pReserved,
	int             flag_relu,
  	int             flag_batch_norm
);

void pulp_nn_indirect_convolution_1D_uint8(
	const uint8_t * pInBuffer,
  	const uint16_t  dim_in_y,
	const uint16_t  ch_in,
  	const int8_t *  pWeight,
	const uint16_t  ch_out,
  	const uint16_t  dim_kernel_y,
  	const uint16_t  padding_y_top,
  	const uint16_t  padding_y_bottom,
  	const uint16_t  stride_y,
  	const int8_t *  bias,
  	const uint16_t  bias_shift,
  	const uint16_t  out_shift,
	const int16_t   out_mult,
	uint8_t *       pOutBuffer,
  	const uint16_t  dim_out_y,
	int64_t *       k,
  	int64_t *       lambda,
	uint8_t **      pIndBuffer,
	const uint16_t	dil,
  	uint8_t *       pReserved,
	int             flag_relu,
  	int             flag_batch_norm
);

uint8_t *pulp_nn_matmul_4x1_uint8_indirect(
	int8_t * pWeight,
	uint8_t ** pInBuffer,
	uint16_t ch_out,
	const uint16_t  dim_kernel_y,
	const uint16_t  ch_in,
	uint16_t dim_ker,
	uint16_t bias_shift,
	uint16_t out_shift,
	int16_t out_mult,
  	int64_t * k,
  	int64_t * lambda,
	int8_t * bias,
	uint8_t * pOut,
	int flag_relu,
  	int flag_batch_norm
);

uint8_t *pulp_nn_matmul_4x2_uint8_indirect(
	int8_t * pWeight,
	uint8_t ** pInBuffer,
	uint16_t ch_out,
	const uint16_t  dim_kernel_y,
	const uint16_t  ch_in,
	uint16_t dim_ker,
	uint16_t bias_shift,
	uint16_t out_shift,
	int16_t out_mult,
  	int64_t * k,
  	int64_t * lambda,
	int8_t * bias,
	uint8_t * pOut,
	int flag_relu,
  	int flag_batch_norm
);