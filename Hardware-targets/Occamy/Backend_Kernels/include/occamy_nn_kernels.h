/*
 * pulp_nn_kernels.h
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


#include "dory.h"
#include "stdint.h"
void __attribute__((noinline)) occamy_conv_opt_fp64(
    kernel* k);

void __attribute__((noinline)) occamy_conv_opt_fp32(
    kernel* k);

/**
 * @brief implementation of a single-precision fp DEPTHWISE convolutional kernel
 * for DORY trials. Currently does a direct convolution without im2col.
 * The memory layout of input/output feature map is HxWxC, resp. CoxFhxFwxCi.
 * Fuses multiple layers together (Conv2d, Batchnorm, Relu) that can be enabled
 * with a flag
 *
 * @param pInBuffer pointer to the input feature map
 * @param dim_in_x width of input feature map
 * @param dim_in_y height of input feature map
 * @param ch_in number of input channels (SIMD restricts multiple of 2)
 * @param pWeight pointer to weights
 * @param ch_out number of output channels (must be equal ch_in)
 * @param dim_kernel_x width of kernel
 * @param dim_kernel_y height of kernel
 * @param padding_y_top number of pixels padded on the top
 * @param padding_y_bottom number of pixels padded on the bottom
 * @param padding_x_left number of pixels padded on the left
 * @param padding_x_right number of pixels padded on the right
 * @param stride_x stride in x direction
 * @param stride_y stride in y direction
 * @param bias bias of convolution (currently not used)
 * @param bias_shift bias shift of convolution (currently not used)
 * @param out_shift shift factor for requantization (not used for floating
 * point)
 * @param out_mult mult factor for requantization (not used for floating point)
 * @param pOutBuffer pointer to output feature map
 * @param dim_out_x width of output feature map
 * @param dim_out_y height of output feature map
 * @param k multiplication factor for BatchNorm
 * @param lambda bias for BatchNorm
 * @param flag_relu RELU activation flag
 * @param flag_batch_norm BatchNorm flag
 * @param flag_y_accumulate_start indicates that output feature map is
 * initizialized with zeros
 * @param flag_y_accumulate_end indicates that BN, RELU can be performed
 * @param memory_chan Not used
 */
void __attribute__((noinline)) occamy_conv_dw_opt_fp32(
    kernel* k);

/**
 * @brief implementation of a single-precision fp convolutional kernel
 * for DORY trials. Currently does a direct convolution without im2col.
 * The memory layout of input feature map is C x H x W, resp. Co x Fh x Fw x Ci
 * for weights Howevever, the output memory layout is H x W x C. This kernel
 * should be used for the first layers in a network where Ci is very small and
 * usually odd numbered. Fuses multiple layers together (Conv2d, Batchnorm,
 * Relu) that can be enabled with a flag
 *
 * @param pInBuffer pointer to the input feature map
 * @param dim_in_x width of input feature map
 * @param dim_in_y height of input feature map
 * @param ch_in number of input channels
 * @param pWeight pointer to weights
 * @param ch_out number of output channels
 * @param dim_kernel_x width of kernel (restricted to even numbers -> zero pad)
 * @param dim_kernel_y height of kernel
 * @param padding_y_top number of pixels padded on the top
 * @param padding_y_bottom number of pixels padded on the bottom
 * @param padding_x_left number of pixels padded on the left
 * @param padding_x_right number of pixels padded on the right
 * @param stride_x stride in x direction
 * @param stride_y stride in y direction
 * @param bias bias of convolution (currently not used)
 * @param bias_shift bias shift of convolution (currently not used)
 * @param out_shift shift factor for requantization (not used for floating
 * point)
 * @param out_mult mult factor for requantization (not used for floating point)
 * @param pOutBuffer pointer to output feature map
 * @param dim_out_x width of output feature map
 * @param dim_out_y height of output feature map
 * @param k multiplication factor for BatchNorm
 * @param lambda bias for BatchNorm
 * @param flag_relu RELU activation flag
 * @param flag_batch_norm BatchNorm flag
 * @param flag_y_accumulate_start indicates that output feature map is
 * initizialized with zeros
 * @param flag_y_accumulate_end indicates that BN, RELU can be performed
 * @param memory_chan Not used
 */
void __attribute__((noinline)) occamy_conv_chw_opt_fp32(
    kernel* k);

/**
 * @brief helper function that implements Batch Normalization and ReLU
 *
 * @param pBuffer pointer to the feature map
 * @param dim_x width of feature map
 * @param dim_y height of feature map
 * @param ch number of channels (SIMD restricts multiple of 2)
 * @param out_shift_mul_factor shift factor by multiplication (to be
 * removed)
 * @param out_clip clipping factor (to be removed)
 * @param k multiplication factor for BatchNorm
 * @param lambda bias for BatchNorm
 * @param flag_relu RELU activation flag
 * @param flag_batch_norm BatchNorm flag
 * @param compute_id compute hart id
 * @param compute_num number of worker cores
 */

void __attribute__((noinline))
bn_relu(const float* pBuffer, const uint16_t dim_x, const uint16_t dim_y,
        const uint16_t ch, float* k, float* lambda, int flag_relu,
        int flag_batch_norm);

void occamy_conv_naive(
  kernel* k
);

void occamy_conv_naive_no_padding(
  const float *   pInBuffer,
  const uint16_t  dim_in_x,
  const uint16_t  dim_in_y,
  const uint16_t  ch_in,
  const float *   pWeight,
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
  float *         pOutBuffer,
  const uint16_t  dim_out_x,
  const uint16_t  dim_out_y,
  float *         k,
  float *         lambda,
  uint8_t *       pIm2ColBuffer,
  int             flag_relu,
  int             flag_batch_norm,
  int             flag_y_accumulate_start,
  int             flag_y_accumulate_end,
  unsigned int *  memory_chan
);
void __attribute__ ((noinline)) occamy_pool_naive(
  kernel* k
);