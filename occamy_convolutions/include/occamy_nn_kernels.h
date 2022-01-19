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


#include "stdint.h"

void __attribute__((noinline)) occamy_conv_opt_fp64(
    const double* pInBuffer, const uint16_t dim_in_x, const uint16_t dim_in_y,
    const uint16_t ch_in, const double* pWeight, const uint16_t ch_out,
    const uint16_t dim_kernel_x, const uint16_t dim_kernel_y,
    const uint16_t padding_y_top, const uint16_t padding_y_bottom,
    const uint16_t padding_x_left, const uint16_t padding_x_right,
    const uint16_t stride_x, const uint16_t stride_y, const int8_t* bias,
    const uint16_t bias_shift, const uint16_t out_shift,
    const uint16_t out_mult, double* pOutBuffer, const uint16_t dim_out_x,
    const uint16_t dim_out_y, double* k, double* lambda, double* pIm2ColBuffer,
    int flag_relu, int flag_batch_norm, int flag_y_accumulate_start,
    int flag_y_accumulate_end, unsigned int* memory_chan);

void __attribute__((noinline)) occamy_conv_opt_fp32(
    const float* pInBuffer, 
    const uint16_t dim_in_x, 
    const uint16_t dim_in_y,
    const uint16_t ch_in, 
    const float* pWeight, 
    const uint16_t ch_out,
    const uint16_t dim_kernel_x, 
    const uint16_t dim_kernel_y,
    const uint16_t padding_y_top, 
    const uint16_t padding_y_bottom,
    const uint16_t padding_x_left, 
    const uint16_t padding_x_right,
    const uint16_t stride_x, 
    const uint16_t stride_y, 
    const int8_t* bias,
    const uint16_t bias_shift, 
    const uint16_t out_shift,
    const uint16_t out_mult, 
    float* pOutBuffer, 
    const uint16_t dim_out_x,
    const uint16_t dim_out_y, 
    float* k, float* lambda, 
    float* pIm2ColBuffer,
    int flag_relu, 
    int flag_batch_norm, 
    int flag_y_accumulate_start,
    int flag_y_accumulate_end, 
    unsigned int* memory_chan);

void occamy_conv_naive(
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
);

void occamy_conv_naive_no_padding(
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
);
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
);