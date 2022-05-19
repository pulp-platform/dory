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

void __attribute__ ((noinline))  pulp_nn_add (
  uint8_t * Im_in_1,             // pointer to the input feature map1
  uint8_t * Im_in_2,             // pointer to the input feature map2
  uint8_t * Im_out,            // pointer to the output
  uint16_t out_mult1,            // paramter to requantize
  uint16_t out_mult2,            // paramter to requantize
  uint16_t out_shift,            // paramter to requantize
  uint16_t  dim_im_in_h,
  uint16_t  dim_im_in_w,
  uint16_t  ch_im_in          // number of channels of the IFM
);

void pulp_nn_avgpool (
  uint8_t *  Im_in,
  uint8_t * Im_out,
  int32_t   lambda,
  uint16_t  out_shift,
  int32_t   out_add,
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
  uint16_t  stride_y,
  int       flag_requant
);


void pulp_nn_conv_Ho_parallel(
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
);

void pulp_nn_conv_Co_parallel(
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
);

void pulp_nn_conv_HoWo_parallel(
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
);

void pulp_nn_pointwise_HoWo_parallel(
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
);

void pulp_nn_pointwise_Ho_parallel(
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
);

void pulp_nn_pointwise_Co_parallel(
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
);

void pulp_nn_linear(
  uint8_t *pInBuffer,
  int8_t *bias,
  uint8_t *pOutBuffer,
  int8_t *pWeights,
  int32_t *k,
  int32_t *lambda,
  uint16_t out_mult,
  int8_t out_shift,
  uint16_t dim_vec,
  uint16_t num_o_neurons,
  int flag_relu,
  int flag_batch_norm
);

void pulp_nn_linear_out_32(
  uint8_t *pInBuffer,
  int8_t *bias,
  int32_t *pOutBuffer,
  int8_t *pWeights,
  uint16_t dim_vec,
  uint16_t num_o_neurons
);

uint8_t * pulp_nn_matmul(
  const int8_t *  pWeight,
  uint8_t *       pInBuffer,
  uint16_t        ch_out,
  uint16_t        num_col_im2col,
  uint16_t        out_shift,
  uint16_t        out_mult,
  int32_t *       k,
  int32_t *       lambda,
  const int8_t *  bias,
  uint8_t *       pOut,
  uint8_t *       pOut2,
  int             flag_relu,
  int             flag_batch_norm
);

void pulp_nn_maxpool (
  uint8_t * Im_in,             // pointer to the input feature map
  uint8_t * Im_out,               // pointer to the output
  uint16_t  dim_im_in_x,       // spatial dimension of the input feature map
  uint16_t  dim_im_in_y,
  uint16_t  ch_im_in,          // number of channels of the IFM
  uint16_t  dim_im_out_x,        // reduced spatial dimension of output
  uint16_t  dim_im_out_y,
  uint16_t  dim_kernel_x,        // spatial dimension of the pooling filter
  uint16_t  dim_kernel_y,        // spatial dimension of the pooling filter
  uint16_t  padding_t,           // amount of padding
  uint16_t  padding_b,           // amount of padding
  uint16_t  padding_l,           // amount of padding
  uint16_t  padding_r,           // amount of padding
  uint16_t  stride_x,              // amount of stride
  uint16_t  stride_y
);

void pulp_nn_depthwise_generic(
  const uint8_t * Im_in,
  uint8_t *       bufferC,
  const int8_t *  bias,
  uint8_t *       Im_out,
  const int8_t *  wt,
  uint8_t *       bufferB,
  int32_t *       k,
  int32_t *       lambda,
  uint16_t        out_mult,
  uint16_t        out_shift,
  const uint16_t  dim_im_in_x,
  const uint16_t  dim_im_in_y,
  const uint16_t  ch_im_in,
  const uint16_t  dim_im_out_x,
  const uint16_t  dim_im_out_y,
  const uint16_t  ch_im_out,
  const uint16_t  dim_kernel_x,
  const uint16_t  dim_kernel_y,
  const uint16_t  padding_y_top,
  const uint16_t  padding_y_bottom,
  const uint16_t  padding_x_left,
  const uint16_t  padding_x_right,
  const uint16_t  stride_x,
  const uint16_t  stride_y,
  int8_t          FLAG_RELU,
  int8_t          FLAG_BATCH_NORM
);

void pulp_nn_depthwise_generic_less_4_weights(
  const uint8_t * Im_in,
  uint8_t *       bufferC,
  const int8_t *  bias,
  uint8_t *       Im_out,
  const int8_t *  wt,
  uint8_t *       bufferB,
  int32_t *       k,
  int32_t *       lambda,
  uint16_t        out_mult,
  uint16_t        out_shift,
  const uint16_t  dim_im_in_x,
  const uint16_t  dim_im_in_y,
  const uint16_t  ch_im_in,
  const uint16_t  dim_im_out_x,
  const uint16_t  dim_im_out_y,
  const uint16_t  ch_im_out,
  const uint16_t  dim_kernel_x,
  const uint16_t  dim_kernel_y,
  const uint16_t  padding_y_top,
  const uint16_t  padding_y_bottom,
  const uint16_t  padding_x_left,
  const uint16_t  padding_x_right,
  const uint16_t  stride_x,
  const uint16_t  stride_y,
  int8_t          FLAG_RELU,
  int8_t          FLAG_BATCH_NORM
);

void pulp_nn_depthwise_3x3_s1(
      uint8_t * In_Img,
      uint8_t * Out_Img,
      int R,
      int C,
      int CH,
      int p_l,
      int p_r,
      int8_t  * Kernel,
      int8_t out_shift,
      uint16_t out_mult,
      int32_t * kappa,
      int32_t * lambda,
      int flag_relu,
      int flag_batch_norm,
      unsigned int * memory_chan
);
