/*
 * kernels.h
 * Angelo Garofalo <angelo.garofalo@unibo.it>
 * Nazareno Bruschi <nazareno.bruschi@studio.unibo.it>
 * Francesco Conti <f.conti@unibo.it>
 * 
 * Copyright (C) 2019 ETH Zurich, University of Bologna.
 * All rights reserved.
 */
void pulp_nn_add_u8_1D (
	uint8_t * Im_in_1,             // pointer to the input feature map1
	uint8_t * Im_in_2,             // pointer to the input feature map2
	uint16_t  ch_im_in,          // number of channels of the IFM
	uint16_t  dim_im_in_h,      
	uint8_t * Im_out,            // pointer to the output
	int16_t out_mult1,            // paramter to requantize
	int16_t out_mult2,            // paramter to requantize
	uint16_t out_shift            // paramter to requantize
);

void pulp_nn_avgpool_u8_1D (
  uint8_t *  Im_in,            // pointer to the input feature map
  uint16_t  dim_im_in_y,	// spatial dimension of the input feature map
  uint16_t  ch_im_in,         // number of channels of the IFM
  uint16_t  dim_kernel_y,       // spatial dimension of the pooling filter
  uint16_t  padding,          // amount of padding
  uint16_t  stride,           // amount of stride     
  uint16_t  dim_im_out_y,	// reduced spatial dimension of output
  uint8_t *  Im_out,           // pointer to the output
  uint32_t * pOutBufferAcc,
  int8_t    flag_acc_buff_out,
  int8_t    flag_first_ch_out,
  int       flag_relu,
  const uint16_t  out_shift,
  const int16_t  out_mult
);

void pulp_nn_linear_uint8(
	uint8_t *pIn,
	int8_t *pWeights,
	uint16_t dim_vec,
	uint16_t num_o_neurons,
	uint16_t bias_shift,
	uint16_t out_shift,
	int8_t *bias,
	uint8_t *pOut
);

void pulp_nn_maxpool_u8_1D (
	uint8_t * Im_in,             // pointer to the input feature map
	uint16_t  dim_im_in_y,		// spatial dimension of the input feature map
	uint16_t  ch_im_in,          // number of channels of the IFM
	uint16_t  dim_kernel,        // spatial dimension of the pooling filter
	uint16_t  stride,            // amount of stride
	uint16_t  dim_im_out_y,	     // reduced spatial dimension of output
	uint8_t * Im_out            // pointer to the output
);

void pulp_nn_relu_u8_1D(
  int8_t * data,
  uint16_t dim_im_in_y,
  uint16_t ch_im_in,
  uint8_t * out
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

void pulp_nn_convolution_1D_uint8_lite(
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
	const uint8_t * zero,
	const uint16_t	dil,
  	uint8_t *        pReserved,
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

void pulp_nn_convolution_1D_uint8_2buff(
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
	uint8_t *       k,
  	uint8_t *       lambda,
	uint8_t *       pIm2ColBuffer1, 
	uint8_t *       pIm2ColBuffer2,
	const uint8_t * zero,
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

unsigned long pulp_nn_indirect_convolution_1D_uint8_profile(
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
	uint8_t **       pIndBuffer,
	const uint8_t * zero, 
	const uint16_t	dil,
  	uint8_t *        pReserved,
	int             flag_relu,
  	int             flag_batch_norm
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

unsigned long pulp_nn_convolution_1D_uint8_profile(
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
	const uint8_t * zero, 
	const uint16_t	dil,
  	uint8_t *        pReserved,
	int             flag_relu,
  	int             flag_batch_norm
);

