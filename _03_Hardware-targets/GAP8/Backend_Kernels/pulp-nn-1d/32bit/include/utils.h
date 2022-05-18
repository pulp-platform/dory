/*
 * utils.h
 * Angelo Garofalo <angelo.garofalo@unibo.it>
 * Nazareno Bruschi <nazareno.bruschi@studio.unibo.it>
 * Francesco Conti <f.conti@unibo.it>
 * 
 * Copyright (C) 2019 ETH Zurich, University of Bologna.
 * All rights reserved.
 */

#ifdef GAP_SDK
#include "pulp.h"
#endif

uint8_t pulp_nn_add_quant_u8 (
  uint8_t pix1,            
  uint8_t pix2,
  int16_t m1,
  int16_t m2,
  uint8_t  d
); 

void pulp_nn_compare_and_replace_if_larger_uint8(
	uint8_t * base,
	uint8_t * target,
	uint16_t  length
);

void pulp_zero_mem_dma(
	uint8_t * pBuffer,
	int       size,
	uint8_t * pZero
);

void pulp_zero_mem(
	uint8_t * pBuffer,
	int       size
);

void pulp_nn_im2col_uint8(
	uint8_t * pInput,
	uint8_t * pOutput,
	unsigned int blockSize
);

void pulp_nn_im2col_uint8_dmafree(
	uint8_t * pInput, 
	uint8_t * pOutput, 
	unsigned int blockSize
);

void pulp_nn_avg_and_replace_uint8(
  uint8_t * base,           // baseline for comparison
  uint8_t * target,         // compare target
  uint16_t length          // data size
);

uint8_t pulp_nn_bn_quant_u8 (
  int32_t phi,
  int32_t k,
  int32_t lambda,
  uint16_t  d
);

uint8_t pulp_nn_quant_u8(
  int32_t phi,
  int16_t m,
  uint16_t  d
);




