#include "rt/rt_api.h"
#include "utils.h"
#include "kernels.h"
#include "mchan_test.h"

#define max4(a,b)  		    __builtin_pulp_maxu4(a,b)
#define avg4(a,b)         __builtin_pulp_avg4(a,b)
#define clip8(x)                __builtin_pulp_clipu_r(x, 255)

uint8_t pulp_nn_add_quant_u8 (
  uint8_t pix1,            
  uint8_t pix2,
  int16_t m1,
  int16_t m2,
  uint8_t  d
) {
  /* Integer Batch Normalization */
  int32_t integer_image = pix1*m1 + pix2*m2;
  /* Quantization */
  int16_t x = (integer_image) >> d;
  uint8_t res = clip8(x);
  return res;
}



void pulp_nn_compare_and_replace_if_larger_uint8(uint8_t * base,
						                                    uint8_t * target,
						                                    uint16_t length)
{
  uint8_t *pIn = base;
  uint8_t *pCom = target;
  v4u inp;
  v4u com;
  uint16_t cnt = length >> 2;

  while(cnt > 0u)
  {
    inp = *((v4u*)pIn);
    com = *((v4u*)pCom);
    pCom+=4;

    *((v4u*)pIn) = max4(inp, com);
    pIn+=4;
    cnt--;
  }

  uint16_t left = length & 0x3;
  while (left>0u)
  {
    if(*pIn<*pCom)
      *pIn=*pCom;
    pIn++;
    pCom++;
    left--;
  }
}


void pulp_nn_avg_and_replace_uint8(uint8_t * base,
                                  uint8_t * target,
                                  uint16_t length)
{
  uint8_t *pIn = base;
  uint8_t *pCom = target;
  v4s inp;
  v4s com;
  uint16_t cnt = length >> 2;

  while(cnt > 0u)
  {
    inp = *((v4s*)pIn);
    com = *((v4s*)pCom);
    pCom+=4;

    *((v4s*)pIn) = avg4(inp, com);
    pIn+=4;
    cnt--;
  }
}


void pulp_zero_mem(uint8_t * pBuffer, int size)
{
  v4s* pDst = (v4s *)pBuffer;
  int lfover = size &0x3;
    for (int i=0; i<(size>>2); i++)
    {
      *((v4s*) pBuffer) = (v4s){0,0,0,0};
        pBuffer+=4;
    }
    while(lfover)
    {
      *pBuffer++=0;
      lfover--;
    }
}

void pulp_zero_mem_dma(uint8_t * pBuffer, int size, uint8_t * pZero){
	mchan_transfer(size, 1, 1, 0, 1, 0, 0, (unsigned int) pZero, (unsigned int) pBuffer, 0, 0);
}

void pulp_nn_im2col_uint8_dmafree(uint8_t * pInput, uint8_t * pOutput, unsigned int blockSize)
{
  unsigned int blkCnt = blockSize >> 2u;
  unsigned int lfover = blockSize & 0x3;

  for (unsigned int i = 0; i<blkCnt; i++)
  {
    *((v4s*)pOutput) = *((v4s*) pInput);
    pInput+=4;
    pOutput+=4;
  }
  while(lfover)
  {
    *((uint8_t*)pOutput) = *((uint8_t*)pInput);
    pOutput++;
    pInput++;
    lfover--;
  }
}


void pulp_nn_im2col_uint8(uint8_t * pInput, uint8_t * pOutput, unsigned int blockSize)
{
  	mchan_transfer(blockSize, 1, 1, 0, 1, 0, 0, (unsigned int) pInput, (unsigned int) pOutput, 0, 0);
}


uint8_t pulp_nn_bn_quant_u8 (
  int32_t phi,
  int32_t k,
  int32_t lambda,
  uint16_t  d
) {
  /* Integer Batch Normalization */
  int32_t integer_image_phi = (k * phi) + lambda;
  /* Quantization */
  int32_t x = integer_image_phi >> d;
  uint8_t res = clip8(x);
  return res;
}


uint8_t pulp_nn_quant_u8(
  int32_t phi,
  int16_t m,
  uint16_t  d
) {
  /* Quantization */
  int32_t x = (m * phi) >> d;
  uint8_t res = clip8(x);
  return res;
}



