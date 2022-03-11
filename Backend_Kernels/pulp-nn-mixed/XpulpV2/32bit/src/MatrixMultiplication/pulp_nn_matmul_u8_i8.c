/*
 * pulp_nn_matmul_u8_i8.c
 * Nazareno Bruschi <nazareno.bruschi@unibo.it>
 *
 * Copyright (C) 2019-2020 University of Bologna
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

#include "pmsis.h"
#include "pulp_nn_utils.h"


uint8_t *pulp_nn_matmul_u8_i8(
                        uint8_t *pIn,
                        int8_t *pBias,
                        uint8_t *pOut,
                        uint8_t *pOut2,
                        int8_t *pWeight,
                        int32_t *pKappa,
                        int32_t *pLambda,
                        uint16_t out_mult,
                        uint16_t out_shift,
                        uint16_t num_col_im2col,
                        uint16_t ch_out,
                        uint8_t flag_relu,
                        uint8_t flag_batch_norm)
{
  v4s vecA;
  v4s vecA2;
  v4s vecA3;
  v4s vecA4;
  v4u vecB;
  v4u vecB2;

  uint16_t ch_out_r = ch_out;
  uint16_t num_col_im2col_w = num_col_im2col;

  //uint8_t *pOut2 = pOut + ch_out_r;
  int8_t *pA = pWeight;

  uint16_t chan_left = ch_out & 0x3;

  for(int i=0; i < (ch_out >> 2); i++)
  {
    uint8_t *pB =  pIn;
    uint8_t *pB2 = (pB + num_col_im2col);
    int8_t *pA2 = (pA + num_col_im2col_w);
    int8_t *pA3 = (pA2 + num_col_im2col_w);
    int8_t *pA4 = (pA3 + num_col_im2col_w);

    int sum = 0;
    int sum2 = 0;
    int sum3 = 0;
    int sum4 = 0;
    int sum5 = 0;
    int sum6 = 0;
    int sum7 = 0;
    int sum8 = 0;

    if (pBias != NULL)
    {
      sum = ((int) (*pBias++));
      sum2 = ((int) (*pBias++));      
      sum3 = ((int) (*pBias++));      
      sum4 = ((int) (*pBias++));

      sum5 = sum;
      sum6 = sum2;
      sum7 = sum3;
      sum8 = sum4;
    }

    for(int j=0; j<(num_col_im2col_w >> 2); j++)
    {
      vecA = *((v4s*)pA);
      vecA2 = *((v4s*)pA2);
      vecA3 = *((v4s*)pA3);
      vecA4 = *((v4s*)pA4);

      vecB = *((v4u*)pB);
      vecB2 = *((v4u*)pB2);

      sum = SumDotp4(vecB, vecA, sum );
      sum2 = SumDotp4(vecB, vecA2, sum2);
      sum3 = SumDotp4(vecB, vecA3, sum3);
      sum4 = SumDotp4(vecB, vecA4, sum4);

      sum5 = SumDotp4(vecB2, vecA, sum5);
      sum6 = SumDotp4(vecB2, vecA2, sum6);
      sum7 = SumDotp4(vecB2, vecA3, sum7);
      sum8 = SumDotp4(vecB2, vecA4, sum8);

      pA+=4;
      pA2+=4;
      pA3+=4;
      pA4+=4;

      pB+=4;
      pB2+=4;
    }
    uint16_t col_cnt_im2col = num_col_im2col & 0x3;
    while (col_cnt_im2col)
    {
      int8_t inA = *pA++;
      int8_t inA2 = *pA2++;
      int8_t inA3 = *pA3++;
      int8_t inA4 = *pA4++;
      uint8_t inB = *pB++;
      uint8_t inB2 = *pB2++;
      asm volatile("": : :"memory");
      sum += inA * inB;
      sum2 += inA2 * inB;
      sum3 += inA3 * inB;
      sum4 += inA4 * inB;
      sum5 += inA * inB2;
      sum6 += inA2 * inB2;
      sum7 += inA3 * inB2;
      sum8 += inA4 * inB2;

      col_cnt_im2col--;
    }
    if (flag_batch_norm && flag_relu)
    {
      *pOut = pulp_nn_bn_quant_u8(sum, *pKappa, *pLambda, out_shift);
      pOut++;
      *pOut2 = pulp_nn_bn_quant_u8(sum5, *pKappa, *pLambda, out_shift);
      pOut2++;
      pKappa++;
      pLambda++;

      *pOut = pulp_nn_bn_quant_u8(sum2, *pKappa, *pLambda, out_shift);
      pOut++;
      *pOut2 = pulp_nn_bn_quant_u8(sum6, *pKappa, *pLambda, out_shift);
      pOut2++;
      pKappa++;
      pLambda++;

      *pOut = pulp_nn_bn_quant_u8(sum3, *pKappa, *pLambda, out_shift);
      pOut++;
      *pOut2 = pulp_nn_bn_quant_u8(sum7, *pKappa, *pLambda, out_shift);
      pOut2++;
      pKappa++;
      pLambda++;

      *pOut = pulp_nn_bn_quant_u8(sum4, *pKappa, *pLambda, out_shift);
      pOut++;
      *pOut2 = pulp_nn_bn_quant_u8(sum8, *pKappa, *pLambda, out_shift);
      pOut2++;
      pKappa++;
      pLambda++;
    }
    else
    {
      if (flag_relu == 1)
      {
        *pOut = pulp_nn_quant_u8(sum, out_mult, out_shift);
        pOut++;
        *pOut = pulp_nn_quant_u8(sum2, out_mult, out_shift);
        pOut++;
        *pOut = pulp_nn_quant_u8(sum3, out_mult, out_shift);
        pOut++;
        *pOut = pulp_nn_quant_u8(sum4, out_mult, out_shift);
        pOut++;

        *pOut2 = pulp_nn_quant_u8(sum5, out_mult, out_shift);
        pOut2++;
        *pOut2 = pulp_nn_quant_u8(sum6, out_mult, out_shift);
        pOut2++;
        *pOut2 = pulp_nn_quant_u8(sum7, out_mult, out_shift);
        pOut2++;
        *pOut2 = pulp_nn_quant_u8(sum8, out_mult, out_shift);
        pOut2++;
      }
      else
      {
        *pOut = (uint8_t) clip8(sum >> out_shift);
        pOut++;
        *pOut = (uint8_t) clip8(sum2 >> out_shift);
        pOut++;
        *pOut = (uint8_t) clip8(sum3 >> out_shift);
        pOut++;
        *pOut = (uint8_t) clip8(sum4 >> out_shift);
        pOut++;

        *pOut2 = (uint8_t) clip8(sum5 >> out_shift);
        pOut2++;
        *pOut2 = (uint8_t) clip8(sum6 >> out_shift);
        pOut2++;
        *pOut2 = (uint8_t) clip8(sum7 >> out_shift);
        pOut2++;
        *pOut2 = (uint8_t) clip8(sum8 >> out_shift);
        pOut2++;
      }
    }
    pA+=(3 * num_col_im2col_w);
  }
   while(chan_left)
  {
    uint8_t *pB = pIn;
    uint8_t *pB2 = (pB + num_col_im2col);
    int sum = 0;
    if (pBias != NULL)
      sum = ((int) (*pBias++));    
    int sum2 = sum;

    for(int j=0; j < (num_col_im2col_w >> 2); j++)
    {
      vecA = *((v4s*) pA);
      vecB = *((v4u*) pB);
      vecB2 = *((v4u*) pB2);

      sum = SumDotp4(vecB, vecA, sum);
      sum2 = SumDotp4(vecB2, vecA, sum2);

      pA+=4;
      pB+=4;
      pB2+=4;
    }
    uint16_t col_cnt_im2col = num_col_im2col & 0x3;
    while(col_cnt_im2col)
    {
      int8_t inA = *pA++;
      uint8_t inB = *pB++;
      uint8_t inB2 = *pB2++;
      asm volatile("": : :"memory");
      sum += inA * inB;
      sum2 += inA * inB2;

      col_cnt_im2col--;
    }
    if (flag_batch_norm && flag_relu)
    {
      *pOut = pulp_nn_bn_quant_u8(sum, *pKappa, *pLambda, out_shift);
      pOut++;
      *pOut2 = pulp_nn_bn_quant_u8(sum2, *pKappa, *pLambda, out_shift);
      pOut2++;
      pKappa++;
      pLambda++;
    }
    else
    {
      if (flag_relu == 1)
      {
        *pOut = pulp_nn_quant_u8(sum, out_mult, out_shift);
        pOut++;
        *pOut2 = pulp_nn_quant_u8(sum2, out_mult, out_shift);
        pOut2++;
      }
      else
      {
        *pOut = (uint8_t) clip8(sum >> out_shift);
        pOut++;
        *pOut2 = (uint8_t) clip8(sum2 >> out_shift);
        pOut2++;
      }
    }
    chan_left--;
  }
  pOut+=ch_out_r;
  return pOut;
}
