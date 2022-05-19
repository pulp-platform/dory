/*
 * pulp_nn_matmul_u8_u8_i2.c
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



uint8_t *pulp_nn_matmul_u8_u8_i2(
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
  v4s vecA[4];
  v4s vecA2[4];
  v4s vecA3[4];
  v4s vecA4[4];
  v4u vecB3;
  v4u vecB4;
  v4u vecB5;
  v4u vecB6;
  v4u vecB7;
  v4u vecB8;
  v4u vecB;
  v4u vecB2;

  uint16_t ch_out_r = ch_out;
  uint16_t num_col_im2col_w = num_col_im2col >> 2;

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
      vecB = *((v4u*)pB);
      vecB2 = *((v4u*)pB2);
      vecB3 = *((v4u*)(pB + 4));
      vecB4 = *((v4u*)(pB2 + 4));
      vecB5 = *((v4u*)(pB + 8));
      vecB6 = *((v4u*)(pB2 + 8));
      vecB7 = *((v4u*)(pB + 12));
      vecB8 = *((v4u*)(pB2 + 12));

      pB+=16;
      pB2+=16;

      pA = pulp_nn_i2_to_i8(pA,vecA);

      sum = SumDotp4(vecB, vecA[0], sum);
      sum5 = SumDotp4(vecB2, vecA[0], sum5);
      sum = SumDotp4(vecB3, vecA[1], sum);
      sum5 = SumDotp4(vecB4, vecA[1], sum5);
      sum = SumDotp4(vecB5, vecA[2], sum);
      sum5 = SumDotp4(vecB6, vecA[2], sum5);
      sum = SumDotp4(vecB7, vecA[3], sum);
      sum5 = SumDotp4(vecB8, vecA[3], sum5);

      pA2 = pulp_nn_i2_to_i8(pA2,vecA2);

      sum2 = SumDotp4(vecB, vecA2[0], sum2);
      sum6 = SumDotp4(vecB2, vecA2[0], sum6);
      sum2 = SumDotp4(vecB3, vecA2[1], sum2);
      sum6 = SumDotp4(vecB4, vecA2[1], sum6);
      sum2 = SumDotp4(vecB5, vecA2[2], sum2);
      sum6 = SumDotp4(vecB6, vecA2[2], sum6);
      sum2 = SumDotp4(vecB7, vecA2[3], sum2);
      sum6 = SumDotp4(vecB8, vecA2[3], sum6);

      pA3 = pulp_nn_i2_to_i8(pA3,vecA3);

      sum3 = SumDotp4(vecB, vecA3[0], sum3);
      sum7 = SumDotp4(vecB2, vecA3[0], sum7);
      sum3 = SumDotp4(vecB3, vecA3[1], sum3);
      sum7 = SumDotp4(vecB4, vecA3[1], sum7);
      sum3 = SumDotp4(vecB5, vecA3[2], sum3);
      sum7 = SumDotp4(vecB6, vecA3[2], sum7);
      sum3 = SumDotp4(vecB7, vecA3[3], sum3);
      sum7 = SumDotp4(vecB8, vecA3[3], sum7);

      pA4 = pulp_nn_i2_to_i8(pA4,vecA4);

      sum4 = SumDotp4(vecB, vecA4[0], sum4);
      sum8 = SumDotp4(vecB2, vecA4[0], sum8);
      sum4 = SumDotp4(vecB3, vecA4[1], sum4);
      sum8 = SumDotp4(vecB4, vecA4[1], sum8);
      sum4 = SumDotp4(vecB5, vecA4[2], sum4);
      sum8 = SumDotp4(vecB6, vecA4[2], sum8);
      sum4 = SumDotp4(vecB7, vecA4[3], sum4);
      sum8 = SumDotp4(vecB8, vecA4[3], sum8);
    }
    uint16_t col_cnt_im2col = num_col_im2col & 0xf;
    while (col_cnt_im2col > 0)
    {
      int8_t inA = (int8_t) bitext((int) *pA, 2, 0);
      int8_t inA2 = (int8_t) bitext((int) *pA2, 2, 0);
      int8_t inA3 = (int8_t) bitext((int) *pA3, 2, 0);
      int8_t inA4 = (int8_t) bitext((int) *pA4, 2, 0);
      uint8_t inB = *pB++;
      uint8_t inB2 = *pB2++;
      sum += inA * inB;
      sum2 += inA2 * inB;
      sum3 += inA3 * inB;
      sum4 += inA4 * inB;
      sum5 += inA * inB2;
      sum6 += inA2 * inB2;
      sum7 += inA3 * inB2;
      sum8 += inA4 * inB2;
      inA = (int8_t) bitext((int) *pA, 2, 2);
      inA2 = (int8_t) bitext((int) *pA2, 2, 2);
      inA3 = (int8_t) bitext((int) *pA3, 2, 2);
      inA4 = (int8_t) bitext((int) *pA4, 2, 2);
      inB = *pB++;
      inB2 = *pB2++;
      sum += inA * inB;
      sum2 += inA2 * inB;
      sum3 += inA3 * inB;
      sum4 += inA4 * inB;
      sum5 += inA * inB2;
      sum6 += inA2 * inB2;
      sum7 += inA3 * inB2;
      sum8 += inA4 * inB2;
      inA = (int8_t) bitext((int) *pA, 2, 4);
      inA2 = (int8_t) bitext((int) *pA2, 2, 4);
      inA3 = (int8_t) bitext((int) *pA3, 2, 4);
      inA4 = (int8_t) bitext((int) *pA4, 2, 4);
      inB = *pB++;
      inB2 = *pB2++;
      sum += inA * inB;
      sum2 += inA2 * inB;
      sum3 += inA3 * inB;
      sum4 += inA4 * inB;
      sum5 += inA * inB2;
      sum6 += inA2 * inB2;
      sum7 += inA3 * inB2;
      sum8 += inA4 * inB2;
      inA = (int8_t) bitext((int) *pA, 2, 6);
      inA2 = (int8_t) bitext((int) *pA2, 2, 6);
      inA3 = (int8_t) bitext((int) *pA3, 2, 6);
      inA4 = (int8_t) bitext((int) *pA4, 2, 6);
      inB = *pB++;
      inB2 = *pB2++;
      sum += inA * inB;
      sum2 += inA2 * inB;
      sum3 += inA3 * inB;
      sum4 += inA4 * inB;
      sum5 += inA * inB2;
      sum6 += inA2 * inB2;
      sum7 += inA3 * inB2;
      sum8 += inA4 * inB2;

      pA++;
      pA2++;
      pA3++;
      pA4++;
      col_cnt_im2col-=4;
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
      vecB = *((v4u*)pB);
      vecB2 = *((v4u*)pB2);
      vecB3 = *((v4u*)(pB + 4));
      vecB4 = *((v4u*)(pB2 + 4));
      vecB5 = *((v4u*)(pB + 8));
      vecB6 = *((v4u*)(pB2 + 8));
      vecB7 = *((v4u*)(pB + 12));
      vecB8 = *((v4u*)(pB2 + 12));

      pA = pulp_nn_i2_to_i8(pA,vecA);

      sum = SumDotp4(vecB, vecA[0], sum);
      sum2 = SumDotp4(vecB2, vecA[0], sum2);
      sum = SumDotp4(vecB3, vecA[1], sum);
      sum2 = SumDotp4(vecB4, vecA[1], sum2);
      sum = SumDotp4(vecB5, vecA[2], sum);
      sum2 = SumDotp4(vecB6, vecA[2], sum2);
      sum = SumDotp4(vecB7, vecA[3], sum);
      sum2 = SumDotp4(vecB8, vecA[3], sum2);

      pB+=16;
      pB2+=16;
    }
    uint16_t col_cnt_im2col = num_col_im2col & 0xf;
    while(col_cnt_im2col > 0)
    {
      int8_t inA = (int8_t) bitext((int) *pA, 2, 0);
      uint8_t inB = *pB++;
      uint8_t inB2 = *pB2++;
      sum += inA * inB;
      sum2 += inA * inB2;
      inA = (int8_t) bitext((int) *pA, 2, 2);
      inB = *pB++;
      inB2 = *pB2++;
      sum += inA * inB;
      sum2 += inA * inB2;
      inA = (int8_t) bitext((int) *pA, 2, 4);
      inB = *pB++;
      inB2 = *pB2++;
      sum += inA * inB;
      sum2 += inA * inB2;
      inA = (int8_t) bitext((int) *pA, 2, 6);
      inB = *pB++;
      inB2 = *pB2++;
      sum += inA * inB;
      sum2 += inA * inB2;

      pA++;
      col_cnt_im2col-=4;
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
