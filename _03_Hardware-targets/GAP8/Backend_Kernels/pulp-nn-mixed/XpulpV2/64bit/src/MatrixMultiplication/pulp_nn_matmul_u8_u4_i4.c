/*
 * pulp_nn_matmul_u8_u4_i4.c
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



uint8_t *pulp_nn_matmul_u8_u4_i4(
                        uint8_t *pIn,
                        int8_t *pBias,
                        uint8_t *pOut,
                        uint8_t *pOut2,
                        int8_t *pWeight,
                        int64_t *pKappa,
                        int64_t *pLambda,
                        uint16_t out_mult,
                        uint16_t out_shift,
                        uint16_t num_col_im2col,
                        uint16_t ch_out,
                        uint8_t flag_relu,
                        uint8_t flag_batch_norm)
{
  int8_t mask = 0xf0;
  int8_t n_mask = ~ mask;
  int8_t off = 0x04;
  v4s vecA[2];
  v4s vecA2[2];
  v4s vecA3[2];
  v4s vecA4[2];
  v4u vecB3;
  v4u vecB4;
  v4u vecB;
  v4u vecB2;

  uint16_t ch_out_r = ch_out >> 1;
  uint16_t num_col_im2col_w = num_col_im2col >> 1;

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

      pB+=8;
      pB2+=8;

      pA = pulp_nn_i4_to_i8(pA,vecA);

      sum = SumDotp4(vecB, vecA[0], sum);
      sum5 = SumDotp4(vecB2, vecA[0], sum5);

      sum = SumDotp4(vecB3, vecA[1], sum);
      sum5 = SumDotp4(vecB4, vecA[1], sum5);

      pA2 = pulp_nn_i4_to_i8(pA2,vecA2);

      sum2 = SumDotp4(vecB, vecA2[0], sum2);
      sum6 = SumDotp4(vecB2, vecA2[0], sum6);

      sum2 = SumDotp4(vecB3, vecA2[1], sum2);
      sum6 = SumDotp4(vecB4, vecA2[1], sum6);

      pA3 = pulp_nn_i4_to_i8(pA3,vecA3);

      sum3 = SumDotp4(vecB, vecA3[0], sum3);
      sum7 = SumDotp4(vecB2, vecA3[0], sum7);

      sum3 = SumDotp4(vecB3, vecA3[1], sum3);
      sum7 = SumDotp4(vecB4, vecA3[1], sum7);

      pA4 = pulp_nn_i4_to_i8(pA4,vecA4);

      sum4 = SumDotp4(vecB, vecA4[0], sum4);
      sum8 = SumDotp4(vecB2, vecA4[0], sum8);

      sum4 = SumDotp4(vecB3, vecA4[1], sum4);
      sum8 = SumDotp4(vecB4, vecA4[1], sum8);
    }
    uint16_t col_cnt_im2col = num_col_im2col & 0x7;
    while (col_cnt_im2col > 0)
    {
      int8_t inA = (int8_t) bitext((int) *pA, 4, 0);
      int8_t inA2 = (int8_t) bitext((int) *pA2, 4, 0);
      int8_t inA3 = (int8_t) bitext((int) *pA3, 4, 0);
      int8_t inA4 = (int8_t) bitext((int) *pA4, 4, 0);
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
      inA = (int8_t) bitext((int) *pA, 4, 4);
      inA2 = (int8_t) bitext((int) *pA2, 4, 4);
      inA3 = (int8_t) bitext((int) *pA3, 4, 4);
      inA4 = (int8_t) bitext((int) *pA4, 4, 4);
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
      col_cnt_im2col-=2;
    }
    if (flag_batch_norm && flag_relu)
    {
      sum = pulp_nn_bn_quant_u4(sum, *pKappa, *pLambda, out_shift);
      sum5 = pulp_nn_bn_quant_u4(sum5, *pKappa, *pLambda, out_shift);
      pKappa++;
      pLambda++;
      sum2 = pulp_nn_bn_quant_u4(sum2, *pKappa, *pLambda, out_shift);
      sum6 = pulp_nn_bn_quant_u4(sum6, *pKappa, *pLambda, out_shift);
      *pOut = bitins(sum, n_mask, sum2, mask, off);
      *pOut2 = bitins(sum5, n_mask, sum6, mask, off);
      pKappa++;
      pLambda++;
      pOut++;
      pOut2++;
      sum3 = pulp_nn_bn_quant_u4(sum3, *pKappa, *pLambda, out_shift);
      sum7 = pulp_nn_bn_quant_u4(sum7, *pKappa, *pLambda, out_shift);
      pKappa++;
      pLambda++;
      sum4 = pulp_nn_bn_quant_u4(sum4, *pKappa, *pLambda, out_shift);
      sum8 = pulp_nn_bn_quant_u4(sum8, *pKappa, *pLambda, out_shift);
      pKappa++;
      pLambda++;
      *pOut = bitins(sum3, n_mask, sum4, mask, off);
      *pOut2 = bitins(sum7, n_mask, sum8, mask, off);
      pOut++;
      pOut2++;
    }
    else
    {
      if (flag_relu == 1)
      {
        sum = pulp_nn_quant_u4(sum, out_mult, out_shift);
        sum2 = pulp_nn_quant_u4(sum2, out_mult, out_shift);
        *pOut = bitins(sum, n_mask, sum2, mask, off);
        pOut++;
        sum3 = pulp_nn_quant_u4(sum3, out_mult, out_shift);
        sum4 = pulp_nn_quant_u4(sum4, out_mult, out_shift);
        *pOut = bitins(sum3, n_mask, sum4, mask, off);
        pOut++;

        sum5 = pulp_nn_quant_u4(sum5, out_mult, out_shift);
        sum6 = pulp_nn_quant_u4(sum6, out_mult, out_shift);
        *pOut2 = bitins(sum5, n_mask, sum6, mask, off);
        pOut2++;
        sum7 = pulp_nn_quant_u4(sum7, out_mult, out_shift);
        sum8 = pulp_nn_quant_u4(sum8, out_mult, out_shift);
        *pOut2 = bitins(sum7, n_mask, sum8, mask, off);
        pOut2++;
      }
      else
      {
        sum = (uint8_t) clip4(sum >> out_shift);
        sum2 = (uint8_t) clip4(sum2 >> out_shift);
        *pOut = bitins(sum, n_mask, sum2, mask, off);
        pOut++;
        sum3 = (uint8_t) clip4(sum3 >> out_shift);
        sum4 = (uint8_t) clip4(sum4 >> out_shift);
        *pOut = bitins(sum3, n_mask, sum4, mask, off);
        pOut++;

        sum5 = (uint8_t) clip4(sum5 >> out_shift);
        sum6 = (uint8_t) clip4(sum6 >> out_shift);
        *pOut2 = bitins(sum5, n_mask, sum6, mask, off);
        pOut2++;
        sum7 = (uint8_t) clip4(sum7 >> out_shift);
        sum8 = (uint8_t) clip4(sum8 >> out_shift);
        *pOut2 = bitins(sum7, n_mask, sum8, mask, off);
        pOut2++;
      }
    }
    pA+=(3 * num_col_im2col_w);
  }
   uint16_t i = 0;
   while(chan_left)
  {
    uint8_t *pB = pIn;
    uint8_t *pB2 = (pB + num_col_im2col);
    int sum = 0;
    if (pBias != NULL)
      sum = ((int) (*pBias++));    
    int sum2 = sum;

    uint8_t out[2];
    uint8_t out2[2];
    for(int j=0; j < (num_col_im2col_w >> 2); j++)
    {
      vecB = *((v4u*)pB);
      vecB2 = *((v4u*)pB2);
      vecB3 = *((v4u*)(pB + 4));
      vecB4 = *((v4u*)(pB2 + 4));

      pA = pulp_nn_i4_to_i8(pA,vecA);

      sum = SumDotp4(vecB, vecA[0], sum);
      sum2 = SumDotp4(vecB2, vecA[0], sum2);

      sum = SumDotp4(vecB3, vecA[1], sum);
      sum2 = SumDotp4(vecB4, vecA[1], sum2);

      pB+=8;
      pB2+=8;
    }
    uint16_t col_cnt_im2col = num_col_im2col & 0x7;
    while(col_cnt_im2col > 0)
    {
      int8_t inA = (int8_t) bitext((int) *pA, 4, 0);
      uint8_t inB = *pB++;
      uint8_t inB2 = *pB2++;
      sum += inA * inB;
      sum2 += inA * inB2;
      inA = (int8_t) bitext((int) *pA, 4, 4);
      inB = *pB++;
      inB2 = *pB2++;
      sum += inA * inB;
      sum2 += inA * inB2;

      pA++;
      col_cnt_im2col-=2;
    }
    if (flag_batch_norm && flag_relu)
    {
      uint8_t i_o = i & 0x01;
      out[i_o] = pulp_nn_bn_quant_u4(sum, *pKappa, *pLambda, out_shift);
      out2[i_o] = pulp_nn_bn_quant_u4(sum2, *pKappa, *pLambda, out_shift);
      pKappa++;
      pLambda++;
      if(i_o == 0x01)
      {
        *pOut = bitins(out[0], n_mask, out[1], mask, off);
        *pOut2 = bitins(out2[0], n_mask, out2[1], mask, off);
        pOut++;
        pOut2++;
      }
    }
    else
    {
      if (flag_relu == 1)
      {
        uint8_t i_o = i & 0x01;
        out[i_o] = pulp_nn_quant_u4(sum, out_mult, out_shift);
        out2[i_o] = pulp_nn_quant_u4(sum2, out_mult, out_shift);
        if(i_o == 0x01)
        {
          *pOut = bitins(out[0], n_mask, out[1], mask, off);
          *pOut2 = bitins(out2[0], n_mask, out2[1], mask, off);
          pOut++;
          pOut2++;
        }
      }
      else
      {
        uint8_t i_o = i & 0x01;
        out[i_o] = (uint8_t) clip4(sum >> out_shift);
        out2[i_o] = (uint8_t) clip4(sum2 >> out_shift);
        if(i_o == 0x01)
        {
          *pOut = bitins(out[0], n_mask, out[1], mask, off);
          *pOut2 = bitins(out2[0], n_mask, out2[1], mask, off);
          pOut++;
          pOut2++;
        }
      }
    }
    i++;
    chan_left--;
  }
  pOut+=ch_out_r;
  return pOut;
}
