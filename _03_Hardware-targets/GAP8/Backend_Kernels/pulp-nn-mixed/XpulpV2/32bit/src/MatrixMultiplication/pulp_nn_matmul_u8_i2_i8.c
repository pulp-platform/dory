/*
 * pulp_nn_matmul_u8_i2_i8.c
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



int8_t *pulp_nn_matmul_u8_i2_i8(
                        uint8_t *pIn,
                        int8_t *pBias,
                        int8_t *pOut,
                        int8_t *pOut2,
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
  int8_t mask2 = 0x0c;
  int8_t n_mask2 = ~ mask2;
  int8_t mask4 = 0x30;
  int8_t n_mask4 = ~ mask4;
  int8_t mask6 = 0xc0;
  int8_t n_mask6 = ~ mask6;
  int8_t off2 = 2;
  int8_t off4 = 4;
  int8_t off6 = 6;
  v4s vecA;
  v4s vecA2;
  v4s vecA3;
  v4s vecA4;
  v4u vecB;
  v4u vecB2;

  uint16_t ch_out_r = ch_out >> 2;
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
    while (col_cnt_im2col > 0)
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
      sum = pulp_nn_bn_quant_i2(sum, *pKappa, *pLambda, out_shift);
      sum5 = pulp_nn_bn_quant_i2(sum5, *pKappa, *pLambda, out_shift);
      pKappa++;
      pLambda++;
      sum2 = pulp_nn_bn_quant_i2(sum2, *pKappa, *pLambda, out_shift);
      sum6 = pulp_nn_bn_quant_i2(sum6, *pKappa, *pLambda, out_shift);
      pKappa++;
      pLambda++;
      sum3 = pulp_nn_bn_quant_i2(sum3, *pKappa, *pLambda, out_shift);
      sum7 = pulp_nn_bn_quant_i2(sum7, *pKappa, *pLambda, out_shift);
      pKappa++;
      pLambda++;
      sum4 = pulp_nn_bn_quant_i2(sum4, *pKappa, *pLambda, out_shift);
      sum8 = pulp_nn_bn_quant_i2(sum8, *pKappa, *pLambda, out_shift);
      pKappa++;
      pLambda++;
      sum = bitins(sum, n_mask2, sum2, mask2, off2);
      sum = bitins(sum, n_mask4, sum3, mask4, off4);
      *pOut = bitins(sum, n_mask6, sum4, mask6, off6);
      sum5 = bitins(sum5, n_mask2, sum6, mask2, off2);
      sum5 = bitins(sum5, n_mask4, sum7, mask4, off4);
      *pOut2 = bitins(sum5, n_mask6, sum8, mask6, off6);
      pOut2++;
      pOut++;
    }
    else
    {
      if (flag_relu == 1)
      {
        sum = pulp_nn_quant_i2(sum, out_mult, out_shift);
        sum2 = pulp_nn_quant_i2(sum2, out_mult, out_shift);
        sum3 = pulp_nn_quant_i2(sum3, out_mult, out_shift);
        sum4 = pulp_nn_quant_i2(sum4, out_mult, out_shift);
        sum = bitins(sum, n_mask2, sum2, mask2, off2);
        sum = bitins(sum, n_mask4, sum3, mask4, off4);
        *pOut = bitins(sum, n_mask6, sum4, mask6, off6);
        pOut++;
        sum5 = pulp_nn_quant_i2(sum5, out_mult, out_shift);
        sum6 = pulp_nn_quant_i2(sum6, out_mult, out_shift);
        sum7 = pulp_nn_quant_i2(sum7, out_mult, out_shift);
        sum8 = pulp_nn_quant_i2(sum8, out_mult, out_shift);
        sum5 = bitins(sum5, n_mask2, sum6, mask2, off2);
        sum5 = bitins(sum5, n_mask4, sum7, mask4, off4);
        *pOut2 = bitins(sum5, n_mask6, sum8, mask6, off6);
        pOut2++;
      }
      else
      {
        sum = (uint8_t) clip2(sum >> out_shift);
        sum2 = (uint8_t) clip2(sum2 >> out_shift);
        sum3 = (uint8_t) clip2(sum3 >> out_shift);
        sum4 = (uint8_t) clip2(sum4 >> out_shift);
        sum = bitins(sum, n_mask2, sum2, mask2, off2);
        sum = bitins(sum, n_mask4, sum3, mask4, off4);
        *pOut = bitins(sum, n_mask6, sum4, mask6, off6);
        pOut++;

        sum5 = (uint8_t) clip2(sum5 >> out_shift);
        sum6 = (uint8_t) clip2(sum6 >> out_shift);
        sum7 = (uint8_t) clip2(sum7 >> out_shift);
        sum8 = (uint8_t) clip2(sum8 >> out_shift);
        sum5 = bitins(sum5, n_mask2, sum6, mask2, off2);
        sum5 = bitins(sum5, n_mask4, sum7, mask4, off4);
        *pOut2 = bitins(sum5, n_mask6, sum8, mask6, off6);
        pOut2++;
      }
    }
    pA+=(3 * num_col_im2col_w);
  }
  pOut+=ch_out_r;
  return pOut;
}
