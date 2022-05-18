/*
 * xpulp_nn_matmul_i2_i2_i2_4x4.c
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



uint8_t * __attribute__((noinline)) xpulp_nn_matmul_i2_i2_i2_4x4(
                        int8_t *pIn,
                        int8_t *pBias,
                        int8_t *pOut,
                        int8_t *pOut2,
                        int8_t *pOut3,
                        int8_t *pOut4,
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
  int8_t mask2 = 0x0c;
  int8_t n_mask2 = ~ mask2;
  int8_t mask4 = 0x30;
  int8_t n_mask4 = ~ mask4;
  int8_t mask6 = 0xc0;
  int8_t n_mask6 = ~ mask6;
  int8_t off2 = 2;
  int8_t off4 = 4;
  int8_t off6 = 6;

  uint16_t ch_out_r = PACK_INT2_SIZE(ch_out);

  uint16_t num_col_im2col_w = PACK_INT2_SIZE(num_col_im2col);
  uint16_t num_col_im2col_a = PACK_INT2_SIZE(num_col_im2col);

  //uint8_t *pOut2 = pOut + ch_out_r;
  int8_t *pA = pWeight;

  uint16_t chan_left = ch_out & 0x3;

  for(int i=0; i < (ch_out >> 2); i++)
  {
    int8_t *pB =  pIn;
    int8_t *pB2 = (pB + num_col_im2col_a);
    int8_t *pB3 = (pB2 + num_col_im2col_a);
    int8_t *pB4 = (pB3 + num_col_im2col_a);

    int32_t *ptrB  = (int32_t *) pB;
    int32_t *ptrB2 = (int32_t *) pB2;
    int32_t *ptrB3 = (int32_t *) pB3;
    int32_t *ptrB4 = (int32_t *) pB4;

    int8_t *pA2 = (pA + num_col_im2col_w);
    int8_t *pA3 = (pA2 + num_col_im2col_w);
    int8_t *pA4 = (pA3 + num_col_im2col_w);

    int32_t *ptrA  = (int32_t *) pA ;
    int32_t *ptrA2 = (int32_t *) pA2;
    int32_t *ptrA3 = (int32_t *) pA3;
    int32_t *ptrA4 = (int32_t *) pA4;

    ptrA  = MacLoadInit(1, 0, 0, 0, ptrA);
    ptrA2 = MacLoadInit(1, 0, 1, 0, ptrA2);
    ptrA3 = MacLoadInit(1, 0, 2, 0, ptrA3);
    ptrA4 = MacLoadInit(1, 0, 3, 0, ptrA4);

    ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);

    int sum = 0;
    int sum2 = 0;
    int sum3 = 0;
    int sum4 = 0;
    int sum5 = 0;
    int sum6 = 0;
    int sum7 = 0;
    int sum8 = 0;

    int sum9  = 0;
    int sum10 = 0;
    int sum11 = 0;
    int sum12 = 0;
    int sum13 = 0;
    int sum14 = 0;
    int sum15 = 0;
    int sum16 = 0;

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

    for(int j=0; j<(num_col_im2col >> 4); j++)
    {
      ptrB2 = MacLoadInit(0, 1, 0, 1, ptrB2);

      sum  = MacLoads16(0, 0, 0, 0, ptrA, sum);
      sum2 = MacLoads16(0, 0, 1, 0, ptrA2, sum2);
      sum3 = MacLoads16(0, 0, 2, 0, ptrA3, sum3);
      sum4 = MacLoads16(0, 1, 3, 0, ptrB3, sum4);
      ptrB3 = MacLoadUpdate(ptrB3);
      

      sum5 = MacLoads16(0, 0, 0, 1, ptrA, sum5);
      sum6 = MacLoads16(0, 0, 1, 1, ptrA2, sum6);
      sum7 = MacLoads16(0, 0, 2, 1, ptrA3, sum7);
      sum8 = MacLoads16(0, 1, 3, 1, ptrB4, sum8);
      ptrB4 = MacLoadUpdate(ptrB4);

      sum9  = MacLoads16(0, 0, 0, 0, ptrA, sum9);
      sum10 = MacLoads16(0, 0, 1, 0, ptrA2, sum10);
      sum11 = MacLoads16(0, 0, 2, 0, ptrA3, sum11);
      sum12 = MacLoads16(0, 1, 3, 0, ptrB, sum12);
      ptrB = MacLoadUpdate(ptrB);

      sum13 = MacLoads16(1, 0, 0, 1, ptrA, sum13);
      ptrA  = MacLoadUpdate(ptrA);
      sum14 = MacLoads16(1, 0, 1, 1, ptrA2, sum14);
      ptrA2 = MacLoadUpdate(ptrA2);
      sum15 = MacLoads16(1, 0, 2, 1, ptrA3, sum15);
      ptrA3 = MacLoadUpdate(ptrA3);
      sum16 = MacLoads16(1, 0, 3, 1, ptrA4, sum16);
      ptrA4 = MacLoadUpdate(ptrA4);

    }

    int col_cnt_im2col = num_col_im2col & 0xf;

    if(col_cnt_im2col)
    {
      uint16_t loop_cnt_im2col_w = (num_col_im2col >> 4) << 2;
      pA+=loop_cnt_im2col_w;
      pA2+=loop_cnt_im2col_w;
      pA3+=loop_cnt_im2col_w;
      pA4+=loop_cnt_im2col_w;

      uint16_t loop_cnt_im2col_a = (num_col_im2col >> 4) << 2;
      pB+=loop_cnt_im2col_a;
      pB2+=loop_cnt_im2col_a;

      do
      {
        int8_t inA = (int8_t) bitext((int) *pA, 2, 0);
        int8_t inA2 = (int8_t) bitext((int) *pA2, 2, 0);
        int8_t inA3 = (int8_t) bitext((int) *pA3, 2, 0);
        int8_t inA4 = (int8_t) bitext((int) *pA4, 2, 0);

        int8_t inB = (int8_t)bitext((int32_t) *pB, 2, 0);
        int8_t inB2 = (int8_t)bitext((int32_t) *pB2, 2, 0);

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

        inB = (int8_t)bitext((int32_t) *pB, 2, 2);
        inB2 = (int8_t)bitext((int32_t) *pB2, 2, 2);

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

        inB = (int8_t)bitext((int32_t) *pB, 2, 4);
        inB2 = (int8_t)bitext((int32_t) *pB2, 2, 4);

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

        inB = (int8_t)bitext((int32_t) *pB, 2, 6);
        inB2 = (int8_t)bitext((int32_t) *pB2, 2, 6);

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

        pB++;
        pB2++;

        col_cnt_im2col-=4;
      } while(col_cnt_im2col > 0);
      pA-=num_col_im2col_w;
    }
    if (flag_batch_norm && flag_relu)
    {
      sum = pulp_nn_bn_quant_i2(sum, *pKappa, *pLambda, out_shift);
      sum5 = pulp_nn_bn_quant_i2(sum5, *pKappa, *pLambda, out_shift);
      sum9 = pulp_nn_bn_quant_i2(sum9, *pKappa, *pLambda, out_shift);
      sum13 = pulp_nn_bn_quant_i2(sum13, *pKappa, *pLambda, out_shift);
      pKappa++;
      pLambda++;
      sum2 = pulp_nn_bn_quant_i2(sum2, *pKappa, *pLambda, out_shift);
      sum6 = pulp_nn_bn_quant_i2(sum6, *pKappa, *pLambda, out_shift);
      sum10 = pulp_nn_bn_quant_i2(sum10, *pKappa, *pLambda, out_shift);
      sum14 = pulp_nn_bn_quant_i2(sum14, *pKappa, *pLambda, out_shift);
      pKappa++;
      pLambda++;
      sum3 = pulp_nn_bn_quant_i2(sum3, *pKappa, *pLambda, out_shift);
      sum7 = pulp_nn_bn_quant_i2(sum7, *pKappa, *pLambda, out_shift);
      sum11 = pulp_nn_bn_quant_i2(sum11, *pKappa, *pLambda, out_shift);
      sum15 = pulp_nn_bn_quant_i2(sum15, *pKappa, *pLambda, out_shift);
      pKappa++;
      pLambda++;
      sum4 = pulp_nn_bn_quant_i2(sum4, *pKappa, *pLambda, out_shift);
      sum8 = pulp_nn_bn_quant_i2(sum8, *pKappa, *pLambda, out_shift);
      sum12 = pulp_nn_bn_quant_i2(sum12, *pKappa, *pLambda, out_shift);
      sum16 = pulp_nn_bn_quant_i2(sum16, *pKappa, *pLambda, out_shift);
      pKappa++;
      pLambda++;
      sum = bitins(sum, n_mask2, sum2, mask2, off2);
      sum = bitins(sum, n_mask4, sum3, mask4, off4);
      *pOut = bitins(sum, n_mask6, sum4, mask6, off6);
      sum5 = bitins(sum5, n_mask2, sum6, mask2, off2);
      sum5 = bitins(sum5, n_mask4, sum7, mask4, off4);
      *pOut2 = bitins(sum5, n_mask6, sum8, mask6, off6);
      sum9 = bitins(sum9, n_mask2, sum10, mask2, off2);
      sum9 = bitins(sum9, n_mask4, sum11, mask4, off4);
      *pOut3 = bitins(sum9, n_mask6, sum12, mask6, off6);
      sum13 = bitins(sum13, n_mask2, sum14, mask2, off2);
      sum13 = bitins(sum13, n_mask4, sum15, mask4, off4);
      *pOut4 = bitins(sum13, n_mask6, sum16, mask6, off6);
      pOut4++;
      pOut3++;
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
        sum9 = pulp_nn_quant_i2(sum9, out_mult, out_shift);
        sum10 = pulp_nn_quant_i2(sum10, out_mult, out_shift);
        sum11 = pulp_nn_quant_i2(sum11, out_mult, out_shift);
        sum12 = pulp_nn_quant_i2(sum12, out_mult, out_shift);
        sum9 = bitins(sum9, n_mask2, sum10, mask2, off2);
        sum9 = bitins(sum9, n_mask4, sum11, mask4, off4);
        *pOut3 = bitins(sum9, n_mask6, sum12, mask6, off6);
        pOut3++;
        sum13 = pulp_nn_quant_i2(sum13, out_mult, out_shift);
        sum14 = pulp_nn_quant_i2(sum14, out_mult, out_shift);
        sum15 = pulp_nn_quant_i2(sum15, out_mult, out_shift);
        sum16 = pulp_nn_quant_i2(sum16, out_mult, out_shift);
        sum13 = bitins(sum13, n_mask2, sum14, mask2, off2);
        sum13 = bitins(sum13, n_mask4, sum15, mask4, off4);
        *pOut4 = bitins(sum13, n_mask6, sum16, mask6, off6);
        pOut4++;
      }
      else
      {
        sum = (int8_t) clips2(sum >> out_shift);
        sum2 = (int8_t) clips2(sum2 >> out_shift);
        sum3 = (int8_t) clips2(sum3 >> out_shift);
        sum4 = (int8_t) clips2(sum4 >> out_shift);
        sum = bitins(sum, n_mask2, sum2, mask2, off2);
        sum = bitins(sum, n_mask4, sum3, mask4, off4);
        *pOut = bitins(sum, n_mask6, sum4, mask6, off6);
        pOut++;

        sum5 = (int8_t) clips2(sum5 >> out_shift);
        sum6 = (int8_t) clips2(sum6 >> out_shift);
        sum7 = (int8_t) clips2(sum7 >> out_shift);
        sum8 = (int8_t) clips2(sum8 >> out_shift);
        sum5 = bitins(sum5, n_mask2, sum6, mask2, off2);
        sum5 = bitins(sum5, n_mask4, sum7, mask4, off4);
        *pOut2 = bitins(sum5, n_mask6, sum8, mask6, off6);
        pOut2++;

        sum9 = (int8_t) clips2(sum9 >> out_shift);
        sum10 = (int8_t) clips2(sum10 >> out_shift);
        sum11 = (int8_t) clips2(sum11 >> out_shift);
        sum12 = (int8_t) clips2(sum12 >> out_shift);
        sum9 = bitins(sum9, n_mask2, sum10, mask2, off2);
        sum9 = bitins(sum9, n_mask4, sum11, mask4, off4);
        *pOut3 = bitins(sum9, n_mask6, sum12, mask6, off6);
        pOut3++;

        sum13 = (int8_t) clips2(sum13 >> out_shift);
        sum14 = (int8_t) clips2(sum14 >> out_shift);
        sum15 = (int8_t) clips2(sum15 >> out_shift);
        sum16 = (int8_t) clips2(sum16 >> out_shift);
        sum13 = bitins(sum13, n_mask2, sum14, mask2, off2);
        sum13 = bitins(sum13, n_mask4, sum15, mask4, off4);
        *pOut4 = bitins(sum13, n_mask6, sum16, mask6, off6);
        pOut4++;
      }
    }
    pA+=(4 * num_col_im2col_w);
  }
  pOut += 3 * ch_out_r;
  return pOut;
}
