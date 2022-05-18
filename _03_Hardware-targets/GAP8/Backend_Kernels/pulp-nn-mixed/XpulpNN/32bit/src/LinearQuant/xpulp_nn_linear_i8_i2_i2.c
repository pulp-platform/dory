/*
 * xpulp_nn_linear_i8_i2_i2.c
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



void __attribute__((noinline)) xpulp_nn_linear_i8_i2_i2(
                        int8_t *pIn,
                        int8_t *pBias,
                        int8_t *pOut,
                        int8_t *pWeight,
                        int32_t *pKappa,
                        int32_t *pLambda,
                        uint16_t out_mult,
                        uint16_t out_shift,
                        uint16_t dim_vec,
                        uint16_t num_o_neurons,
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
  uint16_t dim_vec_in = PACK_INT8_SIZE(dim_vec);
  uint16_t dim_vec_wt = PACK_INT2_SIZE(dim_vec);

  int core_id = pi_core_id();
  int Log2Core = log2(NUM_CORES);
  int chunk = (num_o_neurons >> Log2Core) + ((num_o_neurons & (NUM_CORES-1))!=0);
  int neuron_left = 0;
  if (chunk & 0x3)
  {
      neuron_left = (4 - (chunk & 0x7));
  }
  int start = min((chunk + neuron_left) * core_id, num_o_neurons);
  int stop = min(start + chunk + neuron_left, num_o_neurons);

  v4s vecB[4];
  v4s vecA[4];
  v4s vecA2[4];
  v4s vecA3[4];
  v4s vecA4[4];

  int8_t *pOutBuffer = (int8_t *) pOut + (start >> 2);

  int i;

  int32_t *k1 = pKappa + start;
  int32_t *lambda1 = pLambda + start;

  for(i=start; i<stop; i+=4)
  {
    int sum = 0;
    int sum2 = 0;
    int sum3 = 0;
    int sum4 = 0;

    if (pBias != NULL)
    {
      sum = ((int) (pBias[i]));
      sum2 = (pBias[i + 1]);
      sum3 = (pBias[i + 2]);
      sum4 = (pBias[i + 3]);
    }

    int8_t *pB = pIn;
    int8_t *pA = pWeight + (i * dim_vec_wt);
    int8_t *pA2 = pA + dim_vec_wt;
    int8_t *pA3 = pA2 + dim_vec_wt;
    int8_t *pA4 = pA3 + dim_vec_wt;
    pA  = pulp_nn_i2_to_i8(pA , vecA);
    pA2  = pulp_nn_i2_to_i8(pA2 , vecA2);
    pA3  = pulp_nn_i2_to_i8(pA3 , vecA3);
    pA4  = pulp_nn_i2_to_i8(pA4 , vecA4);

    int32_t *startA;
    int32_t *startA2;
    int32_t *startA3;
    int32_t *startA4;

    asm volatile("mv %0, %1":"=r"(startA):"r"(vecA));
    asm volatile("mv %0, %1":"=r"(startA2):"r"(vecA2));
    asm volatile("mv %0, %1":"=r"(startA3):"r"(vecA3));
    asm volatile("mv %0, %1":"=r"(startA4):"r"(vecA4));
    int32_t *ptrA  = (int32_t *) vecA ;
    int32_t *ptrA2  = (int32_t *) vecA2 ;
    int32_t *ptrA3  = (int32_t *) vecA3 ;
    int32_t *ptrA4  = (int32_t *) vecA4 ;
    int32_t *ptrB  = (int32_t *) pB ;

    ptrA  = MacLoadInit(1, 0, 0, 0, ptrA);
    ptrA2  = MacLoadInit(1, 0, 1, 0, ptrA2);
    ptrA3  = MacLoadInit(1, 0, 2, 0, ptrA3);
    ptrA4  = MacLoadInit(1, 0, 3, 0, ptrA4);

    ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);


    for(int j=0; j < (dim_vec >> 4); j++)
    {
      sum = MacLoads4(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);
      sum2 = MacLoads4(1, 0, 1, 0, ptrA2, sum2);
      ptrA2 = MacLoadUpdate(ptrA2);
      sum3 = MacLoads4(1, 0, 2, 0, ptrA3, sum3);
      ptrA3 = MacLoadUpdate(ptrA3);
      sum4 = MacLoads4(1, 0, 3, 0, ptrA4, sum4);
      ptrA4 = MacLoadUpdate(ptrA4);

      ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);

      sum = MacLoads4(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);
      sum2 = MacLoads4(1, 0, 1, 0, ptrA2, sum2);
      ptrA2 = MacLoadUpdate(ptrA2);
      sum3 = MacLoads4(1, 0, 2, 0, ptrA3, sum3);
      ptrA3 = MacLoadUpdate(ptrA3);
      sum4 = MacLoads4(1, 0, 3, 0, ptrA4, sum4);
      ptrA4 = MacLoadUpdate(ptrA4);

      ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);

      sum = MacLoads4(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);
      sum2 = MacLoads4(1, 0, 1, 0, ptrA2, sum2);
      ptrA2 = MacLoadUpdate(ptrA2);
      sum3 = MacLoads4(1, 0, 2, 0, ptrA3, sum3);
      ptrA3 = MacLoadUpdate(ptrA3);
      sum4 = MacLoads4(1, 0, 3, 0, ptrA4, sum4);
      ptrA4 = MacLoadUpdate(ptrA4);

      ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);
      pA  = pulp_nn_i2_to_i8(pA , vecA);
      pA2  = pulp_nn_i2_to_i8(pA2 , vecA2);
      pA3  = pulp_nn_i2_to_i8(pA3 , vecA3);
      pA4  = pulp_nn_i2_to_i8(pA4 , vecA4);

      ptrA   = MacLoadAssign(startA);
      ptrA2   = MacLoadAssign(startA2);
      ptrA3   = MacLoadAssign(startA3);
      ptrA4   = MacLoadAssign(startA4);
      sum = MacLoads4(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);
      sum2 = MacLoads4(1, 0, 1, 0, ptrA2, sum2);
      ptrA2 = MacLoadUpdate(ptrA2);
      sum3 = MacLoads4(1, 0, 2, 0, ptrA3, sum3);
      ptrA3 = MacLoadUpdate(ptrA3);
      sum4 = MacLoads4(1, 0, 3, 0, ptrA4, sum4);
      ptrA4 = MacLoadUpdate(ptrA4);

      ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);
    }
    uint16_t col_cnt = dim_vec & 0xf;
    if(col_cnt)
    {
      pA-=4;
      pA2-=4;
      pA3-=4;
      pA4-=4;
      pB=((dim_vec >> 4) << 4);
      do
      {
        int8_t inB = *pB;
        pB++;
        int8_t inB2 = *pB;
        pB++;
        int8_t inB3 = *pB;
        pB++;
        int8_t inB4 = *pB;
        pB++;
        int8_t inA =  (int8_t) bitext((int) *pA, 2, 0);
        int8_t inA2 = (int8_t) bitext((int) *pA, 2, 2);
        int8_t inA3 = (int8_t) bitext((int) *pA, 2, 4);
        int8_t inA4 = (int8_t) bitext((int) *pA, 2, 6);
        pA++;
        sum += inA * inB;
        sum += inA2 * inB2;
        sum += inA3 * inB3;
        sum += inA4 * inB4;
        inA =  (int8_t) bitext((int) *pA2, 2, 0);
        inA2 = (int8_t) bitext((int) *pA2, 2, 2);
        inA3 = (int8_t) bitext((int) *pA2, 2, 4);
        inA4 = (int8_t) bitext((int) *pA2, 2, 6);
        pA2++;
        sum2 += inA * inB;
        sum2 += inA2 * inB2;
        sum2 += inA3 * inB3;
        sum2 += inA4 * inB4;
        inA =  (int8_t) bitext((int) *pA3, 2, 0);
        inA2 = (int8_t) bitext((int) *pA3, 2, 2);
        inA3 = (int8_t) bitext((int) *pA3, 2, 4);
        inA4 = (int8_t) bitext((int) *pA3, 2, 6);
        pA3++;
        sum3 += inA * inB;
        sum3 += inA2 * inB2;
        sum3 += inA3 * inB3;
        sum3 += inA4 * inB4;
        inA =  (int8_t) bitext((int) *pA4, 2, 0);
        inA2 = (int8_t) bitext((int) *pA4, 2, 2);
        inA3 = (int8_t) bitext((int) *pA4, 2, 4);
        inA4 = (int8_t) bitext((int) *pA4, 2, 6);
        pA4++;
        sum4 += inA * inB;
        sum4 += inA2 * inB2;
        sum4 += inA3 * inB3;
        sum4 += inA4 * inB4;
        col_cnt-=4;
      }while (col_cnt);
    }
    if (flag_batch_norm && flag_relu)
    {
      sum = pulp_nn_bn_quant_i2(sum, *k1, *lambda1, out_shift);
      sum2 = pulp_nn_bn_quant_i2(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
      sum3 = pulp_nn_bn_quant_i2(sum3, *(k1 + 2), *(lambda1 + 2), out_shift);
      sum4 = pulp_nn_bn_quant_i2(sum4, *(k1 + 3), *(lambda1 + 3), out_shift);
      k1+=4;
      lambda1+=4;
      sum = bitins(sum, n_mask2, sum2, mask2, off2);
      sum = bitins(sum, n_mask4, sum3, mask4, off4);
      *pOutBuffer = bitins(sum, n_mask6, sum4, mask6, off6);
      pOutBuffer++;
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
        *pOutBuffer = bitins(sum, n_mask6, sum4, mask6, off6);
        pOutBuffer++;
      }
      else
      {
        sum = (int8_t) clips2(sum >> out_shift);
        sum2 = (int8_t) clips2(sum2 >> out_shift);
        sum3 = (int8_t) clips2(sum3 >> out_shift);
        sum4 = (int8_t) clips2(sum4 >> out_shift);
        sum = bitins(sum, n_mask2, sum2, mask2, off2);
        sum = bitins(sum, n_mask4, sum3, mask4, off4);
        *pOutBuffer = bitins(sum, n_mask6, sum4, mask6, off6);
        pOutBuffer++;
      }
    }
  }
  pi_cl_team_barrier(0);
}
