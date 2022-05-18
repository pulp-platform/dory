/*
 * xpulp_nn_linear_i8_i2_i8.c
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



void __attribute__((noinline)) xpulp_nn_linear_i8_i2_i8(
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
  uint16_t dim_vec_wt = PACK_INT8_SIZE(dim_vec);

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

  v4s vecB[1];
  v4s vecA[1];
  v4s vecA2[1];
  v4s vecA3[1];
  v4s vecA4[1];

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
    int32_t *ptrA  = (int32_t *) pA ;
    int32_t *ptrA2  = (int32_t *) pA2 ;
    int32_t *ptrA3  = (int32_t *) pA3 ;
    int32_t *ptrA4  = (int32_t *) pA4 ;

    int32_t *ptrB  = (int32_t *) pB ;

    ptrA  = MacLoadInit(1, 0, 0, 0, ptrA);
    ptrA2  = MacLoadInit(1, 0, 1, 0, ptrA2);
    ptrA3  = MacLoadInit(1, 0, 2, 0, ptrA3);
    ptrA4  = MacLoadInit(1, 0, 3, 0, ptrA4);

    ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);


    for(int j=0; j < (dim_vec >> 2); j++)
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

    }
    uint16_t col_cnt = dim_vec & 0x3;
    if(col_cnt)
    {
      pA=((dim_vec >> 2) << 2);
      pA2+=((dim_vec >> 2) << 2);
      pA3+=((dim_vec >> 2) << 2);
      pA4+=((dim_vec >> 2) << 2);
      pB=((dim_vec >> 2) << 2);
      do
      {
        int8_t inB = *pB;
        pB++;
        int8_t inA = *pA;
        pA++;
        sum += inA * inB;
        inA = *pA2;
        pA2++;

        sum2 += inA * inB;
        inA = *pA3;
        pA3++;

        sum3 += inA * inB;

        inA = *pA4;
        pA4++;

        sum4 += inA * inB;
        col_cnt--;
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
