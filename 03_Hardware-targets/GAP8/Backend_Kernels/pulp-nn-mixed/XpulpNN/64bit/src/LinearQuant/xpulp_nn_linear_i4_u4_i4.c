/*
 * xpulp_nn_linear_i4_u4_i4.c
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



void __attribute__((noinline)) xpulp_nn_linear_i4_u4_i4(
                        int8_t *pIn,
                        int8_t *pBias,
                        uint8_t *pOut,
                        int8_t *pWeight,
                        int64_t *pKappa,
                        int64_t *pLambda,
                        uint16_t out_mult,
                        uint16_t out_shift,
                        uint16_t dim_vec,
                        uint16_t num_o_neurons,
                        uint8_t flag_relu,
                        uint8_t flag_batch_norm)
{
  int8_t mask = 0xf0;
  int8_t n_mask = ~ mask;
  int8_t off = 0x04;
  uint16_t dim_vec_in = PACK_INT4_SIZE(dim_vec);
  uint16_t dim_vec_wt = PACK_INT4_SIZE(dim_vec);

  int core_id = pi_core_id();
  int Log2Core = log2(NUM_CORES);
  int chunk = (num_o_neurons >> Log2Core) + ((num_o_neurons & (NUM_CORES-1))!=0);
  int start = min((chunk << (chunk == 1)) * core_id, num_o_neurons);
  int stop = min(start + (chunk << (chunk == 1)), num_o_neurons);

  v4s vecB[2];
  v4s vecA[2];
  v4s vecA2[2];

  uint8_t *pOutBuffer = (uint8_t *) pOut + (start >> 1);

  int i;

  int64_t *k1 = pKappa + start;
  int64_t *lambda1 = pLambda + start;

  for(i=start; i<stop; i+=2)
  {
    int sum = 0;
    int sum2 = 0;

    if (pBias != NULL)
    {
      sum = ((int) (pBias[i]));
      sum2 = (pBias[i + 1]);
    }

    int8_t *pB = pIn;
    int8_t *pA = pWeight + (i * dim_vec_wt);
    int8_t *pA2 = pA + dim_vec_wt;
    int32_t *ptrA  = (int32_t *) pA ;
    int32_t *ptrA2  = (int32_t *) pA2 ;

    int32_t *ptrB  = (int32_t *) pB ;

    ptrA  = MacLoadInit(1, 0, 0, 0, ptrA);
    ptrA2  = MacLoadInit(1, 0, 1, 0, ptrA2);

    ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);


    for(int j=0; j < (dim_vec >> 3); j++)
    {
      sum = MacLoads8(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);
      sum2 = MacLoads8(1, 0, 1, 0, ptrA2, sum2);
      ptrA2 = MacLoadUpdate(ptrA2);

      ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);

    }
    uint16_t col_cnt = dim_vec & 0x7;
    if(col_cnt)
    {
      pA=((dim_vec >> 3) << 2);
      pA2+=((dim_vec >> 3) << 2);
      pB=((dim_vec >> 3) << 2);
      do
      {
        int8_t inB  = (int8_t) bitext((int32_t) *pB, 4, 0);
        int8_t inB2 = (int8_t) bitext((int32_t) *pB, 4, 4);
        pB++;
        int8_t inA  = (int8_t) bitext((int) *pA, 4, 0);
        int8_t inA2 = (int8_t) bitext((int) *pA, 4, 4);
        pA++;
        sum += inA * inB;
        sum += inA2 * inB2;
        inA =  (int8_t) bitext((int) *pA2, 4, 0);
        inA2 = (int8_t) bitext((int) *pA2, 4, 4);
        pA2++;
        sum2 += inA * inB;
        sum2 += inA2 * inB2;
        col_cnt-=2;
      }while (col_cnt);
    }
    if (flag_batch_norm && flag_relu)
    {
      sum = pulp_nn_bn_quant_u4(sum, *k1, *lambda1, out_shift);
      sum2 = pulp_nn_bn_quant_u4(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
      *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
      pOutBuffer++;
      k1+=2;
      lambda1+=2;
    }
    else
    {
      if (flag_relu == 1)
      {
        sum = pulp_nn_quant_u4(sum, out_mult, out_shift);
        sum2 = pulp_nn_quant_u4(sum2, out_mult, out_shift);
        *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
        pOutBuffer++;
      }
      else
      {
        sum = (uint8_t) clip4(sum >> out_shift);
        sum2 = (uint8_t) clip4(sum2 >> out_shift);
        *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
        pOutBuffer++;
      }
    }
  }
  pi_cl_team_barrier(0);
}
