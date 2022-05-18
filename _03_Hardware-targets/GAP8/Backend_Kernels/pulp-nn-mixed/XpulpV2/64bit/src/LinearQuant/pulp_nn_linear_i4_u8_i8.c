/*
 * pulp_nn_linear_i4_u8_i8.c
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



void pulp_nn_linear_i4_u8_i8(
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
    uint16_t dim_vec_in = dim_vec >> 1;
    uint16_t dim_vec_wt = dim_vec;

    int core_id = pi_core_id();
    int Log2Core = log2(NUM_CORES);
    int chunk = (num_o_neurons >> Log2Core) + ((num_o_neurons & (NUM_CORES-1))!=0);
    int start = min(chunk * core_id, num_o_neurons);
    int stop = min(start + chunk, num_o_neurons);

    v4s vecA[2];
    v4s vecB[2];
    v4s vecB2[2];

    uint8_t *pOutBuffer = (uint8_t *) pOut + start;
    int lft_neurons = chunk & 0x01;
    int stop_even = stop - lft_neurons;

    int i;
    int64_t *k1 = pKappa + start;
    int64_t *lambda1 = pLambda + start;

    for(i=start; i<stop_even; i+=2)
    {
        int sum = 0;
        int sum2 = 0;

        int8_t *pA = pIn;
        int8_t *pB = pWeight + (i * dim_vec_wt);
        int8_t *pB2 = pB + dim_vec_wt;

        for (int j=0; j<(dim_vec >> 3); j++)
        {
          pulp_nn_i4_to_i8(pA,vecA);
          vecB[0] = *((v4s*)pB);
          vecB2[0] = *((v4s*)pB2);
          pB+=4;
          pB2+=4;
          vecB[1] = *((v4s*)pB);
          vecB2[1] = *((v4s*)pB2);
          sum = SumDotps4(vecA[0], vecB[0], sum);
          sum = SumDotps4(vecA[1], vecB[1], sum);
          sum2 = SumDotps4(vecA[0], vecB2[0], sum2);
          sum2 = SumDotps4(vecA[1], vecB2[1], sum2);
          pA+=4;
          pB+=4;
          pB2+=4;
        }
        uint16_t col_cnt = dim_vec & 0x7;
        while (col_cnt)
        {
          int8_t inA = (int8_t) bitext((int32_t) *pA, 4, 0);
          int8_t inA2 = (int8_t) bitext((int32_t) *pA, 4, 4);
          pA++;
          int8_t inB = *pB;
          pB++;
          int8_t inB2 = *pB;
          pB++;
          int8_t inB3 = *pB2;
          pB2++;
          int8_t inB4 = *pB2;
          pB2++;
          sum += inA * inB;
          sum += inA2 * inB2;
          sum2 += inA * inB3;
          sum2 += inA2 * inB4;
          col_cnt--;
        }
        if (flag_batch_norm && flag_relu)
        {
          *pOutBuffer = pulp_nn_bn_quant_u8(sum, *k1, *lambda1, out_shift);
          pOutBuffer++;
          *pOutBuffer = pulp_nn_bn_quant_u8(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
          pOutBuffer++;
          k1+=2;
          lambda1+=2;
        }
        else
        {
          if (flag_relu == 1)
          {
            *pOutBuffer = pulp_nn_quant_u8(sum, out_mult, out_shift);
            pOutBuffer++;
            *pOutBuffer = pulp_nn_quant_u8(sum2, out_mult, out_shift);
            pOutBuffer++;
          }
          else
          {
            *pOutBuffer = (uint8_t) clip8(sum >> out_shift);
            pOutBuffer++;
            *pOutBuffer = (uint8_t) clip8(sum2 >> out_shift);
            pOutBuffer++;
          }
        }
    }
    if (lft_neurons && (stop - start) > 0)
    {
        int sum = 0;

        int8_t *pA = pIn;
        int8_t *pB = pWeight + (i * dim_vec_wt);

        for (int j=0; j<(dim_vec >> 3); j++)
        {
            pulp_nn_i4_to_i8(pA,vecA);
            vecB[0] = *((v4s*)pB);
            pB+=4;
            vecB[1] = *((v4s*)pB);
            sum = SumDotps4(vecA[0], vecB[0], sum);
            sum = SumDotps4(vecA[1], vecB[1], sum);
            pA+=4;
            pB+=4;
        }
        uint16_t col_cnt = dim_vec & 0x7;
        while (col_cnt)
        {
          int8_t inA = (int8_t) bitext((int32_t) *pA, 4, 0);
          int8_t inA2 = (int8_t) bitext((int32_t) *pA, 4, 4);
          pA++;
          int8_t inB = *pB;
          pB++;
          int8_t inB2 = *pB;
          pB++;
          sum += inA * inB;
          sum += inA2 * inB2;
          col_cnt--;
        }
        if (flag_batch_norm && flag_relu)
        {
          *pOutBuffer = pulp_nn_bn_quant_u8(sum, *pKappa, *pLambda, out_shift);
          pOutBuffer++;
          pKappa++;
          pLambda++;
        }
        else
        {
          if (flag_relu == 1)
          {
            *pOutBuffer = pulp_nn_quant_u8(sum, out_mult, out_shift);
            pOutBuffer++;
          }
          else
          {
            *pOutBuffer = (uint8_t) clip8(sum >> out_shift);
            pOutBuffer++;
          }
        }
    }
    pi_cl_team_barrier(0);
}
