/*
 * pulp_nn_linear_i2_u4_i4.c
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



void pulp_nn_linear_i2_u4_i4(
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
    uint16_t dim_vec_in = dim_vec >> 2;
    uint16_t dim_vec_wt = dim_vec >> 1;

    int core_id = pi_core_id();
    int Log2Core = log2(NUM_CORES);
    int chunk = (num_o_neurons >> Log2Core) + ((num_o_neurons & (NUM_CORES-1))!=0);
    int start = min((chunk << (chunk == 1)) * core_id, num_o_neurons);
    int stop = min(start + (chunk << (chunk == 1)), num_o_neurons);

    v4s vecA[4];
    v4s vecB[4];
    v4s vecB2[4];

    uint8_t *pOutBuffer = (uint8_t *) pOut + (start >> 1);

    int i;
    int64_t *k1 = pKappa + start;
    int64_t *lambda1 = pLambda + start;

    for(i=start; i<stop; i+=2)
    {
        int sum = 0;
        int sum2 = 0;

        int8_t *pA = pIn;
        int8_t *pB = pWeight + (i * dim_vec_wt);
        int8_t *pB2 = pB + dim_vec_wt;

        for (int j=0; j<(dim_vec >> 4); j++)
        {
          pulp_nn_i2_to_i8(pA,vecA);
          pulp_nn_i4_to_i8(pB,vecB);
          pulp_nn_i4_to_i8(pB2,vecB2);
          pB+=4;
          pB2+=4;
          pulp_nn_i4_to_i8(pB,vecB + 2);
          pulp_nn_i4_to_i8(pB2,vecB2 + 2);
          sum = SumDotps4(vecA[0], vecB[0], sum);
          sum = SumDotps4(vecA[1], vecB[1], sum);
          sum = SumDotps4(vecA[2], vecB[2], sum);
          sum = SumDotps4(vecA[3], vecB[3], sum);
          sum2 = SumDotps4(vecA[0], vecB2[0], sum2);
          sum2 = SumDotps4(vecA[1], vecB2[1], sum2);
          sum2 = SumDotps4(vecA[2], vecB2[2], sum2);
          sum2 = SumDotps4(vecA[3], vecB2[3], sum2);
          pA+=4;
          pB+=4;
          pB2+=4;
        }
        uint16_t col_cnt = dim_vec & 0xf;
        while (col_cnt)
        {
          int8_t inA = (int8_t) bitext((int32_t) *pA, 2, 0);
          int8_t inA2 = (int8_t) bitext((int32_t) *pA, 2, 2);
          int8_t inA3 = (int8_t) bitext((int32_t) *pA, 2, 4);
          int8_t inA4 = (int8_t) bitext((int32_t) *pA, 2, 6);
          pA++;
          int8_t inB = (int8_t) bitext((int) *pB, 4, 0);
          int8_t inB2 = (int8_t) bitext((int) *pB, 4, 4);
          pB++;
          int8_t inB3 = (int8_t) bitext((int) *pB, 4, 0);
          int8_t inB4 = (int8_t) bitext((int) *pB, 4, 4);
          pB++;
          int8_t inB5 = (int8_t) bitext((int) *pB2, 4, 0);
          int8_t inB6 = (int8_t) bitext((int) *pB2, 4, 4);
          pB2++;
          int8_t inB7 = (int8_t) bitext((int) *pB2, 4, 0);
          int8_t inB8 = (int8_t) bitext((int) *pB2, 4, 4);
          pB2++;
          sum += inA * inB;
          sum += inA2 * inB2;
          sum += inA3 * inB3;
          sum += inA4 * inB4;
          sum2 += inA * inB5;
          sum2 += inA2 * inB6;
          sum2 += inA3 * inB7;
          sum2 += inA4 * inB8;
          col_cnt--;
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
