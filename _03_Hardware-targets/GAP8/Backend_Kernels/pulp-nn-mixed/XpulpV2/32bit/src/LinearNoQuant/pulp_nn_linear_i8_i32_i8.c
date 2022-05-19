/*
 * pulp_nn_linear_i8_i32_i8.c
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


void pulp_nn_linear_i8_i32_i8(
                  int8_t *pIn,
                  int8_t *pBias,
                  int8_t *pOut,
                  int8_t *pWeight,
                  uint16_t dim_vec,
                  uint16_t num_o_neurons)
{
    uint16_t dim_vec_in = dim_vec;
    uint16_t dim_vec_wt = dim_vec;

    int core_id = pi_core_id();
    int Log2Core = log2(NUM_CORES);
    int chunk = (num_o_neurons >> Log2Core) + ((num_o_neurons & (NUM_CORES-1))!=0);
    int start = min(chunk * core_id, num_o_neurons);
    int stop = min(start + chunk, num_o_neurons);

    v4s vecA;
    v4s vecB;
    v4s vecB2;

    int32_t *pOutBuffer = (int32_t *) pOut + start;

    int lft_neurons = chunk & 0x01;
    int stop_even = stop - lft_neurons;
    int i;

    for(i=start; i<stop_even; i+=2)
    {
        int sum = 0;
        int sum2 = 0;

        int8_t *pA = pIn;
        int8_t *pB = pWeight + (i * dim_vec_wt);
        int8_t *pB2 = pB + dim_vec_wt;

        for (int j=0; j<(dim_vec >> 2); j++)
        {
               vecA = *((v4s*)pA);
               vecB = *((v4s*)pB);
               vecB2 = *((v4s*)pB2);
             sum = SumDotps4(vecA, vecB, sum);
             sum2 = SumDotps4(vecA, vecB2, sum2);
                  pA+=4;
                  pB+=4;
                  pB2+=4;
        }
            uint16_t col_cnt = dim_vec & 0x3;
            while (col_cnt)
            {
                  int8_t inA = *pA;
                  pA++;
                  int8_t inB = *pB;
                  pB++;
                  int8_t inB5 = *pB2;
                  pB2++;
                  sum += inA * inB;
                sum2 += inA * inB5;
                  col_cnt--;
            }
        *pOutBuffer = sum;
        pOutBuffer++;
        *pOutBuffer = sum2;
        pOutBuffer++;
    }
    if (lft_neurons && (stop - start) > 0)
    {
        int sum = 0;

        int8_t *pA = pIn;
        int8_t *pB = pWeight + (i * dim_vec_wt);

        for (int j=0; j<(dim_vec >> 2); j++)
        {
            vecA = *((v4s*)pA);
            vecB = *((v4s*)pB);
            sum = SumDotps4(vecA, vecB, sum);
           pA+=4;
           pB+=4;
        }
            uint16_t col_cnt = dim_vec & 0x3;
            while (col_cnt)
            {
                  int8_t inA = *pA;
                  pA++;
                  int8_t inB = *pB;
                  pB++;
                  sum += inA * inB;
                  col_cnt--;
            }
        *pOutBuffer = sum;
        pOutBuffer++;
    }
    pi_cl_team_barrier(0);
}
