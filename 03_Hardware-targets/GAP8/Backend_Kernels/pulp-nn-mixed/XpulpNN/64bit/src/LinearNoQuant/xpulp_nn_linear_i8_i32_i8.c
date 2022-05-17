/*
 * xpulp_nn_linear_i8_i32_i8.c
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


void __attribute__((noinline)) xpulp_nn_linear_i8_i32_i8(
                  int8_t *pIn,
                  int8_t *pBias,
                  int8_t *pOut,
                  int8_t *pWeight,
                  uint16_t dim_vec,
                  uint16_t num_o_neurons)
{
  uint16_t dim_vec_in = PACK_INT8_SIZE(dim_vec);
  uint16_t dim_vec_wt = PACK_INT8_SIZE(dim_vec);

  int core_id = pi_core_id();
  int Log2Core = log2(NUM_CORES);
  int chunk = (num_o_neurons >> Log2Core) + ((num_o_neurons & (NUM_CORES-1))!=0);
  int start = min(chunk * core_id, num_o_neurons);
  int stop = min(start + chunk, num_o_neurons);

  int32_t *pOutBuffer = (int32_t *) pOut + start;

  int32_t vecB[1];

  for(int i=start; i<stop; i++)
  {
    int sum = 0;

    if (pBias != NULL)
    {
      sum = ((int) (pBias[i]));
    }

    int8_t *pA = pWeight + (i * dim_vec_wt);

    int8_t *pB = pIn;

    int32_t *ptrA  = (int32_t *) pA ;

    int32_t *ptrB  = pB ;

    ptrA  = MacLoadInit(1, 0, 0, 0, ptrA);

    ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);


    for(int j=0; j < (dim_vec >> 2); j++)
    {
      sum = MacLoads4(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);

      ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);

    }
    uint16_t col_cnt = dim_vec & 0x3;
    if(col_cnt)
    {
      pA=((dim_vec >> 2) << 2);
      pB=((dim_vec >> 2) << 2);
      do
      {
        int8_t inA = *pA;
        pA++;
        int8_t inB = *pB;
        pB++;
        sum += inA * inB;
        col_cnt--;
      }while (col_cnt);
    }
    *pOutBuffer = sum;
    pOutBuffer++;
  }
  pi_cl_team_barrier(0);
}
