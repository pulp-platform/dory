/*
 * ${config.filename}
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


void __attribute__((noinline)) ${config.fn_name}(
                  uint8_t *pIn,
                  int8_t *pBias,
                  uint8_t *pOut,
                  int8_t *pWeight,
                  uint16_t dim_vec,
                  uint16_t num_o_neurons)
{
  uint16_t dim_vec_in = PACK_INT${config.kernel.in_data_t}_SIZE(dim_vec);
  uint16_t dim_vec_wt = PACK_INT${config.kernel.wt_data_t}_SIZE(dim_vec);

  int core_id = pi_core_id();
  int Log2Core = log2(NUM_CORES);
  int chunk = (num_o_neurons >> Log2Core) + ((num_o_neurons & (NUM_CORES-1))!=0);
  int start = min(chunk * core_id, num_o_neurons);
  int stop = min(start + chunk, num_o_neurons);

  int32_t *pOutBuffer = (int32_t *) pOut + start;

%if config.kernel.wt_data_t < config.kernel.in_data_t:
  int32_t vecA[${int(config.max_precision/config.kernel.wt_data_t)}];
%endif
%if config.kernel.in_data_t <= config.kernel.wt_data_t:
  uint32_t vecB[${int(config.max_precision/config.kernel.in_data_t)}];
%endif

  for(int i=start; i<stop; i++)
  {
    int sum = 0;

    if (pBias != NULL)
    {
      sum = ((int) (pBias[i]));
    }

    int8_t *pA = pWeight + (i * dim_vec_wt);

    uint8_t *pB = pIn;

%if config.kernel.in_data_t != config.kernel.wt_data_t:
%if config.kernel.wt_data_t < config.kernel.in_data_t:
    pA  = ${config.unpack_wt_fn}(pA , vecA);

    int32_t *startA;

    asm volatile("mv %0, %1":"=r"(startA):"r"(vecA));

    int32_t *ptrA  = (int32_t *) vecA ;
%else:
    int32_t *ptrA  = (int32_t *) pA ;
%endif
%if config.kernel.in_data_t < config.kernel.wt_data_t:
    pB  = ${config.unpack_in_fn}(pB , vecB);

    uint32_t *startB;

    asm volatile("mv %0, %1":"=r"(startB):"r"(vecB));

    uint32_t *ptrB  = (uint32_t *) vecB;
%else:
    uint32_t *ptrB  = (uint32_t *) pB ;
%endif
%else:
    int32_t *ptrA  = (int32_t *) pA ;

    uint32_t *ptrB  = (uint32_t *) pB ;
%endif

    ptrA  = MacLoadInit(1, 0, 0, 0, ptrA);

    ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);

<%! import math %>
%if config.kernel.wt_data_t < config.kernel.in_data_t:
    for(int j=0; j < (dim_vec >> ${int(math.log2((int(32/config.max_precision))*(int(config.kernel.in_data_t/config.kernel.wt_data_t))))}); j++)
%elif config.kernel.in_data_t <= config.kernel.wt_data_t:
    for(int j=0; j < (dim_vec >> ${int(math.log2((int(32/config.max_precision))*(int(config.kernel.wt_data_t/config.kernel.in_data_t))))}); j++)
%endif
    {
      sum = MacLoad${int(32/config.max_precision)}(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);

      ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);

%if config.kernel.in_data_t != config.kernel.wt_data_t:
%if (int(config.kernel.in_data_t/config.kernel.wt_data_t) == 4) or (int(config.kernel.wt_data_t/config.kernel.in_data_t) == 4):
      sum = MacLoad${int(32/config.max_precision)}(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);

      ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);

      sum = MacLoad${int(32/config.max_precision)}(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);

      ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);
%endif
%if config.kernel.wt_data_t < config.kernel.in_data_t:
      pA  = ${config.unpack_wt_fn}(pA , vecA);

      ptrA   = MacLoadAssign(startA);
%endif
%if config.kernel.in_data_t < config.kernel.wt_data_t:
      pB  = ${config.unpack_in_fn}(pB , vecB);

      ptrB   = MacLoadAssign(startB);
%endif
      sum = MacLoad${int(32/config.max_precision)}(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);

      ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);
%endif
    }
%if config.kernel.wt_data_t < config.kernel.in_data_t:
    uint16_t col_cnt = dim_vec & ${hex(((int(32/config.max_precision))*(int(config.kernel.in_data_t/config.kernel.wt_data_t)))-1)};
%elif config.kernel.in_data_t <= config.kernel.wt_data_t:
    uint16_t col_cnt = dim_vec & ${hex(((int(32/config.max_precision))*(int(config.kernel.wt_data_t/config.kernel.in_data_t)))-1)};
%endif
    if(col_cnt)
    {
%if config.kernel.wt_data_t < config.kernel.in_data_t:
      pA-=4;
%else:
      pA=((dim_vec >> ${int(math.log2((int(32/config.max_precision))*(int(config.kernel.wt_data_t/config.kernel.in_data_t))))}) << ${int(2+int(math.log2(int(config.kernel.wt_data_t/config.kernel.in_data_t))))});
%endif
%if config.kernel.in_data_t < config.kernel.wt_data_t:
      pB-=4;
%else:
      pB=((dim_vec >> ${int(math.log2((int(32/config.max_precision))*(int(config.kernel.in_data_t/config.kernel.wt_data_t))))}) << ${int(2+int(math.log2(int(config.kernel.in_data_t/config.kernel.wt_data_t))))});
%endif
      do
      {
%if config.less_precision == 2:
%if config.kernel.wt_data_t == 2:
        int8_t inA =  (int8_t) bitext((int) *pA, 2, 0);
        int8_t inA2 = (int8_t) bitext((int) *pA, 2, 2);
        int8_t inA3 = (int8_t) bitext((int) *pA, 2, 4);
        int8_t inA4 = (int8_t) bitext((int) *pA, 2, 6);
        pA++;
%elif config.kernel.wt_data_t == 4:
        int8_t inA =  (int8_t) bitext((int) *pA, 4, 0);
        int8_t inA2 = (int8_t) bitext((int) *pA, 4, 4);
        pA++;
        int8_t inA3 = (int8_t) bitext((int) *pA, 4, 0);
        int8_t inA4 = (int8_t) bitext((int) *pA, 4, 4);
        pA++;
%elif config.kernel.wt_data_t == 8:
        int8_t inA = *pA;
        pA++;
        int8_t inA2 = *pA;
        pA++;
        int8_t inA3 = *pA;
        pA++;
        int8_t inA4 = *pA;
        pA++;
%endif
%if config.kernel.in_data_t == 2:
        uint8_t inB =  (uint8_t) bitextu((unsigned int) *pB, 2, 0);
        uint8_t inB2 = (uint8_t) bitextu((unsigned int) *pB, 2, 2);
        uint8_t inB3 = (uint8_t) bitextu((unsigned int) *pB, 2, 4);
        uint8_t inB4 = (uint8_t) bitextu((unsigned int) *pB, 2, 6);
        pB++;
%elif config.kernel.in_data_t == 4:
        uint8_t inB =  (uint8_t) bitextu((unsigned int) *pB, 4, 0);
        uint8_t inB2 = (uint8_t) bitextu((unsigned int) *pB, 4, 4);
        pB++;
        uint8_t inB3 = (uint8_t) bitextu((unsigned int) *pB, 4, 0);
        uint8_t inB4 = (uint8_t) bitextu((unsigned int) *pB, 4, 4);
        pB++;
%elif config.kernel.in_data_t == 8:
        uint8_t inB = *pB;
        pB++;
        uint8_t inB2 = *pB;
        pB++;
        uint8_t inB3 = *pB;
        pB++;
        uint8_t inB4 = *pB;
        pB++;
%endif
        sum += inA * inB;
        sum += inA2 * inB2;
        sum += inA3 * inB3;
        sum += inA4 * inB4;
        col_cnt-=4;
%elif config.less_precision == 4:
%if config.kernel.wt_data_t == 4:
        int8_t inA  = (int8_t) bitext((int) *pA, 4, 0);
        int8_t inA2 = (int8_t) bitext((int) *pA, 4, 4);
        pA++;
%elif config.kernel.wt_data_t == 8:
        int8_t inA = *pA;
        pA++;
        int8_t inA2 = *pA;
        pA++;
%endif
%if config.kernel.in_data_t == 4:
        uint8_t inB  = (uint8_t) bitextu((unsigned int) *pB, 4, 0);
        uint8_t inB2 = (uint8_t) bitextu((unsigned int) *pB, 4, 4);
        pB++;
%elif config.kernel.in_data_t == 8:
        uint8_t inB = *pB;
        pB++;
        uint8_t inB2 = *pB;
        pB++;
%endif
        sum += inA * inB;
        sum += inA2 * inB2;
        col_cnt-=2;
%elif config.less_precision == 8:
        int8_t inA = *pA;
        pA++;
        uint8_t inB = *pB;
        pB++;
        sum += inA * inB;
        col_cnt--;
%endif
      }while (col_cnt);
    }
    *pOutBuffer = sum;
    pOutBuffer++;
  }
  pi_cl_team_barrier(0);
}
