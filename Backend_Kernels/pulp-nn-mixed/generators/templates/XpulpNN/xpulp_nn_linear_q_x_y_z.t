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
%if config.kernel.act_prec == '32bit':
                        int32_t *pKappa,
                        int32_t *pLambda,
%elif config.kernel.act_prec == '64bit':
                        int64_t *pKappa,
                        int64_t *pLambda,
%endif
                        uint16_t out_mult,
                        uint16_t out_shift,
                        uint16_t dim_vec,
                        uint16_t num_o_neurons,
                        uint8_t flag_relu,
                        uint8_t flag_batch_norm)
{
%if config.kernel.out_data_t == 2:
  int8_t mask2 = 0x0c;
  int8_t n_mask2 = ~ mask2;
  int8_t mask4 = 0x30;
  int8_t n_mask4 = ~ mask4;
  int8_t mask6 = 0xc0;
  int8_t n_mask6 = ~ mask6;
  int8_t off2 = 2;
  int8_t off4 = 4;
  int8_t off6 = 6;
%elif config.kernel.out_data_t == 4:
  int8_t mask = 0xf0;
  int8_t n_mask = ~ mask;
  int8_t off = 0x04;
%endif
  uint16_t dim_vec_in = PACK_INT${config.kernel.in_data_t}_SIZE(dim_vec);
  uint16_t dim_vec_wt = PACK_INT${config.kernel.wt_data_t}_SIZE(dim_vec);

  int core_id = pi_core_id();
  int Log2Core = log2(NUM_CORES);
  int chunk = (num_o_neurons >> Log2Core) + ((num_o_neurons & (NUM_CORES-1))!=0);
%if config.kernel.out_data_t == 8:
  int start = min(chunk * core_id, num_o_neurons);
  int stop = min(start + chunk, num_o_neurons);
%elif config.kernel.out_data_t == 4:
  int start = min((chunk << (chunk == 1)) * core_id, num_o_neurons);
  int stop = min(start + (chunk << (chunk == 1)), num_o_neurons);
%elif config.kernel.out_data_t == 2:
  int neuron_left = 0;
  if (chunk & 0x3)
  {
      neuron_left = (4 - (chunk & 0x7));
  }
  int start = min((chunk + neuron_left) * core_id, num_o_neurons);
  int stop = min(start + chunk + neuron_left, num_o_neurons);
%endif

  v4u vecB[${int(8/config.less_precision)}];
  v4s vecA[${int(8/config.less_precision)}];
%if config.kernel.out_data_t < 8:
  v4s vecA2[${int(8/config.less_precision)}];
%if config.kernel.out_data_t == 2:
  v4s vecA3[${int(8/config.less_precision)}];
  v4s vecA4[${int(8/config.less_precision)}];
%endif
%endif

%if config.kernel.out_data_t == 8:
  uint8_t *pOutBuffer = (uint8_t *) pOut + start;
%elif config.kernel.out_data_t == 4:
  uint8_t *pOutBuffer = (uint8_t *) pOut + (start >> 1);
%elif config.kernel.out_data_t == 2:
  uint8_t *pOutBuffer = (uint8_t *) pOut + (start >> 2);
%endif

  int i;
%if config.kernel.act_prec == '32bit':
  int32_t *k1 = pKappa + start;
  int32_t *lambda1 = pLambda + start;
%elif config.kernel.act_prec == '64bit':
  int64_t *k1 = pKappa + start;
  int64_t *lambda1 = pLambda + start;
%endif

%if config.kernel.out_data_t < 8:
  for(i=start; i<stop; i+=${int(8/config.kernel.out_data_t)})
%else:
  for(i=start; i<stop; i++)
%endif
  {
    int sum = 0;
%if config.kernel.out_data_t < 8:
    int sum2 = 0;
%if config.kernel.out_data_t == 2:
    int sum3 = 0;
    int sum4 = 0;
%endif
%endif

    if (pBias != NULL)
    {
      sum = ((int) (pBias[i]));
%if config.kernel.out_data_t < 8:
      sum2 = (pBias[i + 1]);
%if config.kernel.out_data_t == 2:
      sum3 = (pBias[i + 2]);
      sum4 = (pBias[i + 3]);
%endif
%endif
    }

    uint8_t *pB = pIn;
    int8_t *pA = pWeight + (i * dim_vec_wt);
%if config.kernel.out_data_t < 8:
    int8_t *pA2 = pA + dim_vec_wt;
%if config.kernel.out_data_t == 2:
    int8_t *pA3 = pA2 + dim_vec_wt;
    int8_t *pA4 = pA3 + dim_vec_wt;
%endif
%endif
%if config.kernel.in_data_t != config.kernel.wt_data_t:
%if config.kernel.wt_data_t < config.kernel.in_data_t:
    pA  = ${config.unpack_wt_fn}(pA , vecA);
%if config.kernel.out_data_t < 8:
    pA2  = ${config.unpack_wt_fn}(pA2 , vecA2);
%if config.kernel.out_data_t == 2:
    pA3  = ${config.unpack_wt_fn}(pA3 , vecA3);
    pA4  = ${config.unpack_wt_fn}(pA4 , vecA4);
%endif
%endif

    int32_t *startA;
%if config.kernel.out_data_t < 8:
    int32_t *startA2;
%if config.kernel.out_data_t == 2:
    int32_t *startA3;
    int32_t *startA4;
%endif
%endif

    asm volatile("mv %0, %1":"=r"(startA):"r"(vecA));
%if config.kernel.out_data_t < 8:
    asm volatile("mv %0, %1":"=r"(startA2):"r"(vecA2));
%if config.kernel.out_data_t == 2:
    asm volatile("mv %0, %1":"=r"(startA3):"r"(vecA3));
    asm volatile("mv %0, %1":"=r"(startA4):"r"(vecA4));
%endif
%endif
    int32_t *ptrA  = (int32_t *) vecA ;
%if config.kernel.out_data_t < 8:
    int32_t *ptrA2  = (int32_t *) vecA2 ;
%if config.kernel.out_data_t == 2:
    int32_t *ptrA3  = (int32_t *) vecA3 ;
    int32_t *ptrA4  = (int32_t *) vecA4 ;
%endif
%endif
%else:
    int32_t *ptrA  = (int32_t *) pA ;
%if config.kernel.out_data_t < 8:
    int32_t *ptrA2  = (int32_t *) pA2 ;
%if config.kernel.out_data_t == 2:
    int32_t *ptrA3  = (int32_t *) pA3 ;
    int32_t *ptrA4  = (int32_t *) pA4 ;
%endif
%endif
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
%if config.kernel.out_data_t < 8:
    int32_t *ptrA2  = (int32_t *) pA2 ;
%if config.kernel.out_data_t == 2:
    int32_t *ptrA3  = (int32_t *) pA3 ;
    int32_t *ptrA4  = (int32_t *) pA4 ;
%endif
%endif

    uint32_t *ptrB  = (uint32_t *) pB ;
%endif

    ptrA  = MacLoadInit(1, 0, 0, 0, ptrA);
%if config.kernel.out_data_t < 8:
    ptrA2  = MacLoadInit(1, 0, 1, 0, ptrA2);
%if config.kernel.out_data_t == 2:
    ptrA3  = MacLoadInit(1, 0, 2, 0, ptrA3);
    ptrA4  = MacLoadInit(1, 0, 3, 0, ptrA4);
%endif
%endif

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
%if config.kernel.out_data_t < 8:
      sum2 = MacLoad${int(32/config.max_precision)}(1, 0, 1, 0, ptrA2, sum2);
      ptrA2 = MacLoadUpdate(ptrA2);
%if config.kernel.out_data_t == 2:
      sum3 = MacLoad${int(32/config.max_precision)}(1, 0, 2, 0, ptrA3, sum3);
      ptrA3 = MacLoadUpdate(ptrA3);
      sum4 = MacLoad${int(32/config.max_precision)}(1, 0, 3, 0, ptrA4, sum4);
      ptrA4 = MacLoadUpdate(ptrA4);
%endif
%endif

      ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);

%if config.kernel.in_data_t != config.kernel.wt_data_t:
%if (int(config.kernel.in_data_t/config.kernel.wt_data_t) == 4) or (int(config.kernel.wt_data_t/config.kernel.in_data_t) == 4):
      sum = MacLoad${int(32/config.max_precision)}(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);
%if config.kernel.out_data_t < 8:
      sum2 = MacLoad${int(32/config.max_precision)}(1, 0, 1, 0, ptrA2, sum2);
      ptrA2 = MacLoadUpdate(ptrA2);
%if config.kernel.out_data_t == 2:
      sum3 = MacLoad${int(32/config.max_precision)}(1, 0, 2, 0, ptrA3, sum3);
      ptrA3 = MacLoadUpdate(ptrA3);
      sum4 = MacLoad${int(32/config.max_precision)}(1, 0, 3, 0, ptrA4, sum4);
      ptrA4 = MacLoadUpdate(ptrA4);
%endif
%endif

      ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);

      sum = MacLoad${int(32/config.max_precision)}(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);
%if config.kernel.out_data_t < 8:
      sum2 = MacLoad${int(32/config.max_precision)}(1, 0, 1, 0, ptrA2, sum2);
      ptrA2 = MacLoadUpdate(ptrA2);
%if config.kernel.out_data_t == 2:
      sum3 = MacLoad${int(32/config.max_precision)}(1, 0, 2, 0, ptrA3, sum3);
      ptrA3 = MacLoadUpdate(ptrA3);
      sum4 = MacLoad${int(32/config.max_precision)}(1, 0, 3, 0, ptrA4, sum4);
      ptrA4 = MacLoadUpdate(ptrA4);
%endif
%endif

      ptrB  = MacLoadInit(0, 1, 0, 0, ptrB);
%endif
%if config.kernel.wt_data_t < config.kernel.in_data_t:
      pA  = ${config.unpack_wt_fn}(pA , vecA);
%if config.kernel.out_data_t < 8:
      pA2  = ${config.unpack_wt_fn}(pA2 , vecA2);
%if config.kernel.out_data_t == 2:
      pA3  = ${config.unpack_wt_fn}(pA3 , vecA3);
      pA4  = ${config.unpack_wt_fn}(pA4 , vecA4);
%endif
%endif

      ptrA   = MacLoadAssign(startA);
%if config.kernel.out_data_t < 8:
      ptrA2   = MacLoadAssign(startA2);
%if config.kernel.out_data_t == 2:
      ptrA3   = MacLoadAssign(startA3);
      ptrA4   = MacLoadAssign(startA4);
%endif
%endif
%endif
%if config.kernel.in_data_t < config.kernel.wt_data_t:
      pB  = ${config.unpack_in_fn}(pB , vecB);

      ptrB   = MacLoadAssign(startB);
%endif
      sum = MacLoad${int(32/config.max_precision)}(1, 0, 0, 0, ptrA, sum);
      ptrA = MacLoadUpdate(ptrA);
%if config.kernel.out_data_t < 8:
      sum2 = MacLoad${int(32/config.max_precision)}(1, 0, 1, 0, ptrA2, sum2);
      ptrA2 = MacLoadUpdate(ptrA2);
%if config.kernel.out_data_t == 2:
      sum3 = MacLoad${int(32/config.max_precision)}(1, 0, 2, 0, ptrA3, sum3);
      ptrA3 = MacLoadUpdate(ptrA3);
      sum4 = MacLoad${int(32/config.max_precision)}(1, 0, 3, 0, ptrA4, sum4);
      ptrA4 = MacLoadUpdate(ptrA4);
%endif
%endif

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
%if config.kernel.out_data_t < 8:
      pA2-=4;
%if config.kernel.out_data_t == 2:
      pA3-=4;
      pA4-=4;
%endif
%endif
%else:
      pA=((dim_vec >> ${int(math.log2((int(32/config.max_precision))*(int(config.kernel.wt_data_t/config.kernel.in_data_t))))}) << ${int(2+int(math.log2(int(config.kernel.wt_data_t/config.kernel.in_data_t))))});
%if config.kernel.out_data_t < 8:
      pA2+=((dim_vec >> ${int(math.log2((int(32/config.max_precision))*(int(config.kernel.wt_data_t/config.kernel.in_data_t))))}) << ${int(2+int(math.log2(int(config.kernel.wt_data_t/config.kernel.in_data_t))))});
%if config.kernel.out_data_t == 2:
      pA3+=((dim_vec >> ${int(math.log2((int(32/config.max_precision))*(int(config.kernel.wt_data_t/config.kernel.in_data_t))))}) << ${int(2+int(math.log2(int(config.kernel.wt_data_t/config.kernel.in_data_t))))});
      pA4+=((dim_vec >> ${int(math.log2((int(32/config.max_precision))*(int(config.kernel.wt_data_t/config.kernel.in_data_t))))}) << ${int(2+int(math.log2(int(config.kernel.wt_data_t/config.kernel.in_data_t))))});
%endif
%endif
%endif
%if config.kernel.in_data_t < config.kernel.wt_data_t:
      pB-=4;
%else:
      pB=((dim_vec >> ${int(math.log2((int(32/config.max_precision))*(int(config.kernel.in_data_t/config.kernel.wt_data_t))))}) << ${int(2+int(math.log2(int(config.kernel.in_data_t/config.kernel.wt_data_t))))});
%endif
      do
      {
%if config.less_precision == 2:
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
        sum += inA * inB;
        sum += inA2 * inB2;
        sum += inA3 * inB3;
        sum += inA4 * inB4;
%if config.kernel.out_data_t < 8:
%if config.kernel.wt_data_t == 2:
        inA =  (int8_t) bitext((int) *pA2, 2, 0);
        inA2 = (int8_t) bitext((int) *pA2, 2, 2);
        inA3 = (int8_t) bitext((int) *pA2, 2, 4);
        inA4 = (int8_t) bitext((int) *pA2, 2, 6);
        pA2++;
%elif config.kernel.wt_data_t == 4:
        inA =  (int8_t) bitext((int) *pA2, 4, 0);
        inA2 = (int8_t) bitext((int) *pA2, 4, 4);
        pA2++;
        inA3 = (int8_t) bitext((int) *pA2, 4, 0);
        inA4 = (int8_t) bitext((int) *pA2, 4, 4);
        pA2++;
%elif config.kernel.wt_data_t == 8:
        inA = *pA2;
        pA2++;
        inA2 = *pA2;
        pA2++;
        inA3 = *pA2;
        pA2++;
        inA4 = *pA2;
        pA2++;
%endif
        sum2 += inA * inB;
        sum2 += inA2 * inB2;
        sum2 += inA3 * inB3;
        sum2 += inA4 * inB4;
%if config.kernel.out_data_t == 2:
%if config.kernel.wt_data_t == 2:
        inA =  (int8_t) bitext((int) *pA3, 2, 0);
        inA2 = (int8_t) bitext((int) *pA3, 2, 2);
        inA3 = (int8_t) bitext((int) *pA3, 2, 4);
        inA4 = (int8_t) bitext((int) *pA3, 2, 6);
        pA3++;
%elif config.kernel.wt_data_t == 4:
        inA =  (int8_t) bitext((int) *pA3, 4, 0);
        inA2 = (int8_t) bitext((int) *pA3, 4, 4);
        pA3++;
        inA3 = (int8_t) bitext((int) *pA3, 4, 0);
        inA4 = (int8_t) bitext((int) *pA3, 4, 4);
        pA3++;
%elif config.kernel.wt_data_t == 8:
        inA = *pA3;
        pA3++;
        inA2 = *pA3;
        pA3++;
        inA3 = *pA3;
        pA3++;
        inA4 = *pA3;
        pA3++;
%endif
        sum3 += inA * inB;
        sum3 += inA2 * inB2;
        sum3 += inA3 * inB3;
        sum3 += inA4 * inB4;
%if config.kernel.wt_data_t == 2:
        inA =  (int8_t) bitext((int) *pA4, 2, 0);
        inA2 = (int8_t) bitext((int) *pA4, 2, 2);
        inA3 = (int8_t) bitext((int) *pA4, 2, 4);
        inA4 = (int8_t) bitext((int) *pA4, 2, 6);
        pA4++;
%elif config.kernel.wt_data_t == 4:
        inA =  (int8_t) bitext((int) *pA4, 4, 0);
        inA2 = (int8_t) bitext((int) *pA4, 4, 4);
        pA4++;
        inA3 = (int8_t) bitext((int) *pA4, 4, 0);
        inA4 = (int8_t) bitext((int) *pA4, 4, 4);
        pA4++;
%elif config.kernel.wt_data_t == 8:
        inA = *pA4;
        pA4++;
        inA2 = *pA4;
        pA4++;
        inA3 = *pA4;
        pA4++;
        inA4 = *pA4;
        pA4++;
%endif
        sum4 += inA * inB;
        sum4 += inA2 * inB2;
        sum4 += inA3 * inB3;
        sum4 += inA4 * inB4;
%endif
%endif
        col_cnt-=4;
%elif config.less_precision == 4:
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
        sum += inA * inB;
        sum += inA2 * inB2;
%if config.kernel.out_data_t < 8:
%if config.kernel.wt_data_t == 4:
        inA =  (int8_t) bitext((int) *pA2, 4, 0);
        inA2 = (int8_t) bitext((int) *pA2, 4, 4);
        pA2++;
%elif config.kernel.wt_data_t == 8:
        inA = *pA2;
        pA2++;
        inA2 = *pA2;
        pA2++;
%endif
        sum2 += inA * inB;
        sum2 += inA2 * inB2;
%if config.kernel.out_data_t == 2:
%if config.kernel.wt_data_t == 4:
        inA =  (int8_t) bitext((int) *pA3, 4, 0);
        inA2 = (int8_t) bitext((int) *pA3, 4, 4);
        pA3++;
%elif config.kernel.wt_data_t == 8:
        inA = *pA3;
        pA3++;
        inA2 = *pA3;
        pA3++;
%endif
        sum3 += inA * inB;
        sum3 += inA2 * inB2;
%if config.kernel.wt_data_t == 4:
        inA =  (int8_t) bitext((int) *pA4, 4, 0);
        inA2 = (int8_t) bitext((int) *pA4, 4, 4);
        pA4++;
%elif config.kernel.wt_data_t == 8:
        inA = *pA4;
        pA4++;
        inA2 = *pA4;
        pA4++;
%endif
        sum4 += inA * inB;
        sum4 += inA2 * inB2;
%endif
%endif
        col_cnt-=2;
%elif config.less_precision == 8:
        uint8_t inB = *pB;
        pB++;
        int8_t inA = *pA;
        pA++;
        sum += inA * inB;
%if config.kernel.out_data_t < 8:
        inA = *pA2;
        pA2++;

        sum2 += inA * inB;
%if config.kernel.out_data_t == 2:
        inA = *pA3;
        pA3++;

        sum3 += inA * inB;

        inA = *pA4;
        pA4++;

        sum4 += inA * inB;
%endif
%endif
        col_cnt--;
%endif
      }while (col_cnt);
    }
    if (flag_batch_norm && flag_relu)
    {
%if config.kernel.out_data_t == 8:
      *pOutBuffer = ${config.bn_fn}(sum, *k1, *lambda1, out_shift);
      pOutBuffer++;
%elif config.kernel.out_data_t == 4:
      sum = ${config.bn_fn}(sum, *k1, *lambda1, out_shift);
      sum2 = ${config.bn_fn}(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
      *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
      pOutBuffer++;
      k1+=2;
      lambda1+=2;
%elif config.kernel.out_data_t == 2:
      sum = ${config.bn_fn}(sum, *k1, *lambda1, out_shift);
      sum2 = ${config.bn_fn}(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
      sum3 = ${config.bn_fn}(sum3, *(k1 + 2), *(lambda1 + 2), out_shift);
      sum4 = ${config.bn_fn}(sum4, *(k1 + 3), *(lambda1 + 3), out_shift);
      k1+=4;
      lambda1+=4;
      sum = bitins(sum, n_mask2, sum2, mask2, off2);
      sum = bitins(sum, n_mask4, sum3, mask4, off4);
      *pOutBuffer = bitins(sum, n_mask6, sum4, mask6, off6);
      pOutBuffer++;
%endif
    }
    else
    {
      if (flag_relu == 1)
      {
%if config.kernel.out_data_t == 8:
        *pOutBuffer = ${config.relu_fn}(sum, out_mult, out_shift);
        pOutBuffer++;
%elif config.kernel.out_data_t == 4:
        sum = ${config.relu_fn}(sum, out_mult, out_shift);
        sum2 = ${config.relu_fn}(sum2, out_mult, out_shift);
        *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
        pOutBuffer++;
%elif config.kernel.out_data_t == 2:
        sum = ${config.relu_fn}(sum, out_mult, out_shift);
        sum2 = ${config.relu_fn}(sum2, out_mult, out_shift);
        sum3 = ${config.relu_fn}(sum3, out_mult, out_shift);
        sum4 = ${config.relu_fn}(sum4, out_mult, out_shift);
        sum = bitins(sum, n_mask2, sum2, mask2, off2);
        sum = bitins(sum, n_mask4, sum3, mask4, off4);
        *pOutBuffer = bitins(sum, n_mask6, sum4, mask6, off6);
        pOutBuffer++;
%endif
      }
      else
      {
%if config.kernel.out_data_t == 8:
        *pOutBuffer = (uint8_t) clip8(sum >> out_shift);
        pOutBuffer++;
%elif config.kernel.out_data_t == 4:
        sum = (uint8_t) clip4(sum >> out_shift);
        sum2 = (uint8_t) clip4(sum2 >> out_shift);
        *pOutBuffer = bitins(sum, n_mask, sum2, mask, off);
        pOutBuffer++;
%elif config.kernel.out_data_t == 2:
        sum = (uint8_t) clip2(sum >> out_shift);
        sum2 = (uint8_t) clip2(sum2 >> out_shift);
        sum3 = (uint8_t) clip2(sum3 >> out_shift);
        sum4 = (uint8_t) clip2(sum4 >> out_shift);
        sum = bitins(sum, n_mask2, sum2, mask2, off2);
        sum = bitins(sum, n_mask4, sum3, mask4, off4);
        *pOutBuffer = bitins(sum, n_mask6, sum4, mask6, off6);
        pOutBuffer++;
%endif
      }
    }
  }
  pi_cl_team_barrier(0);
}
