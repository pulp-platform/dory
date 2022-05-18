/*
 * pulp_nn_linear_u8_i2_i4.c
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



void pulp_nn_linear_u8_i2_i4(
                        uint8_t *pIn,
                        int8_t *pBias,
                        int8_t *pOut,
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
    int8_t mask2 = 0x0c;
    int8_t n_mask2 = ~ mask2;
    int8_t mask4 = 0x30;
    int8_t n_mask4 = ~ mask4;
    int8_t mask6 = 0xc0;
    int8_t n_mask6 = ~ mask6;
    int8_t off2 = 2;
    int8_t off4 = 4;
    int8_t off6 = 6;
    uint16_t dim_vec_in = dim_vec;
    uint16_t dim_vec_wt = dim_vec >> 1;

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

    v4u vecA[2];
    v4s vecB[2];
    v4s vecB2[2];
    v4s vecB3[2];
    v4s vecB4[2];

    int8_t *pOutBuffer = (int8_t *) pOut + (start >> 2);

    int i;
    int64_t *k1 = pKappa + start;
    int64_t *lambda1 = pLambda + start;

    for(i=start; i<stop; i+=4)
    {
        int sum = 0;
        int sum2 = 0;
        int sum3 = 0;
        int sum4 = 0;

        uint8_t *pA = pIn;
        int8_t *pB = pWeight + (i * dim_vec_wt);
        int8_t *pB2 = pB + dim_vec_wt;
        int8_t *pB3 = pB2 + dim_vec_wt;
        int8_t *pB4 = pB3 + dim_vec_wt;

        for (int j=0; j<(dim_vec >> 3); j++)
        {
          vecA[0] = *((v4u*)pA);
          pA+=4;
          vecA[1] = *((v4u*)pA);
          pulp_nn_i4_to_i8(pB,vecB);
          pulp_nn_i4_to_i8(pB2,vecB2);
          pulp_nn_i4_to_i8(pB3,vecB3);
          pulp_nn_i4_to_i8(pB4,vecB4);
          sum = SumDotp4(vecA[0], vecB[0], sum);
          sum = SumDotp4(vecA[1], vecB[1], sum);
          sum2 = SumDotp4(vecA[0], vecB2[0], sum2);
          sum2 = SumDotp4(vecA[1], vecB2[1], sum2);
          sum3 = SumDotp4(vecA[0], vecB3[0], sum3);
          sum3 = SumDotp4(vecA[1], vecB3[1], sum3);
          sum4 = SumDotp4(vecA[0], vecB4[0], sum4);
          sum4 = SumDotp4(vecA[1], vecB4[1], sum4);
          pA+=4;
          pB+=4;
          pB2+=4;
          pB3+=4;
          pB4+=4;
        }
        uint16_t col_cnt = dim_vec & 0x7;
        while (col_cnt)
        {
          uint8_t inA = *pA;
          pA++;
          uint8_t inA2 = *pA;
          pA++;
          int8_t inB = (int8_t) bitext((int) *pB, 4, 0);
          int8_t inB2 = (int8_t) bitext((int) *pB, 4, 4);
          pB++;
          int8_t inB3 = (int8_t) bitext((int) *pB2, 4, 0);
          int8_t inB4 = (int8_t) bitext((int) *pB2, 4, 4);
          pB2++;
          int8_t inB5 = (int8_t) bitext((int) *pB3, 4, 0);
          int8_t inB6 = (int8_t) bitext((int) *pB3, 4, 4);
          pB3++;
          int8_t inB7 = (int8_t) bitext((int) *pB4, 4, 0);
          int8_t inB8 = (int8_t) bitext((int) *pB4, 4, 4);
          pB4++;
          sum += inA * inB;
          sum += inA2 * inB2;
          sum2 += inA * inB3;
          sum2 += inA2 * inB4;
          sum3 += inA * inB5;
          sum3 += inA2 * inB6;
          sum4 += inA * inB7;
          sum4 += inA2 * inB8;
          col_cnt--;
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
            sum = (int8_t)  clips2(sum >> out_shift);
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
