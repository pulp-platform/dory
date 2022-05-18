/*
 * pulp_nn_avgpool_i4_u2.c
 * Georg Rutishauser <georgr@iis.ee.ethz.ch>
 * Inspired by CMSIS-NN AvgPool at https://github.com/ARM-software/CMSIS_5/blob/develop/CMSIS/NN/Source/PoolingFunctions/arm_avgpool_s8.c
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
#include "pulp_nn_kernels.h"




#define bitins(dst,not_mask_imm,src,mask_imm,off) __builtin_pulp_binsert(dst,not_mask_imm,src,mask_imm,off)
#define bitext_u(x,size,off) __builtin_pulp_bextractu(x,size,off)
#define bitext(x,size,off) __builtin_pulp_bextract(x,size,off)

void __attribute__ ((noinline))  pulp_nn_avgpool_i4_u2(
  int8_t * pIn,
  uint8_t * pOut,
  int32_t lambda,
  uint16_t out_shift,
  int32_t out_add,
  uint16_t dim_im_in_x,
  uint16_t dim_im_in_y,
  uint16_t ch_im_in,
  uint16_t dim_im_out_x,
  uint16_t dim_im_out_y,
  uint16_t dim_kernel_x,
  uint16_t dim_kernel_y,
  uint16_t padding_t,
  uint16_t padding_b,
  uint16_t padding_l,
  uint16_t padding_r,
  uint16_t stride_x,
  uint16_t stride_y,
  int flag_requant
)
{
  /* parallelization */
  int core_id = pi_core_id();
  int n_cores = NUM_CORES;
  if (dim_im_out_y < NUM_CORES)
  {
    n_cores = dim_im_out_y;
  }
  int Log2Core = log2(n_cores);
  int chunck = (dim_im_out_y >> Log2Core) + ((dim_im_out_y & (n_cores -1))!=0);
  int start = chunck * core_id;
  int stop = min(start + chunck, dim_im_out_y);
  int   i_x, i_y;



  uint32_t kernel_size_tot = dim_kernel_x * dim_kernel_y;
  int ch_im_in_r = ch_im_in >> 1;
  int ch_im_out_r = ch_im_in >> 2;
  int32_t sum[2] = {0};
  for (i_y = start; i_y < stop; i_y++)
    {
        for (i_x = 0; i_x < dim_im_out_x; i_x++)
        {
            int k_y_start, k_y_end;
            int k_x_start, k_x_end;

            const int8_t *pTmp, *pTmpInner;
            int8_t *pDst;
            // in data width: 4
            // out data width: 2
            // -> we need to do an output write every 2
            //    input channel "block" (1 byte) iterations
            uint32_t in_iter_cnt = 0;

            k_y_start = maxs32(0, i_y * stride_y - padding_b);
            k_y_end = mins32(i_y * stride_y - padding_t + dim_kernel_y, dim_im_in_y);

            k_x_start = maxs32(0, i_x * stride_x - padding_l);
            k_x_end = mins32(i_x * stride_x - padding_r + dim_kernel_x, dim_im_in_x);

            pTmp = pIn;
            pDst = &pOut[ch_im_out_r * (i_x + i_y * dim_im_out_x)];
            int k_x, k_y;
            uint8_t out_el = 0;

            for (int ch_cnt = 0; ch_cnt < ch_im_in_r; ch_cnt++)
            {
              sum[0] = 0;
              sum[1] = 0;
                for (k_y = k_y_start; k_y < k_y_end; k_y++)
                {
                    for (k_x = k_x_start; k_x < k_x_end; k_x++)
                    {
                        pTmpInner = pTmp + (ch_im_in_r * (k_x + k_y * dim_im_in_x));
                        int8_t cur_chans = *pTmpInner;

                        sum[0] += (int32_t) bitext((int) cur_chans, 4, 0);

                        sum[1] += (int32_t) bitext((int) cur_chans, 4, 4);
                    }
                }
                int32_t out_large;
                if (flag_requant) {
                  out_large = (sum[0] * lambda + out_add) >> out_shift;
                    out_el |= (clip2(out_large)  << (in_iter_cnt * 4 + 0));
                  out_large = (sum[1] * lambda + out_add) >> out_shift;
                    out_el |= (clip2(out_large)  << (in_iter_cnt * 4 + 2));
                  } else {
                  out_large = sum[0] / kernel_size_tot;
                  out_el |= (clip2(out_large)  << (in_iter_cnt * 4 + 0));
                  out_large = sum[1] / kernel_size_tot;
                  out_el |= (clip2(out_large)  << (in_iter_cnt * 4 + 2));
                }
                if (in_iter_cnt++ == 1) {
                    pDst[(ch_cnt >> (1))] = out_el;
                    in_iter_cnt = 0;
                    out_el = 0;
                }
                pTmp++;
            }
        }
    }
 pi_cl_team_barrier(0);
}
