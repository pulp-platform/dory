/*
 * pooling_layer_template.c
 * Alessio Burrello <alessio.burrello@unibo.it>
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

#include "${func_name}.h"
% if ULTRA_VERBOSE:
#define VERBOSE_PRINT(...) printf(__VA_ARGS__)
% endif
void ${func_name}(layer* layer_i) {
  unsigned int l2_x =         layer_i->L2_input;
  unsigned int l2_y =         layer_i->L2_output;
  unsigned int out_shift =    layer_i->out_shift;

  int p_r, p_l, p_t, p_b;
  p_t = ${padding_top};
  p_l = ${padding_left};
  p_b = ${padding_bottom};
  p_r = ${padding_right};
% if 'Max' in optional:
  //TO FIX: RIGHT NOW THE MAXPOOL IS NOT IMPLEMENTED, BUT SIMPLY IDENTICAL TO AVGPOOL
  for (int k = 0; k < ${nof}; ++k)
  {
    for (int j = 0; j < ${y_w}; ++j)
    {
      for (int i = 0; i < ${y_h}; ++i)
      {
        int sum = 0;
        for (int z = 0; z < ${fs1}; ++z)
        {
          for (int t = 0; t < ${fs2}; ++t)
          {
              sum += *(uint8_t *) (l2_x + k + j * ${nof * stride} + i * ${nof * y_w * stride * stride} + z * ${nof} + t * ${nof * y_w * stride});
          }
        }
      *(uint8_t *) (l2_y + k + j * ${nof} + i * ${nof * y_w}) = (sum / ${fs1*fs2});
      }
    }
  } 
% else:
  //chunk_index = 0;
  for (int k = 0; k < ${nof}; ++k)
  {
    for (int j = 0; j < ${y_w}; ++j)
    {
      for (int i = 0; i < ${y_h}; ++i)
      {
        int sum = 0;
        for (int z = 0; z < ${fs1}; ++z)
        {
          for (int t = 0; t < ${fs2}; ++t)
          {
              sum += *(uint8_t *) (l2_x + k + j * ${nof * stride} + i * ${nof * y_w * stride * stride} + z * ${nof} + t * ${nof * y_w * stride});
          }
        }
      *(uint8_t *) (l2_y + k + j * ${nof} + i * ${nof * y_w}) = (sum / ${fs1*fs2});
      }
    }
  }
% endif
}
