/*
 * add_layer_template.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 * Thorir Mar Ingolfsson <thoriri@iis.ee.ethz.ch>
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
${verbose_log}

#include "${func_name}.h"
% if sdk == 'gap_sdk':
#include "pulp.h"
  % endif
#include "pmsis.h"
#include "dory_get_tile.h"
#include "dory_dma.h"
#include "pulp_nn_kernels.h"

% if ULTRA_VERBOSE:
#define VERBOSE_PRINT(...) printf(__VA_ARGS__)
% endif




void ${func_name}(
  void *args
) {
  unsigned int *real_arg = (unsigned int *) args;
  unsigned int l3_x =(unsigned int)  real_arg[0];
  unsigned int l3_y =(unsigned int)  real_arg[1];
  unsigned int l3_W =(unsigned int)  real_arg[2];
  unsigned int l2_x =(unsigned int)  real_arg[3];
  unsigned int l2_x2 =(unsigned int)  real_arg[4];
  unsigned int l2_y =(unsigned int)  real_arg[5];
  unsigned int l2_W =(unsigned int)  real_arg[6];
  unsigned int l1_buffer =(unsigned int)  real_arg[7];
  unsigned int hyperram =(unsigned int)  real_arg[8];
  unsigned int out_mult_in =(unsigned int)  real_arg[9];

  unsigned short x_tile_size_nif;
  unsigned short  x_tile_size_h;
  unsigned short  x_tile_size_w;
  unsigned short  x_tile_size_byte;
  unsigned short  x_length_h_px;
  unsigned short  x_length_nif_byte;
  int pad_offset_h, pad_offset_w;

  ${type} *x;
  ${type} *x2;
  ${type} *y;
  int y_tile_size_nof;
  int y_tile_size_h;
  int y_tile_size_w;
  int y_tile_size_byte;
  int y_length_h_px;
  int y_length_nof_byte;
  // copy first tiles
  //l2_x has input activations
  uint32_t dory_dma_channel = dory_dma_allocate();
  volatile DMA_copy DMA_copy_x, DMA_copy_x2, DMA_copy_y;

  DMA_copy_x.hwc_to_chw = 0;
  DMA_copy_x.stride_2d = ${x_stride_w_byte};
  DMA_copy_x.stride_1d = ${x_stride_c_byte};
  DMA_copy_x.dir = 1;
  DMA_copy_x.tid = dory_dma_channel;

  DMA_copy_x2.hwc_to_chw = 0;
  DMA_copy_x2.stride_2d = ${x_stride_w_byte};
  DMA_copy_x2.stride_1d = ${x_stride_c_byte};
  DMA_copy_x2.dir = 1;
  DMA_copy_x2.tid = dory_dma_channel;
  
  DMA_copy_y.hwc_to_chw = 0;
  DMA_copy_y.stride_2d = ${y_stride_w_byte};
  DMA_copy_y.stride_1d = ${y_stride_c_byte};
  DMA_copy_y.dir = 0;
  DMA_copy_y.tid = dory_dma_channel;

  // tile loop indeces
  int _i_nof_load=0, _i_nif_load=0, _i_h_load=0, _i_w_load=0;


  // last-tile flags
  int last_nof, last_nif, last_h, last_w;
  int iter;
  // tile loop nest
  for(iter=0; iter<${tile_dim_nof}*${tile_dim_h}*${tile_dim_w}; iter++) {

    last_nof = (_i_nof_load+1 == ${tile_dim_nof}) ? 1 : 0;
    last_nif = (_i_nof_load+1 == ${tile_dim_nif}) ? 1 : 0;
    last_h = (_i_h_load+1 == ${tile_dim_h}) ? 1 : 0;
    last_w = (_i_w_load+1 == ${tile_dim_w}) ? 1 : 0;

    x_tile_size_nif = (last_nif) ? ${x_tile_size_nif_last} : ${x_tile_size_nif};
    x_tile_size_h   = (last_h)   ? ${x_tile_size_h_last} : ${x_tile_size_h};
    x_tile_size_w   = (last_w)   ? ${x_tile_size_w_last} : ${x_tile_size_w};
    x_tile_size_byte = x_tile_size_nif*x_tile_size_h*x_tile_size_w*${x_data_size_byte}/8;
    x_length_nif_byte = (last_nif)   ? ${x_tile_size_nif_byte_last} : ${x_tile_size_nif_byte};
    // additionally overlap by padding for the first tile after a border one
    //this because in the first tile we use less pixels from x_buffer, since we have the ones of padding

    DMA_copy_x.ext = dory_get_tile_3d(l2_x, _i_h_load, _i_w_load, _i_nif_load, ${x_tile_size_h}, ${x_tile_size_w}, ${x_tile_size_nif}, ${x_w}, ${nif},  ${conv_overlap1}, ${conv_overlap2},0, 0, 0, 0, ${x_data_size_byte});
    DMA_copy_x.loc = (l1_buffer + ${l1_x_offset});
    DMA_copy_x.number_of_2d_copies = x_tile_size_h;
    DMA_copy_x.number_of_1d_copies = x_tile_size_w;
    DMA_copy_x.length_1d_copy = x_length_nif_byte;
    dory_dma_memcpy_async(&DMA_copy_x);
    dory_dma_barrier(&DMA_copy_x);

    DMA_copy_x2.ext = dory_get_tile_3d(l2_x2, _i_h_load, _i_w_load, _i_nif_load, ${x_tile_size_h}, ${x_tile_size_w}, ${x_tile_size_nif}, ${x_w}, ${nif},  ${conv_overlap1}, ${conv_overlap2},0, 0, 0, 0, ${x_data_size_byte2});
    DMA_copy_x2.loc = (l1_buffer + ${l1_x2_offset});
    DMA_copy_x2.number_of_2d_copies = x_tile_size_h;
    DMA_copy_x2.number_of_1d_copies = x_tile_size_w;
    DMA_copy_x2.length_1d_copy = x_length_nif_byte;
    dory_dma_memcpy_async(&DMA_copy_x2);
    dory_dma_barrier(&DMA_copy_x2);

    y_tile_size_h   = (last_h)   ? ${y_tile_size_h_last} : ${y_tile_size_h};
    y_tile_size_w   = (last_w)   ? ${y_tile_size_w_last} : ${y_tile_size_w};

    x = (${type} *) (l1_buffer + ${l1_x_offset});
    x2 = (${type} *) (l1_buffer + ${l1_x2_offset});
    y = (${type} *) (l1_buffer + ${l1_y_offset});

    y_tile_size_nof = (last_nof) ? ${y_tile_size_nof_last} : ${y_tile_size_nof};
    y_tile_size_h   = (last_h)   ? ${y_tile_size_h_last} : ${y_tile_size_h};
    y_tile_size_w   = (last_w)   ? ${y_tile_size_w_last} : ${y_tile_size_w};
    y_tile_size_byte = y_tile_size_nof*y_tile_size_h*y_tile_size_w*${y_data_size_byte}/8;
    y_length_nof_byte = (last_nof)   ? ${y_length_nof_byte_last} : ${y_tile_size_nof_byte};
    asm volatile("": : :"memory");
    pi_cl_team_barrier(0);
    % if optional_type == '8bit':
    pulp_nn_add(
      x,
      x2,
      y,
      ${inmul2},
      ${inmul1},
      ${outshift},
      x_tile_size_w,
      x_tile_size_h,
      x_tile_size_nif
      );
    % else:
    ${"x" if 'hw' in optional_type else ""}pulp_nn_add_${data_type_x[0]}${x_data_size_byte}_${data_type_x2[0]}${x_data_size_byte2}_${data_type_y[0]}${y_data_size_byte}(
      x,
      x2,
      y,
      ${inmul2},
      ${inadd2},
      ${inshift2},
      ${inmul1},
      ${inadd1},
      ${inshift1},
      ${outmul},
      ${outadd},
      ${outshift},
      x_tile_size_w,
      x_tile_size_h,
      x_tile_size_nif,
      1
      );
    % endif

    pi_cl_team_barrier(0);
    // wait for DMA write
    // copying output back to L2
    DMA_copy_y.ext = dory_get_tile_3d(l2_y, _i_h_load, _i_w_load, _i_nof_load, ${y_tile_size_h}, ${y_tile_size_w}, ${y_tile_size_nof}, ${y_w}, ${nof}, 0, 0, 0, 0, 0, 0, ${y_data_size_byte});
    DMA_copy_y.loc = (l1_buffer + ${l1_y_offset});
    DMA_copy_y.number_of_2d_copies = y_tile_size_h;
    DMA_copy_y.number_of_1d_copies = y_tile_size_w;
    DMA_copy_y.length_1d_copy = y_length_nof_byte;
    dory_dma_memcpy_async(&DMA_copy_y); 
    dory_dma_barrier(&DMA_copy_y); 

    // loop nest is nof,h,w,(nif=0)
    _i_w_load += 1;
    if(_i_w_load==${tile_dim_w}) 
    {
      _i_w_load = 0;
      _i_h_load += 1;
      if(_i_h_load==${tile_dim_h}) 
      {
        _i_h_load = 0;
        _i_nif_load += 1;
        _i_nof_load += 1;
      }
    }
  }
% if not TEST:
  dory_dma_free(&DMA_copy_y);
% endif
}
