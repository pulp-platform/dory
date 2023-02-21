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

#include "pulp.h"
#include "dory.h"
% if ULTRA_VERBOSE:
#define VERBOSE_PRINT(...) printf(__VA_ARGS__)
% endif

int32_t ${func_name}(void* l2_x, void* l2_x_2, void* l2_y)  
{
% if int(func_name[-1]) % 2 == 0:
  unsigned int l1_x2       = 0x0;
  unsigned int l1_y       = 128000 - ${int((l1_x2_offset - l1_y_offset)/32)*32+32};
% else:
  unsigned int l1_x2       = 128000 - ${int((l1_y_offset)/32)*32+32};
  unsigned int l1_y       = 0x0;
% endif
  unsigned int l1_x      = ${int(l1_y_offset/32)*32+32};


  //unsigned int l1_x       = 0x0;
  //unsigned int l1_y       = l1_x + ${int(l1_y_offset/32)*32+32};
  //unsigned int l1_x2      = l1_x + ${int(l1_x2_offset/32)*32+32};

  /////////////////////
  // DMA declaration //
  /////////////////////
  uint32_t dory_dma_channel = dory_dma_allocate();
  volatile DMA_copy DMA_copy_x, DMA_copy_x2, DMA_copy_y;
  
  DMA_copy_x.hwc_to_chw = 0;
  DMA_copy_x.stride_2d = ${x_stride_w_byte};
  DMA_copy_x.stride_1d = ${x_stride_c_byte};
  DMA_copy_x.dir = 0;
  DMA_copy_x.dma_channel = dory_dma_channel;

  DMA_copy_x2.hwc_to_chw = 0;
  DMA_copy_x2.stride_2d = ${x_stride_w_byte};
  DMA_copy_x2.stride_1d = ${x_stride_c_byte};
  DMA_copy_x2.dir = 0;
  DMA_copy_x2.dma_channel = dory_dma_channel;

  DMA_copy_y.hwc_to_chw = 0;
  DMA_copy_y.stride_2d = ${int(y_w * y_h * y_data_size_byte / 8.0)};
  DMA_copy_y.stride_1d = ${int(y_w * y_data_size_byte / 8.0)};
  DMA_copy_y.dir = 1;
  DMA_copy_y.dma_channel = dory_dma_channel;

  volatile int p_r, p_l, p_t, p_b;

  volatile int  x_tile_size_nif;
  volatile int  x_tile_size_h;
  volatile int  x_tile_size_w;
  volatile int  x_tile_size_byte;
  volatile int  x_length_nif_byte;

  volatile int  W_tile_size_nof;
  volatile int  W_tile_size_nif;
  volatile int  W_tile_size_byte;
  volatile int W_length_nif_byte;

  volatile int y_tile_size_h;
  volatile int y_tile_size_w;
  volatile int y_tile_size_byte;
  volatile int y_length_nof_byte;
  // last-tile flags
  int iter, _i_nof=0, _i_nof_pre=0, _i_nif=0, _i_nif_pre=0, _i_h=0, _i_h_pre=0, _i_w=0, _i_w_pre=0;

  int total_tiles = ${tile_dim_nof * tile_dim_h * tile_dim_w};
  // tile loop nest
  for(iter=0; iter < total_tiles; iter++) {

    // check if last in any dimension

    x_tile_size_nif = (_i_nif+1   == ${tile_dim_nif}) ? ${x_tile_size_nif_last} : ${x_tile_size_nif};
    x_tile_size_h   = (_i_h+1     == ${tile_dim_h})   ? ${x_tile_size_h_last} : ${x_tile_size_h};
    x_tile_size_w   = (_i_w+1     == ${tile_dim_w})   ? ${x_tile_size_w_last} : ${x_tile_size_w};
    x_tile_size_byte = x_tile_size_nif*x_tile_size_h*x_tile_size_w*${x_data_size_byte}/8;
    x_length_nif_byte = (_i_nif+1 == ${tile_dim_nif})   ? ${x_tile_size_nif_byte_last} : ${x_tile_size_nif_byte};
    y_tile_size_h   = (_i_h+1     == ${tile_dim_h})   ? ${y_tile_size_h_last} : ${y_tile_size_h};
    y_tile_size_w   = (_i_w+1     == ${tile_dim_w})   ? ${y_tile_size_w_last} : ${y_tile_size_w};
    y_length_nof_byte = (_i_nof+1   == ${tile_dim_nof}) ? ${(y_tile_size_nof_last + 15) // 16 * 16} : ${y_tile_size_nof};

    uint32_t l2_x_tile = dory_get_tile_3d(l2_x,   _i_nif, _i_h, _i_w, ${x_tile_size_nif}, ${x_tile_size_h}, ${x_tile_size_w}, ${x_h}, ${x_w},0, ${conv_overlap1}, ${conv_overlap2}, 0, 0, 0, ${x_data_size_byte});
    uint32_t l2_x2_tile = dory_get_tile_3d(l2_x_2, _i_nif, _i_h, _i_w, ${x_tile_size_nif}, ${x_tile_size_h}, ${x_tile_size_w}, ${x_h}, ${x_w},0, ${conv_overlap1}, ${conv_overlap2}, 0, 0, 0, ${x_data_size_byte});

% if node.previous_layer_tiles != 1:
    DMA_copy_x.ext = l2_x_tile;
    DMA_copy_x.loc = l1_x2;
    DMA_copy_x.number_of_2d_copies = x_length_nif_byte;
    DMA_copy_x.number_of_1d_copies = x_tile_size_h;
    DMA_copy_x.length_1d_copy = x_tile_size_w;
    dory_dma_memcpy_async_digital(DMA_copy_x);
    dory_dma_barrier_digital(DMA_copy_x);
% endif

    DMA_copy_x2.ext = l2_x2_tile;
    DMA_copy_x2.loc = l1_x;
    DMA_copy_x2.number_of_2d_copies = x_length_nif_byte;
    DMA_copy_x2.number_of_1d_copies = x_tile_size_h;
    DMA_copy_x2.length_1d_copy = x_tile_size_w;
    dory_dma_memcpy_async_digital(DMA_copy_x2);
    dory_dma_barrier_digital(DMA_copy_x2);

    Layer_parameters kernel;
    kernel.padding = 0x0000;
    kernel.c = x_tile_size_nif;
    kernel.k = W_tile_size_nof;
    kernel.cx = x_tile_size_w;
    kernel.cy = x_tile_size_h;
    kernel.fx = ${fs1};
    kernel.fy = ${fs2};
    kernel.ox = y_tile_size_w;
    kernel.oy = y_tile_size_h;
    kernel.activation_function = 0;
    kernel.output_shift = ${out_shift};
    kernel.dilation = 1;
    kernel.stride = 0;

    dory_cores_barrier_digital();
    element_wise_sum(l2_x2_tile, l1_x, l2_x_tile, l1_x2, l1_y, &kernel);
    dory_cores_barrier_digital();

    uint32_t l2_y_tile = dory_get_tile_3d(l2_y, _i_nof, _i_h, _i_w, ${W_tile_size_nof}, ${y_tile_size_h}, ${y_tile_size_w}, ${y_h}, ${y_w}, 0, 0, 0, 0, 0, 0, ${y_data_size_byte});
    _i_nof_pre = _i_nof;
    _i_nif_pre = _i_nif;
    _i_h_pre   = _i_h;
    _i_w_pre   = _i_w;
      _i_w += 1;
      if(_i_w == ${tile_dim_w}) 
      {
        _i_w = 0;
        _i_h += 1;
        if(_i_h == ${tile_dim_h}) 
        {
          _i_h = 0;
      % if flag_DW == 1:
        _i_nif += 1;
      % endif
          _i_nof += 1;
        }
      }

% if tile_dim_nof * tile_dim_h * tile_dim_w > 1 or node.branch_out == 1:
    DMA_copy_y.ext = l2_y_tile;
    DMA_copy_y.loc = l1_y;
    DMA_copy_y.number_of_2d_copies = y_length_nof_byte;
    DMA_copy_y.number_of_1d_copies = y_tile_size_h;
    DMA_copy_y.length_1d_copy = y_tile_size_w;
    dory_dma_memcpy_async_digital(DMA_copy_y); 
    dory_dma_barrier_digital(DMA_copy_y);
% endif
  }

  dory_dma_deallocate(dory_dma_channel);
  return 0;
}

