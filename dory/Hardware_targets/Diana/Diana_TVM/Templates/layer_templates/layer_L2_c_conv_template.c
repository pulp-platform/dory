/*
 * layer_template.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 * Francesco Conti <f.conti@unibo.it>
 *
 * Copyright (C) 2018-2020 University of Bologna
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

% if node.weight_bits == 2:
L2_DATA uint32_t Weights_${func_name}[${weights_dimensions}] = {
% else:
L2_DATA uint8_t Weights_${func_name}[${weights_dimensions}] = {
% endif
${weights_vectors}};

int32_t ${func_name}(void* l2_x, void* l2_y)
{
  //////////////////////////////////////////////////////////////////////////
  // arguments assigning: keeping same interface between L2 and L3 memory //
  //////////////////////////////////////////////////////////////////////////
% if W_data_size_byte == 2:
  unsigned int l2_BN	  = Weights_${func_name} + ${int((64 if nif < 64 else nif) * (128 if nof < 128 else nof) * fs1 * fs2 * W_data_size_byte / 8)};
% endif
% if int(func_name[-1]) % 2 == 0 or node.skip_L2_L1 == False:
  unsigned int l1_x       = 0x0;
% if W_data_size_byte == 2:
  // unsigned int l1_y       = 131072 - ${int((l1_W_offset - l1_y_offset)/32)*32+32}*2;
  // NOW IT IS FIXED: TO ADJUST!!!
  unsigned int l1_y       = 0x0 + 16*512*4; 
% else:
  // unsigned int l1_y       = 131072 - ${int((l1_W_offset - l1_y_offset)/32)*32+32};
  unsigned int l1_y       = l1_x + ${int((l1_y_offset)/32)*32+32}; 
% endif
% else:
% if W_data_size_byte == 2:
  unsigned int l1_x       = 131072 - ${int((l1_y_offset)/32)*32+32}*2;
% else:
  unsigned int l1_x       = 131072 - ${int((l1_y_offset)/32)*32+32};
% endif
  unsigned int l1_y       = 0x0;
% endif
  unsigned int l1_weights = 0x0;
  /////////////////////
  // DMA declaration //
  /////////////////////
  uint32_t dory_dma_channel = dory_dma_allocate();
  volatile DMA_copy DMA_copy_x, DMA_copy_y;

  DMA_copy_x.hwc_to_chw = 0;
  DMA_copy_x.stride_2d = ${int(x_w * x_h * x_data_size_byte / 8.0)};
  DMA_copy_x.stride_1d = ${int(x_w * x_data_size_byte / 8.0)};
  DMA_copy_x.dir = 0;
  DMA_copy_x.dma_channel = dory_dma_channel;
  
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

% if flag_DW == 0:
  int total_tiles = ${tile_dim_nof * tile_dim_nif * tile_dim_h * tile_dim_w};
% else:
  int total_tiles = ${tile_dim_nof * tile_dim_h * tile_dim_w};
% endif
  // tile loop nest
  for(iter=0; iter < total_tiles; iter++) {
    // check if last in any dimension
% if "Gemm" in optional:
    x_tile_size_nif = (_i_nif+1   == ${tile_dim_nif}) ? ${(x_tile_size_nif_last + 15) // 16 * 16} : ${(x_tile_size_nif + 15) // 16 * 16};
% else:
    x_tile_size_nif = (_i_nif+1 == ${tile_dim_nif}) ? ${x_tile_size_nif_last} : ${x_tile_size_nif};
% endif
    x_tile_size_h   = (_i_h+1     == ${tile_dim_h})   ? ${x_tile_size_h_last} : ${x_tile_size_h};
% if W_data_size_byte == 8 and 'Gemm' not in optional:
    x_tile_size_w   = (_i_w+1     == ${tile_dim_w})   ? ${(x_tile_size_w_last + 15) // 16 * 16} : ${(x_tile_size_w + 15) // 16 * 16};
    y_tile_size_w   = (_i_w+1     == ${tile_dim_w})   ? ${(y_tile_size_w_last + 15) // 16 * 16} : ${(y_tile_size_w + 15) // 16 * 16};
% else:
    x_tile_size_w   = (_i_w+1     == ${tile_dim_w})   ? ${x_tile_size_w_last} : ${x_tile_size_w };
    y_tile_size_w   = (_i_w+1     == ${tile_dim_w})   ? ${(y_tile_size_w_last)} : ${(y_tile_size_w)};
% endif
    x_tile_size_byte = x_tile_size_nif*x_tile_size_h*x_tile_size_w*${x_data_size_byte}/8;
% if 'Gemm' in optional:
    x_length_nif_byte = (_i_nif+1 == ${tile_dim_nif})   ? ${(x_tile_size_nif_byte_last + 15) // 16 * 16} : ${(x_tile_size_nif_byte + 15) // 16 * 16};
% else:
    x_length_nif_byte = (_i_nif+1 == ${tile_dim_nif})   ? ${x_tile_size_nif_byte_last} : ${x_tile_size_nif_byte};
% endif
    y_tile_size_h   = (_i_h+1     == ${tile_dim_h})   ? ${y_tile_size_h_last} : ${y_tile_size_h};
% if W_data_size_byte == 8:
    y_length_nof_byte = (_i_nof+1   == ${tile_dim_nof}) ? ${(W_tile_size_nof_last + 15) // 16 * 16} : ${(W_tile_size_nof + 15) // 16 * 16};
    W_tile_size_nof = (_i_nof+1   == ${tile_dim_nof}) ? ${(W_tile_size_nof_last + 15) // 16 * 16} : ${(W_tile_size_nof + 15) // 16 * 16};
% else:
    y_length_nof_byte = (_i_nof+1   == ${tile_dim_nof}) ? ${(W_tile_size_nof_last)} : ${(W_tile_size_nof)};
    W_tile_size_nof = (_i_nof+1   == ${tile_dim_nof}) ? ${(W_tile_size_nof_last)} : ${(W_tile_size_nof)};
% endif

% if node.previous_layer_tiles > 1 or node.skip_L2_L1 == False:
% if W_data_size_byte == 2:
    {
      int channel_number = (int)((x_length_nif_byte+63)/64);
      int block_number = (int)((x_tile_size_h * x_tile_size_w * channel_number + 511) / 512);
      for (int blocks_index = 0; blocks_index < block_number; blocks_index++)
      {
        int byte_transfer = 0; 
        if (blocks_index == (block_number-1))
          byte_transfer = 16 * (((x_tile_size_h * x_tile_size_w * channel_number) % 512) ? ((x_tile_size_h * x_tile_size_w * channel_number) % 512) : 512);
        else
          byte_transfer = 16 * 512;
        memcpy_analog(l2_x + blocks_index*byte_transfer, l1_x+blocks_index*16*512, byte_transfer * 4, DMA_copy_x.dir, 4 );
        dory_dma_barrier_analog(DMA_copy_x);
      }
    }
% elif W_data_size_byte == 8:
    int pad_offset_h=0, pad_offset_w=0;
    if(_i_h > 0)
      pad_offset_h = ${padding_top};
    if(_i_w > 0)
      pad_offset_w = ${padding_left};
    uint32_t l2_x_tile = dory_get_tile_3d(l2_x, _i_nif, _i_h, _i_w, ${x_tile_size_nif}, ${x_tile_size_h}, ${x_tile_size_w}, ${x_h}, ${x_w},0, ${conv_overlap1}, ${conv_overlap2}, 0, pad_offset_h, pad_offset_w, ${x_data_size_byte});
    DMA_copy_x.ext = l2_x_tile;
    DMA_copy_x.loc = l1_x;
    DMA_copy_x.number_of_2d_copies = x_length_nif_byte;
    DMA_copy_x.number_of_1d_copies = x_tile_size_h;
    DMA_copy_x.length_1d_copy = x_tile_size_w;
    dory_dma_memcpy_async_digital(DMA_copy_x);
    dory_dma_barrier_digital(DMA_copy_x); 
% endif
% endif

    Layer_parameters kernel;
% if W_data_size_byte == 2:
    kernel.padding = 0x0000;
    if (_i_h == 0)
      kernel.padding  = ${padding_top}<<2;
    if (_i_w == ${tile_dim_w}-1)
      kernel.padding += ${padding_right}<<4;
    if (_i_h == ${tile_dim_h}-1)
      kernel.padding += ${padding_bottom};
    if (_i_w == 0)
      kernel.padding += ${padding_left}<<6;
% elif W_data_size_byte == 8:
    kernel.padding = 0x0000;
    if (_i_h == 0)
      kernel.padding  = ${padding_top}<<8;
    if (_i_w == ${tile_dim_w}-1)
      kernel.padding += ${padding_right};
    if (_i_h == ${tile_dim_h}-1)
      kernel.padding += ${padding_bottom}<<12;
    if (_i_w == 0)
      kernel.padding += 0<<4; // it should be (the nearer multiple of 32 of W) - W
% endif
    kernel.c = x_tile_size_nif;
    kernel.k = W_tile_size_nof;
    kernel.cx = x_tile_size_w;
    kernel.cy = x_tile_size_h;
    kernel.fx = ${fs1};
    kernel.fy = ${fs2};
    kernel.oy = y_tile_size_h;
% if data_type_y == 'int':
    kernel.activation_function = 0;
% elif data_type_y == 'uint':
    kernel.activation_function = 1;
% endif
    kernel.output_shift = ${out_shift};
    kernel.dilation = 1;
% if W_data_size_byte == 8:
    kernel.ox = y_tile_size_w;
    kernel.stride = ${1 if stride > 1 else 0};
% elif W_data_size_byte == 2:
    kernel.ox_unroll = 1;
    for (int i = 0; i < 2; i++)
    {
      if (((kernel.ox_unroll * 2 * kernel.k) <= 512) && ((kernel.c * ${fs2} * (${fs1} + kernel.ox_unroll * 2 - 1)) <= 1152))
        kernel.ox_unroll = kernel.ox_unroll * 2;
    }
    kernel.stride = ${stride};
    kernel.ox = (int) y_tile_size_w / kernel.ox_unroll;
% endif
    uint32_t l2_W_tile = Weights_${func_name} + _i_nof * ${W_tile_size_nof * int((nif + 15) // 16 * 16 * fs1 * fs2 + b_data_size_byte / 8)};
% if W_data_size_byte == 8:
    dory_cores_barrier_digital();
% if 'Gemm' in optional: 
    digital_fully_connected(l2_x_tile, l1_x, l2_W_tile, l1_weights, l1_y, &kernel);
% elif flag_DW == 1:
    digital_depthwise_conv_2d(l2_x_tile, l1_x, l2_W_tile, l1_weights, l1_y, &kernel);
% elif flag_DW == 0: 
    digital_conv_2d(l2_x_tile, l1_x, l2_W_tile, l1_weights, l1_y, &kernel);
% endif 
    dory_cores_barrier_digital();
% elif W_data_size_byte == 2:
    dory_cores_barrier_analog();
% if 'Gemm' in optional: 
    analog_fully_connected(l2_x, l1_x, l2_W_tile, l1_weights, l1_y, &kernel);
% elif flag_DW == 1:
    analog_depthwise_conv_2d(l2_x, l1_x, l2_W_tile, l2_BN, l1_weights, l1_y, &kernel);
% elif flag_DW == 0: 
    analog_conv_2d(l2_x, l1_x, l2_W_tile, l2_BN, l1_weights, l1_y, &kernel);
% endif 
    dory_cores_barrier_analog();
% endif

    uint32_t l2_y_tile = dory_get_tile_3d(l2_y, _i_nof, _i_h, _i_w, ${W_tile_size_nof}, ${y_tile_size_h}, ${y_tile_size_w}, ${y_h}, ${y_w}, 0, 0, 0, 0, 0, 0, ${y_data_size_byte});
    _i_nof_pre = _i_nof;
    _i_nif_pre = _i_nif;
    _i_h_pre   = _i_h;
    _i_w_pre   = _i_w;
  % if tile_dim_nif != 1 and flag_DW == 0:
    // loop nest is nof,h,w,nif
    _i_nif += 1;
    if(_i_nif==${tile_dim_nif}) 
    {
      _i_nif = 0;
  % endif
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
  % if tile_dim_nif != 1 and flag_DW == 0:
    }
  % endif

% if tile_dim_nof * tile_dim_nif * tile_dim_h * tile_dim_w > 1 or node.branch_out == 1 or node.skip_L2_L1 == False:
% if W_data_size_byte == 2:
    {
      int channel_number = (int)((y_length_nof_byte+63)/64);
      int block_number = (int)((y_tile_size_h * y_tile_size_w * channel_number + 511) / 512);
      for (int blocks_index = 0; blocks_index < block_number; blocks_index++)
      {
        int byte_transfer = 0; 
        if (blocks_index == (block_number-1))
          byte_transfer = 16 * (((y_tile_size_h * y_tile_size_w * channel_number) % 512) ? ((y_tile_size_h * y_tile_size_w * channel_number) % 512) : 512);
        else
          byte_transfer = 16 * 512;
        memcpy_analog(l2_y + blocks_index*byte_transfer, l1_y+blocks_index*16*512, byte_transfer * 4, DMA_copy_y.dir, 4 );
        dory_dma_barrier_analog(DMA_copy_y);
      }
    }
% elif W_data_size_byte == 8:
    DMA_copy_y.ext = l2_y_tile;
    DMA_copy_y.loc = l1_y;
    DMA_copy_y.number_of_2d_copies = y_length_nof_byte;
    DMA_copy_y.number_of_1d_copies = y_tile_size_h;
    DMA_copy_y.length_1d_copy = y_tile_size_w;
    dory_dma_memcpy_async_digital(DMA_copy_y);
    dory_dma_barrier_digital(DMA_copy_y); 
% endif
% endif
  }
  dory_dma_deallocate(dory_dma_channel);
  return 0;
}
