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

#include "${func_name}.h"
% if sdk == 'gap_sdk':
#include "pulp.h"
   % endif
#include "pmsis.h"
#include "dory_get_tile.h"
#include "dory_dma.h"
#include "pulp_nn_kernels.h"

void ${func_name}(
  void *args
) {
  //////////////////////////////////////////////////////////////////////////
  // arguments assigning: keeping same interface between L2 and L3 memory //
  //////////////////////////////////////////////////////////////////////////
  unsigned int *real_arg = (unsigned int *) args;
  unsigned int l3_x =(unsigned int)  real_arg[0];
  unsigned int l3_y =(unsigned int)  real_arg[1];
  unsigned int l3_W =(unsigned int)  real_arg[2];
  unsigned int l2_x =(unsigned int)  real_arg[3];
  unsigned int l2_x_2 =(unsigned int)  real_arg[4];
  unsigned int l2_y =(unsigned int)  real_arg[5];
  unsigned int l2_W =(unsigned int)  real_arg[6];
  unsigned int l1_buffer =(unsigned int)  real_arg[7];
  unsigned int hyperram =(unsigned int)  real_arg[8];
  unsigned int out_mult =(uint16_t)  real_arg[9];
  unsigned int out_shift = (uint16_t) real_arg[10];

  /////////////////////
  // DMA declaration //
  /////////////////////
  uint32_t dory_dma_channel = dory_dma_allocate();
  volatile DMA_copy DMA_copy_k_0, DMA_copy_k_1, DMA_copy_lambda_0, DMA_copy_lambda_1;
  volatile DMA_copy DMA_copy_W_0, DMA_copy_W_1,  DMA_copy_x, DMA_copy_y;

  DMA_copy_x.hwc_to_chw = 0;
  DMA_copy_x.stride_2d = ${l1_x_stride_w};
  DMA_copy_x.stride_1d = ${l1_x_stride_channels};
  DMA_copy_x.dir = 1;
  DMA_copy_x.tid = dory_dma_channel;

  /* NODE 0 Parameters */
% if has_bias_0 == 1:
  volatile DMA_copy DMA_copy_bias_0;
  DMA_copy_bias_0.hwc_to_chw = 0;
  DMA_copy_bias_0.stride_2d = 0;
  DMA_copy_bias_0.stride_1d = 0;
  DMA_copy_bias_0.dir = 1;
  DMA_copy_bias_0.tid = dory_dma_channel;

% endif
  DMA_copy_k_0.hwc_to_chw = 0;
  DMA_copy_k_0.stride_2d = 0;
  DMA_copy_k_0.stride_1d = 0;
  DMA_copy_k_0.number_of_2d_copies = 1;
  DMA_copy_k_0.number_of_1d_copies = 1;
  DMA_copy_k_0.dir = 1;
  DMA_copy_k_0.tid = dory_dma_channel;

  DMA_copy_lambda_0.hwc_to_chw = 0;
  DMA_copy_lambda_0.stride_2d = 0;
  DMA_copy_lambda_0.stride_1d = 0;
  DMA_copy_lambda_0.number_of_2d_copies = 1;
  DMA_copy_lambda_0.number_of_1d_copies = 1;
  DMA_copy_lambda_0.dir = 1;
  DMA_copy_lambda_0.tid = dory_dma_channel;
  
  DMA_copy_W_0.hwc_to_chw = 0;
  DMA_copy_W_0.stride_2d = ${l1_W_stride_nof_byte_0};
  DMA_copy_W_0.stride_1d = ${l1_W_stride_hw_byte_0};
% if flag_DW_0 == 0:
  DMA_copy_W_0.number_of_2d_copies = ${l1_W_output_channels_0};
  DMA_copy_W_0.number_of_1d_copies = ${fsx0 * fsy0};
% else:
  DMA_copy_W_0.number_of_2d_copies = 1;
  DMA_copy_W_0.number_of_1d_copies = 1;
% endif
  DMA_copy_W_0.dir = 1;
  DMA_copy_W_0.tid = dory_dma_channel;

  /* NODE 1 Parameters */
% if has_bias_1 == 1:
  volatile DMA_copy DMA_copy_bias_1;
  DMA_copy_bias_1.hwc_to_chw = 0;
  DMA_copy_bias_1.stride_2d = 0;
  DMA_copy_bias_1.stride_1d = 0;
  DMA_copy_bias_1.dir = 1;
  DMA_copy_bias_1.tid = dory_dma_channel;

% endif
  DMA_copy_k_1.hwc_to_chw = 0;
  DMA_copy_k_1.stride_2d = 0;
  DMA_copy_k_1.stride_1d = 0;
  DMA_copy_k_1.number_of_2d_copies = 1;
  DMA_copy_k_1.number_of_1d_copies = 1;
  DMA_copy_k_1.dir = 1;
  DMA_copy_k_1.tid = dory_dma_channel;

  DMA_copy_lambda_1.hwc_to_chw = 0;
  DMA_copy_lambda_1.stride_2d = 0;
  DMA_copy_lambda_1.stride_1d = 0;
  DMA_copy_lambda_1.number_of_2d_copies = 1;
  DMA_copy_lambda_1.number_of_1d_copies = 1;
  DMA_copy_lambda_1.dir = 1;
  DMA_copy_lambda_1.tid = dory_dma_channel;

  DMA_copy_W_1.hwc_to_chw = 0;
  DMA_copy_W_1.stride_2d = ${l1_W_stride_nof_byte_1};
  DMA_copy_W_1.stride_1d = ${l1_W_stride_hw_byte_1};
% if flag_DW_1 == 0:
  DMA_copy_W_1.number_of_2d_copies = ${l1_W_output_channels_1};
  DMA_copy_W_1.number_of_1d_copies = ${fsx1 * fsy1};
% else:
  DMA_copy_W_1.number_of_2d_copies = 1;
  DMA_copy_W_1.number_of_1d_copies = 1;
% endif
  DMA_copy_W_1.dir = 1;
  DMA_copy_W_1.tid = dory_dma_channel;

  DMA_copy_y.hwc_to_chw = 0;
  DMA_copy_y.stride_2d = ${l1_y_stride_w};
  DMA_copy_y.stride_1d = ${l1_y_stride_channels};
  DMA_copy_y.dir = 0;
  DMA_copy_y.tid = dory_dma_channel;

  volatile int p_r_0, p_l_0, p_t_0, p_b_0;
  volatile int p_r_1, p_l_1, p_t_1, p_b_1;
  volatile int pad_offset_h, pad_offset_w;
  volatile unsigned short x_tile_size_nif;
  volatile unsigned short x_tile_size_h;
  volatile unsigned short x_tile_size_w;
  volatile unsigned short y_tile_size_nof;
  volatile unsigned short y_tile_size_h;
  volatile unsigned short y_tile_size_w;
  volatile unsigned short W_tile_size_nof_0, W_tile_size_nof_1;
  volatile unsigned short W_tile_size_nif_0, W_tile_size_nif_1;
  int n_rows;
  volatile ${type} *x, *W0, *W1, *y, *b0, *b1;
% if FLAG_BATCHNORM_0 == 1:
  volatile int${activation_data_bit_0}_t *k0;
  volatile int${activation_data_bit_0}_t *lambda0;
% endif
% if FLAG_BATCHNORM_1 == 1:
  volatile int${activation_data_bit_1}_t *k1;
  volatile int${activation_data_bit_1}_t *lambda1;
% endif
  // last-tile flags
  int iter;
  // tile loop indeces
  int _i_nof_load=0, _i_nif_load=0, _i_h_load=0, _i_w_load=0;
  int _i_nof_exec=1, _i_nif_exec=1, _i_h_exec=1, _i_w_exec=1;
  volatile ${type} *im2col, *support_buffer;
  //// TO FIX: RIGHT NOW THE CONSTRAINT FOR THE SUPPORT BUFFER IS NOT WORKING PERFECTLY
  im2col = l1_buffer + ${buffer_l1_all};
  support_buffer = l1_buffer + ${buffer_l1_all};// + im2col_dim};

  //////////////////////
  // Bias transfering //
  //////////////////////
% if has_bias_0 == 1:
  DMA_copy_bias_0.ext = (uint32_t) l2_W+${l2_off_bias_0};
  DMA_copy_bias_0.loc = (uint32_t) (l1_buffer + ${l1_b_offset_0});
  DMA_copy_bias_0.number_of_2d_copies = 1;
  DMA_copy_bias_0.number_of_1d_copies = 1;
  DMA_copy_bias_0.length_1d_copy = (uint16_t) ${l2_b_size_byte_0};
  dory_dma_memcpy_async(&DMA_copy_bias_0);
  dory_dma_barrier(&DMA_copy_bias_0);

% endif
% if has_bias_1 == 1:
  DMA_copy_bias_1.ext = (uint32_t) l2_W+${l2_off_bias_1};
  DMA_copy_bias_1.loc = (uint32_t) (l1_buffer + ${l1_b_offset_1});
  DMA_copy_bias_1.number_of_2d_copies = 1;
  DMA_copy_bias_1.number_of_1d_copies = 1;
  DMA_copy_bias_1.length_1d_copy = (uint16_t) ${l2_b_size_byte_1};
  dory_dma_memcpy_async(&DMA_copy_bias_1);
  dory_dma_barrier(&DMA_copy_bias_1);

% endif
  pi_cl_team_barrier(0);

  int total_tiles = ${tile_dim_nof * tile_dim_nif * tile_dim_h * tile_dim_w};
  // tile loop nest
  for(iter=0; iter < total_tiles; iter++) {
    // check if last in any dimension
      x_tile_size_nif  = (_i_nif_load+1 == ${tile_dim_nif}) ? ${l1_input_channels_last}   : ${l1_input_channels};
      x_tile_size_h    = (_i_h_load+1   == ${tile_dim_h})   ? ${l1_x_h_last}              : ${l1_x_h};
      x_tile_size_w    = (_i_w_load+1   == ${tile_dim_w})   ? ${l1_x_w_last}              : ${l1_x_w};
      y_tile_size_nof  = (_i_nof_load+1 == ${tile_dim_nof}) ? ${l1_output_channels_last}  : ${l1_output_channels};
      y_tile_size_h    = (_i_h_load+1   == ${tile_dim_h})   ? ${l1_y_h_last}              : ${l1_y_h};
      y_tile_size_w    = (_i_w_load+1   == ${tile_dim_w})   ? ${l1_y_w_last}              : ${l1_y_w};
      W_tile_size_nof_0= (_i_nof_load+1 == ${tile_dim_nof}) ? ${l1_W_output_channels_last_0}: ${l1_W_output_channels_0};
      W_tile_size_nif_0= (_i_nif_load+1 == ${tile_dim_nif}) ? ${l1_W_input_channels_last_0} : ${l1_W_input_channels_0};
      W_tile_size_nof_1= (_i_nof_load+1 == ${tile_dim_nof}) ? ${l1_W_output_channels_last_1}: ${l1_W_output_channels_1};
      W_tile_size_nif_1= (_i_nif_load+1 == ${tile_dim_nif}) ? ${l1_W_input_channels_last_1} : ${l1_W_input_channels_1};
      // additionally overlap by padding for the first tile after a border one
      //this because in the first tile we use less pixels from x_buffer, since we have the ones of padding
      pad_offset_h=0, pad_offset_w=0;
      if(_i_h_load > 0)
        pad_offset_h = ${padding_top_0};
      if(_i_w_load > 0)
        pad_offset_w = ${padding_left_0};
      // transfer of next input tile in double buffering
      if (_i_nif_load!=_i_nif_exec || _i_w_load!=_i_w_exec || _i_h_load!=_i_h_exec)
      {
        DMA_copy_x.ext = dory_get_tile_3d(l2_x, _i_h_load, _i_w_load, _i_nif_load, ${l1_x_h}, ${l1_x_w}, ${l1_input_channels}, ${l2_x_w_0}, ${l2_input_channels_0*g_0},  ${conv_overlaph_0}, ${conv_overlapw_0},0, pad_offset_h, pad_offset_w, 0, ${x_data_bit_0});
        DMA_copy_x.loc = (l1_buffer + ${l1_x_offset});
        DMA_copy_x.number_of_2d_copies = x_tile_size_h;
        DMA_copy_x.number_of_1d_copies = x_tile_size_w;
        DMA_copy_x.length_1d_copy = (int) x_tile_size_nif * ${x_data_bit_0} / 8.0;
        dory_dma_memcpy_async(&DMA_copy_x);
        dory_dma_barrier(&DMA_copy_x);
      }
      if (_i_nif_load!=_i_nif_exec)
      {
        % if flag_DW_0 == 0:
        DMA_copy_W_0.ext = dory_get_tile_3d(l2_W, _i_nof_load, 0, _i_nif_load, ${l1_W_output_channels_0}, ${fsx0 * fsy0}, ${l1_W_input_channels_0}, ${fsx0 * fsy0}, ${l2_input_channels_0}, 0,0,0,0,0,0, ${W_data_bit_0});
        DMA_copy_W_0.number_of_2d_copies = W_tile_size_nof_0;
        DMA_copy_W_0.length_1d_copy = (int) W_tile_size_nif_0 * ${W_data_bit_0} / 8.0;
        % else:
        DMA_copy_W_0.ext = dory_get_tile_3d(l2_W, _i_nof_load, 0, 0, ${l1_W_output_channels_0}, ${fsx0 * fsy0}, ${l1_W_input_channels_0}, ${fsx0 * fsy0}, ${l2_input_channels_0}, 0,0,0,0,0,0, ${W_data_bit_0});
        DMA_copy_W_0.length_1d_copy = (int) W_tile_size_nof_0 * ${W_data_bit_0} * ${fsx0 * fsy0} / 8;
        % endif
        DMA_copy_W_0.loc = (l1_buffer + ${l1_W_offset_0});
        dory_dma_memcpy_async(&DMA_copy_W_0);
        dory_dma_barrier(&DMA_copy_W_0);
        % if FLAG_BATCHNORM_0 == 1:
        // TO FIX!!!!!!!!!!!!!!!!!!!!
        DMA_copy_k_0.ext = (uint32_t) l2_W;//+${l2_off_k_0} + ${l1_k_size_0}*_i_nof_load;
        DMA_copy_k_0.loc = (uint32_t) l1_buffer + ${l1_k_offset_0};
        DMA_copy_k_0.length_1d_copy = (uint16_t) W_tile_size_nof_0 * ${int(activation_data_bit_0/8)};
        dory_dma_memcpy_async(&DMA_copy_k_0);
        dory_dma_barrier(&DMA_copy_k_0);

        DMA_copy_lambda_0.ext = (uint32_t) l2_W;//+${l2_off_lambda_0} + ${l1_lambda_size_0}*_i_nof_load;
        DMA_copy_lambda_0.loc = (uint32_t) l1_buffer + ${l1_lambda_offset_0};
        DMA_copy_lambda_0.length_1d_copy = (uint16_t) W_tile_size_nof_0 * ${int(activation_data_bit_0/8)};
        dory_dma_memcpy_async(&DMA_copy_lambda_0);
        dory_dma_barrier(&DMA_copy_lambda_0);
        % endif
      }
      // transfer of next weight tile if changed input or output channels
      if (_i_nof_load!=_i_nof_exec)
      {
        % if flag_DW_1 == 0:
        DMA_copy_W_1.ext = dory_get_tile_3d(l2_W, _i_nof_load, 0, _i_nif_load, ${l1_W_output_channels_1}, ${fsx1 * fsy1}, ${l1_W_input_channels_1}, ${fsx1 * fsy1}, ${l2_input_channels_1}, 0,0,0,0,0,0, ${W_data_bit_1});
        DMA_copy_W_1.number_of_2d_copies = W_tile_size_nof_1;
        DMA_copy_W_1.length_1d_copy = (int) W_tile_size_nif_1 * ${W_data_bit_1} / 8.0;
        % else:
        DMA_copy_W_1.ext = dory_get_tile_3d(l2_W, _i_nof_load, 0, 0, ${l1_W_output_channels_1}, ${fsx1 * fsy1}, ${l1_W_input_channels_1}, ${fsx1 * fsy1}, ${l2_input_channels_1}, 0,0,0,0,0,0, ${W_data_bit_1});
        DMA_copy_W_1.length_1d_copy = (int) W_tile_size_nof_1 * ${W_data_bit_1} * ${fsx1 * fsy1} / 8;
        % endif
        DMA_copy_W_1.loc = (l1_buffer + ${l1_W_offset_1});
        dory_dma_memcpy_async(&DMA_copy_W_1);
        dory_dma_barrier(&DMA_copy_W_1);
        % if FLAG_BATCHNORM_1 == 1:

        DMA_copy_k_1.ext = (uint32_t) l2_W+${l2_off_k_1} + ${l1_k_size_1}*_i_nof_load;
        DMA_copy_k_1.loc = (uint32_t) l1_buffer + ${l1_k_offset_1};
        DMA_copy_k_1.length_1d_copy = (uint16_t) W_tile_size_nof_1 * ${int(activation_data_bit_1/8)};
        dory_dma_memcpy_async(&DMA_copy_k_1);
        dory_dma_barrier(&DMA_copy_k_1);

        DMA_copy_lambda_1.ext = (uint32_t) l2_W+${l2_off_lambda_1} + ${l1_lambda_size_1}*_i_nof_load;
        DMA_copy_lambda_1.loc = (uint32_t) l1_buffer + ${l1_lambda_offset_1};
        DMA_copy_lambda_1.length_1d_copy = (uint16_t) W_tile_size_nof_1 * ${int(activation_data_bit_1/8)};
        dory_dma_memcpy_async(&DMA_copy_lambda_1);
        dory_dma_barrier(&DMA_copy_lambda_1);
        % endif
      }
    // creation of the pointers to input, output, weights, lambda and k
    p_r_0 = 0; p_r_1 = 0;
    p_l_0 = 0; p_l_1 = 0;
    p_t_0 = 0; p_t_1 = 0;
    p_b_0 = 0; p_b_1 = 0;
    if (_i_h_load == 0){
      p_t_0 = ${padding_top_0}; p_t_1 = ${padding_top_1};}
    if (_i_w_load == 0){
      p_l_0 = ${padding_left_0}; p_l_1 = ${padding_left_1};}
    if (_i_h_load == ${tile_dim_h}-1){
      p_b_0 = ${padding_bottom_0}; p_b_1 = ${padding_bottom_1};}
    if (_i_w_load == ${tile_dim_w}-1){
      p_r_0 = ${padding_right_0}; p_r_1 = ${padding_right_1};}

    asm volatile("": : :"memory");

    x = (${type} *) (l1_buffer + ${l1_x_offset});
    % if FLAG_BATCHNORM_0 == 1:
    k0 = (int${activation_data_bit_0}_t *) (l1_buffer + ${l1_k_offset_0});
    lambda0 = (int${activation_data_bit_0}_t *) (l1_buffer + ${l1_lambda_offset_0});
    % endif
    % if FLAG_BATCHNORM_1 == 1:
    k1 = (int${activation_data_bit_1}_t *) (l1_buffer + ${l1_k_offset_1});
    lambda1 = (int${activation_data_bit_1}_t *) (l1_buffer + ${l1_lambda_offset_1});
    % endif
    % if has_bias_0 == 1:
    b0 = (${type} *) (l1_buffer + ${l1_b_offset_0} + _i_nof_load*${l1_bias_size_0});
    % endif
    % if has_bias_1 == 1:
    b1 = (${type} *) (l1_buffer + ${l1_b_offset_1} + _i_nof_load*${l1_bias_size_1});
    % endif
    W0 = (${type} *) (l1_buffer + ${l1_W_offset_0});
    W1 = (${type} *) (l1_buffer + ${l1_W_offset_1});
    y = (${type} *) (l1_buffer + ${l1_y_offset});
    n_rows = y_tile_size_h > 4 ? 4 : y_tile_size_h;

    pi_cl_team_barrier(0);
    asm volatile("": : :"memory");
% if "PW_DW" in func_name:
    pulp_nn_pw_dw_HWC_HWC_CHW_channels_lazy(
% elif "DW_PW" in func_name:
    pulp_nn_dw_pw_CHW_CHW_HWC_rows_lazy(
% endif
      x,
      im2col, // 8xKS x KS
      support_buffer, // RIGHE_DW_OUT x CH_IN x W_OUT x H_OUT
      b1,
      b0,
      y,
      W0,
      W1,
      k0,
      k1,
      lambda0,
      lambda1,
      out_mult,
      out_mult,
      out_shift,
      out_shift,
      x_tile_size_w, x_tile_size_h, x_tile_size_nif,
      y_tile_size_w, y_tile_size_h, y_tile_size_nof,
      ${fsy0}, ${fsx0},
      p_t_0, p_b_0, p_l_0, p_r_0,
      ${stride_0}, ${stride_0},
      ${FLAG_RELU_0}, ${FLAG_RELU_1}, 
      ${FLAG_BATCHNORM_0}, ${FLAG_BATCHNORM_1},
      n_rows);

    pi_cl_team_barrier(0);

    DMA_copy_y.ext = dory_get_tile_3d(l2_y, _i_h_load, _i_w_load, _i_nof_load, ${l1_y_h}, ${l1_y_w}, ${l1_output_channels}, ${l2_y_w_1}, ${int(l2_output_channels_1 * factor)}, 0, 0, 0, 0, 0, 0, ${y_data_bit_1});
    DMA_copy_y.loc = (l1_buffer + ${l1_y_offset});
    DMA_copy_y.number_of_2d_copies = y_tile_size_h;
    DMA_copy_y.number_of_1d_copies = y_tile_size_w;
    DMA_copy_y.length_1d_copy = (int) y_tile_size_nof * ${y_data_bit_1} / 8.0;
    dory_dma_memcpy_async(&DMA_copy_y); 
    dory_dma_barrier(&DMA_copy_y);  
    // update prev iterators
    _i_nof_exec = _i_nof_load;
    _i_nif_exec = _i_nif_load;
    _i_h_exec = _i_h_load;
    _i_w_exec = _i_w_load;

    _i_w_load += 1;
    if(_i_w_load==${tile_dim_w}) 
    {
      _i_w_load = 0;
      _i_h_load += 1;
      if(_i_h_load==${tile_dim_h}) 
      {
        _i_h_load = 0;
    % if flag_DW == 1:
      _i_nif_load += 1;
    % endif
        _i_nof_load += 1;
      }
    }
    pi_cl_team_barrier(0);

  }
  // wait for final write
  dory_dma_free(&DMA_copy_y);
}
