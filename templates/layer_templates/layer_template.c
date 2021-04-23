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
% if ULTRA_VERBOSE:
#define VERBOSE_PRINT(...) printf(__VA_ARGS__)
% endif

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
  unsigned int out_mult_in =(unsigned int)  real_arg[9];
  unsigned int inmul1 = (unsigned int) real_arg[10];
  unsigned int inmul2 = (unsigned int) real_arg[11];
  unsigned int out_shift_in = (unsigned int) real_arg[12];

  //////////////////////////
  // Variable declaration //
  //////////////////////////
  unsigned int dma_evt;
  volatile int p_r, p_l, p_t, p_b;
% if tile_dim_nif*tile_dim_h*tile_dim_w != 1:
  volatile  unsigned short x_tile_size_nif;
  volatile unsigned short  x_tile_size_h;
  volatile unsigned short  x_tile_size_w;
  volatile unsigned short  x_tile_size_byte;
  volatile unsigned short  x_length_nif_byte;
  volatile int pad_offset_h, pad_offset_w;
% endif  
  volatile unsigned short  W_tile_size_nof;
  volatile unsigned short  W_tile_size_nif;
  volatile unsigned short  W_tile_size_byte;
  volatile unsigned short W_length_nif_byte;
  volatile ${type} *x;
  volatile ${type} *W;
  volatile ${type} *y;
  volatile ${type} *b;
% if FLAG_BATCHNORM == 1:
% if act_dim_bit == 32:
  volatile int32_t *k;
  volatile int32_t *lambda;
% else:
  volatile int64_t *k;
  volatile int64_t *lambda;
% endif
% endif
  volatile int x_tile_size_nif_exec;
  volatile int x_tile_size_h_exec;
  volatile int x_tile_size_w_exec;
  volatile int y_tile_size_nof;
  volatile int y_tile_size_h;
  volatile int y_tile_size_w;
  volatile int y_tile_size_byte;
  volatile int y_length_nof_byte;
  volatile int db_x;
  volatile int db_W;
  volatile int db_act;
  volatile int db_y;
  volatile int exec_db_x;
  volatile int exec_db_W;
  volatile int exec_db_act;
  volatile pi_cl_dma_copy_t copy_k;
  volatile pi_cl_dma_copy_t copy_lambda;
  // double buffering state
  int db_state_x=0;
  int db_state_W=0;
  int db_state_y=1;
  // last-tile flags
  int iter;
  // tile loop indeces
  int _i_nof_load=0, _i_nif_load=0, _i_h_load=0, _i_w_load=0;
  int _i_nof_exec=0, _i_nif_exec=0, _i_h_exec=0, _i_w_exec=0;
% if has_bias == 1:
  int has_bias = ${has_bias};
% endif
  volatile ${type} *im2col;
  im2col = l1_buffer + ${buffer_l1_all};
% if flag_DW == 1:
  volatile ${type} *pwt_buffer;
  pwt_buffer = im2col + ${im2col_dim};
% endif
% if FLAG_RELU == 1:
  uint16_t out_mult = out_mult_in;
  uint16_t out_shift = out_shift_in;
% endif
  /////////////////////////////////////
  /// Not Double buffered transfers ///
  /////////////////////////////////////
% if has_bias == 1:
  if(pi_core_id()==0)
  {
    pi_cl_dma_copy_t copy;
    copy.dir = PI_CL_DMA_DIR_EXT2LOC;
    copy.merge = 0;
    copy.size = (uint16_t) ${b_size_byte};
    copy.id = 0;
    copy.ext = (uint32_t) l2_W+${l2_off_bias};
    copy.loc = (uint32_t) (l1_buffer + ${l1_b_offset});
    pi_cl_dma_memcpy(&copy);  
    pi_cl_dma_wait(&copy);
  }
% endif
% if FLAG_BATCHNORM == 1:
  if(pi_core_id()==0)
  {
    copy_k.dir = PI_CL_DMA_DIR_EXT2LOC;
    copy_k.merge = 0;
    copy_k.size = (uint16_t) ${k_tile_size_byte_transfer};
    copy_k.id = 0;
    copy_k.ext = (uint32_t) l2_W+${l2_off_k};
    copy_k.loc = (uint32_t) l1_buffer + ${l1_k_offset};
    pi_cl_dma_memcpy(&copy_k);   
    copy_lambda.dir = PI_CL_DMA_DIR_EXT2LOC;
    copy_lambda.merge = 0;
    copy_lambda.size = (uint16_t) ${lambda_tile_size_byte_transfer};
    copy_lambda.id = 0;
    copy_lambda.ext = (uint32_t) l2_W+${l2_off_lambda};
    copy_lambda.loc = (uint32_t) l1_buffer + ${l1_lambda_offset};
    pi_cl_dma_memcpy(&copy_lambda);                                                   
    pi_cl_dma_wait(&copy_k);                                                    
    pi_cl_dma_wait(&copy_lambda);
  }
% endif
  pi_cl_team_barrier(0);
% if chip == 'GAP8v3':
  //////////////////////////////////////////////////////////////
  // Allocation of one channel per each core for DMA transfer //
  //////////////////////////////////////////////////////////////
% if dma_parallelization == '8-cores':
  dma_evt = mchan_alloc();
% elif dma_parallelization == '1-core':
  if (pi_core_id()==0)
    dma_evt = mchan_alloc();
% endif
% endif
  ////////////////////////////
  // First tile transfering //
  ////////////////////////////
% if dma_parallelization == '1-core':
  if (pi_core_id()==0)
  {
% endif
  % if flag_DW == 1:
  dory_dma_memcpy_3d_custom_hwc_to_chw(
  % else:
  dory_dma_memcpy_3d_custom(
  % endif
  l2_x, // ext
  (l1_buffer + ${l1_x_offset}) + 0, // loc
  ${x_tile_size_byte}, // size: dimension of the buffer
  ${x_stride_w_byte}, // stride_1: stride for the 3d copy: if we have to copy on n_features axis, this is the stride to change from first 2D space to the next ones.
  ${x_stride_c_byte}, // stride_0: stride to be passed to 2d_copy: the dimension w of the in image
  ${x_tile_size_h},// length_2: how many 2_d copies we need -> the dimension of the tile in n_features direction
  ${x_tile_size_nif_byte}, // length_0: legnth of the 1_d copy, the length of tile in w direction
  1, // dir
  &dma_evt // copy
  );
  % if flag_DW == 1:
  dory_dma_memcpy_3d_custom_blocking(
  % else:
  dory_dma_memcpy_3d_custom(
  % endif
  l2_W, // ext
  (l1_buffer + ${l1_W_offset}) + 0, // loc offset caused by size of tile_x*2 (double_buffer) and tile_y*2 (double buffer)
  ${W_tile_size_byte}, // size: dimension of matrix of weight * bytes_per_weight
  ${W_stride_nof_byte}, // stride_1: stride for the 3d copy: if we have to copy on n_features axis, this is the stride to change from first 2D space to the next ones.
  ${W_stride_hw_byte}, // stride_0: stride to be passed to 2d_copy: the dimension w of the in image
  ${W_tile_size_nof}, // length_2: how many 2_d copies we need -> the dimension of the tile in n_features direction
  ${W_tile_nif_byte}, // length_0: legnth of the 1_d copy, the length of tile in w direction
  1, // dir
  &dma_evt // copy
  );
  % if chip == 'GAP8v3':
  mchan_barrier(dma_evt);
  % endif
% if dma_parallelization == '1-core':
  }
% endif
  pi_cl_team_barrier(0);


  // tile loop nest
% if flag_DW == 0:
  for(iter=0; iter<${tile_dim_nof}*${tile_dim_nif}*${tile_dim_h}*${tile_dim_w}; iter++) {
% else:
  for(iter=0; iter<${tile_dim_nof}*${tile_dim_h}*${tile_dim_w}; iter++) {
% endif
  % if tile_dim_nif != 1 and flag_DW == 0:
    // loop nest is nof,h,w,nif
    _i_nif_load += 1;
    if(_i_nif_load==${tile_dim_nif}) 
    {
      _i_nif_load = 0;
      _i_w_load += 1;
      if(_i_w_load==${tile_dim_w}) 
      {
        _i_w_load = 0;
        _i_h_load += 1;
        if(_i_h_load==${tile_dim_h}) 
        {
          _i_h_load = 0;
          _i_nof_load += 1;
        }
      }
    }
  % else:
    // loop nest is nof,h,w,(nif=0)
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
  % endif
    // check if last in any dimension

    // compute double buffering offsets and update db state
    db_x = !db_state_x ? ${x_tile_size_byte} : 0;
    db_W = !db_state_W ? ${W_tile_size_byte} : 0;
    db_y = !db_state_y ? ${y_tile_size_byte} : 0;
% if FLAG_BATCHNORM == 1:
    db_act = !db_state_W ? ${k_tile_size_byte_transfer} : 0;
% endif
  % if tile_dim_nif*tile_dim_h*tile_dim_w != 1:
    exec_db_x = db_state_x ? ${x_tile_size_byte} : 0;
  % else:
    exec_db_x = 0;
  % endif
    db_state_x = ! db_state_x;
    exec_db_W = db_state_W ? ${W_tile_size_byte} : 0;
% if FLAG_BATCHNORM == 1:
    exec_db_act = db_state_W ? ${k_tile_size_byte_transfer} : 0;
% endif
    if (_i_nif_load!=_i_nif_exec || _i_nof_load!=_i_nof_exec)
      db_state_W = ! db_state_W;
    //switch all double buffering offset and y only after that all n_input_features have been analyzed: we need to pass all n_in to produce a single fil
///////// POSSIBLE BUG FIX!!!!! DB_STATE_Y NOT SWITCHED /////////////

    // double buffered reads
  % if flag_DW == 0:
    if(iter<${tile_dim_nof}*${tile_dim_nif}*${tile_dim_h}*${tile_dim_w}-1) 
    {
  % else:
    if(iter<${tile_dim_nof}*${tile_dim_h}*${tile_dim_w}-1) 
    {
      asm volatile("": : :"memory");
  % endif
    % if tile_dim_nif*tile_dim_h*tile_dim_w != 1:
      x_tile_size_nif = (_i_nif_load+1 == ${tile_dim_nif}) ? ${x_tile_size_nif_last} : ${x_tile_size_nif};
      x_tile_size_h   = (_i_h_load+1 == ${tile_dim_h})   ? ${x_tile_size_h_last} : ${x_tile_size_h};
      x_tile_size_w   = (_i_w_load+1 == ${tile_dim_w})   ? ${x_tile_size_w_last} : ${x_tile_size_w};
      x_tile_size_byte = x_tile_size_nif*x_tile_size_h*x_tile_size_w*${x_data_size_byte}/8;
      x_length_nif_byte = (_i_nif_load+1 == ${tile_dim_nif})   ? ${x_tile_size_nif_byte_last} : ${x_tile_size_nif_byte};
      // additionally overlap by padding for the first tile after a border one
      //this because in the first tile we use less pixels from x_buffer, since we have the ones of padding
      pad_offset_h=0, pad_offset_w=0;
      if(_i_h_load > 0)
        pad_offset_h = ${padding_top};
      if(_i_w_load > 0)
        pad_offset_w = ${padding_left};
    % endif
      y_tile_size_h   = (_i_h_load+1 == ${tile_dim_h})   ? ${y_tile_size_h_last} : ${y_tile_size_h};
      y_tile_size_w   = (_i_w_load+1 == ${tile_dim_w})   ? ${y_tile_size_w_last} : ${y_tile_size_w};
      W_tile_size_nof = (_i_nof_load+1 == ${tile_dim_nof}) ? ${W_tile_size_nof_last} : ${W_tile_size_nof};
      W_tile_size_nif = (_i_nif_load+1 == ${tile_dim_nif}) ? ${W_tile_size_nif_last} : ${W_tile_size_nif};
    % if flag_DW == 1:
      W_tile_size_byte = W_tile_size_nof*W_tile_size_nif*${fs1}*${fs2};
    % else:
      W_tile_size_byte = W_tile_size_nof*W_tile_size_nif*${W_data_size_byte}*${fs1}*${fs2}/8;
    % endif
      W_length_nif_byte = (_i_nif_load+1 == ${tile_dim_nif}) ? ${W_tile_size_nif_byte_last} : ${W_tile_nif_byte};
    // transfer of next input tile in double buffering
    % if tile_dim_nif*tile_dim_h*tile_dim_w != 1:
% if dma_parallelization == '1-core':
      if (pi_core_id()==0)
      {
% endif
    % if flag_DW == 1:
      dory_dma_memcpy_3d_custom_hwc_to_chw(
    % else:
      dory_dma_memcpy_3d_custom(
    % endif
      dory_get_tile_3d(l2_x, _i_h_load, _i_w_load, _i_nif_load, ${x_tile_size_h}, ${x_tile_size_w}, ${x_tile_size_nif}, ${x_w}, ${nif*g},  ${conv_overlap1}, ${conv_overlap2},0, pad_offset_h, pad_offset_w, 0, ${x_data_size_byte}), // extern
      (l1_buffer + ${l1_x_offset}) + db_x, // loc
      x_tile_size_byte, // size: dimension of the buffer
      ${x_stride_w_byte}, // stride_1: stride for the 3d copy: if we have to copy on n_features axis, this is the stride to change from first 2D space to the next ones.
      ${x_stride_c_byte}, // stride_0: stride to be passed to 2d_copy: the dimension w of the in image
      x_tile_size_h,// length_2: how many 2_d copies we need -> the dimension of the tile in n_features direction
      x_length_nif_byte, // length_0: legnth of the 1_d copy, the length of tile in w direction
      1, // dir
      &dma_evt // copy
      );
% if dma_parallelization == '1-core':
      }
% endif
    % endif
      // transfer of next weight tile if changed input or output channels
      if (_i_nif_load!=_i_nif_exec || _i_nof_load!=_i_nof_exec)
      {
% if dma_parallelization == '1-core':
        if (pi_core_id()==0)
        {
% endif
        % if flag_DW == 1:
        dory_dma_memcpy_3d_custom_blocking(
        % else:
        dory_dma_memcpy_3d_custom_weights(
        % endif
      % if flag_DW == 0:
        dory_get_tile_3d(l2_W, _i_nof_load, 0, _i_nif_load, ${W_tile_size_nof}, ${fs1}*${fs2}, ${W_tile_size_nif}, ${fs1}*${fs2}, ${nif}, 0,0,0,0,0,0, ${W_data_size_byte}), // ext
      % else:
        dory_get_tile_3d(l2_W, _i_nof_load, 0, 0, ${W_tile_size_nof*8/W_data_size_byte}, ${fs1}*${fs2}, ${W_tile_size_nif}, ${fs1}*${fs2}, ${nif}, 0,0,0,0,0,0, ${W_data_size_byte}), // ext
      % endif
        (l1_buffer + ${l1_W_offset}) + db_W, // loc
        W_tile_size_byte, // size: dimension of matrix of weight * bytes_per_weight
        ${W_stride_nof_byte}, // stride_1: stride for the 3d copy: if we have to copy on n_features axis, this is the stride to change from first 2D space to the next ones.
        ${W_stride_hw_byte}, // stride_0: stride to be passed to 2d_copy: the dimension w of the in image
        W_tile_size_nof, // length_2: how many 2_d copies we need -> the dimension of the tile in n_features direction
        W_length_nif_byte, // length_0: legnth of the 1_d copy, the length of tile in w direction
        1, // dir
        &dma_evt // copy
        );
% if dma_parallelization == '1-core':
        }
% endif
% if FLAG_BATCHNORM == 1:
        if(pi_core_id()==0)
        {
          copy_k.dir = PI_CL_DMA_DIR_EXT2LOC;
          copy_k.merge = 0;
          copy_k.size = (uint16_t) W_tile_size_nof * ${int(act_dim_bit/8)};
          copy_k.id = 0;
          copy_k.ext = (uint32_t) l2_W+${l2_off_k} + ${k_tile_size_byte_transfer}*_i_nof_load;
          copy_k.loc = (uint32_t) l1_buffer + ${l1_k_offset} + db_act;
          pi_cl_dma_memcpy(&copy_k);   
          copy_lambda.dir = PI_CL_DMA_DIR_EXT2LOC;
          copy_lambda.merge = 0;
          copy_lambda.size = (uint16_t) W_tile_size_nof * ${int(act_dim_bit/8)};
          copy_lambda.id = 0;
          copy_lambda.ext = (uint32_t) l2_W+${l2_off_lambda} + ${lambda_tile_size_byte_transfer}*_i_nof_load;
          copy_lambda.loc = (uint32_t) l1_buffer + ${l1_lambda_offset} + db_act;
          pi_cl_dma_memcpy(&copy_lambda);      
        }
% endif
      }
    }
    // creation of the pointers to input, output, weights, lambda and k
% if flag_DW == 1:
    asm volatile("": : :"memory");
% endif
    x = (${type} *) (l1_buffer + ${l1_x_offset} + exec_db_x);
% if FLAG_BATCHNORM == 1:
% if act_dim_bit == 32:
    k = (int32_t *) (l1_buffer + ${l1_k_offset} + exec_db_act);
    lambda = (int32_t *) (l1_buffer + ${l1_lambda_offset} + exec_db_act);
% else:
    k = (int64_t *) (l1_buffer + ${l1_k_offset} + exec_db_act);
    lambda = (int64_t *) (l1_buffer + ${l1_lambda_offset} + exec_db_act);
% endif
% endif
% if has_bias == 1:
    b = (${type} *) (l1_buffer + ${l1_b_offset} + _i_nof_exec*${bias_tile_size_byte});
% endif
    W = (${type} *) (l1_buffer + ${l1_W_offset} + exec_db_W);
    y = (${type} *) (l1_buffer + ${l1_y_offset} + db_y);
    // parameter passed to the kernel. Input and output sizes
    x_tile_size_nif_exec = (_i_nif_exec+1 == ${tile_dim_nif}) ? ${x_tile_size_nif_last} : ${x_tile_size_nif};
    x_tile_size_h_exec   = (_i_h_exec+1 == ${tile_dim_h})   ? ${x_tile_size_h_last} : ${x_tile_size_h};
    x_tile_size_w_exec   = (_i_w_exec+1 == ${tile_dim_w})   ? ${x_tile_size_w_last} : ${x_tile_size_w};
    y_tile_size_nof = (_i_nof_exec+1 == ${tile_dim_nof}) ? ${y_tile_size_nof_last} : ${y_tile_size_nof};
    y_tile_size_h   = (_i_h_exec+1 == ${tile_dim_h})   ? ${y_tile_size_h_last} : ${y_tile_size_h};
    y_tile_size_w   = (_i_w_exec+1 == ${tile_dim_w})   ? ${y_tile_size_w_last} : ${y_tile_size_w};
    y_tile_size_byte = y_tile_size_nof*y_tile_size_h*y_tile_size_w*${y_data_size_byte}/8;
    y_length_nof_byte = (_i_nof_exec+1 == ${tile_dim_nof})   ? ${y_length_nof_byte_last} : ${y_tile_size_nof_byte};
    p_r = 0;
    p_l = 0;
    p_t = 0;
    p_b = 0;
    if (_i_h_exec == 0)
      p_t = ${padding_top};
    if (_i_w_exec == 0)
      p_l = ${padding_left};
    if (_i_h_exec == ${tile_dim_h}-1)
      p_b = ${padding_bottom};
    if (_i_w_exec == ${tile_dim_w}-1)
      p_r = ${padding_right};

    pi_cl_team_barrier(0);
  % if tile_dim_nof*tile_dim_nif*tile_dim_h*tile_dim_w==1:
    asm volatile("": : :"memory");
  % endif
  % if flag_DW==1:
    asm volatile("": : :"memory");
  % endif
% if flag_DW == 0:
  % if optional_type == '8bit' or optional_type == '1D_Conv':
    % if 'Relu0' in func_name:
    pulp_nn_conv_Ho_parallel(
    % elif '_last' in func_name and ('Gemm' in func_name or 'MatMul' in func_name):
    pulp_nn_linear_out_32( 
    x,
    W,
    x_tile_size_nif_exec,
    y_tile_size_nof,
    0, 0, 1, 1, 0, 0,
    y,
    0, 0, &dma_evt );
    % elif 'Gemm' in func_name or 'MatMul' in func_name:
    pulp_nn_linear( 
    x,
    W,
    x_tile_size_nif_exec,
    y_tile_size_nof,
    0, 0, 
      % if FLAG_RELU == 1:
    out_shift,
    out_mult,
      % else:
    0,
    0,
      % endif
      % if FLAG_BATCHNORM == 1:
    k,
    lambda,
      % else:
    0,
    0,
      % endif
    y,
    ${FLAG_RELU}, ${FLAG_BATCHNORM}, &dma_evt );  
    % elif fs1*fs2>1 or 'Gemm' in func_name or stride>1:
    pulp_nn_conv_Ho_parallel(
    % else:
    pulp_nn_pointwise_HoWo_parallel(
    % endif
  % elif 'mixed-sw' in optional_type  and ('Conv' in func_name):
    pulp_nn_conv_u${x_data_size_byte}_u${y_data_size_byte}_i${W_data_size_byte}(
  % elif 'mixed-hw' in optional_type  and ('Conv' in func_name):
    xpulp_nn_conv_u${x_data_size_byte}_u${y_data_size_byte}_i${W_data_size_byte}(
  % elif 'mixed-sw' in optional_type  and ('Gemm' in func_name or 'MatMul' in func_name):
    pulp_nn_linear_u${x_data_size_byte}_i${y_data_size_byte}_i${W_data_size_byte}( 
      x,
      W,
      x_tile_size_nif_exec,
      y_tile_size_nof,
      0, 0, 
      % if FLAG_RELU == 1:
      out_shift,
      out_mult,
      % else:
      0,
      0,
      % endif
      % if FLAG_BATCHNORM == 1:
      k,
      lambda,
      % else:
      0,
      0,
      % endif
      y,
      ${FLAG_RELU}, ${FLAG_BATCHNORM}, &dma_evt );  
  % elif 'mixed-hw' in optional_type  and ('Gemm' in func_name or 'MatMul' in func_name):
    pulp_nn_linear_u${x_data_size_byte}_i${y_data_size_byte}_i${W_data_size_byte}( 
      x,
      W,
      x_tile_size_nif_exec,
      y_tile_size_nof,
      0, 0, 
      % if FLAG_RELU == 1:
      out_shift,
      out_mult,
      % else:
      0,
      0,
      % endif
      % if FLAG_BATCHNORM == 1:
      k,
      lambda,
      % else:
      0,
      0,
      % endif
      y,
      ${FLAG_RELU}, ${FLAG_BATCHNORM}, &dma_evt );  
  % endif
% else:
  % if optional_type == '8bit':
    % if fs1 == 3 and fs2 == 3 and stride==1:
    pulp_nn_depthwise_generic(
    % elif fs1*fs2 < 4:
    pulp_nn_depthwise_generic_less_4_weights(
    % else:
    pulp_nn_depthwise_generic(
    % endif
  % elif optional_type == 'mixed-sw':
    pulp_nn_depthwise_u${x_data_size_byte}_u${y_data_size_byte}_i${W_data_size_byte}(
  % elif optional_type == 'mixed-hw':
    xpulp_nn_depthwise_u${x_data_size_byte}_u${y_data_size_byte}_i${W_data_size_byte}( 
  % endif
% endif
% if 'Gemm' not in func_name and 'MatMul' not in func_name:
    x,
    x_tile_size_w_exec,
    x_tile_size_h_exec,
    x_tile_size_nif_exec,
    W,
    y_tile_size_nof,
    ${fs2},
    ${fs1},
    p_t,
    p_b,
    p_l,
    p_r,
    ${stride},
    ${stride},
  % if has_bias:
    b,
  % else:
    NULL,
  % endif
    ${has_bias},
  % if FLAG_RELU == 1:
    out_shift,
    out_mult,
  % else:
    0,
    0,
  % endif
    y,
    y_tile_size_w,
    y_tile_size_h,
  % if FLAG_BATCHNORM == 1:
    k,
    lambda,
  % else:
    0,
    0,
  % endif
    im2col,
  % if flag_DW == 1:
    pwt_buffer,
  % endif
    ${FLAG_RELU},
    ${FLAG_BATCHNORM},
    &dma_evt
    );
% endif
    pi_cl_team_barrier(0);
% if tile_dim_nif != 1 and flag_DW == 0:
    if(_i_nif_load == 0) 
    {
% endif
      // wait for DMA write/read
% if chip == 'GAP8v3':
% if dma_parallelization == '1-core':
      if (pi_core_id()==0)
      {
% endif
      mchan_barrier(dma_evt);
% if dma_parallelization == '1-core':
      }
% endif
% endif   

% if FLAG_BATCHNORM == 1:    
% if flag_DW == 0:
    if(iter<${tile_dim_nof}*${tile_dim_nif}*${tile_dim_h}*${tile_dim_w}-1) 
    {
% else:
    if(iter<${tile_dim_nof}*${tile_dim_h}*${tile_dim_w}-1) 
    {  
% endif 
      if(pi_core_id()==0 && (_i_nif_load!=_i_nif_exec || _i_nof_load!=_i_nof_exec))
      {                                       
        pi_cl_dma_wait(&copy_k);                                                    
        pi_cl_dma_wait(&copy_lambda);
      }
    }
% endif       
% if dma_parallelization == '1-core':
        if (pi_core_id()==0)
        {
% endif                        
% if flag_DW == 1:
        dory_dma_memcpy_3d_custom_blocking(
% else:
        dory_dma_memcpy_3d_custom_out(
% endif
        dory_get_tile_3d(l2_y, _i_h_exec, _i_w_exec, _i_nof_exec, ${y_tile_size_h}, ${y_tile_size_w}, ${y_tile_size_nof}, ${y_w}, ${int(nof*factor)}, 0, 0, 0, 0, 0, 0, ${y_data_size_byte}), // ext
        (l1_buffer + ${l1_y_offset}) + db_y, // loc
        y_tile_size_byte, // size
        ${y_stride_w_byte}, // stride_1
        ${y_stride_c_byte}, // stride_0
        y_tile_size_h, // length_2
        y_length_nof_byte, // length_0
        0, // dir
        &dma_evt // copy
        );
% if dma_parallelization == '1-core':
        }
% endif
% if tile_dim_nif != 1 and flag_DW == 0:
    }
% endif
    // update prev iterators
    db_state_y = ! db_state_y; 
    _i_nof_exec = _i_nof_load;
    _i_nif_exec = _i_nif_load;
    _i_h_exec = _i_h_load;
    _i_w_exec = _i_w_load;
    pi_cl_team_barrier(0);
  }

% if not TEST:
  // wait for final write
  % if chip == 'GAP8v3':
% if dma_parallelization == '1-core':
  if (pi_core_id()==0)
  {
% endif
  mchan_barrier(dma_evt);
  mchan_free(dma_evt);
% if dma_parallelization == '1-core':
  }
% endif
  % endif
% endif
}
