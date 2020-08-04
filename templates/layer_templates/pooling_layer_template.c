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
${verbose_log}

#include "${func_name}.h"
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
  unsigned int l2_x_2 =(unsigned int)  real_arg[4];
  unsigned int l2_y =(unsigned int)  real_arg[5];
  unsigned int l2_W =(unsigned int)  real_arg[6];
  unsigned int l1_buffer =(unsigned int)  real_arg[7];
  unsigned int hyperram =(unsigned int)  real_arg[8];
  unsigned int out_mult_in =(unsigned int)  real_arg[9];
  unsigned int inmul1 = (unsigned int) real_arg[10];
  unsigned int inmul2 = (unsigned int) real_arg[11];
  unsigned int out_shift_in = (unsigned int) real_arg[12];
  unsigned int dma_evt;
  int p_r, p_l, p_t, p_b;
  int last_nof_exec;
  int last_nif_exec;
  int last_h_exec;
  int last_w_exec;
% if tile_dim_nif*tile_dim_h*tile_dim_w != 1:
  unsigned short x_tile_size_nif;
  unsigned short x_tile_size_h;
  unsigned short x_tile_size_w;
  unsigned short x_tile_size_byte;
  unsigned short x_length_h_px;
  unsigned short x_length_nif_byte;
  int pad_offset_h, pad_offset_w;
% endif  
  ${type} *x;
  ${type} *y;
  int x_tile_size_nif_exec;
  int x_tile_size_h_exec;
  int x_tile_size_w_exec;
  int y_tile_size_nof;
  int y_tile_size_h;
  int y_tile_size_w;
  int y_tile_size_byte;
  int y_length_h_px;
  int y_length_nof_byte;
  int db_x;
  int db_y;
  int exec_db_x;
  int exec_db_W;
 ${type} *im2col;
  im2col = l1_buffer + ${buffer_l1_all};
  % if chip == 'GAP8v3':
  dma_evt = mchan_alloc();
  % endif
  // copy first tiles
  //l2_x has input activations
  dory_dma_memcpy_3d_custom(
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
  % if chip == 'GAP8v3':
  // wait for x read
  mchan_barrier(dma_evt);
  % endif
  // tile loop indeces
  int _i_nof_load=0, _i_nif_load=0, _i_h_load=0, _i_w_load=0;
  int _i_nof_exec=0, _i_nif_exec=0, _i_h_exec=0, _i_w_exec=0;

  // double buffering state
  int db_state_x=0;
  int db_state_y=1;
  int db_state_acc_out=1;
  int flag_first_ch_out;

  // last-tile flags
  int last_nof_load = (${tile_dim_nof} == 1) ? 1 : 0;
  int last_nif_load = (${tile_dim_nif} == 1) ? 1 : 0;
  int last_h_load = (${tile_dim_h} == 1) ? 1 : 0;
  int last_w_load = (${tile_dim_w} == 1) ? 1 : 0;
  int iter;
% if FLAG_RELU == 1:
  uint16_t out_mult = out_mult_in;
  uint16_t out_shift = out_shift_in;
% endif
  // tile loop nest
  for(iter=0; iter<${tile_dim_nof}*${tile_dim_h}*${tile_dim_w}; iter++) {
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
    if (_i_nof_exec==0)
      flag_first_ch_out = 1;
    else
      flag_first_ch_out = 0;
    // wait for x,W read
    // check if last in any dimension
    last_nof_exec = last_nof_load;
    last_nif_exec = last_nif_load;
    last_h_exec = last_h_load;
    last_w_exec = last_w_load;
    last_nof_load = (_i_nof_load+1 == ${tile_dim_nof}) ? 1 : 0;
    last_nif_load = (_i_nof_load+1 == ${tile_dim_nif}) ? 1 : 0;
    last_h_load = (_i_h_load+1 == ${tile_dim_h}) ? 1 : 0;
    last_w_load = (_i_w_load+1 == ${tile_dim_w}) ? 1 : 0;

    // compute double buffering offsets and update db state
    db_x = !db_state_x ? ${x_tile_size_byte} : 0;
    db_y = !db_state_y ? ${y_tile_size_byte} : 0;
% if tile_dim_nif*tile_dim_h*tile_dim_w != 1:
    exec_db_x = db_state_x ? ${x_tile_size_byte} : 0;
% else:
    exec_db_x = 0;
% endif
    db_state_x = ! db_state_x;

    //switch all double buffering offset and y only after that all n_input_features have been analyzed: we need to pass all n_in to produce a single filter_out
    db_state_y = ! db_state_y;
    if(iter<${tile_dim_nof}*${tile_dim_h}*${tile_dim_w}-1) 
    {
% if tile_dim_nif*tile_dim_h*tile_dim_w != 1:
      x_tile_size_nif = (last_nif_load) ? ${x_tile_size_nif_last} : ${x_tile_size_nif};
      x_tile_size_h   = (last_h_load)   ? ${x_tile_size_h_last} : ${x_tile_size_h};
      x_tile_size_w   = (last_w_load)   ? ${x_tile_size_w_last} : ${x_tile_size_w};
      x_tile_size_byte = x_tile_size_nif*x_tile_size_h*x_tile_size_w*${x_data_size_byte}/8;
      x_length_nif_byte = (last_nif_load)   ? ${x_tile_size_nif_byte_last} : ${x_tile_size_nif_byte};
      // additionally overlap by padding for the first tile after a border one
      //this because in the first tile we use less pixels from x_buffer, since we have the ones of padding
      pad_offset_h=0, pad_offset_w=0;
      if(_i_h_load > 0)
        pad_offset_h = ${padding_top};
      if(_i_w_load > 0)
        pad_offset_w = ${padding_left};

      dory_dma_memcpy_3d_custom(
        dory_get_tile_3d(l2_x, _i_h_load, _i_w_load, _i_nif_load, ${x_tile_size_h}, ${x_tile_size_w}, ${x_tile_size_nif}, ${x_w}, ${nif},  ${conv_overlap1}, ${conv_overlap2},0, pad_offset_h, pad_offset_w, 0, ${x_data_size_byte}), // extern
        (l1_buffer + ${l1_x_offset}) + db_x, // loc
        x_tile_size_byte, // size: dimension of the buffer
        ${x_stride_w_byte}, // stride_1: stride for the 3d copy: if we have to copy on n_features axis, this is the stride to change from first 2D space to the next ones.
        ${x_stride_c_byte}, // stride_0: stride to be passed to 2d_copy: the dimension w of the in image
        x_tile_size_h,// length_2: how many 2_d copies we need -> the dimension of the tile in n_features direction
        x_length_nif_byte, // length_0: legnth of the 1_d copy, the length of tile in w direction
        1, // dir
        &dma_evt // copy
        );
% endif
      y_tile_size_h   = (last_h_load)   ? ${y_tile_size_h_last} : ${y_tile_size_h};
      y_tile_size_w   = (last_w_load)   ? ${y_tile_size_w_last} : ${y_tile_size_w};
    }
    x = (${type} *) (l1_buffer + ${l1_x_offset} + exec_db_x);
    y = (${type} *) (l1_buffer + ${l1_y_offset} + db_y);
   
    x_tile_size_nif_exec = (last_nif_exec) ? ${x_tile_size_nif_last} : ${x_tile_size_nif};
    x_tile_size_h_exec   = (last_h_exec)   ? ${x_tile_size_h_last} : ${x_tile_size_h};
    x_tile_size_w_exec   = (last_w_exec)   ? ${x_tile_size_w_last} : ${x_tile_size_w};
  
    y_tile_size_nof = (last_nof_exec) ? ${y_tile_size_nof_last} : ${y_tile_size_nof};
    y_tile_size_h   = (last_h_exec)   ? ${y_tile_size_h_last} : ${y_tile_size_h};
    y_tile_size_w   = (last_w_exec)   ? ${y_tile_size_w_last} : ${y_tile_size_w};
    y_tile_size_byte = y_tile_size_nof*y_tile_size_h*y_tile_size_w*${y_data_size_byte}/8;
    y_length_nof_byte = (last_nof_exec)   ? ${y_length_nof_byte_last} : ${y_tile_size_nof_byte};
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
  
// aggiungere padding su tutti i lati, acc_out, and filter asymettric
  % if 'Max' in optional:
    pulp_nn_maxpool(
  % else:
    pulp_nn_avgpool(
  % endif
    x,
    x_tile_size_w_exec,
    x_tile_size_h_exec,
    x_tile_size_nif_exec,
% if 'Max' in optional:
    ${fs1},
% else:
    ${fs2},
    ${fs1},
% endif
    p_t,
% if 'Max' in optional:
    p_b,
    p_l,
    p_r,
% endif
    ${stride},
    y_tile_size_w,
    y_tile_size_h,
    im2col,
    y,
    0,
    0,
% if 'Max' in optional:
    flag_first_ch_out
% else:
    flag_first_ch_out,
    ${FLAG_RELU},
% if FLAG_RELU == 1:
    out_shift,
    out_mult
% else:
    0,
    0
% endif    
% endif
    );
    pi_cl_team_barrier(0);
    % if chip == 'GAP8v3':
    mchan_barrier(dma_evt);
    % endif
    // transfering of output to L2
    dory_dma_memcpy_3d_custom(
      dory_get_tile_3d(l2_y, _i_h_exec, _i_w_exec, _i_nof_exec, ${y_tile_size_h}, ${y_tile_size_w}, ${y_tile_size_nof}, ${y_w}, ${nof}, 0, 0, 0, 0, 0, 0, ${y_data_size_byte}), // ext
      (l1_buffer + ${l1_y_offset}) + db_y, // loc
      y_tile_size_byte, // size
      ${y_stride_w_byte}, // stride_1
      ${y_stride_c_byte}, // stride_0
      y_tile_size_h, // length_2
      y_length_nof_byte, // length_0
      0, // dir
      &dma_evt // copy
    );
    // update prev iterators
    _i_nof_exec = _i_nof_load;
    _i_nif_exec = _i_nif_load;
    _i_h_exec = _i_h_load;
    _i_w_exec = _i_w_load;
  }
% if not TEST:
  // wait for final write
  % if chip == 'GAP8v3':
  mchan_barrier(dma_evt);
  mchan_free(dma_evt);
  % endif

% endif
}
