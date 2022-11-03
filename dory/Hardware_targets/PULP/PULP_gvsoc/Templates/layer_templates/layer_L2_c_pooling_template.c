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
  unsigned int l2_x_2 =(unsigned int)  real_arg[4];
  unsigned int l2_y =(unsigned int)  real_arg[5];
  unsigned int l2_W =(unsigned int)  real_arg[6];
  unsigned int l1_buffer =(unsigned int)  real_arg[7];
  unsigned int hyperram =(unsigned int)  real_arg[8];
  unsigned int out_shift_in = (unsigned int) real_arg[10];
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
  volatile DMA_copy DMA_copy_x, DMA_copy_y;
  int dma_id = dory_dma_allocate();
  // copy first tiles
  //l2_x has input activations

  DMA_copy_x.hwc_to_chw = 0;
  DMA_copy_x.stride_2d = ${x_stride_w_byte};
  DMA_copy_x.stride_1d = ${x_stride_c_byte};
  DMA_copy_x.dir = 1;
  DMA_copy_x.tid = dma_id;

  DMA_copy_y.hwc_to_chw = 0;
  DMA_copy_y.stride_2d = ${y_stride_w_byte};
  DMA_copy_y.stride_1d = ${y_stride_c_byte};
  DMA_copy_y.dir = 0;
  DMA_copy_y.tid = dma_id;

  DMA_copy_x.ext = l2_x;
  DMA_copy_x.loc = (l1_buffer + ${l1_x_offset}) + 0;
  DMA_copy_x.number_of_2d_copies = ${x_tile_size_h};
  DMA_copy_x.number_of_1d_copies = ${x_tile_size_w};
  DMA_copy_x.length_1d_copy = ${x_tile_size_nif_byte};
  dory_dma_memcpy_async(&DMA_copy_x);
  dory_dma_barrier(&DMA_copy_x);
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

      DMA_copy_x.ext = dory_get_tile_3d(l2_x, _i_h_load, _i_w_load, _i_nif_load, ${x_tile_size_h}, ${x_tile_size_w}, ${x_tile_size_nif}, ${x_w}, ${nif},  ${conv_overlap1}, ${conv_overlap2},0, pad_offset_h, pad_offset_w, 0, ${x_data_size_byte});
      DMA_copy_x.loc = (l1_buffer + ${l1_x_offset}) + db_x;
      DMA_copy_x.number_of_2d_copies = x_tile_size_h;
      DMA_copy_x.number_of_1d_copies = x_tile_size_w;
      DMA_copy_x.length_1d_copy = x_length_nif_byte;
      dory_dma_memcpy_async(&DMA_copy_x);
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
    % if optional_type == 'mixed-sw':
    pulp_nn_maxpool_${data_type_y[0]}${y_data_size_byte}(
    % elif optional_type == 'mixed-hw':
    xpulp_nn_maxpool_${data_type_y[0]}${y_data_size_byte}(
    % else:
    pulp_nn_maxpool(
    % endif
  % else:
    % if 'mixed' in optional_type:
    ${"x" if "hw" in optional_type else ""}pulp_nn_avgpool_${data_type_x[0]}${x_data_size_byte}_${data_type_y[0]}${y_data_size_byte}(
    % else:
    pulp_nn_avgpool(
    % endif
  % endif
    x, y,
    % if 'Max' not in optional:
    ${out_mul},
    ${out_shift},
    ${out_add},
  % endif
    x_tile_size_w_exec,
    x_tile_size_h_exec,
    x_tile_size_nif_exec,
    y_tile_size_w,
    y_tile_size_h,
    ${fs2},
    ${fs1},
    p_t,
    p_b,
    p_l,
    p_r,
    ${stride},
    ${stride}${"," if "Max" not in optional else ""}
% if 'Max' not in optional:
    ${FLAG_RELU}
% endif
    );
    dory_dma_barrier(&DMA_copy_x);
    dory_dma_barrier(&DMA_copy_y);
    pi_cl_team_barrier(0);


    // transfering of output to L2
    DMA_copy_y.ext = dory_get_tile_3d(l2_y, _i_h_exec, _i_w_exec, _i_nof_exec, ${y_tile_size_h}, ${y_tile_size_w}, ${y_tile_size_nof}, ${y_w}, ${nof}, 0, 0, 0, 0, 0, 0, ${y_data_size_byte});
    DMA_copy_y.loc = (l1_buffer + ${l1_y_offset}) + db_y;
    DMA_copy_y.number_of_2d_copies = y_tile_size_h;
    DMA_copy_y.number_of_1d_copies = y_tile_size_w;
    DMA_copy_y.length_1d_copy = y_length_nof_byte;
    dory_dma_memcpy_async(&DMA_copy_y);
    // update prev iterators
    _i_nof_exec = _i_nof_load;
    _i_nif_exec = _i_nif_load;
    _i_h_exec = _i_h_load;
    _i_w_exec = _i_w_load;
    pi_cl_team_barrier(0);
  }
% if not TEST:
  // wait for final write
    dory_dma_barrier(&DMA_copy_y);
    dory_dma_free(&DMA_copy_y);
% endif
}
