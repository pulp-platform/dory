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

void ${func_name}(void *args) 
{
  unsigned int *real_arg =    (unsigned int *) args;
  unsigned int l2_x =         (unsigned int) real_arg[0];
  unsigned int l2_x_2 =       (unsigned int) real_arg[1];
  unsigned int l2_y =         (unsigned int) real_arg[2];
  unsigned int l2_W =         (unsigned int) real_arg[3];
  unsigned int out_mult_in =  (unsigned int) real_arg[4];
  unsigned int inmul1 =       (unsigned int) real_arg[5];
  unsigned int inmul2 =       (unsigned int) real_arg[6];
  unsigned int out_shift_in = (unsigned int) real_arg[7];
  /// allocation of Occamy
  ${type} *memory_cluster = (${type} *)snrt_cluster_memory().start;
  unsigned int l1_buffer = (unsigned int) memory_cluster;
  volatile DMA_copy DMA_copy_k, DMA_copy_lambda;
  volatile DMA_copy DMA_copy_W, DMA_copy_x, DMA_copy_y;
% if has_bias == 1:
  volatile DMA_copy DMA_copy_bias;
  DMA_copy_bias.hwc_to_chw = 0;
  DMA_copy_bias.stride_2d = 0;
  DMA_copy_bias.stride_1d = 0;
  DMA_copy_bias.dir = 1;
  DMA_copy_bias.dma_channel = NULL;

% endif
  DMA_copy_k.hwc_to_chw = 0;
  DMA_copy_k.stride_2d = 0;
  DMA_copy_k.stride_1d = 0;
  DMA_copy_k.dir = 1;
  DMA_copy_k.dma_channel = NULL;

  DMA_copy_lambda.hwc_to_chw = 0;
  DMA_copy_lambda.stride_2d = 0;
  DMA_copy_lambda.stride_1d = 0;
  DMA_copy_lambda.dir = 1;
  DMA_copy_lambda.dma_channel = NULL;
  
  % if flag_DW == 1:
  DMA_copy_x.hwc_to_chw = 1;
  % else:
  DMA_copy_x.hwc_to_chw = 0;
  % endif  
  DMA_copy_x.stride_2d = ${x_stride_w_byte};
  DMA_copy_x.stride_1d = ${x_stride_c_byte};
  DMA_copy_x.dir = 1;
  DMA_copy_x.dma_channel = NULL;
  
  DMA_copy_W.hwc_to_chw = 0;
  DMA_copy_W.stride_2d = ${W_stride_nof_byte};
  DMA_copy_W.stride_1d = ${W_stride_hw_byte};
  DMA_copy_W.dir = 1;
  DMA_copy_W.dma_channel = NULL;
  
  DMA_copy_y.hwc_to_chw = 0;
  DMA_copy_y.stride_2d = ${y_stride_w_byte};
  DMA_copy_y.stride_1d = ${y_stride_c_byte};
  DMA_copy_y.dir = 0;
  DMA_copy_y.dma_channel = NULL;

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
  volatile ${type} *x, *W, *y, *b;
% if FLAG_BATCHNORM == 1:
% if act_dim_bit == 32:
  volatile ${type} *k;
  volatile ${type} *lambda;
% else:
  volatile ${type} *k;
  volatile ${type} *lambda;
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
  int has_bias = 1;
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
  ////////////////////////////
  // First tile transfering //
  ////////////////////////////
% if has_bias == 1:
  DMA_copy_bias.ext = (uint32_t) l2_W+${l2_off_bias};
  DMA_copy_bias.loc = (uint32_t) (l1_buffer + ${l1_b_offset});
  DMA_copy_bias.number_of_2d_copies = 1;
  DMA_copy_bias.number_of_1d_copies = 1;
  DMA_copy_bias.length_1d_copy = (uint16_t) ${b_size_byte};
  dory_dma_memcpy_async(DMA_copy_bias);
  dory_dma_barrier(DMA_copy_bias);

% endif
% if FLAG_BATCHNORM == 1:
  DMA_copy_k.ext = (uint32_t) l2_W+${l2_off_k}+${k_tile_size_byte_transfer}*${int(tile_dim_nof/number_of_clusters)}*snrt_cluster_idx();
  DMA_copy_k.loc = (uint32_t) l1_buffer + ${l1_k_offset};
  DMA_copy_k.number_of_2d_copies = 1;
  DMA_copy_k.number_of_1d_copies = 1;
  DMA_copy_k.length_1d_copy = (uint16_t) ${k_tile_size_byte_transfer};
  dory_dma_memcpy_async(DMA_copy_k);
  dory_dma_barrier(DMA_copy_k);

  DMA_copy_lambda.ext = (uint32_t) l2_W+${l2_off_k} + ${l2_off_lambda-l2_off_k}+${k_tile_size_byte_transfer}*${int(tile_dim_nof/number_of_clusters)}*snrt_cluster_idx();
  DMA_copy_lambda.loc = (uint32_t) l1_buffer + ${l1_lambda_offset};
  DMA_copy_lambda.number_of_2d_copies = 1;
  DMA_copy_lambda.number_of_1d_copies = 1;
  DMA_copy_lambda.length_1d_copy = (uint16_t) ${lambda_tile_size_byte_transfer};
  dory_dma_memcpy_async(DMA_copy_lambda);
  dory_dma_barrier(DMA_copy_lambda);

% endif

  DMA_copy_x.ext = l2_x;
  DMA_copy_x.loc = l1_buffer + (${l1_x_offset} + 0);
  DMA_copy_x.number_of_2d_copies = ${x_tile_size_h};
  DMA_copy_x.number_of_1d_copies = ${x_tile_size_w};
  DMA_copy_x.length_1d_copy = ${x_tile_size_nif_byte};
  dory_dma_memcpy_async(DMA_copy_x);
  dory_dma_barrier(DMA_copy_x);

  DMA_copy_W.ext = l2_W + ${fs1 * fs2 * W_tile_size_nof * W_tile_nif_byte}*${int(tile_dim_nof/number_of_clusters)}*snrt_cluster_idx();
  DMA_copy_W.loc = l1_buffer + (${l1_W_offset} + 0);
  DMA_copy_W.number_of_2d_copies = ${W_tile_size_nof};
  DMA_copy_W.number_of_1d_copies = ${fs1 * fs2};
  DMA_copy_W.length_1d_copy = ${W_tile_nif_byte};
  dory_dma_memcpy_async(DMA_copy_W);
  dory_dma_barrier(DMA_copy_W);

  float sum = 0;
% if flag_DW == 0:
  int total_tiles = ${tile_dim_nif * tile_dim_h * tile_dim_w * int(tile_dim_nof/number_of_clusters) };
  if (snrt_cluster_idx() == (${number_of_clusters - 1}))
    total_tiles+= ${(tile_dim_nof % number_of_clusters) * tile_dim_h * tile_dim_w * tile_dim_nif};
% else:
  int total_tiles = ${tile_dim_h * tile_dim_w * int(tile_dim_nof/number_of_clusters)};
  if (snrt_cluster_idx() == (${number_of_clusters - 1}))
    total_tiles+= ${(tile_dim_nof % number_of_clusters) * tile_dim_h * tile_dim_w };
% endif
  // tile loop nest
  for(iter=0; iter < total_tiles; iter++) {
  % if tile_dim_nif != 1 and flag_DW == 0:
    // loop nest is nof,h,w,nif
    _i_nif_load += 1;
    if(_i_nif_load==${tile_dim_nif}) 
    {
      _i_nif_load = 0;
  % endif
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
  % if tile_dim_nif != 1 and flag_DW == 0:
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

    if(iter < (total_tiles-1) )
    {
      asm volatile("": : :"memory");
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
      if (snrt_cluster_idx() == (${number_of_clusters - 1}))
        W_tile_size_nof = (_i_nof_load+1 == ${int(tile_dim_nof/number_of_clusters) + tile_dim_nof%number_of_clusters}) ? ${W_tile_size_nof_last} : ${W_tile_size_nof};
      else
        W_tile_size_nof = ${W_tile_size_nof};
      W_tile_size_nif = (_i_nif_load+1 == ${tile_dim_nif}) ? ${W_tile_size_nif_last} : ${W_tile_size_nif};
      % if flag_DW == 1:
      W_tile_size_byte = W_tile_size_nof*W_tile_size_nif*${fs1}*${fs2};
      % else:
      W_tile_size_byte = W_tile_size_nof*W_tile_size_nif*${W_data_size_byte}*${fs1}*${fs2}/8;
      % endif
      W_length_nif_byte = (_i_nif_load+1 == ${tile_dim_nif}) ? ${W_tile_size_nif_byte_last} : ${W_tile_nif_byte};
      // transfer of next input tile in double buffering
      % if tile_dim_nif*tile_dim_h*tile_dim_w != 1:

      DMA_copy_x.ext = dory_get_tile_3d(l2_x, _i_h_load, _i_w_load, _i_nif_load, ${x_tile_size_h}, ${x_tile_size_w}, ${x_tile_size_nif}, ${x_w}, ${nif*g},  ${conv_overlap1}, ${conv_overlap2},0, pad_offset_h, pad_offset_w, 0, ${x_data_size_byte});
      DMA_copy_x.loc = l1_buffer + (${l1_x_offset} + db_x);
      DMA_copy_x.number_of_2d_copies = x_tile_size_h;
      DMA_copy_x.number_of_1d_copies = x_tile_size_w;
      DMA_copy_x.length_1d_copy = x_length_nif_byte;
      dory_dma_memcpy_async(DMA_copy_x);
      % endif
      // transfer of next weight tile if changed input or output channels
      if (_i_nif_load!=_i_nif_exec || _i_nof_load!=_i_nof_exec)
      {
        % if flag_DW == 0:
        DMA_copy_W.ext = dory_get_tile_3d(l2_W,  _i_nof_load + snrt_cluster_idx() * ${int(tile_dim_nof/number_of_clusters)}, 0, _i_nif_load, ${W_tile_size_nof}, ${fs1}*${fs2}, ${W_tile_size_nif}, ${fs1}*${fs2}, ${nif}, 0,0,0,0,0,0, ${W_data_size_byte});
        % else:
        DMA_copy_W.ext = dory_get_tile_3d(l2_W, _i_nof_load + snrt_cluster_idx() * ${int(tile_dim_nof/number_of_clusters)}, 0, 0, ${W_tile_size_nof*8/W_data_size_byte}, ${fs1}*${fs2}, ${W_tile_size_nif}, ${fs1}*${fs2}, ${nif}, 0,0,0,0,0,0, ${W_data_size_byte});
        % endif
        DMA_copy_W.loc = l1_buffer + (${l1_W_offset} + db_W);
        DMA_copy_W.number_of_2d_copies = W_tile_size_nof;
        DMA_copy_W.length_1d_copy = W_length_nif_byte;
        dory_dma_memcpy_async(DMA_copy_W);
        % if FLAG_BATCHNORM == 1:

        DMA_copy_k.ext = (uint32_t) l2_W+${l2_off_k} + ${k_tile_size_byte_transfer} * (_i_nof_load + snrt_cluster_idx() * ${int(tile_dim_nof/number_of_clusters)});
        DMA_copy_k.loc = (uint32_t) l1_buffer + ${l1_k_offset} + db_act;
        DMA_copy_k.length_1d_copy = (uint16_t) W_tile_size_nof * ${int(act_dim_bit/8)};
        dory_dma_memcpy_async(DMA_copy_k);

        DMA_copy_lambda.ext = (uint32_t) l2_W+${l2_off_k} + ${lambda_tile_size_byte_transfer} * (_i_nof_load + snrt_cluster_idx() * ${int(tile_dim_nof/number_of_clusters)})  + ${l2_off_lambda-l2_off_k};
        DMA_copy_lambda.loc = (uint32_t) l1_buffer + ${l1_lambda_offset} + db_act;
        DMA_copy_lambda.length_1d_copy = (uint16_t) W_tile_size_nof * ${int(act_dim_bit/8)};
        dory_dma_memcpy_async(DMA_copy_lambda);
        % endif
      }
    }
    // creation of the pointers to input, output, weights, lambda and k
    % if flag_DW == 1:
    asm volatile("": : :"memory");
    % endif
    x = (${type} *) (l1_buffer + (${l1_x_offset} + exec_db_x));
    % if FLAG_BATCHNORM == 1:
    % if act_dim_bit == 32:
    k = (${type} *) (l1_buffer + ${l1_k_offset} + exec_db_act);
    lambda = (${type} *) (l1_buffer + ${l1_lambda_offset} + exec_db_act);
    % else:
    k = (${type} *) (l1_buffer + ${l1_k_offset} + exec_db_act);
    lambda = (${type} *) (l1_buffer + ${l1_lambda_offset} + exec_db_act);
    % endif
    % endif
    % if has_bias == 1:
    b = (${type} *) (l1_buffer + (${l1_b_offset} + _i_nof_exec*${bias_tile_size_byte}));
    % endif
    W = (${type} *) (l1_buffer + (${l1_W_offset} + exec_db_W));
    y = (${type} *) (l1_buffer + (${l1_y_offset} + db_y));
    // parameter passed to the kernel. Input and output sizes
    x_tile_size_nif_exec = (_i_nif_exec+1 == ${tile_dim_nif}) ? ${x_tile_size_nif_last} : ${x_tile_size_nif};
    x_tile_size_h_exec   = (_i_h_exec+1 == ${tile_dim_h})   ? ${x_tile_size_h_last} : ${x_tile_size_h};
    x_tile_size_w_exec   = (_i_w_exec+1 == ${tile_dim_w})   ? ${x_tile_size_w_last} : ${x_tile_size_w};
    if (snrt_cluster_idx() == (${number_of_clusters - 1}))
      y_tile_size_nof = (_i_nof_exec+1 == ${int(tile_dim_nof/number_of_clusters) + tile_dim_nof%number_of_clusters}) ? ${y_tile_size_nof_last} : ${y_tile_size_nof};
    else
      y_tile_size_nof = ${y_tile_size_nof};
    y_tile_size_h   = (_i_h_exec+1 == ${tile_dim_h})   ? ${y_tile_size_h_last} : ${y_tile_size_h};
    y_tile_size_w   = (_i_w_exec+1 == ${tile_dim_w})   ? ${y_tile_size_w_last} : ${y_tile_size_w};
    y_tile_size_byte = y_tile_size_nof*y_tile_size_h*y_tile_size_w*${y_data_size_byte}/8;
    if (snrt_cluster_idx() == (${number_of_clusters - 1}))
      y_length_nof_byte = (_i_nof_exec+1 == ${int(tile_dim_nof/number_of_clusters) + tile_dim_nof%number_of_clusters})   ? ${y_length_nof_byte_last} : ${y_tile_size_nof_byte};
    else
      y_length_nof_byte = ${y_tile_size_nof_byte};
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
    % if tile_dim_nof*tile_dim_nif*tile_dim_h*tile_dim_w == 1 or flag_DW == 1:
    asm volatile("": : :"memory");
    % endif


  // printf("x: ");
  // sum=0;
  // for (int i = 0; i < (x_tile_size_nif_exec*x_tile_size_h_exec*x_tile_size_w_exec); i++)
  //   sum+=*(x+i);
  // printf("%f ", sum);
  // printf("\n");


    //printf("Tile execution %d of ${func_name}\n", iter);
    % if FLAG_BATCHNORM == 1:
    // printf("First W %f, l %f, k %f\n", *W, *lambda, *k);
    % endif
    occamy_conv_naive(x, x_tile_size_w_exec, x_tile_size_h_exec, x_tile_size_nif_exec,
      W, y_tile_size_nof, ${fs2}, ${fs1},
      p_t, p_b, p_l, p_r, ${stride}, ${stride},
      % if has_bias:
      b,
      % else:
      NULL,
      % endif
      ${has_bias},
      % if FLAG_RELU == 1:
      out_shift, out_mult,
      % else:
      0, 0,
      % endif
      y, y_tile_size_w, y_tile_size_h,
      % if FLAG_BATCHNORM == 1:
      k, lambda,
      % else:
      0, 0,
      % endif
      im2col,
      % if flag_DW == 1:
      pwt_buffer,
      % endif
      ${FLAG_RELU}, ${FLAG_BATCHNORM}, NULL);


  /*printf("y: ");
  sum=0;
  for (int i = 0; i < (y_tile_size_nof*y_tile_size_h*y_tile_size_w); i++)
    sum+=*(y+i);
  printf("%f ", sum);
  printf("\n");*/
    

    % if tile_dim_nif != 1 and flag_DW == 0:
    if(_i_nif_load == 0) 
    {
    % endif
    // wait for DMA write/read
      dory_dma_barrier(DMA_copy_y);
      dory_dma_barrier(DMA_copy_x);
      dory_dma_barrier(DMA_copy_W);

    % if FLAG_BATCHNORM == 1:   
    if(iter < (total_tiles-1) && (_i_nif_load!=_i_nif_exec || _i_nof_load!=_i_nof_exec))
    {                        
      dory_dma_barrier(DMA_copy_k);
      dory_dma_barrier(DMA_copy_lambda);
    }
    % endif      
      DMA_copy_y.ext = dory_get_tile_3d(l2_y, _i_h_exec, _i_w_exec, _i_nof_exec + snrt_cluster_idx() * ${int(tile_dim_nof/number_of_clusters)}, ${y_tile_size_h}, ${y_tile_size_w}, ${y_tile_size_nof}, ${y_w}, ${int(nof*factor)}, 0, 0, 0, 0, 0, 0, ${y_data_size_byte});
      DMA_copy_y.loc = l1_buffer + (${l1_y_offset} + db_y);
      DMA_copy_y.number_of_2d_copies = y_tile_size_h;
      DMA_copy_y.number_of_1d_copies = y_tile_size_w;
      DMA_copy_y.length_1d_copy = y_length_nof_byte;
      dory_dma_memcpy_async(DMA_copy_y);   
% if tile_dim_nif != 1 and flag_DW == 0:
    }
% endif
    // update prev iterators
    db_state_y = ! db_state_y; 
    _i_nof_exec = _i_nof_load;
    _i_nif_exec = _i_nif_load;
    _i_h_exec = _i_h_load;
    _i_w_exec = _i_w_load;
  }

% if not TEST:
  // wait for final write
  dory_dma_barrier(DMA_copy_y);
% endif
}
