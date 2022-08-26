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


void ${func_name}(layer* layer_i) 
{

  /////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////
  ////////////// VARIABLE DECLARATION AND INITIALIZATION //////////////////
  /////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////

  unsigned int l2_x =         layer_i->L2_input;
  unsigned int l2_x_2 =       layer_i->L2_input_add;
  unsigned int l2_y =         layer_i->L2_output;
  unsigned int l2_W =         layer_i->L2_weights;
  unsigned int l2_zeros =     layer_i->l2_zeros;
  
  volatile kernel kernel_i;
  int CLUSTERS = ${number_of_clusters};
  volatile DMA_copy DMA_copy_k, DMA_copy_lambda, DMA_copy_W, DMA_copy_x, DMA_copy_y, DMA_copy_p_top, DMA_copy_p_bottom, DMA_copy_p_left, DMA_copy_p_right, DMA_copy_bias;
  // Memory allocation
  ${type} *memory_cluster = (${type} *)snrt_cluster_memory().start;
  unsigned int l1_buffer = (unsigned int) memory_cluster;
  volatile unsigned short x_length_nif_byte;
% if tile_dim_nif*tile_dim_h*tile_dim_w*number_of_clusters != 1:
  // Input Parameters for DMA loading
  volatile unsigned short x_tile_size_h, x_tile_size_w;
  volatile int pad_offset_h, pad_offset_w;
% endif  
  // Weights Parameters
  volatile unsigned short  W_tile_size_nof, W_length_nif_byte, W_length_nif_byte_last;
  // Input Parameters for execution
  // Output Parameters for execution
  volatile int y_tile_size_nof, y_tile_size_h, y_tile_size_w, y_length_nof_byte;
  // Double Buffering parameters and states
  volatile int db_x, db_W, db_act, db_y, exec_db_x, exec_db_W, exec_db_act;
  int db_state_x=0, db_state_W=0, db_state_y=1;
  // tile loop indeces
  int iter, _i_nof_load=0, _i_nif_load=0, _i_h_load=0, _i_w_load=0, _i_nof_exec=0, _i_nif_exec=0, _i_h_exec=0, _i_w_exec=0;
% if has_bias == 1:
  int has_bias = 1;
% endif

  /////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////
  //////////////// FIRST TILE TRANSFERING OF ALL TENSORS //////////////////
  /////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////

  ////////////////////// BEGIN DMA DEDICATED SECTION //////////////////////
  if (snrt_is_dm_core())
  {
% if number_of_clusters > 1:
    x_length_nif_byte = ${int(x_tile_size_nif_byte)};
    if (snrt_cluster_idx() == (CLUSTERS-1))
    {
      x_length_nif_byte = ${int(x_tile_size_nif_byte_last)};
    }
% else:
    x_length_nif_byte = ${int(x_tile_size_nif_byte)};
% endif
% if has_bias == 1:
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
    
    DMA_copy_p_top.ext = l2_zeros;
    DMA_copy_p_top.hwc_to_chw = 0;
    DMA_copy_p_top.stride_2d = ${x_stride_w_byte+(padding_left+padding_right)*x_stride_c_byte};
    DMA_copy_p_top.stride_1d = ${x_stride_c_byte};
    DMA_copy_p_top.dir = 1;
    DMA_copy_p_top.dma_channel = NULL;

    DMA_copy_p_bottom.ext = l2_zeros;
    DMA_copy_p_bottom.hwc_to_chw = 0;
    DMA_copy_p_bottom.stride_2d = ${x_stride_w_byte+(padding_left+padding_right)*x_stride_c_byte};
    DMA_copy_p_bottom.stride_1d = ${x_stride_c_byte};
    DMA_copy_p_bottom.dir = 1;
    DMA_copy_p_bottom.dma_channel = NULL;

    DMA_copy_p_left.ext = l2_zeros;
    DMA_copy_p_left.hwc_to_chw = 0;
    DMA_copy_p_left.stride_2d = ${padding_left*x_stride_c_byte};
    DMA_copy_p_left.stride_1d = ${x_stride_c_byte};
    DMA_copy_p_left.dir = 1;
    DMA_copy_p_left.dma_channel = NULL;

    DMA_copy_p_right.ext = l2_zeros;
    DMA_copy_p_right.hwc_to_chw = 0;
    DMA_copy_p_right.stride_2d = ${padding_right*x_stride_c_byte};
    DMA_copy_p_right.stride_1d = ${x_stride_c_byte};
    DMA_copy_p_right.dir = 1;
    DMA_copy_p_right.dma_channel = NULL;

% if flag_DW == 1:
  // DMA_copy_x.hwc_to_chw = 1;
    DMA_copy_x.hwc_to_chw = 0;
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

% if has_bias == 1:
    DMA_copy_bias.ext = (uint32_t) l2_W+${l2_off_bias};
    DMA_copy_bias.loc = (uint32_t) (l1_buffer + ${l1_b_offset});
    DMA_copy_bias.number_of_2d_copies = 1;
    DMA_copy_bias.number_of_1d_copies = 1;
    DMA_copy_bias.length_1d_copy = (uint16_t) ${b_size_byte};
    dory_dma_memcpy_async(DMA_copy_bias);

% endif
% if FLAG_BATCHNORM == 1:
    DMA_copy_k.ext = (uint32_t) l2_W+${l2_off_k}+${k_tile_size_byte_transfer}*${int(tile_dim_nof/number_of_clusters)}*snrt_cluster_idx();
    DMA_copy_k.loc = (uint32_t) l1_buffer + ${l1_k_offset};
    DMA_copy_k.number_of_2d_copies = 1;
    DMA_copy_k.number_of_1d_copies = 1;
    DMA_copy_k.length_1d_copy = (uint16_t) ${k_tile_size_byte_transfer};
    dory_dma_memcpy_async(DMA_copy_k);

    DMA_copy_lambda.ext = (uint32_t) l2_W+${l2_off_k} + ${l2_off_lambda-l2_off_k}+${k_tile_size_byte_transfer}*${int(tile_dim_nof/number_of_clusters)}*snrt_cluster_idx();
    DMA_copy_lambda.loc = (uint32_t) l1_buffer + ${l1_lambda_offset};
    DMA_copy_lambda.number_of_2d_copies = 1;
    DMA_copy_lambda.number_of_1d_copies = 1;
    DMA_copy_lambda.length_1d_copy = (uint16_t) ${lambda_tile_size_byte_transfer};
    dory_dma_memcpy_async(DMA_copy_lambda);

% endif
    if (${padding_top} > 0)
    {
      DMA_copy_p_top.loc = l1_buffer + (${l1_x_offset} + 0);
      DMA_copy_p_top.number_of_2d_copies = 1;
      DMA_copy_p_top.number_of_1d_copies = 1;
      DMA_copy_p_top.length_1d_copy = ${padding_top}*x_length_nif_byte*(${padding_left} + ${padding_right}*(${tile_dim_w}==1) + ${x_tile_size_w});
      dory_dma_memcpy_async(DMA_copy_p_top);
    }

    if (${tile_dim_h}==1 && ${padding_bottom} > 0)
    {
      DMA_copy_p_bottom.loc = l1_buffer + (${l1_x_offset} + 0) + x_length_nif_byte*${x_tile_size_w*x_tile_size_h} + ${padding_top}*x_length_nif_byte*(${padding_left} + ${padding_right}*(${tile_dim_w}==1) + ${x_tile_size_w}) + x_length_nif_byte*${padding_left}*${x_tile_size_h} + x_length_nif_byte*${padding_right}*${x_tile_size_h}*(${tile_dim_w}==1);
      DMA_copy_p_bottom.number_of_2d_copies = 1;
      DMA_copy_p_bottom.number_of_1d_copies = 1;
      DMA_copy_p_bottom.length_1d_copy = ${padding_bottom}*x_length_nif_byte*(${padding_left} + ${padding_right}*(${tile_dim_w}==1) + ${x_tile_size_w});
      dory_dma_memcpy_async(DMA_copy_p_bottom);
    }

    if (${padding_left} > 0)
    {
      DMA_copy_p_left.loc = l1_buffer + (${l1_x_offset} + 0) + ${padding_top}*x_length_nif_byte*(${padding_left} + ${padding_right}*(${tile_dim_w}==1) + ${x_tile_size_w});
      DMA_copy_p_left.number_of_2d_copies = 1;
      DMA_copy_p_left.number_of_1d_copies = ${x_tile_size_h};
      DMA_copy_p_left.length_1d_copy = ${padding_left}*x_length_nif_byte;
      DMA_copy_p_left.stride_L1_1d = x_length_nif_byte*(${padding_left} + ${padding_right}*(${tile_dim_w}==1) + ${x_tile_size_w});
      dory_dma_memcpy_async(DMA_copy_p_left);
    }

    if (${tile_dim_w}==1 && ${padding_right} > 0)
    {
      DMA_copy_p_right.loc = l1_buffer + (${l1_x_offset} + 0) + ${padding_top}*x_length_nif_byte*(${padding_left} + ${padding_right}*(${tile_dim_w}==1) + ${x_tile_size_w}) + x_length_nif_byte*(${padding_left} + ${x_tile_size_w});
      DMA_copy_p_right.number_of_2d_copies = 1;
      DMA_copy_p_right.number_of_1d_copies = ${x_tile_size_h};
      DMA_copy_p_right.length_1d_copy = ${padding_right}*x_length_nif_byte;
      DMA_copy_p_right.stride_L1_1d = x_length_nif_byte*(${padding_left} + ${padding_right} + ${x_tile_size_w});
      dory_dma_memcpy_async(DMA_copy_p_right);
    }

% if tile_dim_nif > 1:
% if flag_DW == 0:
    DMA_copy_x.ext = l2_x + x_length_nif_byte*snrt_cluster_idx();
% else:
    DMA_copy_x.ext = l2_x + x_length_nif_byte*${int(tile_dim_nof/number_of_clusters)}*snrt_cluster_idx();
% endif
% else:
    DMA_copy_x.ext = l2_x;
% endif
    DMA_copy_x.loc = l1_buffer + (${l1_x_offset} + 0) + ${padding_top}*x_length_nif_byte*(${padding_left} + ${padding_right}*(${tile_dim_w}==1) + ${x_tile_size_w}) + x_length_nif_byte*${padding_left};
    DMA_copy_x.number_of_2d_copies = ${x_tile_size_h};
    DMA_copy_x.number_of_1d_copies = ${x_tile_size_w};
    DMA_copy_x.length_1d_copy = x_length_nif_byte;
    DMA_copy_x.stride_L1_1d = DMA_copy_x.length_1d_copy;
    DMA_copy_x.stride_L1_2d = DMA_copy_x.length_1d_copy*DMA_copy_x.number_of_1d_copies + x_length_nif_byte*(${padding_left} + ${padding_right}*(${tile_dim_w}==1));
    dory_dma_memcpy_async(DMA_copy_x);

    int cl = 0;
    W_length_nif_byte = ((cl+snrt_cluster_idx()) % CLUSTERS  == (${tile_dim_nif}-1)) ? ${W_tile_nif_byte_last} : ${W_tile_nif_byte}; 
    DMA_copy_W.ext = l2_W +${int(fs1 * fs2 * W_tile_size_nof * W_tile_nif_byte)}*${int(tile_dim_nof/number_of_clusters)}*snrt_cluster_idx();
    DMA_copy_W.loc = l1_buffer + (${l1_W_offset} + 0);
    DMA_copy_W.number_of_2d_copies = ${W_tile_size_nof};
    DMA_copy_W.number_of_1d_copies = ${fs1 * fs2};
% if flag_DW == 0:
    DMA_copy_W.length_1d_copy = (int)W_length_nif_byte/${int(tile_dim_nif)};
% else:
    DMA_copy_W.length_1d_copy = (int)W_length_nif_byte;
% endif
    DMA_copy_W.stride_L1_1d = DMA_copy_W.length_1d_copy;
    DMA_copy_W.stride_L1_2d = DMA_copy_W.length_1d_copy*DMA_copy_W.number_of_1d_copies;
    dory_dma_memcpy_async(DMA_copy_W);
% if tile_dim_nif > 1 and flag_DW == 0:
    for(cl = 1; cl < ${number_of_clusters}; cl++)
    {
      W_length_nif_byte = ((cl+snrt_cluster_idx()) % CLUSTERS  == (${tile_dim_nif}-1)) ? ${W_tile_nif_byte_last} : ${W_tile_nif_byte};
      DMA_copy_W.ext = l2_W +${int(fs1 * fs2 * W_tile_size_nof * W_tile_nif_byte)}*${int(tile_dim_nof/number_of_clusters)}*snrt_cluster_idx() + ${int(W_tile_nif_byte/tile_dim_nif)}*cl;
      DMA_copy_W.loc = l1_buffer + (${l1_W_offset} + 0) + ${W_tile_size_nof * fs1 * fs2 * int(W_tile_nif_byte/tile_dim_nif)}*cl;
      DMA_copy_W.number_of_2d_copies = ${W_tile_size_nof};
      DMA_copy_W.number_of_1d_copies = ${fs1 * fs2};
      DMA_copy_W.length_1d_copy = (int)W_length_nif_byte/${int(tile_dim_nif)};
      DMA_copy_W.stride_L1_1d = DMA_copy_W.length_1d_copy;
      DMA_copy_W.stride_L1_2d = DMA_copy_W.length_1d_copy*DMA_copy_W.number_of_1d_copies;
      dory_dma_memcpy_async(DMA_copy_W);
    }
% endif
  }
  ////////////////////// END DMA DEDICATED SECTION ////////////////////////
  dory_global_barrier();
  float sum = 0;

% if flag_DW == 0:
  int total_tiles = ${tile_dim_nif * tile_dim_h * tile_dim_w * int(tile_dim_nof/number_of_clusters) };
  total_tiles+= ${(tile_dim_nof % number_of_clusters) * tile_dim_h * tile_dim_w * tile_dim_nif};
% else:
  int total_tiles = ${tile_dim_h * tile_dim_w * int(tile_dim_nof/number_of_clusters)};
  total_tiles+= ${(tile_dim_nof % number_of_clusters) * tile_dim_h * tile_dim_w };
% endif


  /////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////
  ////////////////// LOOP OF COMPUTATION OVER ALL TILES ///////////////////
  /////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////

  for(iter=0; iter < total_tiles; iter++) 
  {
% if tile_dim_nif != 1 and flag_DW == 0:
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
% if flag_DW == 1:
        _i_nif_load += 1;
% endif
          _i_nof_load += 1;
        }
      }
    }
% else:
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

    // compute double buffering offsets and update db state
    db_x = !db_state_x ? ${x_tile_size_byte} : 0;
    db_W = !db_state_W ? ${W_tile_size_byte} : 0;
    db_y = !db_state_y ? ${y_tile_size_byte} : 0;
% if FLAG_BATCHNORM == 1:
    db_act = !db_state_W ? ${k_tile_size_byte_transfer} : 0;
% endif
% if tile_dim_nif*tile_dim_h*tile_dim_w*min(x_tile_size_nif,number_of_clusters) != 1:
    exec_db_x = db_state_x ? ${x_tile_size_byte} : 0;
% else:
    exec_db_x = 0;
% endif
    exec_db_W = db_state_W ? ${W_tile_size_byte} : 0;
% if FLAG_BATCHNORM == 1:
    exec_db_act = db_state_W ? ${k_tile_size_byte_transfer} : 0;
% endif
    //switch all double buffering offset and y only after that all n_input_features have been analyzed: we need to pass all n_in to produce a single pixel
    db_state_x = ! db_state_x;
    if (_i_nof_load!=_i_nof_exec)
    {
      db_state_W = ! db_state_W;
    }

    ///////////////////// BEGIN DMA DEDICATED SECTION ///////////////////////
    if (snrt_is_dm_core())
    {
      if(iter < (total_tiles-1) )
      {
        asm volatile("": : :"memory");
% if tile_dim_nif*tile_dim_h*tile_dim_w*min(x_tile_size_nif,number_of_clusters) != 1:
        // X PARAMETERS (H, W, C) DEFINITION
        x_tile_size_h   = (_i_h_load+1 == ${tile_dim_h})   ? ${x_tile_size_h_last} : ${x_tile_size_h};
        x_tile_size_w   = (_i_w_load+1 == ${tile_dim_w})   ? ${x_tile_size_w_last} : ${x_tile_size_w};
% if number_of_clusters > 1:
        x_length_nif_byte = ((_i_nif_load+snrt_cluster_idx()) % CLUSTERS  == (${tile_dim_nif}-1)) ? ${x_tile_size_nif_byte_last} : ${x_tile_size_nif_byte};
% else:
        x_length_nif_byte = ((_i_nif_load)  == (${tile_dim_nif}-1)) ? ${x_tile_size_nif_byte_last} : ${x_tile_size_nif_byte};
% endif

        // additionally overlap by padding for the first tile after a border one because in the first tile we use less pixels from x_buffer
        pad_offset_h=0, pad_offset_w=0;
        if(_i_h_load > 0)
        {
          pad_offset_h = ${padding_top};
        }
        if(_i_w_load > 0)
        {
          pad_offset_w = ${padding_left};
        }

% endif
% if number_of_clusters > 1:
        // W PARAMETERS (NIF, NOF) DEFINITION
        if (snrt_cluster_idx() == (${number_of_clusters - 1}))
        {
          W_tile_size_nof = (_i_nof_load+1 == ${int(tile_dim_nof/number_of_clusters) + tile_dim_nof%number_of_clusters}) ? ${W_tile_size_nof_last} : ${W_tile_size_nof};
        }
        else
        {
          W_tile_size_nof = ${W_tile_size_nof};
        }
% else:
        W_tile_size_nof = (_i_nof_load+1 == ${int(tile_dim_nof)}) ? ${W_tile_size_nof_last} : ${W_tile_size_nof};
% endif

% if tile_dim_nif*tile_dim_h*tile_dim_w*min(x_tile_size_nif,number_of_clusters) != 1:
        if (_i_nif_load !=0 && ${int(flag_DW==0)})
        {
          DMA_copy_x.ext = ((snrt_cluster_idx() + 1) % CLUSTERS) * 131072 + (${l1_x_offset} + exec_db_x);
          DMA_copy_x.loc = l1_buffer + (${l1_x_offset} + db_x);
          DMA_copy_x.number_of_2d_copies = 1;
          DMA_copy_x.number_of_1d_copies = 1;
          DMA_copy_x.stride_2d = x_length_nif_byte*(x_tile_size_h + ${padding_top}*(_i_h_load==0) + ${padding_bottom}*(_i_h_load==${tile_dim_h-1}))*(x_tile_size_w + ${padding_left}*(_i_w_load==0) + ${padding_right}*(_i_w_load==${tile_dim_w-1}));
          DMA_copy_x.stride_1d = x_length_nif_byte*(x_tile_size_h + ${padding_top}*(_i_h_load==0) + ${padding_bottom}*(_i_h_load==${tile_dim_h-1}))*(x_tile_size_w + ${padding_left}*(_i_w_load==0) + ${padding_right}*(_i_w_load==${tile_dim_w-1}));
          DMA_copy_x.length_1d_copy = x_length_nif_byte*(x_tile_size_h + ${padding_top}*(_i_h_load==0) + ${padding_bottom}*(_i_h_load==${tile_dim_h-1}))*(x_tile_size_w + ${padding_left}*(_i_w_load==0) + ${padding_right}*(_i_w_load==${tile_dim_w-1}));
          DMA_copy_x.stride_L1_1d = DMA_copy_x.length_1d_copy;
          DMA_copy_x.stride_L1_2d = DMA_copy_x.length_1d_copy * DMA_copy_x.number_of_1d_copies;
          dory_dma_memcpy_async(DMA_copy_x);
        }
        else
        {
          if (_i_h_load == 0 && ${padding_top} > 0)
          {
            DMA_copy_p_top.loc = l1_buffer + (${l1_x_offset} + db_x);
            DMA_copy_p_top.number_of_2d_copies = 1;
            DMA_copy_p_top.number_of_1d_copies = 1;
            DMA_copy_p_top.length_1d_copy = ${padding_top}*x_length_nif_byte*(${padding_left}*(_i_w_load == 0) + ${padding_right}*(_i_w_load == ${(tile_dim_w-1)}) + x_tile_size_w);
            dory_dma_memcpy_async(DMA_copy_p_top);
          }
          if (_i_h_load == ${(tile_dim_h-1)} && ${padding_bottom} > 0)
          {
            DMA_copy_p_bottom.loc = l1_buffer + (${l1_x_offset} + db_x) + x_length_nif_byte*x_tile_size_w*x_tile_size_h +  (_i_h_load == 0)*${padding_top}*x_length_nif_byte*(${padding_left}*(_i_w_load == 0) + ${padding_right}*(_i_w_load == ${(tile_dim_w-1)}) + x_tile_size_w) + x_length_nif_byte*${padding_left}*(_i_w_load == 0)*x_tile_size_h + x_length_nif_byte*${padding_right}*(_i_w_load == ${(tile_dim_w-1)})*x_tile_size_h;
            DMA_copy_p_bottom.number_of_2d_copies = 1;
            DMA_copy_p_bottom.number_of_1d_copies = 1;
            DMA_copy_p_bottom.length_1d_copy = ${padding_bottom}*x_length_nif_byte*(${padding_left}*(_i_w_load == 0) + ${padding_right}*(_i_w_load == ${(tile_dim_w-1)}) + x_tile_size_w);
            dory_dma_memcpy_async(DMA_copy_p_bottom);
          }
          if (_i_w_load == 0 && ${padding_left} > 0)
          {
            DMA_copy_p_left.loc = l1_buffer + (${l1_x_offset} + db_x) + ${padding_top}*x_length_nif_byte*(${padding_left}*(_i_w_load == 0) + ${padding_right}*(_i_w_load == ${(tile_dim_w-1)}) + x_tile_size_w)*(_i_h_load == 0);
            DMA_copy_p_left.number_of_2d_copies = 1;
            DMA_copy_p_left.number_of_1d_copies = x_tile_size_h;
            DMA_copy_p_left.length_1d_copy = ${padding_left}*x_length_nif_byte;
            DMA_copy_p_left.stride_L1_1d = x_length_nif_byte*(${padding_left}*(_i_w_load == 0) + ${padding_right}*(_i_w_load == ${(tile_dim_w-1)}) + x_tile_size_w);
            dory_dma_memcpy_async(DMA_copy_p_left);
          }
          if (_i_w_load == ${(tile_dim_w-1)} && ${padding_right} > 0)
          {
            DMA_copy_p_right.loc = l1_buffer + (${l1_x_offset} + db_x) + ${padding_top}*x_length_nif_byte*(${padding_left}*(_i_w_load == 0) + ${padding_right}*(_i_w_load == ${(tile_dim_w-1)}) + x_tile_size_w)*(_i_h_load == 0) + x_length_nif_byte*(${padding_left}*(_i_w_load == 0) + x_tile_size_w);
            DMA_copy_p_right.number_of_2d_copies = 1;
            DMA_copy_p_right.number_of_1d_copies = x_tile_size_h;
            DMA_copy_p_right.length_1d_copy = ${padding_right}*x_length_nif_byte;
            DMA_copy_p_right.stride_L1_1d = x_length_nif_byte*(${padding_left}*(_i_w_load == 0) + ${padding_right}*(_i_w_load == ${(tile_dim_w-1)}) + x_tile_size_w);
            dory_dma_memcpy_async(DMA_copy_p_right);
          }
  % if tile_dim_nif > 1 and flag_DW == 0:
          DMA_copy_x.ext = dory_get_tile_3d(l2_x, _i_h_load, _i_w_load, _i_nif_load, ${x_tile_size_h}, ${x_tile_size_w}, ${x_tile_size_nif}, ${x_w}, ${nif*g},  ${conv_overlap1}, ${conv_overlap2},0, pad_offset_h, pad_offset_w, 0, ${x_data_size_byte}) + ${x_tile_size_nif_byte}*snrt_cluster_idx();
  % elif tile_dim_nif > 1 and flag_DW == 1:
          DMA_copy_x.ext = dory_get_tile_3d(l2_x, _i_h_load, _i_w_load, _i_nif_load + snrt_cluster_idx() * ${int(tile_dim_nif/number_of_clusters)}, ${x_tile_size_h}, ${x_tile_size_w}, ${x_tile_size_nif}, ${x_w}, ${nif*g},  ${conv_overlap1}, ${conv_overlap2},0, pad_offset_h, pad_offset_w, 0, ${x_data_size_byte});
  % else:
          DMA_copy_x.ext = dory_get_tile_3d(l2_x, _i_h_load, _i_w_load, _i_nif_load, ${x_tile_size_h}, ${x_tile_size_w}, ${x_tile_size_nif}, ${x_w}, ${nif*g},  ${conv_overlap1}, ${conv_overlap2},0, pad_offset_h, pad_offset_w, 0, ${x_data_size_byte});
  % endif        
          DMA_copy_x.loc = l1_buffer + (${l1_x_offset} + db_x) + ${padding_top}*x_length_nif_byte*(${padding_left}*(_i_w_load == 0) + ${padding_right}*(_i_w_load == ${(tile_dim_w-1)}) + x_tile_size_w)*(_i_h_load == 0) + x_length_nif_byte*${padding_left}*(_i_w_load == 0);
          DMA_copy_x.number_of_2d_copies = x_tile_size_h;
          DMA_copy_x.number_of_1d_copies = x_tile_size_w;
          DMA_copy_x.stride_2d = ${x_stride_w_byte};
          DMA_copy_x.stride_1d = ${x_stride_c_byte};
          DMA_copy_x.length_1d_copy = x_length_nif_byte;
          DMA_copy_x.stride_L1_1d = DMA_copy_x.length_1d_copy;
          DMA_copy_x.stride_L1_2d = DMA_copy_x.length_1d_copy*DMA_copy_x.number_of_1d_copies + x_length_nif_byte*(${padding_left}*(_i_w_load == 0) + ${padding_right}*(_i_w_load == ${(tile_dim_w-1)}));
          dory_dma_memcpy_async(DMA_copy_x);
        }
% endif
        // transfer of next weight tile if changed input or output channels
% if flag_DW == 0:
        if (_i_nof_load!=_i_nof_exec && (iter <  ${tile_dim_nif * tile_dim_h * tile_dim_w * int(tile_dim_nof/number_of_clusters) } || snrt_cluster_idx() == (CLUSTERS-1)))
% else:
        if (_i_nof_load!=_i_nof_exec && (iter <  ${tile_dim_h * tile_dim_w * int(tile_dim_nof/number_of_clusters) } || snrt_cluster_idx() == (CLUSTERS-1)))
% endif
        {
          int cl = 0;
          W_length_nif_byte = ((cl+_i_nif_load+snrt_cluster_idx()) % CLUSTERS  == (${tile_dim_nif}-1)) ? ${W_tile_nif_byte_last} : ${W_tile_nif_byte}; 
          DMA_copy_W.ext = dory_get_tile_3d(l2_W,  _i_nof_load + snrt_cluster_idx() * ${int(tile_dim_nof/number_of_clusters)}, 0, 0, ${W_tile_size_nof}, ${fs1}*${fs2}, ${W_tile_size_nif}, ${fs1}*${fs2}, ${nif}, 0,0,0,0,0,0, ${W_data_size_byte});
          DMA_copy_W.loc = l1_buffer + (${l1_W_offset} + db_W);
          DMA_copy_W.number_of_2d_copies = W_tile_size_nof;
% if tile_dim_nif > 1 and flag_DW == 0:
          DMA_copy_W.length_1d_copy = (int) W_length_nif_byte/${tile_dim_nif};
% else:
          DMA_copy_W.length_1d_copy = (int) W_length_nif_byte;
% endif
          DMA_copy_W.stride_L1_1d = DMA_copy_W.length_1d_copy;
          DMA_copy_W.stride_L1_2d = DMA_copy_W.length_1d_copy * DMA_copy_W.number_of_1d_copies;
          dory_dma_memcpy_async(DMA_copy_W);
% if tile_dim_nif > 1 and flag_DW == 0:
          for(cl = 1; cl < ${number_of_clusters}; cl++)
          {
            W_length_nif_byte = ((cl+_i_nif_load+snrt_cluster_idx()) % CLUSTERS  == (${tile_dim_nif}-1)) ? ${W_tile_nif_byte_last} : ${W_tile_nif_byte};
      	    DMA_copy_W.ext = dory_get_tile_3d(l2_W,  _i_nof_load + snrt_cluster_idx() * ${int(tile_dim_nof/number_of_clusters)}, 0, 0, ${W_tile_size_nof}, ${fs1}*${fs2}, ${W_tile_size_nif}, ${fs1}*${fs2}, ${nif}, 0,0,0,0,0,0, ${W_data_size_byte}) + ${int(W_tile_nif_byte/tile_dim_nif)}*cl;
      		DMA_copy_W.loc = l1_buffer + (${l1_W_offset} + db_W) + ${W_tile_size_nof * fs1 * fs2 * int(W_tile_nif_byte/tile_dim_nif)}*cl;
      	    DMA_copy_W.number_of_2d_copies = W_tile_size_nof;
      	    DMA_copy_W.length_1d_copy = (int) W_length_nif_byte/${tile_dim_nif};
            DMA_copy_W.stride_L1_1d = DMA_copy_W.length_1d_copy;
            DMA_copy_W.stride_L1_2d = DMA_copy_W.length_1d_copy * DMA_copy_W.number_of_1d_copies;
      	    dory_dma_memcpy_async(DMA_copy_W);
          }
% endif
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
    }
    ////////////////////// END DMA DEDICATED SECTION ////////////////////////

    /////////////////// BEGIN COMPUTE DEDICATED SECTION /////////////////////
    if (snrt_is_compute_core())
    {
    // CREATING POINTERS TO TENSORS TO PASS TO THE KERNEL
% if flag_DW == 1:
      asm volatile("": : :"memory");
% endif
      // SETTING PARAMETERS TO PASS TO THE KERNEL FUNCTION
      y_tile_size_h   = (_i_h_exec+1 == ${tile_dim_h}) ? ${y_tile_size_h_last} : ${y_tile_size_h};
      y_tile_size_w   = (_i_w_exec+1 == ${tile_dim_w}) ? ${y_tile_size_w_last} : ${y_tile_size_w};
% if number_of_clusters > 1:
      if (snrt_cluster_idx() == (${number_of_clusters - 1}))
      {
        y_length_nof_byte = (_i_nof_exec+1 == ${int(tile_dim_nof/number_of_clusters) + tile_dim_nof%number_of_clusters})   ? ${y_length_nof_byte_last} : ${y_tile_size_nof_byte};
      }
      else
      {
        y_length_nof_byte = ${y_tile_size_nof_byte};
      }
      if (snrt_cluster_idx() == (${number_of_clusters - 1}))
      {
        y_tile_size_nof = (_i_nof_exec+1 == ${int(tile_dim_nof/number_of_clusters) + tile_dim_nof%number_of_clusters}) ? ${y_tile_size_nof_last} : ${y_tile_size_nof};
      }
      else
      {
        y_tile_size_nof = ${y_tile_size_nof};
      }
% else:
      y_length_nof_byte = (_i_nof_exec+1 == ${int(tile_dim_nof)})   ? ${y_length_nof_byte_last} : ${y_tile_size_nof_byte};
      y_tile_size_nof = (_i_nof_exec+1 == ${int(tile_dim_nof)}) ? ${y_tile_size_nof_last} : ${y_tile_size_nof};
% endif
% if tile_dim_nof*tile_dim_nif*tile_dim_h*tile_dim_w*min(x_tile_size_nif,number_of_clusters) == 1 or flag_DW == 1:
      asm volatile("": : :"memory");
% endif
      kernel_i.pInBuffer = (${type} *) (l1_buffer + (${l1_x_offset} + exec_db_x));
      kernel_i.dim_in_x = (_i_w_exec+1 == ${tile_dim_w}) ? ${x_tile_size_w_last} : ${x_tile_size_w};
      kernel_i.dim_in_y = (_i_h_exec+1 == ${tile_dim_h}) ? ${x_tile_size_h_last} : ${x_tile_size_h};
% if flag_DW == 0:
      kernel_i.ch_in = ${x_tile_size_nif};
% else:
      kernel_i.ch_in = y_tile_size_nof;
% endif
% if tile_dim_nif > 1 and flag_DW == 0:
      kernel_i.pWeight = (${type} *) (l1_buffer + (${l1_W_offset} + exec_db_W + ((snrt_cluster_idx() + _i_nif_exec) % CLUSTERS) * ${int(fs1 * fs2 * W_tile_size_nof * W_tile_nif_byte / tile_dim_nif)}));
% else:
      kernel_i.pWeight = (${type} *) (l1_buffer + (${l1_W_offset} + exec_db_W));
% endif
      kernel_i.ch_out = y_tile_size_nof;
      kernel_i.dim_kernel_x = ${fs2};
      kernel_i.dim_kernel_y = ${fs1};
      kernel_i.padding_y_top = (_i_h_exec == 0) ? ${padding_top} : 0;
      kernel_i.padding_y_bottom = (_i_h_exec == ${tile_dim_h}-1) ? ${padding_bottom} : 0;
      kernel_i.padding_x_left = (_i_w_exec == 0) ? ${padding_left} : 0;
      kernel_i.padding_x_right = (_i_w_exec == ${tile_dim_w}-1) ? ${padding_right} : 0;
      kernel_i.stride_x = ${stride};
      kernel_i.stride_y = ${stride};
% if has_bias:
      kernel_i.bias = (${type} *) (l1_buffer + (${l1_b_offset} + _i_nof_exec*${bias_tile_size_byte}));
% else:
      kernel_i.bias = NULL;
% endif
      kernel_i.bias_shift = ${has_bias};
      kernel_i.out_shift = layer_i->out_shift;
% if FLAG_RELU == 1:
      kernel_i.out_mult = layer_i->out_mult;
% else:
      kernel_i.out_mult = 0;
% endif
      kernel_i.pOutBuffer = (${type} *) (l1_buffer + (${l1_y_offset} + db_y));
      kernel_i.dim_out_x = y_tile_size_w;
      kernel_i.dim_out_y = y_tile_size_h;
% if FLAG_BATCHNORM == 1:
      kernel_i.kappa = (${type} *) (l1_buffer + ${l1_k_offset} + exec_db_act);
      kernel_i.lambda = (${type} *) (l1_buffer + ${l1_lambda_offset} + exec_db_act);
% else:
      kernel_i.kappa = 0;
      kernel_i.lambda = 0;
% endif
      kernel_i.pIm2ColBuffer = (${type} *) (l1_buffer + ${buffer_l1_all});
% if flag_DW == 1:
      kernel_i.flag_relu = ${FLAG_RELU};
      kernel_i.flag_batch_norm = ${FLAG_BATCHNORM};
% else:
      kernel_i.flag_relu = ${FLAG_RELU} && (_i_nif_load==0);
      kernel_i.flag_batch_norm = ${FLAG_BATCHNORM} && (_i_nif_load==0);
% endif
      kernel_i.flag_y_accumulate_start = (_i_nif_exec==0);
      kernel_i.flag_y_accumulate_end = (_i_nif_load==0);
      kernel_i.memory_chan = NULL;
      //occamy_conv_naive
      //occamy_conv_opt_fp32
      //occamy_conv_chw_opt_fp32
      //occamy_conv_dw_opt_fp32

% if flag_DW == 0:
      if (iter <  ${tile_dim_nif * tile_dim_h * tile_dim_w * int(tile_dim_nof/number_of_clusters) } || snrt_cluster_idx() == (CLUSTERS-1))
% else:
      if (iter <  ${tile_dim_h * tile_dim_w * int(tile_dim_nof/number_of_clusters) } || snrt_cluster_idx() == (CLUSTERS-1))
% endif
      {
% if first_layer == 1:
        occamy_conv_chw_opt_fp32(&kernel_i);
% elif flag_DW == 1:
	    occamy_conv_dw_opt_fp32(&kernel_i);
% else:
	    occamy_conv_opt_fp32(&kernel_i);
% endif
      }
      else
      {
        snrt_cluster_hw_barrier();
      }
    }
    else
    {
      snrt_cluster_hw_barrier();
    }

    /////////////////// END COMPUTE DEDICATED SECTION /////////////////////
    dory_global_barrier();
	  /*
	  printf("Tile %d [h,w,c] indexes [%d,%d,%d] dimensions [%d, %d, %d] y: \n", iter, _i_h_exec, _i_w_exec, _i_nof_exec, y_tile_size_h, y_tile_size_w, y_tile_size_nof);
	  sum=0;
	  for (int i = 0; i < (y_tile_size_nof*y_tile_size_h*y_tile_size_w); i++)
	    sum+=*(y+i);
	  printf("%f ", sum);
	  printf("\n");
	  */  
    if (snrt_is_dm_core())
    {
% if tile_dim_nif*min(x_tile_size_nif,number_of_clusters) != 1 and flag_DW == 0:
      if(_i_nif_load == 0) 
      {
% endif
        y_tile_size_h   = (_i_h_exec+1 == ${tile_dim_h}) ? ${y_tile_size_h_last} : ${y_tile_size_h};
        y_tile_size_w   = (_i_w_exec+1 == ${tile_dim_w}) ? ${y_tile_size_w_last} : ${y_tile_size_w};
% if number_of_clusters > 1:
        if (snrt_cluster_idx() == (${number_of_clusters - 1}))
        {
          y_length_nof_byte = (_i_nof_exec+1 == ${int(tile_dim_nof/number_of_clusters) + tile_dim_nof%number_of_clusters})   ? ${y_length_nof_byte_last} : ${y_tile_size_nof_byte};
        }
        else
        {
          y_length_nof_byte = ${y_tile_size_nof_byte};
        }
% else:
        y_length_nof_byte = (_i_nof_exec+1 == ${int(tile_dim_nof)})   ? ${y_length_nof_byte_last} : ${y_tile_size_nof_byte};
% endif
% if flag_DW == 0:
	      if (iter <  ${tile_dim_nif * tile_dim_h * tile_dim_w * int(tile_dim_nof/number_of_clusters) } || snrt_cluster_idx() == (CLUSTERS-1))
% else:
	      if (iter <  ${tile_dim_h * tile_dim_w * int(tile_dim_nof/number_of_clusters) } || snrt_cluster_idx() == (CLUSTERS-1))
% endif
	      {
    	    DMA_copy_y.ext = dory_get_tile_3d(l2_y, _i_h_exec, _i_w_exec, _i_nof_exec + snrt_cluster_idx() * ${int(tile_dim_nof/number_of_clusters)}, ${y_tile_size_h}, ${y_tile_size_w}, ${y_tile_size_nof}, ${y_w}, ${int(nof*factor)}, 0, 0, 0, 0, 0, 0, ${y_data_size_byte});
    	    DMA_copy_y.loc = l1_buffer + (${l1_y_offset} + db_y);
    	    DMA_copy_y.number_of_2d_copies = y_tile_size_h;
    	    DMA_copy_y.number_of_1d_copies = y_tile_size_w;
    	    DMA_copy_y.length_1d_copy = y_length_nof_byte;
    	    dory_dma_memcpy_async(DMA_copy_y); 
    	  }
% if tile_dim_nif*min(x_tile_size_nif,number_of_clusters) != 1 and flag_DW == 0:
      }
% endif
    }
    // update prev iterators
% if tile_dim_nif*min(x_tile_size_nif,number_of_clusters) != 1 and flag_DW == 0:
    if(_i_nif_load == 0) 
    {
% endif
      db_state_y = ! db_state_y;   
% if tile_dim_nif*min(x_tile_size_nif,number_of_clusters) != 1 and flag_DW == 0:
    }
% endif
    _i_nof_exec = _i_nof_load;
    _i_nif_exec = _i_nif_load;
    _i_h_exec = _i_h_load;
    _i_w_exec = _i_w_load;
  }

% if not TEST:
  // wait for final write
  dory_global_barrier();
% endif
}
