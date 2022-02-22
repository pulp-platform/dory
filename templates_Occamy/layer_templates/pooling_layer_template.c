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
  unsigned int l2_x =         layer_i->L2_input;
  unsigned int l2_x_2 =       layer_i->L2_input_add;
  unsigned int l2_y =         layer_i->L2_output;
  unsigned int l2_W =         layer_i->L2_weights;
  unsigned int l2_zeros =     layer_i->l2_zeros;
  unsigned int out_mult_in =  layer_i->out_mult;
  unsigned int inmul1 =       layer_i->inmul1;
  unsigned int inmul2 =       layer_i->inmul2;
  unsigned int out_shift_in = layer_i->out_shift;

  volatile kernel kernel_i;
  /// allocation of Occamy
  ${type} *memory_cluster = (${type} *)snrt_cluster_memory().start;
  unsigned int l1_buffer = (unsigned int) memory_cluster;
  volatile DMA_copy DMA_copy_x, DMA_copy_y;
  DMA_copy_x.hwc_to_chw = 0;
  DMA_copy_x.stride_2d = ${x_stride_w_byte};
  DMA_copy_x.stride_1d = ${x_stride_c_byte};
  DMA_copy_x.dir = 1;
  DMA_copy_x.dma_channel = NULL;
  
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
  // double buffering state
  int db_state_x=0;
  int db_state_y=1;
  // last-tile flags
  int iter;
  // tile loop indeces
  int _i_nof_load=0, _i_nif_load=0, _i_h_load=0, _i_w_load=0;
  int _i_nof_exec=0, _i_nif_exec=0, _i_h_exec=0, _i_w_exec=0;


  DMA_copy_x.ext = l2_x;
  DMA_copy_x.loc = l1_buffer + (${l1_x_offset} + 0);
  DMA_copy_x.number_of_2d_copies = ${x_tile_size_h};
  DMA_copy_x.number_of_1d_copies = ${x_tile_size_w};
  DMA_copy_x.length_1d_copy = ${x_tile_size_nif_byte};
  DMA_copy_x.stride_L1_1d = DMA_copy_x.length_1d_copy;
  DMA_copy_x.stride_L1_2d = DMA_copy_x.length_1d_copy*DMA_copy_x.number_of_1d_copies;
  dory_dma_memcpy_async(DMA_copy_x);
  dory_dma_barrier(DMA_copy_x);


  float sum = 0;
  int total_tiles = ${tile_dim_nof * tile_dim_h * tile_dim_w};
  // tile loop nest
  for(iter=0; iter < total_tiles; iter++) {
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
    // check if last in any dimension

    // compute double buffering offsets and update db state
    db_x = !db_state_x ? ${x_tile_size_byte} : 0;
    db_y = !db_state_y ? ${y_tile_size_byte} : 0;
  % if tile_dim_nif*tile_dim_h*tile_dim_w != 1:
    exec_db_x = db_state_x ? ${x_tile_size_byte} : 0;
  % else:
    exec_db_x = 0;
  % endif
    db_state_x = ! db_state_x;
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
      // transfer of next input tile in double buffering
      % if tile_dim_nif*tile_dim_h*tile_dim_w != 1:

      DMA_copy_x.ext = dory_get_tile_3d(l2_x, _i_h_load, _i_w_load, _i_nif_load, ${x_tile_size_h}, ${x_tile_size_w}, ${x_tile_size_nif}, ${x_w}, ${nif*g},  ${conv_overlap1}, ${conv_overlap2},0, pad_offset_h, pad_offset_w, 0, ${x_data_size_byte});
      DMA_copy_x.loc = l1_buffer + (${l1_x_offset} + db_x);
      DMA_copy_x.number_of_2d_copies = x_tile_size_h;
      DMA_copy_x.number_of_1d_copies = x_tile_size_w;
      DMA_copy_x.length_1d_copy = x_length_nif_byte;
      DMA_copy_x.stride_L1_1d = DMA_copy_x.length_1d_copy;
      DMA_copy_x.stride_L1_2d = DMA_copy_x.length_1d_copy*DMA_copy_x.number_of_1d_copies;
      dory_dma_memcpy_async(DMA_copy_x);
      % endif
    }
    // creation of the pointers to input, output, weights, lambda and k

    kernel_i.pInBuffer = (${type} *) (l1_buffer + (${l1_x_offset} + exec_db_x));
    kernel_i.pOutBuffer = (${type} *) (l1_buffer + (${l1_y_offset} + db_y));
    // parameter passed to the kernel. Input and output sizes
    kernel_i.ch_in = (_i_nif_exec+1 == ${tile_dim_nif}) ? ${x_tile_size_nif_last} : ${x_tile_size_nif};
    kernel_i.dim_in_y   = (_i_h_exec+1 == ${tile_dim_h})   ? ${x_tile_size_h_last} : ${x_tile_size_h};
    kernel_i.dim_in_x   = (_i_w_exec+1 == ${tile_dim_w})   ? ${x_tile_size_w_last} : ${x_tile_size_w};
    kernel_i.ch_out = (_i_nof_exec+1 == ${tile_dim_nof}) ? ${y_tile_size_nof_last} : ${y_tile_size_nof};
    kernel_i.dim_out_y   = (_i_h_exec+1 == ${tile_dim_h})   ? ${y_tile_size_h_last} : ${y_tile_size_h};
    kernel_i.dim_out_x   = (_i_w_exec+1 == ${tile_dim_w})   ? ${y_tile_size_w_last} : ${y_tile_size_w};
    kernel_i.stride_x = ${stride};
    kernel_i.stride_y = ${stride};
    kernel_i.dim_kernel_x = ${fs2};
    kernel_i.dim_kernel_y = ${fs1};
    kernel_i.padding_y_top = (_i_h_exec == 0) ? ${padding_top} : 0;
    kernel_i.padding_y_bottom = (_i_h_exec == ${tile_dim_h}-1) ? ${padding_bottom} : 0;
    kernel_i.padding_x_left = (_i_w_exec == 0) ? ${padding_left} : 0;
    kernel_i.padding_x_right = (_i_w_exec == ${tile_dim_w}-1) ? ${padding_right} : 0;
    y_tile_size_byte = y_tile_size_nof*y_tile_size_h*y_tile_size_w*${y_data_size_byte}/8;
    y_length_nof_byte = (_i_nof_exec+1 == ${tile_dim_nof})   ? ${y_length_nof_byte_last} : ${y_tile_size_nof_byte};
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
    
    occamy_pool_naive(&kernel_i);


  // printf("y: ");
  // sum=0;
  // for (int i = 0; i < (y_tile_size_nof*y_tile_size_h*y_tile_size_w); i++)
  //   sum+=*(y+i);
  // printf("%f ", sum);
  // printf("\n");
    

    // wait for DMA write/read
      dory_dma_barrier(DMA_copy_y);
      dory_dma_barrier(DMA_copy_x);
     
      DMA_copy_y.ext = dory_get_tile_3d(l2_y, _i_h_exec, _i_w_exec, _i_nof_exec, ${y_tile_size_h}, ${y_tile_size_w}, ${y_tile_size_nof}, ${y_w}, ${int(nof*factor)}, 0, 0, 0, 0, 0, 0, ${y_data_size_byte});
      DMA_copy_y.loc = l1_buffer + (${l1_y_offset} + db_y);
      DMA_copy_y.number_of_2d_copies = (_i_h_exec+1 == ${tile_dim_h})   ? ${y_tile_size_h_last} : ${y_tile_size_h};
      DMA_copy_y.number_of_1d_copies = (_i_w_exec+1 == ${tile_dim_w})   ? ${y_tile_size_w_last} : ${y_tile_size_w};
      DMA_copy_y.length_1d_copy = y_length_nof_byte;
      dory_dma_memcpy_async(DMA_copy_y);  
    // update prev iterators
    db_state_y = ! db_state_y; 
    _i_nof_exec = _i_nof_load;
    _i_nif_exec = _i_nif_load;
    _i_h_exec = _i_h_load;
    _i_w_exec = _i_w_load;
  }

  // wait for final write
  dory_dma_barrier(DMA_copy_y);
}
