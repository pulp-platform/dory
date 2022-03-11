/*
 * layer_template.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 * Thorir Mar Ingolfsson <thoriri@iis.ee.ethz.ch>
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
  volatile int p_r, p_l;
  int last_nof_exec;
  int last_nif_exec;
  int last_h_exec;
% if tile_dim_nif*tile_dim_h != 1:
  volatile  unsigned short x_tile_size_nif;
  volatile unsigned short  x_tile_size_h;
  volatile unsigned short  x_tile_size_byte;
  volatile unsigned short  x_length_nif_byte;
  volatile int pad_offset_h;
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
  volatile int y_tile_size_nof;
  volatile int y_tile_size_h;
  volatile int y_tile_size_byte;
  volatile int y_length_nof_byte;
  volatile int db_x;
  volatile int db_W;
  volatile int db_y;
  volatile int exec_db_x;
  volatile int exec_db_W;

  int copy_x_pending = 0;
  int copy_W_pending = 0;
  int copy_y_pending = 0;
  
  // double buffering state
  int db_state_x=0;
  int db_state_acc_in=0;
  int db_state_W=0;
  int db_state_y=1;
  int db_state_acc_out=1;
  // last-tile flags
  int last_nof_load = (${tile_dim_nof} == 1) ? 1 : 0;
  int last_nif_load = (${tile_dim_nif} == 1) ? 1 : 0;
  int last_h_load = (${tile_dim_h} == 1) ? 1 : 0;
  int iter;
  // tile loop indeces
  int _i_nof_load=0, _i_nif_load=0, _i_h_load=0;
  int _i_nof_exec=0, _i_nif_exec=0, _i_h_exec=0;
  int has_bias = ${has_bias};
  volatile ${type} *im2col;
  im2col = l1_buffer + ${buffer_l1_all};
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
    pi_cl_dma_copy_t copy_k;
    copy_k.dir = PI_CL_DMA_DIR_EXT2LOC;
    copy_k.merge = 0;
    copy_k.size = (uint16_t) ${k_size_byte};
    copy_k.id = 0;
    copy_k.ext = (uint32_t) l2_W+${l2_off_k};
    copy_k.loc = (uint32_t) l1_buffer + ${l1_k_offset};
    pi_cl_dma_memcpy(&copy_k);   
    pi_cl_dma_copy_t copy_lambda;
    copy_lambda.dir = PI_CL_DMA_DIR_EXT2LOC;
    copy_lambda.merge = 0;
    copy_lambda.size = (uint16_t) ${lambda_size_byte};
    copy_lambda.id = 0;
    copy_lambda.ext = (uint32_t) l2_W+${l2_off_lambda};
    copy_lambda.loc = (uint32_t) l1_buffer + ${l1_lambda_offset};
    pi_cl_dma_memcpy(&copy_lambda);                                                   
    pi_cl_dma_wait(&copy_k);                                                    
    pi_cl_dma_wait(&copy_lambda);
  }
% endif
  pi_cl_team_barrier(0);

  ////////////////////////////
  // First tile transfering //
  ////////////////////////////
 

  pi_cl_dma_copy_t copy_x;
  pi_cl_dma_copy_t copy_W;
  pi_cl_dma_copy_t copy_y;
  if(pi_core_id()==0)
  {
    copy_x.dir = PI_CL_DMA_DIR_EXT2LOC;
    copy_x.merge = 0;
    copy_x.size = (uint16_t) ${x_tile_size_byte};
    copy_x.id = 0;
    copy_x.ext = (uint32_t) l2_x;
    copy_x.loc = (uint32_t) l1_buffer + ${l1_x_offset};
    pi_cl_dma_memcpy(&copy_x);  
    pi_cl_dma_wait(&copy_x);
    copy_W.dir = PI_CL_DMA_DIR_EXT2LOC;
    copy_W.merge = 0;
    copy_W.size = (uint16_t) ${W_tile_size_byte};
    copy_W.id = 0;
    copy_W.ext = (uint32_t) l2_W;
    copy_W.loc = (uint32_t) l1_buffer + ${l1_W_offset};
    pi_cl_dma_memcpy(&copy_W);                                                   
    pi_cl_dma_wait(&copy_W);
  }
  pi_cl_team_barrier(0);

% if 'performance' in test_location:  
  // perf measurement begin
  pi_perf_conf(1<<PI_PERF_CYCLES);          
  pi_perf_reset();                      
  pi_perf_stop();                       
  pi_perf_start();
% endif
  // tile loop nest
% if flag_DW == 0:
  for(iter=0; iter<${tile_dim_nof}*${tile_dim_nif}*${tile_dim_h}; iter++) {
% else:
  for(iter=0; iter<${tile_dim_nof}*${tile_dim_h}; iter++) {
    // tile_dim_nof: Number of tiles for the output channels 
    // tile_dim_h: Number of tiles in the temporal dimension
    // Number of tiles: Factor of tiles, e.g. 64/32 = 2 tiles.
% endif
  % if tile_dim_nif != 1:
    // loop nest is nof,h,w,nif
    _i_nif_load += 1;
    if(_i_nif_load==${tile_dim_nif}) 
    {
      _i_nif_load = 0;
      _i_h_load += 1;
      if(_i_h_load==${tile_dim_h}) 
      {
        _i_h_load = 0;
        _i_nof_load += 1;
      }
    }
  % else:
    // loop nest is nof,h,w,(nif=0)
    _i_h_load += 1;
    if(_i_h_load==${tile_dim_h}) 
    {
      _i_h_load = 0;
    % if flag_DW == 1:
      _i_nif_load += 1;
    % endif
      _i_nof_load += 1;
    }
  % endif        
    // check if last in any dimension
    last_nof_exec = last_nof_load;
    last_nif_exec = last_nif_load;
    last_h_exec = last_h_load;
    last_nof_load = (_i_nof_load+1 == ${tile_dim_nof}) ? 1 : 0;
    last_nif_load = (_i_nif_load+1 == ${tile_dim_nif}) ? 1 : 0;
    last_h_load = (_i_h_load+1 == ${tile_dim_h}) ? 1 : 0;

    // compute double buffering offsets and update db state
    db_x = !db_state_x ? ${x_tile_size_byte} : 0;
    db_W = !db_state_W ? ${W_tile_size_byte} : 0;
    db_y = !db_state_y ? ${y_tile_size_byte} : 0;
  % if tile_dim_nif*tile_dim_h != 1:
    exec_db_x = db_state_x ? ${x_tile_size_byte} : 0;
  % else:
    exec_db_x = 0;
  % endif
    db_state_x = ! db_state_x;
    exec_db_W = db_state_W ? ${W_tile_size_byte} : 0;
    if (_i_nif_load!=_i_nif_exec || _i_nof_load!=_i_nof_exec)
      db_state_W = ! db_state_W;
    //switch all double buffering offset and y only after that all n_input_features have been analyzed: we need to pass all n_in to produce a single fil
  % if flag_DW == 0:
    if(last_nif_exec) 
      db_state_y = ! db_state_y;
  % endif
    // double buffered reads
  % if flag_DW == 0:
    if(iter<${tile_dim_nof}*${tile_dim_nif}*${tile_dim_h}-1) 
    {
  % else:
    if(iter<${tile_dim_nof}*${tile_dim_h}-1) 
    {
      asm volatile("": : :"memory");
  % endif
    % if tile_dim_nif*tile_dim_h != 1:
      x_tile_size_nif = (last_nif_load) ? ${x_tile_size_nif_last} : ${x_tile_size_nif};
      x_tile_size_h   = (last_h_load)   ? ${x_tile_size_h_last} : ${x_tile_size_h};
      x_tile_size_byte = x_tile_size_nif*x_tile_size_h*${x_data_size_byte}/8;
      x_length_nif_byte = (last_nif_load)   ? ${x_tile_size_nif_byte_last} : ${x_tile_size_nif_byte};
      // additionally overlap by padding for the first tile after a border one
      //this because in the first tile we use less pixels from x_buffer, since we have the ones of padding
      pad_offset_h=0;
      if(_i_h_load > 0)
        pad_offset_h = ${padding_left};
    % endif
      y_tile_size_h   = (last_h_load)   ? ${y_tile_size_h_last} : ${y_tile_size_h};
      W_tile_size_nof = (last_nof_load) ? ${W_tile_size_nof_last} : ${W_tile_size_nof};
      W_tile_size_nif = (last_nif_load) ? ${W_tile_size_nif_last} : ${W_tile_size_nif};
      W_tile_size_byte = W_tile_size_nof*W_tile_size_nif*${W_data_size_byte}*${fs1}/8;
      W_length_nif_byte = (last_nif_load) ? ${W_tile_size_nif_byte_last} : ${W_tile_nif_byte};
    // transfer of next input tile in double buffering
    % if tile_dim_nif*tile_dim_h != 1:
      if(pi_core_id()==0)
      {
        if (copy_W_pending)                                          
        {
          pi_cl_dma_wait(&copy_W);
          copy_W_pending = 0;        	
        }
        if (copy_x_pending)
        {
          pi_cl_dma_wait(&copy_x); 
          copy_x_pending = 0;                                                  
        }    
        if (copy_y_pending)
        {
          pi_cl_dma_wait(&copy_y); 
          copy_y_pending = 0;                                                  
        }  

        copy_x.dir = PI_CL_DMA_DIR_EXT2LOC;
        copy_x.merge = 0;
        copy_x.size = (uint16_t) x_tile_size_byte;
        copy_x.id = 0;
        copy_x.ext = (uint32_t) dory_get_tile_3d(l2_x, _i_h_load, 0, _i_nif_load, ${x_tile_size_h}, 1, ${x_tile_size_nif}, 1, ${nif},  ${conv_overlap1}, 0,0, pad_offset_h, 0, 0, ${x_data_size_byte});
        copy_x.loc = (uint32_t) (l1_buffer + ${l1_x_offset}) + db_x;
        pi_cl_dma_memcpy(&copy_x);  
        copy_x_pending = 1;
      }
    % endif
      if(pi_core_id()==0)
      {
        // transfer of next weight tile if changed input or output channels
        if (_i_nif_load!=_i_nif_exec || _i_nof_load!=_i_nof_exec)
        {
          if (copy_W_pending)                                          
          {
            pi_cl_dma_wait(&copy_W);
            copy_W_pending = 0;        	
          }
          if (copy_x_pending)
          {
            pi_cl_dma_wait(&copy_x); 
            copy_x_pending = 0;                                                  
          } 
          if (copy_y_pending)
          {
            pi_cl_dma_wait(&copy_y); 
            copy_y_pending = 0;                                                  
          } 
          copy_W.dir = PI_CL_DMA_DIR_EXT2LOC;
          copy_W.merge = 0;
          copy_W.size = (uint16_t) W_tile_size_byte;
          copy_W.id = 0;
          copy_W.ext = (uint32_t) dory_get_tile_3d(l2_W, _i_nof_load, 0, _i_nif_load, ${W_tile_size_nof}, ${fs1}, ${W_tile_size_nif}, ${fs1}, ${nif}, 0,0,0,0,0,0, ${W_data_size_byte});
          copy_W.loc = (uint32_t) (l1_buffer + ${l1_W_offset}) + db_W;
          pi_cl_dma_memcpy(&copy_W);  
          copy_W_pending = 1;      
        }
      }
    }
    // creation of the pointers to input, output, weights, lambda and k
    x = (${type} *) (l1_buffer + ${l1_x_offset} + exec_db_x);
% if FLAG_BATCHNORM == 1:
% if act_dim_bit == 32:
    k = (int32_t *) (l1_buffer + ${l1_k_offset} + _i_nof_exec*${k_tile_size_byte});
    lambda = (int32_t *) (l1_buffer + ${l1_lambda_offset} + _i_nof_exec*${lambda_tile_size_byte});
% else:
    k = (int64_t *) (l1_buffer + ${l1_k_offset} + _i_nof_exec*${k_tile_size_byte});
    lambda = (int64_t *) (l1_buffer + ${l1_lambda_offset} + _i_nof_exec*${lambda_tile_size_byte});
% endif
% endif
% if has_bias == 1:
    b = (${type} *) (l1_buffer + ${l1_b_offset} + _i_nof_exec*${bias_tile_size_byte});
% endif
    W = (${type} *) (l1_buffer + ${l1_W_offset} + exec_db_W);
    y = (${type} *) (l1_buffer + ${l1_y_offset} + db_y);
    // parameter passed to the kernel. Input and output sizes
    x_tile_size_nif_exec = (last_nif_exec) ? ${x_tile_size_nif_last} : ${x_tile_size_nif};
    x_tile_size_h_exec   = (last_h_exec)   ? ${x_tile_size_h_last} : ${x_tile_size_h};
    y_tile_size_nof = (last_nof_exec) ? ${y_tile_size_nof_last} : ${y_tile_size_nof};
    y_tile_size_h   = (last_h_exec)   ? ${y_tile_size_h_last} : ${y_tile_size_h};
    y_tile_size_byte = y_tile_size_nof*y_tile_size_h*${y_data_size_byte}/8;
    y_length_nof_byte = (last_nof_exec)   ? ${y_length_nof_byte_last} : ${y_tile_size_nof_byte};
    p_r = 0;
    p_l = 0;
    if (_i_h_exec == 0)
      p_l = ${padding_left};
    if (_i_h_exec == ${tile_dim_h}-1)
      p_r = ${padding_right};

    pi_cl_team_barrier(0);
    % if layer_type == 'normal':
    pulp_nn_convolution_1D_uint8(
    % elif layer_type == 'nodilation':
    pulp_nn_convolution_1D_uint8_nopad_nodilation(
    % elif layer_type == 'indirect':
    pulp_nn_indirect_convolution_1D_uint8(
    % endif
      x, 
      x_tile_size_h_exec, 
      x_tile_size_nif_exec, 
      W, 
      y_tile_size_nof, 
      ${fs1}, 
      p_l, 
      p_r, 
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
      y_tile_size_h, 
      % if FLAG_BATCHNORM == 1:
      k,
      lambda,
      % else:
      0,
      0,
      % endif
      im2col,
      ${dilation}, 
      NULL, 
      ${FLAG_RELU},
      ${FLAG_BATCHNORM}
      ); 
    pi_cl_team_barrier(0);

  % if tile_dim_nif != 1:
    if(last_nif_exec) 
    {
  % endif
      if(pi_core_id()==0)
      {
        if (copy_W_pending)                                          
        {
          pi_cl_dma_wait(&copy_W);
          copy_W_pending = 0;        	
        }
        if (copy_x_pending)
        {
          pi_cl_dma_wait(&copy_x); 
          copy_x_pending = 0;                                                  
        } 
        if (copy_y_pending)
        {
          pi_cl_dma_wait(&copy_y); 
          copy_y_pending = 0;                                                  
        } 
        copy_y.dir = PI_CL_DMA_DIR_LOC2EXT;
        copy_y.merge = 0;
        copy_y.size = (uint16_t) y_tile_size_byte;
        copy_y.stride = ${y_stride_c_byte};
        copy_y.length = y_length_nof_byte;
        copy_y.id = 0;
        copy_y.ext = (uint32_t) dory_get_tile_3d(l2_y, _i_h_exec, 0, _i_nof_exec, ${y_tile_size_h}, 1, ${y_tile_size_nof}, 1, ${int(nof)}, 0, 0, 0, 0, 0, 0, ${y_data_size_byte});
        copy_y.loc = (uint32_t) (l1_buffer + ${l1_y_offset}) + db_y;
        pi_cl_dma_memcpy_2d(&copy_y); 
        copy_y_pending = 1;       
      }
% if tile_dim_nif != 1 and flag_DW == 0:
    }
% endif
    // update prev iterators
    _i_nof_exec = _i_nof_load;
    _i_nif_exec = _i_nif_load;
    _i_h_exec = _i_h_load;
    pi_cl_team_barrier(0);
  }
  if (pi_core_id()==0)
  {
    pi_cl_dma_wait(&copy_y); 
  }
% if 'performance' in test_location:
  // performance measurements: end
  pi_perf_stop();          
  int cid = pi_core_id();   
  int perf_cyc =  pi_perf_read(PI_PERF_CYCLES); 
  int MACs = ${nof*nif*y_h*fs1};
  float perf_MAC =  (float)MACs/perf_cyc;
  if (cid == 0)
  {
    printf("[%d] : num_cycles: %d\n",cid,perf_cyc); 
    printf("[%d] : MACs: %d\n",cid,MACs ); 
    printf("[%d] : MAC/cycle: %f\n",cid,perf_MAC ); 
    printf("[%d] : n. of Cores: %d\n",cid,NUM_CORES); 
  }
% endif
  pi_cl_team_barrier(0);
}
