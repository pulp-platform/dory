/*
 * layer_template_L3.c
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
#include "${func_name}.h"
#include "pmsis.h"
#include "bsp/fs.h"
#include "bsp/fs/readfs.h"
#include "bsp/flash.h"
#include "bsp/ram.h"
#include "bsp/flash/hyperflash.h"
#include "bsp/ram/hyperram.h"
#include "network.h"

void __attribute__ ((noinline)) ${func_name}(void *args)
{
  layer_args_t *layer_args = (layer_args_t *)args;
  const unsigned int l3_x = layer_args->L3_input;
  const unsigned int l3_y = layer_args->L3_output;
  const unsigned int l3_W = layer_args->L3_after_weights;
  const unsigned int l2_x = layer_args->L2_input;
  const unsigned int l2_x_2 = layer_args->bypass;
  const unsigned int l2_y = layer_args->L2_output;
  const unsigned int l2_W = layer_args->L2_weights;
  const unsigned int l1_buffer = layer_args->L1_buffer;
  const unsigned int hyperram = layer_args->ram;
  const unsigned int outmult = layer_args->out_mult;
  const unsigned int out_shift = layer_args->out_shift;

  char *exec_input = l2_x, *transfer_input;
  char *exec_weights = l2_W, *transfer_weights;
  char *exec_output = l2_y, *transfer_output;

% if n_tile_W > 1:
  // weight L3 tiling. Parameters
  char *L2_weights_1 = l2_W;
  char *L2_weights_2 = l2_W + ${weight_dim} + ${lambda_dim} + ${k_dim} + ${bias_dim};

  int d_buffering_weights_t = 0;
  int d_buffering_weights_e = 0;
  pi_cl_ram_req_t buff_req_w1, buff_req_w2, buff_req_w3, buff_req_w4;
  transfer_weights = L2_weights_1;
  exec_weights = L2_weights_1;  
  // first tile transfer. Weights, k, lambda
  if(pi_core_id()==0)
  {
    pi_cl_ram_read(hyperram, l3_W, transfer_weights, ${weight_dim}, &buff_req_w1);
    % if k_dim != 0:
    pi_cl_ram_read(hyperram, l3_W+${(weight_dim+bias_dim)*n_tile_W}, transfer_weights + ${weight_dim} + ${bias_dim}, ${k_dim}, &buff_req_w2);
    pi_cl_ram_read(hyperram, l3_W+${(weight_dim+bias_dim+k_dim)*n_tile_W}, transfer_weights + ${weight_dim} + ${k_dim} + ${bias_dim}, ${lambda_dim}, &buff_req_w3);
    % endif 
    % if bias_dim != 0:
    pi_cl_ram_read(hyperram, l3_W+${weight_dim*n_tile_W}, transfer_weights + ${weight_dim}, ${bias_dim}, &buff_req_w4);
    % endif 
    pi_cl_ram_read_wait(&buff_req_w1);
    % if k_dim != 0:
    pi_cl_ram_read_wait(&buff_req_w2);
    pi_cl_ram_read_wait(&buff_req_w3);
    % endif
    % if bias_dim != 0:
    pi_cl_ram_read_wait(&buff_req_w4);
    % endif 
  }
  // switching buffers
  d_buffering_weights_t = !d_buffering_weights_t;
  transfer_weights = d_buffering_weights_t ? L2_weights_2 : L2_weights_1;
% endif

% if n_tile_x > 1 or input_L3==1:
  // input L3 tiling. Parameters
  char* L2_input_1;
  char* L2_input_2;
  int input_t = 0;
  int input_e = 0;
  pi_cl_ram_req_t buff_req_x1;
  L2_input_1 = l2_x;
  L2_input_2 = l2_x + ${dim_in};
  transfer_input = input_t ? L2_input_2 : L2_input_1;
  exec_input = input_e ? L2_input_2 : L2_input_1;
  // first tile transfer. Input activations
  if(pi_core_id()==0)
  {
    pi_cl_ram_read(hyperram, l3_x, transfer_input, ${dim_in}, &buff_req_x1);
    pi_cl_ram_read_wait(&buff_req_x1);
  }
  input_t = !input_t;
  transfer_input = input_t ? L2_input_2 : L2_input_1;
% endif

% if n_tile_y > 1:
  // output L3 tiling. Parameters
% if verbose == 1:
  int checksum = 0;
% endif
  char* L2_output_1;
  char* L2_output_2;
  int output_t = 0;
  int output_e = 0;
  pi_cl_ram_req_t buff_req_y1;
  L2_output_1 = l2_y;
  L2_output_2 = l2_y + ${dim_out};
  transfer_output = output_t ? L2_output_2 : L2_output_1;
  exec_output = output_e ? L2_output_2 : L2_output_1;
% endif

% if n_tile_y > 1 and n_tile_x > 1:
  // loop over input/output tiles
  for(int j=0; j<${n_tile_x}; j++) 
  {
    if(pi_core_id()==0)
    {
      int shift = 0; 
      if (j==0)
        shift = ${dim_in-int(conv_overlap1*n_in*w_in*BitIn/8) - int(padding*n_in*w_in*BitIn/8)};
      else
        shift = ${dim_in-int(conv_overlap1*n_in*w_in*BitIn/8) - int(padding*n_in*w_in*BitIn/8)} + j*${dim_in-int(conv_overlap1*n_in*w_in*BitIn/8)};
      // read from L3 of the new input tile. The shift is computed based on the overlap
      if (j<${n_tile_x-1})
        pi_cl_ram_read(hyperram, l3_x + shift, transfer_input, ${dim_in}, &buff_req_x1);
    }    
    input_t = !input_t;
    transfer_input = input_t ? L2_input_2 : L2_input_1;
% elif n_tile_y > 1:
  // loop over output tiles
  for(int j=0; j<${n_tile_y}; j++) 
  {
% elif n_tile_x > 1:
  // loop over input tiles
  for(int j=0; j<${n_tile_x}; j++) 
  {
    if(pi_core_id()==0)
    {
      pi_cl_ram_read_wait(&buff_req_x1);
      int shift = 0; 
      if (j==0)
        shift = ${dim_in-int(conv_overlap1*n_in*w_in*BitIn/8) - int(padding*n_in*w_in*BitIn/8)};
      else
        shift = ${dim_in-int(conv_overlap1*n_in*w_in*BitIn/8) - int(padding*n_in*w_in*BitIn/8)} + j*${dim_in-int(conv_overlap1*n_in*w_in*BitIn/8)};
      // read from L3 of the new input tile. The shift is computed based on the overlap
      if (j<${n_tile_x-1})
        pi_cl_ram_read(hyperram, l3_x + shift, transfer_input, ${dim_in}, &buff_req_x1);
    }    
    input_t = !input_t;
    transfer_input = input_t ? L2_input_2 : L2_input_1;
% else:
  int j = 0;
% endif

% if n_tile_W > 1:
  // loop over weight tiles
  for(int k=0; k<${n_tile_W}; k++) 
  {
    if (k < ${n_tile_W-1}) 
    {
      if(pi_core_id()==0) 
      {
        pi_cl_ram_read(hyperram, (l3_W+(k+1)*${weight_dim}), transfer_weights, ${weight_dim}, &buff_req_w1);
        % if k_dim != 0:
        pi_cl_ram_read(hyperram, l3_W+${(weight_dim+bias_dim)*n_tile_W}+ (k+1)*${k_dim}, transfer_weights + ${weight_dim} + ${bias_dim}, ${k_dim}, &buff_req_w2);
        pi_cl_ram_read(hyperram, l3_W+${(weight_dim+bias_dim+k_dim)*n_tile_W} + (k+1)*${lambda_dim}, transfer_weights + ${weight_dim}+ ${k_dim} + ${bias_dim}, ${lambda_dim}, &buff_req_w3);
        % endif
        % if bias_dim != 0:
        pi_cl_ram_read(hyperram, l3_W+${weight_dim*n_tile_W}+ (k+1)*${bias_dim}, transfer_weights + ${weight_dim}, ${bias_dim}, &buff_req_w4);
        % endif 
      }
    }  
% else:
    int k = 0;
% endif

    // execution of L2-L1 layer. Either top, middle or bottom layer.
    pi_cl_team_barrier(0);
    if (j==0)
    {
      unsigned int args[11] = {l3_x,
          l3_y,
          l3_W,
          exec_input,
          l2_x_2,
          dory_get_tile_3d(exec_output, 0, 0, k, ${h_out}, ${w_out}, ${n_out}, ${w_out}, ${n_out * n_tile_W}, 0, 0, 0, 0, 0, 0, ${y_data_size_byte}),
          exec_weights,
          l1_buffer,
          hyperram,
          outmult,
          out_shift};
      % if (n_tile_x > 1 or n_tile_y > 1) and padding > 0:
      ${L2_func_names[1]}(\
      % else:
      ${L2_func_names[0]}(\
      % endif
          args);
    }

% if n_tile_x > 1:
    else if (j==(${n_tile_x-1}))
    {
% else:
    else if (j==(${n_tile_y-1}))
    {
% endif

      unsigned int args[11] = {l3_x,
          l3_y,
          l3_W,
          % if n_tile_x > 1 or n_tile_W > 1:
          exec_input,
          % else:
          dory_get_tile_3d(exec_input, j, 0, 0, ${h_in}, ${w_in}, ${n_in}, ${w_in}, ${n_in}, ${conv_overlap1}, ${conv_overlap2},0, ${padding}, 0, 0, ${x_data_size_byte}),
          % endif
          l2_x_2,
          % if n_tile_y > 1:
          dory_get_tile_3d(exec_output, 0, 0, k, ${h_out}, ${w_out}, ${n_out}, ${w_out}, ${n_out * n_tile_W}, 0, 0, 0, 0, 0, 0, ${y_data_size_byte}),
          % else:
          dory_get_tile_3d(exec_output, j, 0, k, ${h_out}, ${w_out}, ${n_out}, ${w_out}, ${n_out * n_tile_W}, 0, 0, 0, 0, 0, 0, ${y_data_size_byte}),
          % endif
          exec_weights,
          l1_buffer,
          hyperram,
          outmult,
          out_shift};

      % if (n_tile_x > 1 or n_tile_y > 1) and padding > 0:
      ${L2_func_names[2]}(\
      % else:
      ${L2_func_names[0]}(\
      % endif
          args);
    }
    else
    {
      unsigned int args[11] = {l3_x,
          l3_y,
          l3_W,
          % if n_tile_x > 1 or n_tile_W > 1:
          exec_input,
          % else:
          dory_get_tile_3d(exec_input, j, 0, 0, ${h_in}, ${w_in}, ${n_in}, ${w_in}, ${n_in}, ${conv_overlap1}, ${conv_overlap2},0, ${padding}, 0, 0, ${x_data_size_byte}),
          % endif
          l2_x_2,
          % if n_tile_y > 1:
          dory_get_tile_3d(exec_output, 0, 0, k, ${h_out}, ${w_out}, ${n_out}, ${w_out}, ${n_out * n_tile_W}, 0, 0, 0, 0, 0, 0, ${y_data_size_byte}),
          % else:
          dory_get_tile_3d(exec_output, j, 0, k, ${h_out}, ${w_out}, ${n_out}, ${w_out}, ${n_out * n_tile_W}, 0, 0, 0, 0, 0, 0, ${y_data_size_byte}),
          % endif
          exec_weights,
          l1_buffer,
          hyperram,
          outmult,
          out_shift};
      ${L2_func_names[0]}(args);
      }    
      pi_cl_team_barrier(0);
% if n_tile_W > 1:
        if(pi_core_id()==0)
        {
          // waiting for weights, lambda, and k
          pi_cl_ram_read_wait(&buff_req_w1);
          % if k_dim != 0:
          pi_cl_ram_read_wait(&buff_req_w2);
          pi_cl_ram_read_wait(&buff_req_w3);
          % endif
          % if bias_dim != 0:
          pi_cl_ram_read_wait(&buff_req_w4);
          % endif
        }
        d_buffering_weights_e = !d_buffering_weights_e;
        exec_weights = d_buffering_weights_e ? L2_weights_2 : L2_weights_1;
        d_buffering_weights_t = !d_buffering_weights_t;
        transfer_weights = d_buffering_weights_t ? L2_weights_2 : L2_weights_1;
      }   
% endif

% if n_tile_x > 1:
    input_e = !input_e;
    exec_input = input_e ? L2_input_2 : L2_input_1;
% endif

% if n_tile_y > 1:
    if(pi_core_id()==0) 
    {
      % if n_tile_x > 1:
      // waits for input transfer to be ended
      pi_cl_ram_read_wait(&buff_req_x1);
      % endif
      // waits for output transfer to be ended
      if (j > 0)
        pi_cl_ram_write_wait(&buff_req_y1);
      pi_cl_ram_write(hyperram, (l3_y + j*${dim_out}), transfer_output, ${dim_out}, &buff_req_y1);
    % if verbose == 1:
    for(int j=0; j<${dim_out}; j++) 
      checksum += transfer_output[j];
    printf("checksum = %d\n", checksum);
    % endif
    }
    // switching parameters
    output_e = !output_e;
    output_t = !output_t;
    exec_output = output_e ? L2_output_2 : L2_output_1;
    transfer_output = output_t ? L2_output_2 : L2_output_1;
% endif
% if n_tile_x > 1 or n_tile_y > 1:
  }
% endif

% if n_tile_y > 1:
  // last wait
  if (pi_core_id()==0) {
    pi_cl_ram_write_wait(&buff_req_y1);
  }
% endif
  pi_cl_team_barrier(0);
}
