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
#include "${func_name_L3}.h"
% for name in func_name:
#include "${name}.h"
% endfor
#include "dory_get_tile.h"
#include "pmsis.h"
#include "bsp/fs.h"
#include "bsp/fs/readfs.h"
#include "bsp/flash.h"
#include "bsp/ram.h"
#include "bsp/flash/hyperflash.h"
#include "bsp/ram/hyperram.h"
#include "net_utils.h"



void __attribute__ ((noinline)) ${func_name_L3}(void *args)
{
  layer_args_t *layer_args = (layer_args_t *)args;
  const unsigned int l3_x = layer_args->L3_input;
  const unsigned int l3_y = layer_args->L3_output;
  const unsigned int l3_W = layer_args->L3_after_weights;
  const unsigned int l2_x = layer_args->L2_input;
  const unsigned int l2_y = layer_args->L2_output;
  const unsigned int l2_W = layer_args->L2_weights;
  const pi_device_t * ram = (pi_device_t *)layer_args->ram;

  layer_args_t tile_args = *layer_args;

  const struct {
    % if bias_dim != 0:
    void *bias;
    % endif
    % if k_dim != 0:
    void *k;
    void *l;
    % endif
    void *x;
    void *w;
    void *y;
  } db[2] = {
    {
      % if bias_dim != 0:
      .bias = l2_W + ${weight_dim},
      % endif
      % if k_dim != 0:
      .k = l2_W + ${weight_dim + bias_dim},
      .l = l2_W + ${weight_dim + bias_dim + k_dim},
      % endif
      .x = l2_x,
      .w = l2_W,
      .y = l2_y
    },
    {
      % if bias_dim != 0:
      .bias = l2_W + ${weight_dim + bias_dim + k_dim + lambda_dim + weight_dim},
      % endif
      % if k_dim != 0:
      .k = l2_W + ${weight_dim + bias_dim + k_dim + lambda_dim + weight_dim + bias_dim},
      .l = l2_W + ${weight_dim + bias_dim + k_dim + lambda_dim + weight_dim + bias_dim + k_dim},
      % endif
      .x = l2_x + ${dim_in},
      .w = l2_W + ${weight_dim + bias_dim + k_dim + lambda_dim},
      .y = l2_y + ${dim_out}
    }
  };

  int i_db_x = 0, i_db_w = 0, i_db_y = 0;

  % if n_tile_W > 1:
  // weight L3 tiling. Parameters
  pi_cl_ram_req_t req_w, req_k, req_l, req_bias;
  // first tile transfer. Weights, k, lambda
  if(pi_core_id()==0)
  {
    pi_cl_ram_read(ram, l3_W, db[i_db_w].w, ${weight_dim}, &req_w);
    % if k_dim != 0:
    pi_cl_ram_read(ram, l3_W+${(weight_dim+bias_dim)*n_tile_W}, db[i_db_w].k, ${k_dim}, &req_k);
    pi_cl_ram_read(ram, l3_W+${(weight_dim+bias_dim+k_dim)*n_tile_W}, db[i_db_w].l, ${lambda_dim}, &req_l);
    % endif 
    % if bias_dim != 0:
    pi_cl_ram_read(ram, l3_W+${weight_dim*n_tile_W}, db[i_db_w].bias, ${bias_dim}, &req_bias);
    % endif 
    pi_cl_ram_read_wait(&req_w);
    % if k_dim != 0:
    pi_cl_ram_read_wait(&req_k);
    pi_cl_ram_read_wait(&req_l);
    % endif
    % if bias_dim != 0:
    pi_cl_ram_read_wait(&req_bias);
    % endif 
  }
  // switching buffers
  % endif

  % if n_tile_x > 1 or input_L3 == 1:
  // input L3 tiling. Parameters
  pi_cl_ram_req_t req_x;
  // first tile transfer. Input activations
  if(pi_core_id()==0) {
    pi_cl_ram_read(ram, l3_x, db[i_db_x].x, ${dim_in}, &req_x);
    pi_cl_ram_read_wait(&req_x);
  }
  % endif

  % if n_tile_y > 1:
  // output L3 tiling. Parameters
  % if verbose == 1:
  int checksum = 0;
  % endif
  pi_cl_ram_req_t req_y;
  int offset_y = 0;
  % endif

  // read from L3 of the new input tile. The offset_x is computed based on the overlap
  int offset_x = ${dim_in-int(conv_overlap1*n_in*w_in*BitIn/8) - int(padding*n_in*w_in*BitIn/8)};

  % if n_tile_x > 1:
  // loop over input/output tiles
  for(int j = 0; j < ${n_tile_x}; j++) {
    if(pi_core_id()==0) {
      // Fetching next input tile
      if (j > 0) pi_cl_ram_read_wait(&req_x);
      if (j + 1 < ${n_tile_x}) {
        pi_cl_ram_read(ram, l3_x + offset_x, db[!i_db_x].x, ${dim_in}, &req_x);
        offset_x += ${dim_in-int(conv_overlap1*n_in*w_in*BitIn/8)};
      }
    }
  % elif n_tile_y > 1:
  // loop over output tiles
  for(int j = 0; j < ${n_tile_y}; j++) {
  % else:
  int j = 0;
  % endif

  % if n_tile_W > 1:
  // loop over weight tiles

  int offset_w = ${l3_offset_w + weight_dim};
  % if k_dim != 0:
  int offset_k = ${l3_offset_k + k_dim};
  int offset_l = ${l3_offset_l + lambda_dim};
  % endif
  % if bias_dim != 0:
  int offset_b = ${l3_offset_b + bias_dim};
  % endif

  for(int k = 0; k < ${n_tile_W}; k++) {
    if (k < ${n_tile_W-1}) {
      // Fetch next weights
      if(pi_core_id()==0) {
        pi_cl_ram_read(ram, l3_W + offset_w, db[!i_db_w].w, ${weight_dim}, &req_w);
        offset_w += ${weight_dim};
        % if k_dim != 0:
        pi_cl_ram_read(ram, l3_W + offset_k, db[!i_db_w].k, ${k_dim}, &req_k);
        offset_k += ${k_dim};
        pi_cl_ram_read(ram, l3_W + offset_l, db[!i_db_w].l, ${lambda_dim}, &req_l);
        offset_l += ${lambda_dim};
        % endif
        % if bias_dim != 0:
        pi_cl_ram_read(ram, l3_W + offset_b, db[!i_db_w].bias, ${bias_dim}, &req_bias);
        offset_b += ${bias_dim};
        % endif 
      }
    }  
  % else:
    int k = 0;
  % endif

    % if n_tile_x > 1 or n_tile_W > 1:
    tile_args.L2_input = db[i_db_x].x;
    % else:
    tile_args.L2_input = j == 0 ? db[i_db_x].x :
      dory_get_tile_3d(db[i_db_x].x, j, 0, 0, ${h_in}, ${w_in}, ${n_in}, ${w_in}, ${n_in}, ${conv_overlap1}, ${conv_overlap2},0, ${padding}, 0, 0, ${x_data_size_byte});
    % endif
    tile_args.L2_output = dory_get_tile_3d(db[i_db_y].y, ${0 if n_tile_y > 1 else 'j'}, 0, k, ${h_out}, ${w_out}, ${n_out}, ${w_out}, ${n_out * n_tile_W}, 0, 0, 0, 0, 0, 0, ${y_data_size_byte});
    tile_args.L2_weights = db[i_db_w].w;

    // execution of L2-L1 layer. Either top, middle or bottom layer.
    pi_cl_team_barrier(0);

    if (j==0) {
      ${func_name[1] if (n_tile_x > 1 or n_tile_y > 1) and padding > 0 else func_name[0]}((void*)&tile_args);
    } \
% if n_tile_x > 1:
else if (j == (${n_tile_x-1})) {
% else:
else if (j == (${n_tile_y-1})) {
% endif
      ${func_name[2] if (n_tile_x > 1 or n_tile_y > 1) and padding > 0 else func_name[0]}((void*)&tile_args);
    } else {
      ${func_name[0]}((void*)&tile_args);
    }    

      pi_cl_team_barrier(0);
      % if n_tile_W > 1:
        if(pi_core_id()==0)
        {
          // waiting for weights, lambda, and k
          pi_cl_ram_read_wait(&req_w);
          % if k_dim != 0:
          pi_cl_ram_read_wait(&req_k);
          pi_cl_ram_read_wait(&req_l);
          % endif
          % if bias_dim != 0:
          pi_cl_ram_read_wait(&req_bias);
          % endif
        }
        i_db_w = !i_db_w;
      }   
      % endif 
  % if n_tile_x > 1:
    i_db_x = !i_db_x;
  % endif
  % if n_tile_y > 1:
    if(pi_core_id()==0) 
    {
      % if n_tile_x > 1:
      // waits for input transfer to be ended
      pi_cl_ram_read_wait(&req_x);
      % endif
      // waits for output transfer to be ended
      if (j > 0)
        pi_cl_ram_write_wait(&req_y);
      pi_cl_ram_write(ram, l3_y + offset_y, db[i_db_y].y, ${dim_out}, &req_y);
      offset_y += ${dim_out};
    % if verbose == 1:
    for(int iter = 0; iter < ${dim_out}; iter++)
      checksum += db[i_db_y].y[iter];
    printf("checksum = %d\n", checksum);
    % endif
    }
    // switching parameters
    i_db_y = !i_db_y;
  % endif
  % if n_tile_x > 1 or n_tile_y > 1:
  }
  % endif
  % if n_tile_y > 1:
  // last wait
  if(pi_core_id()==0) {
    pi_cl_ram_write_wait(&req_y);
  }
  % endif
  pi_cl_team_barrier(0);
}
