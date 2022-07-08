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
  const unsigned int ram = layer_args->ram;
  const unsigned int outmult = layer_args->out_mult;
  const unsigned int out_shift = layer_args->out_shift;

  pi_cl_ram_req_t req_x, req_w, req_y, req_k, req_l;

  int offset_x = 0, offset_w = 0, offset_k = 0, offset_l = 0, offset_y = 0;

  int i_x = 0, i_w = 0;

  int i_db_x = 0, i_db_y = 0, i_db_w = 0;

  int is_load_w = 1, is_load_x = 1;

  layer_args_t tile_args;
  tile_args.L1_buffer = l1_buffer;
  tile_args.out_shift = out_shift;

  const struct {
% if FLAG_BATCHNORM == 1:
    int k;
    int l;
% endif
    int x;
    int y;
    int w;
  } db[2] = {
    {
% if FLAG_BATCHNORM == 1:
      .k = l2_W + ${weight_dim},
      .l = l2_W + ${weight_dim + k_dim},
% endif
      .x = l2_x,
      .y = l2_y,
      .w = l2_W
    },
    {
% if FLAG_BATCHNORM == 1:
      .k = l2_W + ${weight_dim + k_dim + lambda_dim + bias_dim + weight_dim},
      .l = l2_W + ${weight_dim + k_dim + lambda_dim + bias_dim + weight_dim + k_dim},
% endif
      .x = l2_x + ${dim_in},
      .y = l2_y + ${dim_out},
      .w = l2_W + ${weight_dim + lambda_dim + k_dim + bias_dim}
    }
  };

  const int total_tiles = ${n_tile_W} /*n_tile_W*/ * ${n_tile_x} /*n_tile_x*/;

  for (int i_tile = 0; i_tile < total_tiles; i_tile++) {
    const int x_tile_ptr = db[i_db_x].x;
    const int w_tile_ptr = db[i_db_w].w;
    const int y_tile_ptr = db[i_db_y].y
% if FLAG_BATCHNORM == 1:
    const int k_tile_ptr = db[i_db_w].k;
    const int l_tile_ptr  = db[i_db_w].l;
% endif

    if (is_load_x) {
        pi_cl_ram_read(ram, l3_x + offset_x, x_tile_ptr, ${dim_in}, &req_x);
    }

    if (is_load_w) {
        pi_cl_ram_read(ram, l3_W + offset_w, w_tile_ptr, ${weight_dim}, &req_x);
        % if FLAG_BATCHNORM == 1:
        pi_cl_ram_read(ram, l3_W + ${l3_offset_k - l3_offset_w} + offset_k, k_tile_ptr, ${k_dim}, &req_k);
        pi_cl_ram_read(ram, l3_W + ${l3_offset_l - l3_offset_w} + offset_l, l_tile_ptr, ${lambda_dim}, &req_l);
        % endif
    }

    pi_cl_team_barrier(0);

    tile_args.L2_input = x_tile_ptr;
    tile_args.L2_output = y_tile_ptr;
    tile_args.L2_weights = w_tile_ptr;

    ${L2_func_names[0]}(tile_args);

    pi_cl_ram_write(ram, l3_y + offset_y, y_tile_ptr, ${dim_out}, &req_y);

    const int i_x_prev = i_x, i_w_prev = i_w;

    i_w++;
    if (i_w == ${n_tile_W}) {
        i_w = 0;
        i_x++;
    }

    is_load_x = i_x_prev != i_x;
    is_load_w = i_w_prev != i_w;

    if (is_load_x) i_db_x = !i_db_x;
    if (is_load_w) i_db_w = !i_db_w;
    i_db_y = !i_db_y;
  }

}
