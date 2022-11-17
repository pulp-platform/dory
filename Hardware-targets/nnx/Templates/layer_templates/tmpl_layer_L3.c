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
#include "bsp/ram.h"
#include "network.h"
#include "dory_get_tile.h"

#ifdef CHECKSUM_L3
static int checksum = 0;

#define CHECKSUM_CALCULATE()   ${"\\"}
    uint8_t *y = db[i_db_y].y; ${"\\"}
    for (int i = 0; i < ${dim_out}; i++) checksum += y[i]
#define CHECKSUM_REPORT() printf("checksum = %d\n", checksum)
#else
#define CHECKSUM_CALCULATE()
#define CHECKSUM_REPORT()
#endif


#ifdef MEASURE_L3_TILING
#define PERF_INIT() pi_perf_conf(1<<PI_PERF_CYCLES)

#define PERF_RESTART()             ${"\\"}
  do {                             ${"\\"}
    asm volatile("": : :"memory"); ${"\\"}
    pi_perf_stop();                ${"\\"}
    pi_perf_reset();               ${"\\"}
    pi_perf_start();               ${"\\"}
    asm volatile("": : :"memory"); ${"\\"}
  } while (0)

#define PERF_READ(var)                ${"\\"}
  asm volatile("": : :"memory");      ${"\\"}
  var = pi_perf_read(PI_PERF_CYCLES); ${"\\"}
  asm volatile("": : :"memory")

#define PERF_ACCUMULATE(accumulator, x) accumulator += x

static int cycles_first_l3_dma = 0, cycles_l3_dma = 0, cycles_exec = 0, cycles_l3_postamble = 0;
static int total_cycles_l3_dma = 0, total_cycles_exec = 0;
static int is_first_exec = 1;

#define PERF_REPORT() ${"\\"}
  printf("Measured latency L3 tiling - preamble + first dma: %d, total dma: %d, total execution: %d, postamble: %d\n", ${"\\"}
         cycles_first_l3_dma, total_cycles_l3_dma, total_cycles_exec, cycles_l3_postamble);
#else
#define PERF_INIT()
#define PERF_RESTART()
#define PERF_READ(var)
#define PERF_ACCUMULATE(accumulator, x)
#define PERF_REPORT()
#endif

void __attribute__ ((noinline)) ${func_name}(void *args)
{
  PERF_INIT();
  PERF_RESTART();

  layer_args_t *layer_args = (layer_args_t *)args;
  layer_args_t  tile_args = *layer_args;
  % if not (n_tile_x > 1 or n_tile_y > 1):
  tile_args.padding = PAD_TOP & PAD_BOTTOM;
  % endif

  const void *l3_x = layer_args->L3_input;
  const void *l3_y = layer_args->L3_output;
  const void *l3_W = layer_args->L3_after_weights;
  const void *l2_x = layer_args->L2_input;
  const void *l2_y = layer_args->L2_output;
  const void *l2_W = layer_args->L2_weights;
  const void *ram  = layer_args->ram;

  % if n_tile_x > 1:
  pi_cl_ram_req_t req_x;
  % endif
  % if n_tile_y > 1:
  pi_cl_ram_req_t req_y;
  % endif
  % if n_tile_W > 1:
  pi_cl_ram_req_t req_w, req_bias, req_k, req_l;
  % endif
  % if n_tile_x > 1:

  int offset_x = ${dim_in-int(conv_overlap1*n_in*w_in*BitIn/8) - int(padding_top*n_in*w_in*BitIn/8)};
  % endif
  % if n_tile_y > 1:
  int offset_y = 0;
  % endif

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
  % if n_tile_x > 1 or input_L3 == 1:

  // first tile transfer. Input activations
  pi_cl_ram_read(ram, l3_x, db[i_db_x].x, ${dim_in}, &req_x);
  pi_cl_ram_read_wait(&req_x);
  % endif

  % if n_tile_W > 1:
  // first tile transfer. Weights, k, lambda
  pi_cl_ram_read(ram, l3_W + ${l3_offset_w}, db[i_db_w].w, ${weight_dim}, &req_w);
  % if bias_dim != 0:
  pi_cl_ram_read(ram, l3_W + ${l3_offset_b}, db[i_db_w].bias, ${bias_dim}, &req_bias);
  % endif
  % if k_dim != 0:
  pi_cl_ram_read(ram, l3_W + ${l3_offset_k}, db[i_db_w].k, ${k_dim}, &req_k);
  pi_cl_ram_read(ram, l3_W + ${l3_offset_l}, db[i_db_w].l, ${lambda_dim}, &req_l);
  % endif
  pi_cl_ram_read_wait(&req_w);
  % if k_dim != 0:
  pi_cl_ram_read_wait(&req_k);
  pi_cl_ram_read_wait(&req_l);
  % endif
  % if bias_dim != 0:
  pi_cl_ram_read_wait(&req_bias);
  % endif
  // switching buffers
  % endif

  // read from L3 of the new input tile. The offset_x is computed based on the overlap

  % if n_tile_x > 1:
  // loop over input/output tiles
  for (int j = 0; j < ${n_tile_x}; j++) {
    // Fetching next input tile
    if (j > 0) pi_cl_ram_read_wait(&req_x);
    if (j + 1 < ${n_tile_x}) {
      pi_cl_ram_read(ram, l3_x + offset_x, db[!i_db_x].x, ${dim_in}, &req_x);
      offset_x += ${dim_in-int(conv_overlap1*n_in*w_in*BitIn/8)};
    }
  % elif n_tile_y > 1:
  // loop over output tiles
  for (int j = 0; j < ${n_tile_y}; j++) {
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

    for (int k = 0; k < ${n_tile_W}; k++) {
      if (k + 1 < ${n_tile_W}) {
        // Fetch next weights
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
    % else:
    int k = 0;
    % endif

      % if n_tile_x > 1 or n_tile_W > 1:
      tile_args.L2_input = db[i_db_x].x;
      % else:
      tile_args.L2_input = j == 0 ? db[i_db_x].x :
        dory_get_tile_3d(db[i_db_x].x, j, 0, 0, ${h_in}, ${w_in}, ${n_in}, ${w_in}, ${n_in}, ${conv_overlap1}, ${conv_overlap2}, 0, ${padding_top}, 0, 0, ${x_data_size_byte});
      % endif
      tile_args.L2_output = dory_get_tile_3d(db[i_db_y].y, ${0 if n_tile_y > 1 else 'j'}, 0, k, ${h_out}, ${w_out}, ${n_out}, ${w_out}, ${n_out * n_tile_W}, 0, 0, 0, 0, 0, 0, ${y_data_size_byte});
      tile_args.L2_weights = db[i_db_w].w;
      % if n_tile_x > 1 or n_tile_y > 1:
      if (j == 0)
        tile_args.padding = PAD_TOP;
      else if (j == ${n_tile_x if n_tile_x > 1 else n_tile_y} - 1)
        tile_args.padding = PAD_BOTTOM;
      else
        tile_args.padding = NO_PAD;
      % endif

#ifdef MEASURE_L3_TILING
      if (is_first_exec) {
        PERF_READ(cycles_first_l3_dma);
        PERF_RESTART();
        is_first_exec = 0;
      } else {
        PERF_READ(cycles_l3_dma);
        PERF_RESTART();
        PERF_ACCUMULATE(total_cycles_l3_dma, cycles_l3_dma);
      }
#endif

      // execution of L2-L1 layer. Either top, middle or bottom layer.
      ${L2_func_names[0]}((void *)&tile_args);

      PERF_READ(cycles_exec);
      PERF_RESTART();
      PERF_ACCUMULATE(total_cycles_exec, cycles_exec);

    % if n_tile_W > 1:

      // waiting for weights, lambda, and k
      pi_cl_ram_read_wait(&req_w);
      % if k_dim != 0:
      pi_cl_ram_read_wait(&req_k);
      pi_cl_ram_read_wait(&req_l);
      % endif
      % if bias_dim != 0:
      pi_cl_ram_read_wait(&req_bias);
      % endif

      i_db_w = !i_db_w;
    }
    % endif
    % if n_tile_x > 1:

    i_db_x = !i_db_x;
    % endif
    % if n_tile_y > 1:
    // waits for output transfer to be ended
    if (j > 0) pi_cl_ram_write_wait(&req_y);
    pi_cl_ram_write(ram, l3_y + offset_y, db[i_db_y].y, ${dim_out}, &req_y);
    offset_y += ${dim_out};
    CHECKSUM_CALCULATE();

    i_db_y = !i_db_y;
    % endif
  % if n_tile_x > 1 or n_tile_y > 1:
  }
  % endif

  % if n_tile_y > 1:
  // last wait
  pi_cl_ram_write_wait(&req_y);
  % if verbose:
  CHECKSUM_REPORT();
  % endif
  % endif

  PERF_READ(cycles_l3_postamble);
  PERF_RESTART();
  PERF_REPORT();
}
