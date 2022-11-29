/*
 * layer_template_nnx.c
 * Francesco Conti <f.conti@unibo.it>
 * Alessio Burrello <alessio.burrello@unibo.it>
 * Luka Macan <luka.macan@unibo.it>
 *
 * Copyright (C) 2018-2022 University of Bologna
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
#include "pulp_nnx.h"
#include "network.h"
#include "dory_dma.h"
#include "dory_get_tile.h"

/* Measurement defines
 * To enable any of the special measurements modes, just add the define
 *   - NO_DMA_SYNC - removes the DMA synchronization points
 *   - NO_DMA      - removes the DMA synchronization points and transactions
 */
//#define NO_DMA_SYNC
//#define NO_DMA

#if defined MEASURE_LAYER_COMPONENT || defined MEASURE_EXEC_COMPONENT
% if flag_DW == 0:
#define TOTAL_TILES (${tile_dim_nof} /*tile_dim_nof*/ * ${tile_dim_nif} /*tile_dim_nif*/ * ${tile_dim_h} /*tile_dim_h*/ * ${tile_dim_w} /*tile_dim_w*/)
% else:
#define TOTAL_TILES (${tile_dim_nof} /*tile_dim_nof*/ * ${tile_dim_h} /*tile_dim_h*/ * ${tile_dim_w} /*tile_dim_w*/)
% endif

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

#define PERF_ACCUMULATE(accumulator, var) accumulator += var

#define PERF_ARRAY_DECLARATION(name) static int cycles_ ## name[TOTAL_TILES];

#define PERF_ARRAY_REPORT(name)          ${"\\"}
do {                                     ${"\\"}
  printf(#name " latency:\n");           ${"\\"}
  for (int i = 0; i < TOTAL_TILES; i++)  ${"\\"}
    printf("%d,\n", cycles_ ## name[i]); ${"\\"}
} while (0)
#else
#define PERF_INIT()
#define PERF_RESTART()
#define PERF_READ(var)
#define PERF_ACCUMULATE(accumulator, var)
#define PERF_ARRAY_DECLARATION(name)
#define PERF_ARRAY_REPORT(name)
#endif

#ifdef MEASURE_LAYER_COMPONENT
static int cycles_preamble = 0, cycles_first_conf = 0, cycles_first_dma = 0, cycles_nnx = 0, cycles_postamble = 0;

#define PERF_LAYER_COMPONENT_READ(component) \
  PERF_READ(cycles_ ## component);           \
  PERF_RESTART()

#define PERF_LAYER_COMPONENT_REPORT()                                                               ${"\\"}
  do {                                                                                              ${"\\"}
    printf("Measured time - preamble: %d, first conf: %d, first dma: %d, nnx: %d, postamble: %d\n", ${"\\"}
           cycles_preamble, cycles_first_conf, cycles_first_dma, cycles_nnx, cycles_postamble);     ${"\\"}
  } while (0)
#else  // MEASURE_LAYER_COMPONENT
#define PERF_LAYER_COMPONENT_READ(component)
#define PERF_LAYER_COMPONENT_REPORT()
#endif  // MEASURE_LAYER_COMPONENT

#ifdef MEASURE_EXEC_COMPONENT
typedef enum {
    exec_component_acquire,
    exec_component_dma_memcpy,
    exec_component_dma_barrier,
    n_exec_component
} exec_component_e;

static int cycles_exec_component[TOTAL_TILES][n_exec_component] = {0};

#define PERF_EXEC_COMPONENT_BEGIN(component)                               ${"\\"}
int cycles_ ## component ## _start = 0, cycles_ ## component ## _stop = 0; ${"\\"}
PERF_READ(cycles_ ## component ## _start)

#define PERF_EXEC_COMPONENT_END(component)                         ${"\\"}
PERF_READ(cycles_ ## component ## _stop);                          ${"\\"}
cycles_exec_component[i_tile][exec_component_ ## component] =      ${"\\"}
    cycles_ ## component ## _stop - cycles_ ## component ## _start

#define PERF_EXEC_COMPONENT_REPORT()              ${"\\"}
do {                                              ${"\\"}
  printf("Execution component report:\n");        ${"\\"}
  printf("acquire,dma_memcpy,dma_barrier\n");     ${"\\"}
  for (int i = 0; i < TOTAL_TILES; i++) {         ${"\\"}
    for (int j = 0; j < n_exec_component; j++) {  ${"\\"}
      printf("%d,", cycles_exec_component[i][j]); ${"\\"}
    }                                             ${"\\"}
    printf("\n");                                 ${"\\"}
  }                                               ${"\\"}
} while (0)
#else  // MEASURE_EXEC_COMPONENT
#define PERF_EXEC_COMPONENT_BEGIN(component)
#define PERF_EXEC_COMPONENT_END(component)
#define PERF_EXEC_COMPONENT_REPORT()
#endif  // MEASURE_EXEC_COMPONENT


#ifdef GVSOC_LOGGING
#define GVSOC_LOG_LEVEL 1
#include "pulp_nnx_util.h"
#endif  // GVSOC_LOGGING

#ifdef DEBUG_DMA_COPY
#define dory_dma_memcpy_async(dma)                                                                                             ${"\\"}
  do                                                                                                                           ${"\\"}
  {                                                                                                                            ${"\\"}
    printf(                                                                                                                    ${"\\"}
        "\n[" #dma "] ext:%p, loc:%p, n_2d:%d, s_2d:%d, n_1d:%d, s_1d:%d, l_1d:%d\n",                                          ${"\\"}
        dma.ext, dma.loc, dma.number_of_2d_copies, dma.stride_2d, dma.number_of_1d_copies, dma.stride_1d, dma.length_1d_copy); ${"\\"}
    dory_dma_memcpy_async(dma);                                                                                                ${"\\"}
  } while (0)
#endif

% if ULTRA_VERBOSE:
// #define VERBOSE_PRINT(...) printf(__VA_ARGS__)
#define VERBOSE_PRINT(...)
% endif

#define MIN(a,b) (a < b ? a : b)

// DMA_Y_CONTEXT_SIZE
// At least NNX_CONTEXT_SIZE + 1 DMA_copy_y configurations are needed because output
// is always 2 phases late, so there are 2 configurations for previous stages
// and 1 for the current. It can be done differently but it sacrifices code
// readability which was prioritized at the moment.
// Size of 4 has been assigned to have index calculation done with only masking.
#define DMA_Y_CONTEXT_SIZE 4
#define DMA_Y_MASK 0x3
#define DMA_Y_INDEX(n) (n & DMA_Y_MASK)

static int increment_i_dma_y(int i) {
  return (i + 1) != DMA_Y_CONTEXT_SIZE ? i + 1 : 0; 
}

void ${func_name}(
  void *args
) {
  /////////////
  // Logging //
  /////////////

#ifdef GVSOC_LOGGING
  nnx_activate_gvsoc_logging(GVSOC_LOG_LEVEL);
#endif

  PERF_INIT();
  PERF_RESTART();

  //////////////////////////
  // Arguments assignment //
  //////////////////////////

  // Keep the same interface between L2 and L3 memory
  layer_args_t *layer_args = (layer_args_t *) args;
  const unsigned int l2_x = layer_args->L2_input;
  const unsigned int l2_y = layer_args->L2_output;
  const unsigned int l2_W = layer_args->L2_weights;
% if FLAG_BATCHNORM == 1:
  const unsigned int l2_scale = l2_W + ${l2_k_offset};
  const unsigned int l2_bias  = l2_W + ${l2_lambda_offset};
% endif
  const unsigned int l1_buffer = layer_args->L1_buffer;
  const unsigned int out_shift = layer_args->out_shift;
  nnx_padding_t padding = {
    .top    = layer_args->padding & PAD_TOP ? ${padding_top} : DONT_PAD,
    .right  = ${padding_right},
    .bottom = layer_args->padding & PAD_BOTTOM ? ${padding_bottom} : DONT_PAD,
    .left   = ${padding_left},
    .value = 0
  };

  /////////////////////
  // DMA declaration //
  /////////////////////

  DMA_copy DMA_copy_W, DMA_copy_x;
% if FLAG_BATCHNORM == 1:
  DMA_copy DMA_copy_k, DMA_copy_lambda;
% endif
  DMA_copy DMA_copy_y[DMA_Y_CONTEXT_SIZE];

  //////////////////
  // DMA defaults //
  //////////////////

  DMA_copy_x.stride_2d = ${l1_x_dma_stride_2d};
  DMA_copy_x.stride_1d = ${l1_x_dma_stride_1d};
  DMA_copy_x.dir = 1;
  
  DMA_copy_W.number_of_2d_copies = 1;
  DMA_copy_W.stride_2d = 0;
  DMA_copy_W.number_of_1d_copies = 1;
  DMA_copy_W.stride_1d = 0;
  DMA_copy_W.dir = 1;

% if FLAG_BATCHNORM == 1:
  DMA_copy_k.stride_2d = 0;
  DMA_copy_k.number_of_2d_copies = 1;
  DMA_copy_k.stride_1d = 0;
  DMA_copy_k.number_of_1d_copies = 1;
  DMA_copy_k.dir = 1;

  DMA_copy_lambda.stride_2d = 0;
  DMA_copy_lambda.number_of_2d_copies = 1;
  DMA_copy_lambda.stride_1d = 0;
  DMA_copy_lambda.number_of_1d_copies = 1;
  DMA_copy_lambda.dir = 1;
% endif
  
  for (int i = 0; i < DMA_Y_CONTEXT_SIZE; i++) {
    DMA_copy_y[i].stride_2d = ${l1_y_dma_stride_2d};
    DMA_copy_y[i].stride_1d = ${l1_y_dma_stride_1d};
    DMA_copy_y[i].dir = 0;
  }

% if has_bias == 1:
  DMA_copy DMA_copy_bias;
  DMA_copy_bias.stride_2d = 0;
  DMA_copy_bias.stride_1d = 0;
  DMA_copy_bias.dir = 1;
% endif

  //////////////////////////
  // Variable declaration //
  //////////////////////////

  int y_tile_size_h = ${y_tile_size_h};
  int y_tile_size_w = ${y_tile_size_w};
  int y_length_nof_byte = ${y_tile_size_nof_byte};

  int x_tile_size_h = ${x_tile_size_h};
  int x_tile_size_w = ${x_tile_size_w};
  int x_length_nif_byte = ${x_tile_size_nif_byte};

  int W_tile_size_nof = ${W_tile_size_nof};
  int W_tile_size_nif = ${W_tile_size_nif};
  int W_tile_ko_len = ${l1_W_tile_ko_len};

  // Tile loop indices
  int i_nof = 0, i_nif = 0, i_h = 0, i_w = 0;

  // Double buffer pointer indices
  int i_db_x = 0, i_db_y = 0, i_db_w = 0;

  // Store iterator
  int i_store_y = 0;

  // Load flags (first tile must be loaded)
  int is_load_w = 1, is_load_x = 1;

  ////////////////////////
  // Double buffer init //
  ////////////////////////

  const int l1_buffer_x = l1_buffer + ${l1_x_offset};
  const int l1_buffer_y = l1_buffer + ${l1_y_offset};
  const int l1_buffer_w = l1_buffer + ${l1_W_offset};
% if FLAG_BATCHNORM:
  const int l1_buffer_scale = l1_buffer + ${l1_k_offset};
  const int l1_buffer_bias = l1_buffer + ${l1_lambda_offset};
% endif

  const struct {
% if FLAG_BATCHNORM == 1:
    int scale;
    int bias;
% endif
    int x;
    int y;
    int w;
  } db[2] = {
    {
% if FLAG_BATCHNORM == 1:
      .scale = l1_buffer_scale,
      .bias = l1_buffer_bias,
% endif
      .x = l1_buffer_x,
      .y = l1_buffer_y,
      .w = l1_buffer_w
    },
    {
% if FLAG_BATCHNORM == 1:
      .scale = l1_buffer_scale + ${l1_k_tile_size},
      .bias = l1_buffer_bias + ${l1_lambda_tile_size},
% endif
      .x = l1_buffer_x + ${l1_x_tile_size},
      .y = l1_buffer_y + ${l1_y_tile_size},
      .w = l1_buffer_w + ${l1_W_tile_size}
    }
  };

  //////////////////////
  // Accelerator init //
  //////////////////////

  nnx_init();

  ///////////////////////
  // NNX task defaults //
  ///////////////////////

  enum nnx_task_e {
    NNX_TASK_BODY,
    NNX_TASK_REMAINDER,
    NNX_TASK_COUNT
  };

  nnx_task_t  nnx_tasks[NNX_TASK_COUNT];
  nnx_task_t *nnx_task_to_offload;


  nnx_weights_t nnx_weights = {
    .data = db[i_db_w].w,
    .height = ${fs1},
    .width = ${fs2},
    .depth = ${W_tile_size_nif},
    .n_weights = ${W_tile_size_nof},
    .bitwidth = ${W_data_size},
    .offset_factor = ${-(2**(W_data_size-1))},
    .offset_mode = weightOffsetModeLayerWise
  };

  nnx_feature_t nnx_input = {
    .data = db[i_db_x].x,
    .height = ${x_tile_size_h},
    .width = ${x_tile_size_w},
    .depth = ${x_tile_size_nif},
    .bitwidth = featureBitwidth8Bit
  };

  nnx_feature_t nnx_output = {
    .data = db[i_db_y].y,
    .height = ${y_tile_size_h},
    .width = ${y_tile_size_w},
    .depth = ${y_tile_size_nof},
    .bitwidth = featureBitwidth8Bit
  };

  const nnx_norm_t norm = {
    .mode  = ${"normMode32Bit" if act_dim_bit == 32 else "normMode8Bit" if act_dim_bit == 8 else "FLAG_UNUSED"},
    .flag_bias  = FLAG_USED,
    .flag_shift = FLAG_UNUSED
  };

  const nnx_quant_t quant = {
    .shift_amount = out_shift,
    .mode = quantMode8Bit,
    .function = quantFunctionRelu,
    .flag_rounding = FLAG_UNUSED
  };

  /////////////////
  // Total tiles //
  /////////////////

% if flag_DW == 0:
  const int total_tiles = ${tile_dim_nof} /*tile_dim_nof*/ * ${tile_dim_nif} /*tile_dim_nif*/ * ${tile_dim_h} /*tile_dim_h*/ * ${tile_dim_w} /*tile_dim_w*/;
% else:
  const int total_tiles = ${tile_dim_nof} /*tile_dim_nof*/ * ${tile_dim_h} /*tile_dim_h*/ * ${tile_dim_w} /*tile_dim_w*/;
% endif

  ///////////////////
  // NNX task init //
  ///////////////////

  const stride = ${stride};

  const nnx_padding_t padding_init = {
    .top = ${'DONT_PAD' if tile_dim_h > 1 else 'padding.top'},
    .right = ${'DONT_PAD' if tile_dim_w > 1 else 'padding.right'},
    .bottom = ${'DONT_PAD' if tile_dim_h > 1 else 'padding.bottom'},
    .left = ${'DONT_PAD' if tile_dim_w > 1 else 'padding.left'},
    .value = 0
  };

  for (int i = 0; i < MIN(NNX_TASK_COUNT, total_tiles); i++) {
    nnx_task_init(&nnx_tasks[i]);
    int err = nnx_conv_${fs1}x${fs2}${'_dw' if flag_DW else ''}(&(nnx_tasks[i].cfg), nnx_weights, nnx_input, nnx_output, padding_init, stride);
    nnx_norm_quant(&(nnx_tasks[i].cfg), norm, quant);
    nnx_pad_input(&(nnx_tasks[i].cfg), padding_init);
  }

  PERF_LAYER_COMPONENT_READ(preamble);


//  /$$$$$$$$ /$$$$$$ /$$       /$$$$$$$$       /$$        /$$$$$$   /$$$$$$  /$$$$$$$ 
// |__  $$__/|_  $$_/| $$      | $$_____/      | $$       /$$__  $$ /$$__  $$| $$__  $$
//    | $$     | $$  | $$      | $$            | $$      | $$  \ $$| $$  \ $$| $$  \ $$
//    | $$     | $$  | $$      | $$$$$         | $$      | $$  | $$| $$  | $$| $$$$$$$/
//    | $$     | $$  | $$      | $$__/         | $$      | $$  | $$| $$  | $$| $$____/ 
//    | $$     | $$  | $$      | $$            | $$      | $$  | $$| $$  | $$| $$      
//    | $$    /$$$$$$| $$$$$$$$| $$$$$$$$      | $$$$$$$$|  $$$$$$/|  $$$$$$/| $$      
//    |__/   |______/|________/|________/      |________/ \______/  \______/ |__/      

  for (int i_tile = 0; i_tile < total_tiles; i_tile++)
  {
                                                                                        
//   /$$$$$$   /$$$$$$  /$$   /$$ /$$$$$$$$ /$$$$$$  /$$$$$$  /$$   /$$ /$$$$$$$  /$$$$$$$$
//  /$$__  $$ /$$__  $$| $$$ | $$| $$_____/|_  $$_/ /$$__  $$| $$  | $$| $$__  $$| $$_____/
// | $$  \__/| $$  \ $$| $$$$| $$| $$        | $$  | $$  \__/| $$  | $$| $$  \ $$| $$      
// | $$      | $$  | $$| $$ $$ $$| $$$$$     | $$  | $$ /$$$$| $$  | $$| $$$$$$$/| $$$$$   
// | $$      | $$  | $$| $$  $$$$| $$__/     | $$  | $$|_  $$| $$  | $$| $$__  $$| $$__/   
// | $$    $$| $$  | $$| $$\  $$$| $$        | $$  | $$  \ $$| $$  | $$| $$  \ $$| $$      
// |  $$$$$$/|  $$$$$$/| $$ \  $$| $$       /$$$$$$|  $$$$$$/|  $$$$$$/| $$  | $$| $$$$$$$$
//  \______/  \______/ |__/  \__/|__/      |______/ \______/  \______/ |__/  |__/|________/

    const int x_tile_ptr     = db[i_db_x].x;
    const int w_tile_ptr     = db[i_db_w].w;
% if FLAG_BATCHNORM == 1:
    const int scale_tile_ptr = db[i_db_w].scale;
    const int bias_tile_ptr  = db[i_db_w].bias;
% endif
    const int y_tile_ptr     = db[i_db_y].y;

    ///////////////////////
    // DMA configuration //
    ///////////////////////

    if (is_load_x) {
      x_tile_size_h = (i_h + 1 == ${tile_dim_h}) ? ${x_tile_size_h_last} : ${x_tile_size_h};
      x_tile_size_w = (i_w + 1 == ${tile_dim_w}) ? ${x_tile_size_w_last} : ${x_tile_size_w};
      x_length_nif_byte = (i_nif + 1 == ${tile_dim_nif}) ? ${x_tile_size_nif_byte_last} : ${x_tile_size_nif_byte};

      // additionally overlap by padding for the first tile after a border one
      // this because in the first tile we use less pixels from x_buffer, since we have the ones of padding
      const int pad_offset_h = i_h > 0 ? padding.top : 0;
      const int pad_offset_w = i_w > 0 ? padding.left : 0;

      DMA_copy_x.ext = dory_get_tile_3d(l2_x, i_h, i_w, i_nif, ${x_tile_size_h}, ${x_tile_size_w}, ${x_tile_size_nif}, ${x_w}, ${nif*g}, ${conv_overlap1}, ${conv_overlap2}, 0, pad_offset_h, pad_offset_w, 0, ${x_data_size_byte});
      DMA_copy_x.loc = x_tile_ptr;
      DMA_copy_x.number_of_2d_copies = x_tile_size_h;
      DMA_copy_x.number_of_1d_copies = x_tile_size_w;
      DMA_copy_x.length_1d_copy = x_length_nif_byte;
    }

    if (is_load_w) {
      W_tile_size_nof = (i_nof + 1 == ${tile_dim_nof}) ? ${W_tile_size_nof_last} : ${W_tile_size_nof};
      W_tile_size_nif = (i_nif + 1 == ${tile_dim_nif}) ? ${W_tile_size_nif_last} : ${W_tile_size_nif};

      W_tile_ko_len = (i_nof + 1 == ${tile_dim_nof}) ? ${l1_W_tile_ko_len_last} : ${l1_W_tile_ko_len};

      DMA_copy_W.ext = l2_W + ${l1_W_tile_ko_len * l1_W_tile_ki_size} * i_nof;
      DMA_copy_W.loc = w_tile_ptr;
      DMA_copy_W.length_1d_copy = W_tile_ko_len * ${l1_W_tile_ki_size};

% if FLAG_BATCHNORM == 1:
      DMA_copy_k.ext = l2_scale + ${k_tile_size_byte_transfer} * i_nof;
      DMA_copy_k.loc = scale_tile_ptr;
      DMA_copy_k.length_1d_copy = W_tile_size_nof * ${int(act_dim_bit/8)};

      DMA_copy_lambda.ext = l2_bias + ${lambda_tile_size_byte_transfer} * i_nof;
      DMA_copy_lambda.loc = bias_tile_ptr;
      DMA_copy_lambda.length_1d_copy = W_tile_size_nof * ${int(bias_bits/8)};
% endif
    }

    y_tile_size_h = (i_h + 1 == ${tile_dim_h}) ? ${y_tile_size_h_last} : ${y_tile_size_h};
    y_tile_size_w = (i_w + 1 == ${tile_dim_w}) ? ${y_tile_size_w_last} : ${y_tile_size_w};
    y_length_nof_byte = (i_nof + 1 == ${tile_dim_nof}) ? ${y_length_nof_byte_last} : ${y_tile_size_nof_byte};

    DMA_copy_y[DMA_Y_INDEX(i_tile)].ext = dory_get_tile_3d(l2_y, i_h, i_w, i_nof, ${y_tile_size_h}, ${y_tile_size_w}, ${y_tile_size_nof}, ${y_w}, ${int(nof*factor)}, 0, 0, 0, 0, 0, 0, ${y_data_size_byte});
    DMA_copy_y[DMA_Y_INDEX(i_tile)].loc = y_tile_ptr;
    DMA_copy_y[DMA_Y_INDEX(i_tile)].number_of_2d_copies = y_tile_size_h;
    DMA_copy_y[DMA_Y_INDEX(i_tile)].number_of_1d_copies = y_tile_size_w;
    DMA_copy_y[DMA_Y_INDEX(i_tile)].length_1d_copy = y_length_nof_byte;

    ////////////////////////
    // NE16 configuration //
    ////////////////////////

    int is_border_tile = 0
  % if tile_dim_nif != 1:
      || i_nif + 1 == ${tile_dim_nif}
  % endif
  % if tile_dim_h != 1:
      || i_h + 1 == ${tile_dim_h}
  % endif
  % if tile_dim_w != 1:
      || i_w + 1 == ${tile_dim_w}
  % endif
  % if tile_dim_nof != 1:
      || i_nof + 1 == ${tile_dim_nof}
  % endif
    ;

    nnx_task_to_offload = is_border_tile ? &nnx_tasks[NNX_TASK_REMAINDER] : &nnx_tasks[NNX_TASK_BODY];

    nnx_padding_t tile_padding = {
      .top    = i_h == 0 ? padding.top : DONT_PAD,
      .right  = i_w == ${tile_dim_w} - 1 ? padding.right : DONT_PAD,
      .bottom = i_h == ${tile_dim_h} - 1 ? padding.bottom : DONT_PAD,
      .left   = i_w == 0 ? padding.left : DONT_PAD,
      .value = 0
    };

    % if stride != 2:
    if (is_border_tile) {
      nnx_conv_${fs1}x${fs2}${'_dw' if flag_DW else ''}_update_dims(&(nnx_task_to_offload->cfg),
          y_tile_size_h, y_tile_size_w, x_tile_size_w, W_tile_size_nof, W_tile_size_nif,
          x_tile_size_w, y_tile_size_w, tile_padding);
    }

    nnx_pad_input(&(nnx_task_to_offload->cfg), tile_padding);

    % endif
    const x_tile_ptr_w_padding = x_tile_ptr - (tile_padding.top * x_tile_size_w + tile_padding.left) * ${x_tile_size_nif};
    % if stride != 2:
    nnx_task_to_offload->infeat_ptr = x_tile_ptr_w_padding;
    nnx_task_to_offload->outfeat_ptr = y_tile_ptr;
    % endif
    nnx_task_to_offload->weights_ptr = w_tile_ptr;
    % if FLAG_BATCHNORM == 1:
    nnx_task_to_offload->scale_ptr = scale_tile_ptr;
    nnx_task_to_offload->scale_bias_ptr = bias_tile_ptr;
    % endif


//  /$$        /$$$$$$   /$$$$$$  /$$$$$$$ 
// | $$       /$$__  $$ /$$__  $$| $$__  $$
// | $$      | $$  \ $$| $$  \ $$| $$  \ $$
// | $$      | $$  | $$| $$$$$$$$| $$  | $$
// | $$      | $$  | $$| $$__  $$| $$  | $$
// | $$      | $$  | $$| $$  | $$| $$  | $$
// | $$$$$$$$|  $$$$$$/| $$  | $$| $$$$$$$/
// |________/ \______/ |__/  |__/|_______/ 
                                        
    // Acquire implicitly acts as a barrier that waits for the
    // accelerator to not be full i.e. have less than NNX_CONTEXT_SIZE
    // jobs commited.
    // This barrier is required before dma_memcpy so that we don't
    // overwrite the data being used by the accelerator.
    % if stride != 2:
    PERF_EXEC_COMPONENT_BEGIN(acquire);
    nnx_acquire();
    PERF_EXEC_COMPONENT_END(acquire);
    % endif

    if (i_tile == 0) {
        PERF_LAYER_COMPONENT_READ(first_conf);
    }

    PERF_EXEC_COMPONENT_BEGIN(dma_memcpy);
#ifndef NO_DMA
    if (is_load_x) {
      dory_dma_memcpy_async(&DMA_copy_x);
    }
    if (is_load_w) {
      dory_dma_memcpy_async(&DMA_copy_W);
% if FLAG_BATCHNORM == 1:
      dory_dma_memcpy_async(&DMA_copy_k);
      dory_dma_memcpy_async(&DMA_copy_lambda);
% endif
    }
#endif  // NO_DMA


//   /$$$$$$  /$$$$$$$$ /$$$$$$  /$$$$$$$  /$$$$$$$$
//  /$$__  $$|__  $$__//$$__  $$| $$__  $$| $$_____/
// | $$  \__/   | $$  | $$  \ $$| $$  \ $$| $$      
// |  $$$$$$    | $$  | $$  | $$| $$$$$$$/| $$$$$   
//  \____  $$   | $$  | $$  | $$| $$__  $$| $$__/   
//  /$$  \ $$   | $$  | $$  | $$| $$  \ $$| $$      
// |  $$$$$$/   | $$  |  $$$$$$/| $$  | $$| $$$$$$$$
//  \______/    |__/   \______/ |__/  |__/|________/

    const is_store = i_tile > i_store_y + 1;

#ifndef NO_DMA
    if (is_store) {
      dory_dma_memcpy_async(&DMA_copy_y[DMA_Y_INDEX(i_store_y)]);
    }
#endif  // NO_DMA
    PERF_EXEC_COMPONENT_END(dma_memcpy);


//  /$$$$$$$$ /$$   /$$ /$$$$$$$$  /$$$$$$ 
// | $$_____/| $$  / $$| $$_____/ /$$__  $$
// | $$      |  $$/ $$/| $$      | $$  \__/
// | $$$$$    \  $$$$/ | $$$$$   | $$      
// | $$__/     >$$  $$ | $$__/   | $$      
// | $$       /$$/\  $$| $$      | $$    $$
// | $$$$$$$$| $$  \ $$| $$$$$$$$|  $$$$$$/
// |________/|__/  |__/|________/ \______/ 

    % if stride != 2:
    nnx_offload(nnx_task_to_offload);
    % endif

    PERF_EXEC_COMPONENT_BEGIN(dma_barrier);
#ifndef NO_DMA

#ifdef NO_DMA_SYNC_ON_BORDER_TILE
    if (!is_border_tile) {
#endif

#ifdef NO_DMA_SYNC
    // Wait for data to arrive
    if (is_load_x) {
      dory_dma_free(&DMA_copy_x);
    }
    if (is_load_w) {
      dory_dma_free(&DMA_copy_W);
      dory_dma_free(&DMA_copy_k);
      dory_dma_free(&DMA_copy_lambda);
    }

    // Wait to write the data so that not to overwrite it by the next job
    if (is_store) {
      dory_dma_free(&DMA_copy_y[DMA_Y_INDEX(i_store_y)]);
    }
#else
    // Wait for data to arrive
    if (is_load_x) {
      dory_dma_barrier(&DMA_copy_x);
    }
    if (is_load_w) {
      dory_dma_barrier(&DMA_copy_W);
      dory_dma_barrier(&DMA_copy_k);
      dory_dma_barrier(&DMA_copy_lambda);
    }

    // Wait to write the data so that not to overwrite it by the next job
    if (is_store) {
      dory_dma_barrier(&DMA_copy_y[DMA_Y_INDEX(i_store_y)]);
    }
#endif  // NO_DMA_SYNC

#ifdef NO_DMA_SYNC_ON_BORDER_TILE
    } else {
      if (is_load_x) {
        dory_dma_free(&DMA_copy_x);
      }
      if (is_load_w) {
        dory_dma_free(&DMA_copy_W);
        dory_dma_free(&DMA_copy_k);
        dory_dma_free(&DMA_copy_lambda);
      }

      // Wait to write the data so that not to overwrite it by the next job
      if (is_store) {
        dory_dma_free(&DMA_copy_y[DMA_Y_INDEX(i_store_y)]);
      }
    }
#endif  // NO_DMA_SYNC_ON_BORDER_TILE

#endif  // NO_DMA
    PERF_EXEC_COMPONENT_END(dma_barrier);

    if (i_tile == 0) {
        PERF_LAYER_COMPONENT_READ(first_dma);
    }

    % if stride != 2:
    nnx_run_async();
    % else:
    const n_h = DIVNCEIL(y_tile_size_h, 2);
    const n_w = DIVNCEIL(y_tile_size_w, 2);
    const is_odd_h = y_tile_size_h % 2;
    const is_odd_w = y_tile_size_w % 2;
    //printf("[%d] is_odd_h:%s, is_odd_w:%s\n", i_tile, is_odd_h ? "True" : "False", is_odd_w ? "True" : "False");
    //printf("[%d] n_h:%d n_w:%d\n", i_tile, n_h, n_w);

    int x_tile_size_h_to_process = x_tile_size_h + tile_padding.top;
    for (int i = 0; i < n_h; i++) {
      int x_tile_size_w_to_process = x_tile_size_w + tile_padding.left;
      for (int j = 0; j < n_w; j++) {
        const y_extra_h = x_tile_size_h_to_process < 5 ? x_tile_size_h_to_process + tile_padding.bottom - 3 : 0;
        const y_extra_w = x_tile_size_w_to_process < 5 ? x_tile_size_w_to_process + tile_padding.right - 3 : 0;
        const y_subtile_size_h = is_odd_h && i == n_h - 1 ? 1 + y_extra_h : 3;
        const y_subtile_size_w = is_odd_w && j == n_w - 1 ? 1 + y_extra_w : 3;
        const x_subtile_size_h = x_tile_size_h_to_process < 5 ? x_tile_size_h_to_process : 5;
        const x_subtile_size_w = x_tile_size_w_to_process < 5 ? x_tile_size_w_to_process : 5;

        nnx_padding_t subtile_padding = {
          .top    = i == 0       ? tile_padding.top    : DONT_PAD,
          .right  = j == n_w - 1 && x_tile_size_w_to_process < 5 ? tile_padding.right  : DONT_PAD,
          .bottom = i == n_h - 1 && x_tile_size_h_to_process < 5 ? tile_padding.bottom : DONT_PAD,
          .left   = j == 0       ? tile_padding.left   : DONT_PAD,
          .value = 0
        };

        nnx_pad_input(&(nnx_task_to_offload->cfg), subtile_padding);
        nnx_conv_${fs1}x${fs2}${'_dw' if flag_DW else ''}_update_dims(&(nnx_task_to_offload->cfg),
            y_subtile_size_h, y_subtile_size_w, x_subtile_size_w, W_tile_size_nof, W_tile_size_nif,
            x_tile_size_w, y_tile_size_w, subtile_padding);

        nnx_task_to_offload->infeat_ptr =
          dory_get_tile_3d(x_tile_ptr_w_padding,
                           i, j, 0, /* iterator */
                           5, 5, ${x_tile_size_nif}, /* size */
                           x_tile_size_w, ${x_tile_size_nif}, /* stride */
                           1, 1, 0, /* overlap */
                           i > 0 ? subtile_padding.top : 0, j > 0 ? subtile_padding.left : 0, 0, /* offset */
                           ${x_data_size_byte} /* data size */ );
        nnx_task_to_offload->outfeat_ptr =
          dory_get_tile_3d(y_tile_ptr,
                           i, j, 0, /* iterator */
                           2, 2, ${y_tile_size_nof}, /* size */
                           y_tile_size_w, ${y_tile_size_nof}, /* stride */
                           0, 0, 0, /* overlap */
                           0, 0, 0, /* offset */
                           ${y_data_size_byte} /* data size */ );

        asm volatile("": : :"memory");
        nnx_acquire();
        //printf("[loop %d, %d] x_ptr:%p, y_ptr:%p\n", i, j,
        //       nnx_task_to_offload->infeat_ptr, nnx_task_to_offload->outfeat_ptr);
        nnx_offload(nnx_task_to_offload);
        nnx_run_async();

        x_tile_size_w_to_process -= 4;
      }
      x_tile_size_h_to_process -= 4;
    }
    % endif


//  /$$   /$$ /$$$$$$$  /$$$$$$$   /$$$$$$  /$$$$$$$$ /$$$$$$$$       /$$$$$$ /$$   /$$ /$$$$$$$  /$$$$$$  /$$$$$$  /$$$$$$$$  /$$$$$$ 
// | $$  | $$| $$__  $$| $$__  $$ /$$__  $$|__  $$__/| $$_____/      |_  $$_/| $$$ | $$| $$__  $$|_  $$_/ /$$__  $$| $$_____/ /$$__  $$
// | $$  | $$| $$  \ $$| $$  \ $$| $$  \ $$   | $$   | $$              | $$  | $$$$| $$| $$  \ $$  | $$  | $$  \__/| $$      | $$  \__/
// | $$  | $$| $$$$$$$/| $$  | $$| $$$$$$$$   | $$   | $$$$$           | $$  | $$ $$ $$| $$  | $$  | $$  | $$      | $$$$$   |  $$$$$$ 
// | $$  | $$| $$____/ | $$  | $$| $$__  $$   | $$   | $$__/           | $$  | $$  $$$$| $$  | $$  | $$  | $$      | $$__/    \____  $$
// | $$  | $$| $$      | $$  | $$| $$  | $$   | $$   | $$              | $$  | $$\  $$$| $$  | $$  | $$  | $$    $$| $$       /$$  \ $$
// |  $$$$$$/| $$      | $$$$$$$/| $$  | $$   | $$   | $$$$$$$$       /$$$$$$| $$ \  $$| $$$$$$$/ /$$$$$$|  $$$$$$/| $$$$$$$$|  $$$$$$/
//  \______/ |__/      |_______/ |__/  |__/   |__/   |________/      |______/|__/  \__/|_______/ |______/ \______/ |________/ \______/ 

    /////////////////////////
    // Update tile indices //
    /////////////////////////

    % if tile_dim_nif != 1:
    const int i_nif_prev = i_nif;
    % endif
    % if tile_dim_w != 1:
    const int i_w_prev = i_w;
    % endif
    % if tile_dim_h != 1:
    const int i_h_prev = i_h;
    % endif
    % if tile_dim_nof != 1:
    const int i_nof_prev = i_nof;
    % endif

    % if tile_dim_nif != 1 and flag_DW == 0:
    // loop nest is nof,h,w,nif
    i_nif += 1;
    if(i_nif==${tile_dim_nif}) {
      i_nif = 0;
    % endif
      % if tile_dim_w != 1:
      i_w += 1;
      if(i_w==${tile_dim_w}) {
        i_w = 0;
      % endif
        % if tile_dim_h != 1:
        i_h += 1;
        if(i_h==${tile_dim_h}) {
          i_h = 0;
        % endif
          % if flag_DW == 1:
          i_nif += 1;
          % endif
          % if tile_dim_nof != 1:
          i_nof += 1;
          % endif
        % if tile_dim_h != 1:
        }
        % endif
      % if tile_dim_w != 1:
      }
      % endif
    % if tile_dim_nif != 1 and flag_DW == 0:
    }
    % endif

    ///////////////////////
    // Update load flags //
    ///////////////////////

    is_load_w = 0 \
% if tile_dim_nif != 1:
|| i_nif_prev != i_nif \
% endif
% if tile_dim_nof != 1:
|| i_nof_prev != i_nof \
% endif
;

    is_load_x = 0 \
% if tile_dim_nif != 1:
|| i_nif_prev != i_nif \
% endif
% if tile_dim_h != 1:
|| i_h_prev != i_h \
% endif
% if tile_dim_w != 1:
|| i_w_prev != i_w \
% endif
;

    ///////////////////////////
    // Update store iterator //
    ///////////////////////////
    if (is_store) {
      i_store_y += 1;
    }

    ///////////////////////////////////
    // Update double buffer pointers //
    ///////////////////////////////////

    if (is_load_x) i_db_x = !i_db_x;
    if (is_load_w) i_db_w = !i_db_w;
    i_db_y = !i_db_y;
  }


//   /$$$$$$  /$$$$$$$$ /$$$$$$  /$$$$$$$  /$$$$$$$$       /$$$$$$$  /$$$$$$$$ /$$      /$$
//  /$$__  $$|__  $$__//$$__  $$| $$__  $$| $$_____/      | $$__  $$| $$_____/| $$$    /$$$
// | $$  \__/   | $$  | $$  \ $$| $$  \ $$| $$            | $$  \ $$| $$      | $$$$  /$$$$
// |  $$$$$$    | $$  | $$  | $$| $$$$$$$/| $$$$$         | $$$$$$$/| $$$$$   | $$ $$/$$ $$
//  \____  $$   | $$  | $$  | $$| $$__  $$| $$__/         | $$__  $$| $$__/   | $$  $$$| $$
//  /$$  \ $$   | $$  | $$  | $$| $$  \ $$| $$            | $$  \ $$| $$      | $$\  $ | $$
// |  $$$$$$/   | $$  |  $$$$$$/| $$  | $$| $$$$$$$$      | $$  | $$| $$$$$$$$| $$ \/  | $$
//  \______/    |__/   \______/ |__/  |__/|________/      |__/  |__/|________/|__/     |__/

  for (int i = i_store_y; i < total_tiles; i++) {
    if (i < total_tiles - 1) {
      nnx_wait_not_full();
    } else {
      nnx_wait_empty();
    }

#ifndef NO_DMA
    dory_dma_memcpy_async(&DMA_copy_y[DMA_Y_INDEX(i)]);
#endif  // NO_DMA
  }

  PERF_LAYER_COMPONENT_READ(nnx);

#ifndef NO_DMA
  for (int i = i_store_y; i < total_tiles; i++) {
#ifdef NO_DMA_SYNC
    dory_dma_free(&DMA_copy_y[DMA_Y_INDEX(i)]);
#else
    dory_dma_barrier(&DMA_copy_y[DMA_Y_INDEX(i)]);
#endif  // NO_DMA_SYNC
  }
#endif  // NO_DMA

/*
  uint8_t *d = (uint8_t *)db[0].y;
  const h = ${y_tile_size_h};
  const w = ${y_tile_size_w};
  const c = ${y_tile_size_nof};

  printf("\nOutput:\n");
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      for (int k = 0; k < c; k++) {
        printf(" %3d", d[(i * w + j) * c + k]);
      }
      printf(";");
    }
    printf("\n");
  }
  */
% if not TEST:
  // wait for final write
% endif

  // clear NNX for cleanup
  nnx_term();

  PERF_LAYER_COMPONENT_READ(postamble);

  PERF_LAYER_COMPONENT_REPORT();
  PERF_EXEC_COMPONENT_REPORT();
}
