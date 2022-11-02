/*
 * network.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 * Thorir Mar Ingolfsson <thoriri@iis.ee.ethz.ch>
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
<%
l3_supported = DORY_HW_graph[0].HW_description['memory']['levels'] > 2
%>\
#define DEFINE_CONSTANTS
%if not l3_supported:
#include "weights.h"
%endif
#include "pmsis.h"
#include "network.h"
#include "directional_allocator.h"
#include "mem.h"
#include <string.h>
% for layer in list_h:
#include "${layer}"
% endfor

% if sdk == 'pulp-sdk':
#define ICACHE_CTRL_UNIT 0x10201400
#define ICACHE_PREFETCH ICACHE_CTRL_UNIT + 0x1C
% endif

% if verbose:
#define VERBOSE 1
% endif

% if 'Yes' in performance or 'Perf_final' in verbose_level:
static void print_perf(const char *name, const int cycles, const int macs) {
  float perf = (float) macs / cycles;
  printf("\n%s performance:\n", name);
  printf("  - num cycles: %d\n", cycles);
  printf("  - MACs: %d\n", macs );
  printf("  - MAC/cycle: %g\n", perf);
  printf("  - n. of Cores: %d\n\n", NUM_CORES);
}

% endif
% if 'Check_all' in verbose_level:
#ifdef VERBOSE
static void checksum(const char *name, const uint8_t *d, size_t size, uint32_t sum_true) {
    uint32_t sum = 0;
    for (int i = 0; i < size; i++) sum += d[i];

    printf("Checking %s: Checksum ", name);
    if (sum_true == sum)
        printf("OK\n");
    else
        printf("Failed: true [%u] vs. calculated [%u]\n", sum_true, sum);
}
#endif
% endif

% if l3_supported:
#define L3_WEIGHTS_SIZE 4000000
#define L3_INPUT_SIZE 1500000
#define L3_OUTPUT_SIZE 1500000
% endif
static void *L3_weights = NULL;
static void *L3_input = NULL;
static void *L3_output = NULL;

% if l3_supported:
/* Moves the weights and the biases from hyperflash to hyperram */
void network_initialize() {

  L3_weights = ram_malloc(L3_WEIGHTS_SIZE);
  L3_input = ram_malloc(L3_INPUT_SIZE);
  L3_output = ram_malloc(L3_OUTPUT_SIZE);

#ifdef VERBOSE
  printf("\nL3 Buffer alloc initial\t@ %d:\t%s\n", (unsigned int)L3_weights, L3_weights?"Ok":"Failed");
  printf("\nL3 Buffer alloc initial\t@ %d:\t%s\n", (unsigned int)L3_input, L3_input?"Ok":"Failed");
  printf("\nL3 Buffer alloc initial\t@ %d:\t%s\n", (unsigned int)L3_output, L3_output?"Ok":"Failed");
#endif

  void *w_ptr = L3_weights;
  for (int i = 0; i < ${weights_number}; i++) {
    size_t size = load_file_to_ram(w_ptr, L3_weights_files[i]);
    L3_weights_size[i] = size;
    w_ptr += size;
  }
}
% endif

% if l3_supported:
/* Remove RAM memory */
void network_terminate() {
  % if l3_supported:
  ram_free(L3_weights, L3_WEIGHTS_SIZE);
  ram_free(L3_input, L3_INPUT_SIZE);
  ram_free(L3_output, L3_OUTPUT_SIZE);
  % endif
}
% endif

void execute_layer_fork(void *args) {
  layer_args_t *layer_args = (layer_args_t *)args;
  if (pi_core_id() == 0) layer_args->L1_buffer = pmsis_l1_malloc(${l1_buffer});

  switch (layer_args->layer_id)
  {
% for i in range(len(DORY_HW_graph)):
    case ${i}:
      pi_cl_team_fork(NUM_CORES, (void *)${func_name[i]}, args);
      break;
% endfor
  }

  if (pi_core_id() == 0) pmsis_l1_malloc_free(layer_args->L1_buffer, ${l1_buffer});
}

void network_run(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, int exec${", void *L2_input_h" if not l3_supported else ""})
{
/*
  - initial buffer allocation L2 and L1
  - variable declaration
*/
/* ---------------------------------- */
/* -------- SECTION 0 BEGIN --------- */
/* ---------------------------------- */
  void *L2_output = NULL;
  void *L2_input = NULL;
  void *L2_weights = NULL;
  void *L3_weights_curr = L3_weights;
  void *bypass_activations = NULL;

  int dir = 1;
  int residual_number = 0;
  % if not l3_supported:
  int bypass_dimension = 0;
  int left_branch_nodes = 0, right_branch_nodes = 0;
  int z = 0;
  int end_left = 0;
  % endif
  int perf_cyc = 0;
  struct pi_device cluster_dev = {0};
  struct pi_cluster_conf conf;
  struct pi_cluster_task cluster_task = {0};
  // First open the cluster
  pi_cluster_conf_init(&conf);
  conf.id=0;

/* ---------------------------------- */
/* --------- SECTION 0 END ---------- */
/* ---------------------------------- */

/*
  - initial copies from L3 of input
  - copies of weights of first 2 layers
*/
/* ---------------------------------- */
/* -------- SECTION 1 BEGIN --------- */
/* ---------------------------------- */
  % if not l3_supported:
  L2_input = L2_input_h;
% endif
  directional_allocator_init(l2_buffer, l2_buffer_size);

/* ---------------------------------- */
/* --------- SECTION 1 END ---------- */
/* ---------------------------------- */
% if 'Yes' in performance or 'Perf_final' in verbose_level:
  // perf measurement begin
  int cycle_network_execution = 0;
% endif
/* MAIN SECTION
  - for loop over all the layers of the network
  - double buffering using L3
  - check on layers to be executed from L3
  - residual check at the end of each layer
*/
/* ---------------------------------- */
/* -------- SECTION 2 BEGIN --------- */
/* ---------------------------------- */
  int weight_l_cnt = 0; // count how many layers with weights we have processed to increment the weights_L3 pointer
  for (int i = 0; i < ${len(DORY_HW_graph)}; i++) {
/* MEMORY ALLOCATION
  - allocate memory if layer is executed from L3;
  - allocate weights
  - read weights
*/
    L2_output = dmalloc(activations_out_size[i], !dir);
    % if l3_supported:
    if (L3_input_layers[i] == 1)
      L2_input = dmalloc(activations_size[i], dir);

    if (layer_with_weights[i] == 1)
      L2_weights = dmalloc(weights_size[i], dir);

    if (allocate_layer[i] == 1)
      ram_read(L2_weights, L3_weights_curr, weights_size[i]);
    % else:
    L2_weights = Weights_name[i];
% endif

% if 'Check_all' in verbose_level:
#ifdef VERBOSE
        % if l3_supported:
    if (L3_input_layers[i] == 1)
      printf("Input in L3\n");
    else
% endif
    if (i == 0 || branch_change[i-1] == 0) {
      checksum("L2 input", L2_input, activations_size[i], activations_checksum[i][exec]);
% if l3_supported:
      if (allocate_layer[i] == 1)
% else:
      if (layer_with_weights[i])
% endif
        checksum("L2 weights", L2_weights, weights_size[i], weights_checksum[i]);
      else
        printf("Weights in L3\n");
    }
    else
      printf("Switching branch, already checked activation\n");
#endif
% endif

    layer_args_t args = {
      .L3_input = (unsigned int) L3_input,
      .L3_output = (unsigned int) L3_output,
      .L3_after_weights = (unsigned int) L3_weights_curr,
      .L2_input = (unsigned int) L2_input,
      .bypass = (unsigned int) bypass_activations,
      .L2_output = (unsigned int) L2_output,
      .L2_weights = (unsigned int) L2_weights,
      .L1_buffer = 0,
      .ram = (unsigned int) get_ram_ptr(),
      .out_mult = (unsigned int) out_mult_vector[i],
      .out_shift = (unsigned int) out_shift_vector[i],
      .layer_id = i
    };

/*
- Execution of the layers_pointers
*/
% if 'Yes' in performance or 'Perf_final' in verbose_level:
    // perf measurement begin
    pi_perf_conf(1<<PI_PERF_CYCLES);
    pi_perf_reset();
    pi_perf_stop();
    pi_perf_start();
% endif
    pi_cluster_task(&cluster_task, execute_layer_fork, &args);
    pi_open_from_conf(&cluster_dev, &conf);
    if (pi_cluster_open(&cluster_dev))
      return;
    // Then offload an entry point, this will get executed on the cluster controller
    cluster_task.stack_size = ${master_stack};
    cluster_task.slave_stack_size = ${slave_stack};
    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
    // closing of the cluster
    pi_cluster_close(&cluster_dev);
% if 'Yes' in performance or 'Perf_final' in verbose_level:
    // performance measurements: end
    pi_perf_stop();
    perf_cyc =  pi_perf_read(PI_PERF_CYCLES);
    cycle_network_execution += perf_cyc;
% endif

% if 'Yes' in performance:
    print_perf(Layers_name[i], perf_cyc, NODEs_MACS[i]);
% endif

    // TODO: What error?
    // prevents error from compiler
    asm volatile("": : :"memory");
    unsigned int temp = L3_input;
    L3_input = L3_output;
    asm volatile("": : :"memory");
    L3_output = temp;
    asm volatile("": : :"memory");

#ifdef VERBOSE
    printf("Layer %s %d ended: \n", Layers_name[i], i);
% if 'Check_all' in verbose_level:
    % if l3_supported:
    if (L3_output_layers[i]==1) {
      printf("Output in L3. Expected checksum: %d\n", activations_out_checksum[i][exec]);
    } else {
% endif
      checksum(i + 1 < ${len(DORY_HW_graph)} ? "L2 output" : "final output",
               L2_output, activations_out_size[i], activations_out_checksum[i][exec]);
      % if l3_supported:
    }
% endif
    printf("\n");
% elif 'Last' in verbose_level:
    if (i == ${len(DORY_HW_graph) - 1})
        checksum("final layer", L2_output, activations_out_size[i], activations_out_checksum[i][exec]);
% endif
#endif

    // Free memory
    % if l3_supported:
    if (layer_with_weights[i] == 1)
      dfree(weights_size[i], dir);
    dfree(activations_size[i], dir);
    if (branch_input[i] == 1)
      dfree(activations_size[i], dir);
% endif
    L2_input = L2_output;
% if not l3_supported:
    if  (branch_output[i]==1)
      {
        bypass_activations = L2_output;
        bypass_dimension = activations_out_size[i];
      }

    if (i > 0 && branch_output[i-1]==0)
      dfree(activations_size[i], dir);
% endif
    // Residual connections
    if (i < ${len(DORY_HW_graph) - 1}) {
 % if l3_supported:
      if (branch_input[i+1] == 1) {
        bypass_activations = dmalloc(activations_out_size[i], !dir);
        residual_number--;
        ram_read(bypass_activations, layers_pointers[residual_number], activations_out_size[i]);
        ram_free(layers_pointers[residual_number], activations_out_size[i]);
      }

      // TODO I feel like this should look ahead instead of back
      if (i > 0 && branch_output[i-1]==1 && L3_input_layers[i]==1) { // TODO don't understand this condition
        L3_input = ram_malloc(1500000);
      }
      if (branch_output[i]==1 && L3_output_layers[i]==1) {
        ram_free(L3_input + activations_out_size[i], 1500000 - activations_out_size[i]);
        layers_pointers[residual_number] = L3_input;
        residual_number++;
      } else
    if (branch_output[i]==1 || branch_change[i] == 1) {
        layers_pointers[residual_number] = ram_malloc(activations_out_size[i]);
        ram_write(layers_pointers[residual_number], L2_output, activations_out_size[i]);
        residual_number++;
    }

      if (branch_change[i]==1) {
        dfree(activations_out_size[i], !dir);
        L2_input = dmalloc(activations_size[i + 1], !dir);
        ram_read(L2_input, layers_pointers[residual_number - 2], activations_size[i + 1]);
        ram_free(layers_pointers[residual_number - 2], activations_out_size[i + 1]);
      }
      if (L3_output_layers[i] == 1)
        dfree(activations_out_size[i], !dir);
 % else:

      if  (branch_output[i]==1)
      {
        left_branch_nodes = 0;
        right_branch_nodes = 0;
        z = i+1;
        end_left = 0;
        while (branch_input[z] == 0)
        {
          if (end_left == 0)
            left_branch_nodes+=1;
          else
            right_branch_nodes+=1;
          if (branch_change[z] == 1)
            end_left = 1;
          z+=1;
        }
        if ((left_branch_nodes % 2 == 1) && (right_branch_nodes == 0))
          dir = !dir;
        if ((left_branch_nodes % 2 == 0) && (right_branch_nodes > 0))
          dir = !dir;
      }

      if  (branch_change[i]==1)
      {
        L2_input = bypass_activations;
        bypass_activations = L2_output;
        bypass_dimension = activations_out_size[i];
        if (right_branch_nodes % 2 == 1)
          dir = !dir;
      }
% endif
    }
% if l3_supported:
    if (layer_with_weights[i])
       L3_weights_curr += L3_weights_size[weight_l_cnt++];
% endif
    dir = !dir;
  }

  memcpy(L2_output, l2_final_output, activations_out_size[${len(DORY_HW_graph)-1}]);

/* ---------------------------------- */
/* --------- SECTION 2 END ---------- */
/* ---------------------------------- */

/* ---------------------------------- */
/* -------- SECTION 3 BEGIN --------- */
/* ---------------------------------- */

% if 'Perf_final' in verbose_level:
  print_perf("Final", cycle_network_execution, ${MACs});
% endif

/* ---------------------------------- */
/* --------- SECTION 3 END ---------- */
/* ---------------------------------- */
}
