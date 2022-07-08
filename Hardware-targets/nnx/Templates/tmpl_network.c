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
#define DEFINE_CONSTANTS
#include "network.h"
#include "dory.h"
% for layer in list_h:
#include "${layer}"
% endfor

% if sdk == 'pulp-sdk':
#define ICACHE_CTRL_UNIT 0x10201400
#define ICACHE_PREFETCH ICACHE_CTRL_UNIT + 0x1C
% endif
#define FLASH_BUFF_SIZE 128
% if verbose:
#define VERBOSE 1
% endif

static uint8_t flashBuffer[FLASH_BUFF_SIZE];
int memId;
char* L2_output;
char* L2_input;
char* L2_weights;
char* l1_buffer;
char* bypass_activations;

% if 'Check_all' in verbose_level:
#ifdef VERBOSE
static void checksum_last(${DORY_HW_graph[-1].output_activation_type}${DORY_HW_graph[-1].output_activation_bits}_t *d, int size, int sum_true) {
  int sum = 0;
  for (int i = 0; i < size / ${DORY_HW_graph[-1].output_activation_bits // 8}; i++) sum += d[i];

  printf("Checking final layer: Checksum ");
  if(sum_true == sum)
    printf("OK\n");
  else
    printf("Failed: true [%d] vs. calculated [%d]\n", sum_true, sum);
}

static void checksum(char *name, char *d, int size, int sum_true) {
    int sum = 0;
    for (int i = 0; i < size; i++) sum += d[i];

    printf("Checking %s: Checksum ", name);
    if (sum_true == sum)
        printf("OK\n");
    else
        printf("Failed: true [%d] vs. calculated [%d]\n", sum_true, sum);
}
#endif
% endif


/* Moves the weights and the biases from hyperflash to hyperram */
void network_initialize(struct pi_device fs, struct pi_device ram)
{
  pi_fs_file_t *file;
  pi_ram_alloc(&ram, &L3_weights, (uint32_t) 4000000);
  pi_ram_alloc(&ram, &L3_input, (uint32_t) 1500000);
  pi_ram_alloc(&ram, &L3_output, (uint32_t) 1500000);
#ifdef VERBOSE
  printf("\nL3 Buffer alloc initial\t@ %d:\t%s\n", (unsigned int)L3_weights, L3_weights?"Ok":"Failed");
  printf("\nL3 Buffer alloc initial\t@ %d:\t%s\n", (unsigned int)L3_input, L3_input?"Ok":"Failed");
  printf("\nL3 Buffer alloc initial\t@ %d:\t%s\n", (unsigned int)L3_output, L3_output?"Ok":"Failed");
#endif
  unsigned int rdDone = 0;
% if 'Check_all' in verbose_level:
  int layer_number = 0;
  int sum_weights;
% endif
  for (int i=0;i<${weights_number};i++)
  {
% if 'Check_all' in verbose_level:
    if (layer_with_weights[layer_number]==0)
      layer_number +=1;
% endif
    file = pi_fs_open(&fs, L3_weights_files[i], 0);
    if (file == NULL)
    {
      printf("file %s open failed \n", L3_weights_files[i]);
      return -1;
    }
    L3_weights_size[i] = file->size + rdDone;
    int flashBuffSize = FLASH_BUFF_SIZE * sizeof(char);
% if 'Check_all' in verbose_level:
    sum_weights = 0;
% endif
    while(rdDone < (L3_weights_size[i] / sizeof(char)))
    {
      int size = pi_fs_read(file, flashBuffer, flashBuffSize);
% if 'Check_all' in verbose_level:
      for (int t = 0; t < size; t++)
        sum_weights+=flashBuffer[t];
% endif
      pi_ram_write(&ram, L3_weights+rdDone, flashBuffer,size);
      rdDone += size / sizeof(char);
    }
% if 'Check_all' in verbose_level:
    if (weights_checksum[layer_number] == sum_weights)
      printf("Layer %-3d: Checksum = %-12d, FLASH %-12d, Check OK\n", layer_number, weights_checksum[layer_number], sum_weights);
    else
      printf("Layer %-3d: Checksum = %-12d, FLASH %-12d, Check FAILED\n", layer_number, weights_checksum[layer_number], sum_weights);
    layer_number +=1;
% endif
  }
  return 1;
}

/* Remove RAM memory */
void network_terminate(struct pi_device ram)
{
  pi_ram_free(&ram, L3_weights, (uint32_t) 4000000);
  pi_ram_free(&ram, L3_input, (uint32_t) 1500000);
  pi_ram_free(&ram, L3_output, (uint32_t) 1500000);
}

void execute_layer_fork(void *arg)
{
  unsigned int *real_arg = (unsigned int *)arg;
  if (pi_core_id() == 0) // TODO: Why is only core 0 freeing when all the cores have allocated?
      real_arg[7] = pmsis_l1_malloc((uint32_t) ${l1_buffer}); // TODO: What is this l1_buffer?
  switch (real_arg[11])
  {
% for i in range(len(DORY_HW_graph)):
    case ${i}:
      pi_cl_team_fork(NUM_CORES, (void *)${func_name[i]}, arg); // TODO Can't we just pass the pointer to the function?
      break;
% endfor
  }
  if (pi_core_id() == 0) // TODO: Why is only core 0 freeing when all the cores have allocated?
    pmsis_l1_malloc_free(real_arg[7], (uint32_t) ${l1_buffer});
}

void network_run(char *L2_memory_buffer, int L2_memory_dimension, char *L2_output_to_pass, struct pi_device ram)
{

/*
  - initial buffer allocation L2 and L1
  - variable declaration
*/
/* ---------------------------------- */
/* -------- SECTION 0 BEGIN --------- */
/* ---------------------------------- */
  bypass_activations = 0;
  int residual_number = 0;
  int perf_cyc = 0;
  int dir = 1;  // direction of the L2 memory allocation: 1 - begin -> end, 0 - end -> begin
  char *L3_weights_internal = L3_weights;
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

  dory_L2_mem_init((void *)L2_memory_buffer, L2_memory_dimension);

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
  for (int i = 0; i < ${len(DORY_HW_graph)}; i++)
  {
/* MEMORY ALLOCATION
  - allocate memory if layer is executed from L3;
  - allocate weights
  - read weights
*/
    L2_output = dory_L2_alloc(activations_out_size[i], !dir);

    if (L3_input_layers[i] == 1)
      L2_input = dory_L2_alloc(activations_size[i], dir);

    if (layer_with_weights[i] == 1)
      L2_weights = dory_L2_alloc(weights_size[i], dir);

    if (allocate_layer[i] == 1)
      pi_ram_read(&ram, L3_weights_internal, L2_weights, weights_size[i]);

% if 'Check_all' in verbose_level:
#ifdef VERBOSE
    if (L3_input_layers[i] == 1)
      printf("Input in L3\n");
    else if (i == 0 || branch_change[i-1] == 0) {
      printf("Checking input of layer %d...\n", i);
      checksum("L2 input", L2_input, activations_size[i], activations_checksum[i]);
      if (allocate_layer[i] == 1)
        checksum("L2 weights", L2_weights, weights_size[i], weights_checksum[i]);
      else
        printf("Weights in L3\n");
    }
    else
      printf("Switching branch, already checked activation\n");
#endif
% endif

    layer_args_t args = {
      .L3_input = L3_input,
      .L3_output = L3_output,
      .L3_after_weights = L3_weights_internal,
      .L2_input = L2_input,
      .bypass = bypass_activations,
      .L2_output = L2_output,
      .L2_weights = L2_weights,
      .L1_buffer = l1_buffer,
      .ram = &ram,
      .out_mult = out_mult_vector[i],
      .out_shift = out_shift_vector[i],
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
      return -1;
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
    int MACs = NODEs_MACS[i];
    float perf_MAC =  (float)MACs/perf_cyc;
    printf(" Layer %-3d: num_cycles: %-11d,", i, perf_cyc);
    printf(" MACs: %-11d,",MACs );
    printf(" MAC/cycle: %-8f,",perf_MAC );
    printf(" n. of Cores: %d\n",NUM_CORES);
% endif

    // TODO: What error?
    // prevents error from compiler
    asm volatile("": : :"memory");
    unsigned int temp = L3_input;
    L3_input = L3_output;
    asm volatile("": : :"memory");
    L3_output = temp;
    asm volatile("": : :"memory");

% if 'Check_all' in verbose_level:
#ifdef VERBOSE
    printf("Layer %s %d ended: \n", Layers_name[i], i);
    if (i < ${len(DORY_HW_graph) - 1}) {
      if (L3_output_layers[i]==1) {
        printf("Out in L3\n");
      } else {
        checksum("L2 output", L2_output, activations_out_size[i], activations_out_checksum[i]);
      }
      printf("\n");
    } else {
      checksum_last((${DORY_HW_graph[-1].output_activation_type}${DORY_HW_graph[-1].output_activation_bits}_t *)L2_output, activations_out_size[i], activations_out_checksum[i]);
    }
#endif
% elif 'Last' in verbose_level:
    if (i == ${len(DORY_HW_graph) - 1})
        checksum_last((${DORY_HW_graph[-1].output_activation_type}${DORY_HW_graph[-1].output_activation_bits}_t *)L2_output, activations_out_size[i], activations_out_checksum[i]);
% else:
#ifdef VERBOSE
    printf("Layer %s %d ended: \n", Layers_name[i], i);
#endif
% endif

    // Free memory
    if (layer_with_weights[i] == 1)
      dory_L2_free(weights_size[i], dir);
    dory_L2_free(activations_size[i], dir);
    if (branch_input[i] == 1)
      dory_L2_free(activations_size[i], dir);

    // Residual connections
    if (i < ${len(DORY_HW_graph) - 1}) {
      if (branch_input[i+1] == 1) {
        bypass_activations = dory_L2_alloc(activations_out_size[i], !dir);
        residual_number--;
        pi_ram_read(&ram, layers_pointers[residual_number], bypass_activations, activations_out_size[i]);
        pi_ram_free(&ram, layers_pointers[residual_number], activations_out_size[i]);
      }

      // TODO I feel like this should look ahead instead of back
      if (i > 0 && branch_output[i-1]==1 && L3_input_layers[i]==1) { // TODO don't understand this condition
        pi_ram_alloc(&ram, &L3_input, (uint32_t) 1500000);
      }

      if (branch_output[i]==1 && L3_output_layers[i]==1) {
        pi_ram_free(&ram, L3_input + activations_out_size[i], 1500000 - activations_out_size[i]);
        layers_pointers[residual_number] = L3_input;
        residual_number++;
      } else if (branch_output[i]==1 || branch_change[i] == 1) {
        pi_ram_alloc(&ram, &layers_pointers[residual_number], activations_out_size[i]);
        pi_ram_write(&ram, layers_pointers[residual_number], L2_output, activations_out_size[i]);
        residual_number++;
      }

      if (branch_change[i]==1) {
        dory_L2_free(activations_out_size[i], !dir);
        L2_input = dory_L2_alloc(activations_size[i + 1], !dir);
        pi_ram_read(&ram, layers_pointers[residual_number - 2], L2_input, activations_size[i + 1]);
        pi_ram_free(&ram, layers_pointers[residual_number - 2], activations_out_size[i + 1]);
      }

      if (L3_output_layers[i] == 1)
        dory_L2_free(activations_out_size[i], !dir);
    }

    L3_weights_internal += weights_size[i];
    L2_input = L2_output;
    dir = !dir;
  }

  // memcpy(L2_output, L2_output_to_pass, activations_out_size[${len(DORY_HW_graph)}])
  for (int i = 0; i < activations_out_size[${len(DORY_HW_graph)}]; i++) {
    *(L2_output_to_pass + i) = *(L2_output + i);
  }
/* ---------------------------------- */
/* --------- SECTION 2 END ---------- */
/* ---------------------------------- */

/* ---------------------------------- */
/* -------- SECTION 3 BEGIN --------- */
/* ---------------------------------- */

% if 'Perf_final' in verbose_level:
  int MACs = ${MACs};
  float perf_MAC =  (float)MACs/cycle_network_execution;
  printf("\nnum_cycles: %d\n",cycle_network_execution);
  printf("MACs: %d\n",MACs );
  printf("MAC/cycle: %f\n",perf_MAC );
  printf("n. of Cores: %d\n",NUM_CORES);
% endif

/* ---------------------------------- */
/* --------- SECTION 3 END ---------- */
/* ---------------------------------- */
}

