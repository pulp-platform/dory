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
#include "weights.h"
#include "network.h"

% if verbose:
#define VERBOSE 1
% endif

int memId;
char* L2_output;
char* L2_input;
char* L2_weights;
char* l1_buffer;
char* bypass_activations;
int L3_weights_internal;

% if verbose_level == 'Check_all+Perf_final':
#ifdef VERBOSE
// check for input/output acitvation checksum
static void check_layer(char *output, int check_sum_true, int dim) {
  int checksum = 0;
  char *ptr = (char *) output;
  for(int j=0; j<dim; j++) {
    checksum += ptr[j];
  }

  if(check_sum_true == checksum)
    printf("Checksum in/out Layer :\tOk\n");
  else
    printf("Checksum in/out Layer :\tFailed [%u vs. %u]\n", checksum, check_sum_true);
}

static void check_layer_last(${DORY_HW_graph[-1].output_activation_type}${DORY_HW_graph[-1].output_activation_bits}_t *ptr, int check_sum_true, int dim) {
  int checksum = 0;
  for(int j=0; j<dim/${DORY_HW_graph[-1].output_activation_bits // 8}; j++) {
    checksum += ptr[j];
  }

  if(check_sum_true == checksum)
    printf("Checksum final :\tOk\n");
  else
    printf("Checksum final :\tFailed [%d vs. %d]\n", checksum, check_sum_true);
}

// check for weight checksum
static void check_layer_weight(char *weight, int check_sum_true, int dim) {
  int checksum = 0;
  char *ptr = (char *) weight;
  for(int j=0; j<dim; j++) {
    checksum += ptr[j];
  }

  if(check_sum_true == checksum)
    printf("Checksum weight/bias Layer :\tOk\n");
  else
    printf("Checksum weight/bias Layer :\tFailed [%u vs. %u]\n", checksum, check_sum_true);
}
#endif
% endif

/* Moves the weights and the biases from hyperflash to hyperram */
void network_alloc()
{
% if 'Check_all' in verbose_level:
  int sum_weights;
  for (int i=0;i<${len(DORY_HW_graph)};i++)
  {
    if (layer_with_weights[i]==1)
    {
      sum_weights = 0;
      for (int j = 0; j < check_weights_dimension[i]; j++)
      {
        sum_weights+= Weights_name[i][j];
      }
      if (check_weights[i] == sum_weights)
        printf("Layer %-3d: Checksum = %-12d, FLASH %-12d, Check OK\n", i, check_weights[i], sum_weights);
      else
        printf("Layer %-3d: Checksum = %-12d, FLASH %-12d, Check FAILED\n", i, check_weights[i], sum_weights);
    }
  }
% endif
  return 1;
}

/* Remove RAM memory */
void network_free()
{
  return 1;
}

void network_run(char *L2_memory_buffer, int L2_memory_dimension, char *L2_output_to_pass, int begin_end)
{

/*
  - initial buffer allocation L2 and L1
  - variable declaration
*/
/* ---------------------------------- */
/* -------- SECTION 0 BEGIN --------- */
/* ---------------------------------- */
  bypass_activations = 0;
  int begin_end_n = begin_end;
  int L2_memory_buffer_end = L2_memory_buffer + L2_memory_dimension;
  int residual_number = 0;
  int perf_cyc = 0;

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

/*
  - input allocation and copy
*/
  dory_L2_alloc(&L2_memory_buffer,
    &L2_memory_buffer_end,
    &L2_input,
    check_activations_dimension[0],
    begin_end_n // begin is 1, end is 0
    );
  begin_end_n = !begin_end_n;
/*
  - output of the first layer allocation
*/
  dory_L2_alloc(&L2_memory_buffer,
    &L2_memory_buffer_end,
    &L2_output,
    check_activations_out_dimension[0],
    begin_end_n // begin is 1, end is 0
    );
  if(L2_output == NULL) return -1;
  begin_end_n = !begin_end_n;
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
  for(int i = 0; i < ${len(DORY_HW_graph)}; i++)
  {
/* MEMORY ALLOCATION
  - allocate memory if layer is executed from L3;
  - allocate weights
  - read weights
*/
    L2_weights = Weights_name[i];
% if verbose_level == 'Check_all+Perf_final':
#ifdef VERBOSE
    if (i==0) {
      printf("Checking input of layer %d...\n", i);
      check_layer(L2_input, check_activations[i], check_activations_dimension[i]);
      check_layer_weight(L2_weights, check_weights[i], check_weights_dimension[i]);
    }
    else if (branch_change[i-1]==0) {
      printf("Checking input of layer %d...\n", i);
      check_layer(L2_input, check_activations[i], check_activations_dimension[i]);
      check_layer_weight(L2_weights, check_weights[i], check_weights_dimension[i]);
    }
    else
      printf("Switching branch, already checked activation\n");
#endif
% endif
    unsigned int args[6] = {
      L2_input,
      bypass_activations,
      L2_output,
      L2_weights,
      out_shift_vector[i],
      i};

/*
- Execution of the layers_pointers
*/
      functions[i](args);
    
#ifdef VERBOSE
% if verbose_level == 'Check_all+Perf_final':
    printf("Layer %s %d ended: \n", Layers_name[i], i);
    if (i < ${len(DORY_HW_graph) - 1})
    {
      check_layer(L2_output, check_activations_out[i], check_activations_out_dimension[i]);
      printf("\n");
    }
    else
    {
      check_layer_last((${DORY_HW_graph[-1].output_activation_type}${DORY_HW_graph[-1].output_activation_bits}_t *) L2_output, check_activations_out[i], check_activations_out_dimension[i]);
    }
% elif verbose_level == 'Last+Perf_final':
    if (i == ${len(DORY_HW_graph) - 1})
        check_layer_last((${DORY_HW_graph[-1].output_activation_type}${DORY_HW_graph[-1].output_activation_bits}_t *) L2_output, check_activations_out[i], check_activations_out_dimension[i]);
% else:
    printf("Layer %s %d ended: \n", Layers_name[i], i);
% endif
#endif

/* MEMORY DEALLOCATION
*/
    dory_L2_free(&L2_memory_buffer,
      &L2_memory_buffer_end,
      check_activations_dimension[i],
      begin_end_n // begin is 1, end is 0
      );
    if (branch_input[i]==1)
      dory_L2_free(&L2_memory_buffer,
        &L2_memory_buffer_end,
        check_activations_dimension[i],
        begin_end_n // begin is 1, end is 0
        );
    L2_input = L2_output;
    if (i < ${len(DORY_HW_graph) - 1})
    {
      dory_L2_alloc(&L2_memory_buffer,
        &L2_memory_buffer_end,
        &L2_output,
        check_activations_out_dimension[i+1],
        begin_end_n // begin is 1, end is 0
        );
      //switching output and input in the buffer for allocation.
      begin_end_n = !begin_end_n;
    }
  }
  for(int i = 0; i < check_activations_out_dimension[${len(DORY_HW_graph)}]; i++)
  {
    *(L2_output_to_pass + i) = *(L2_output + i);
  }
/* ---------------------------------- */
/* --------- SECTION 2 END ---------- */
/* ---------------------------------- */
}

