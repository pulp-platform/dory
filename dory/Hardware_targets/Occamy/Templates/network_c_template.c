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
#include "mem_controller.h"
#include "network.h"
#include "dory.h"
% for layer in list_h:
#include "${layer}"
% endfor
#include "snrt.h"
#include "printf.h"
#include "data.h"

% if verbose:
#define VERBOSE 1
% endif


// allocation of buffers with parameters needed by the network execution
const float * Weights_tensors[] = {
  ${files_list[1:-1].replace("\"", "")}
};

layer layer_i;
__attribute__((section(".data"))) int layers_pointers[${len(DORY_HW_graph)}];
__attribute__((section(".data"))) char * Layers_name[${len(DORY_HW_graph)}] = {\
% for i in range(len(DORY_HW_graph)):
"${DORY_HW_graph[i].name}"${'' if loop.last else ', '}\
% endfor
};
__attribute__((section(".data"))) int check_activations[${len(DORY_HW_graph)}] = {\
% for i in range(len(DORY_HW_graph)):
${DORY_HW_graph[i].check_sum_in}${'' if loop.last else ', '}\
% endfor
};
__attribute__((section(".data"))) int check_activations_dimension[${len(DORY_HW_graph)}] = {\
% for i in range(len(DORY_HW_graph)):
${int(DORY_HW_graph[i].tiling_dimensions["L2"]["input_activation_memory"])}${'' if loop.last else ', '}\
% endfor
};
__attribute__((section(".data"))) int out_mult_vector[${len(DORY_HW_graph)}] = {\
% for i in range(len(DORY_HW_graph)):
% if "outmul" not in DORY_HW_graph[i].__dict__:
1${'' if loop.last else ', '}\
% else:
${DORY_HW_graph[i].outmul["value"]}${'' if loop.last else ', '}\
% endif
% endfor
};
__attribute__((section(".data"))) int out_shift_vector[${len(DORY_HW_graph)}] = {\
% for i in range(len(DORY_HW_graph)):
% if "outshift" not in DORY_HW_graph[i].__dict__:
0${'' if loop.last else ', '}\
% else:
${DORY_HW_graph[i].outshift["value"]}${'' if loop.last else ', '}\
% endif
% endfor
};
__attribute__((section(".data"))) int inmul1_vector[${len(DORY_HW_graph)}] = {\
% for i in range(len(DORY_HW_graph)):
% if "inmul1" not in DORY_HW_graph[i].__dict__:
0${'' if loop.last else ', '}\
% else:
${DORY_HW_graph[i].inmul1["value"]}${'' if loop.last else ', '}\
% endif
% endfor
};
__attribute__((section(".data"))) int inmul2_vector[${len(DORY_HW_graph)}] = {\
% for i in range(len(DORY_HW_graph)):
% if "inmul2" not in DORY_HW_graph[i].__dict__:
0${'' if loop.last else ', '}\
% else:
${DORY_HW_graph[i].inmul2["value"]}${'' if loop.last else ', '}\
% endif
% endfor
};
__attribute__((section(".data"))) int check_activations_out[${len(DORY_HW_graph)}] = {\
% for i in range(len(DORY_HW_graph)):
${DORY_HW_graph[i].check_sum_out}${'' if loop.last else ', '}\
% endfor
};
__attribute__((section(".data"))) int check_activations_out_dimension[${len(DORY_HW_graph)}] = {\
% for i in range(len(DORY_HW_graph)):
${int(DORY_HW_graph[i].tiling_dimensions["L2"]["output_activation_memory"])}${'' if loop.last else ', '}\
% endfor
};
__attribute__((section(".data"))) int layer_with_weights[${len(DORY_HW_graph)}] = {\
% for i in range(len(DORY_HW_graph)):
% if 'Gemm' in DORY_HW_graph[i].name or 'Conv' in DORY_HW_graph[i].name or 'MatMul' in DORY_HW_graph[i].name: 
1${'' if loop.last else ', '}\
% else:
0${'' if loop.last else ', '}\
% endif
% endfor
};

% if verbose_level == 'Check_all+Perf_final':
#ifdef VERBOSE
// check for input/output acitvation checksum
static void check_layer(char *output, int check_sum_true, int dim) {
  int checksum = 0;
  float *ptr = (float *) output;
  for(int j=0; j<dim; j++) {
    checksum += ptr[j];
  }

  if(check_sum_true == checksum)
    printf("Checksum in/out Layer :\tOk\n");
  else 
    printf("Checksum in/out Layer :\tFailed [%u vs. %u]\n", checksum, check_sum_true);
}

#endif 
% endif


uint32_t L2_weights, L2_output, L2_input_add, L2_input;
__attribute__((section(".data"))) float L2_output_mem[1000000];
__attribute__((section(".data"))) float L2_input_add_mem[1000000];
__attribute__((section(".data"))) float L2_input_mem[1000000];
void network_run()
{   

/* 
  - initial buffer allocation L2 and L1
  - variable declaration
*/
/* ---------------------------------- */
/* -------- SECTION 0 BEGIN --------- */
/* ---------------------------------- */
  uint16_t out_mult = 0;
  uint16_t out_shift = 0;
  uint16_t inmul1 = 0;
  uint16_t inmul2 = 0;
#ifdef VERBOSE
  if (snrt_cluster_compute_core_idx()==0)
    printf("I'm Core 0 from Occamy Cluster. Beginning the neural network computation\n");
#endif
/* ---------------------------------- */
/* --------- SECTION 0 END ---------- */ 
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
  int j = 0;
  for(int i = 0; i < ${len(DORY_HW_graph)}; i++)
  {
    if (layer_with_weights[i] == 1)
    {
      L2_weights = (uint32_t) Weights_tensors[j];
      j++;
    }
    if (i == 0)
    {
      L2_input = input;
      L2_output = L2_output_mem;
    }
    else if ( i % 2 == 0)
    {
      L2_input = L2_input_mem;
      L2_output = L2_output_mem;
    }
    else
    {
      L2_input = L2_output_mem;
      L2_output = L2_input_mem;
    }
% if verbose_level == 'Check_all+Perf_final':
#ifdef VERBOSE
    if(snrt_cluster_compute_core_idx() == 0 && snrt_cluster_idx() == 0)
    {
      if (i==0)
        check_layer(L2_input, check_activations[i], check_activations_dimension[i]);
      else if (branch_change[i-1]==0)
        check_layer(L2_input, check_activations[i], check_activations_dimension[i]);
      else
        printf("Switching branch, already checked activation\n");
    }
#endif  
% endif
    snrt_global_barrier();
    layer_i.L2_input = L2_input;
    layer_i.L2_input_add = L2_input_add;
    layer_i.L2_output = L2_output;
    layer_i.L2_weights = L2_weights;
    layer_i.l2_zeros = l2_zeros;
    layer_i.out_mult = out_mult_vector[i];
    layer_i.out_shift = out_shift_vector[i];
    layer_i.inmul1 = inmul1_vector[i];
    layer_i.inmul2 = inmul2_vector[i];
    switch (i)
    {
% for i in range(len(DORY_HW_graph)):
      case ${i}:
% if performance == 'Yes':
        benchmark_get_cycle();
% endif
        ${func_name[i]}(&layer_i);
% if performance == 'Yes':
        benchmark_get_cycle();
% endif
        break;
% endfor
    }
    snrt_global_barrier();

% if verbose_level == 'Check_all+Perf_final':
#ifdef VERBOSE
    if(snrt_cluster_compute_core_idx() == 0 && snrt_cluster_idx() == 0)
    {
      printf("Layer %s %d ended: \n", Layers_name[i], i);
      check_layer(L2_output, check_activations_out[i], check_activations_out_dimension[i]);
    }    
    snrt_global_barrier();
#endif 
% elif verbose_level == 'Last+Perf_final':
    if(pi_core_id()==0)
      if (i == ${len(DORY_HW_graph) - 1})
          check_layer_last((int32_t *) L2_output, check_activations_out[i], check_activations_out_dimension[i]);
% else:
#ifdef VERBOSE
    if(pi_core_id()==0)
    {
      printf("Layer %s %d ended: \n", Layers_name[i], i);
    }     
#endif   
% endif

  }
/* ---------------------------------- */
/* --------- SECTION 2 END ---------- */
/* ---------------------------------- */

/* ---------------------------------- */
/* -------- SECTION 3 BEGIN --------- */
/* ---------------------------------- */
% if 'Perf_final' in verbose_level:
#ifdef VERBOSE
  int cid = snrt_cluster_compute_core_idx();    
  int MACs = ${MACs};
  if (cid == 0)
  {
    printf("[%d] : Total MACs: %d\n",cid,MACs ); 
  }
#endif
% endif
/* ---------------------------------- */
/* --------- SECTION 3 END ---------- */
/* ---------------------------------- */
}

