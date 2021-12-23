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

static int layers_pointers[${len(PULP_Nodes_Graph)}];


static char * Layers_name[${len(PULP_Nodes_Graph)}] = {\
% for i in range(len(PULP_Nodes_Graph)):
"${PULP_Nodes_Graph[i].name}"${'' if loop.last else ', '}\
% endfor
};
static int L3_layers[${len(PULP_Nodes_Graph)}] = {\
% for i in range(len(PULP_Nodes_Graph)):
% if 'L3' in func_name[i]: 
1${'' if loop.last else ', '}\
% else:
0${'' if loop.last else ', '}\
% endif
% endfor
};
static int L3_input_layers[${len(PULP_Nodes_Graph)}] = {\
% for i in range(len(PULP_Nodes_Graph)):
% if PULP_Nodes_Graph[i].L3_input == 1: 
1${'' if loop.last else ', '}\
% else:
0${'' if loop.last else ', '}\
% endif
% endfor
};
static int L3_output_layers[${len(PULP_Nodes_Graph)}] = {\
% for i in range(len(PULP_Nodes_Graph)):
% if PULP_Nodes_Graph[i].L3_output == 1: 
1${'' if loop.last else ', '}\
% else:
0${'' if loop.last else ', '}\
% endif
% endfor
};
static int L3_weights_layers[${len(PULP_Nodes_Graph)}] = {\
% for i in range(len(PULP_Nodes_Graph)):
% if PULP_Nodes_Graph[i].L3_weights == 1: 
1${'' if loop.last else ', '}\
% else:
0${'' if loop.last else ', '}\
% endif
% endfor
};
static int allocate_layer[${len(PULP_Nodes_Graph)}] = {\
% for i in range(len(PULP_Nodes_Graph)):
% if PULP_Nodes_Graph[i].L3_allocation!=1 and ('Gemm' in PULP_Nodes_Graph[i].name or 'Conv' in PULP_Nodes_Graph[i].name or 'MatMul' in PULP_Nodes_Graph[i].name): 
1${'' if loop.last else ', '}\
% else:
0${'' if loop.last else ', '}\
% endif
% endfor
};
static int branch_input[${len(PULP_Nodes_Graph)}] = {\
% for i in range(len(PULP_Nodes_Graph)):
% if PULP_Nodes_Graph[i].branch_in == 1:
1${'' if loop.last else ', '}\
% else:
0${'' if loop.last else ', '}\
% endif
% endfor
};
static int branch_output[${len(PULP_Nodes_Graph)}] = {\
% for i in range(len(PULP_Nodes_Graph)):
% if PULP_Nodes_Graph[i].branch_out == 1:
1${'' if loop.last else ', '}\
% else:
0${'' if loop.last else ', '}\
% endif
% endfor
};
static int branch_change[${len(PULP_Nodes_Graph)}] = {\
% for i in range(len(PULP_Nodes_Graph)):
% if PULP_Nodes_Graph[i].branch_change == 1:
1${'' if loop.last else ', '}\
% else:
0${'' if loop.last else ', '}\
% endif
% endfor
};
static int branch_last[${len(PULP_Nodes_Graph)}] = {\
% for i in range(len(PULP_Nodes_Graph)):
% if PULP_Nodes_Graph[i].branch_last == 1:
1${'' if loop.last else ', '}\
% else:
0${'' if loop.last else ', '}\
% endif
% endfor
};
static int check_weights[${len(PULP_Nodes_Graph)}] = {\
% for i in range(len(PULP_Nodes_Graph)):
${PULP_Nodes_Graph[i].check_sum_w}${'' if loop.last else ', '}\
% endfor
};
static int check_weights_dimension[${len(PULP_Nodes_Graph)}] = {\
% for i in range(len(PULP_Nodes_Graph)):
% if i == 0:
${int(PULP_Nodes_Graph[i].weights_dimension * PULP_Nodes_Graph[0].weight_bits / 8.0)}${'' if loop.last else ', '}\
% else:
${int((PULP_Nodes_Graph[i].weights_dimension - PULP_Nodes_Graph[i-1].weights_dimension) * PULP_Nodes_Graph[0].weight_bits / 8.0)}${'' if loop.last else ', '}\
% endif
% endfor
};
static int cumulative_weights_dimension[${len(PULP_Nodes_Graph)}] = {\
% for i in range(len(PULP_Nodes_Graph)):
% if i == 0: 
0${'' if loop.last else ', '}\
% else:
${int((PULP_Nodes_Graph[i-1].weights_dimension_L3))}${'' if loop.last else ', '}\
% endif
% endfor
};
static int check_activations[${len(PULP_Nodes_Graph)}] = {\
% for i in range(len(PULP_Nodes_Graph)):
${PULP_Nodes_Graph[i].check_sum_in}${'' if loop.last else ', '}\
% endfor
};
static int check_activations_dimension[${len(PULP_Nodes_Graph)}] = {\
% for i in range(len(PULP_Nodes_Graph)):
${int(PULP_Nodes_Graph[i].input_activation_dimensions)}${'' if loop.last else ', '}\
% endfor
};
static int check_activations_dimension_L3_in[${len(PULP_Nodes_Graph)}] = {\
% for i in range(len(PULP_Nodes_Graph)):
${int(PULP_Nodes_Graph[i].input_activation_dimensions_L3)}${'' if loop.last else ', '}\
% endfor
};
static int check_activations_dimension_L3_out[${len(PULP_Nodes_Graph)}] = {\
% for i in range(len(PULP_Nodes_Graph)):
${int(PULP_Nodes_Graph[i].output_activation_dimensions_L3)}${'' if loop.last else ', '}\
% endfor
};
static int out_mult_vector[${len(PULP_Nodes_Graph)}] = {\
% for i in range(len(PULP_Nodes_Graph)):
% if PULP_Nodes_Graph[i].outmul == 'empty':
0${'' if loop.last else ', '}\
% else:
${PULP_Nodes_Graph[i].outmul}${'' if loop.last else ', '}\
% endif
% endfor
};
static int out_shift_vector[${len(PULP_Nodes_Graph)}] = {\
% for i in range(len(PULP_Nodes_Graph)):
% if PULP_Nodes_Graph[i].outshift == 'empty':
0${'' if loop.last else ', '}\
% else:
${PULP_Nodes_Graph[i].outshift}${'' if loop.last else ', '}\
% endif
% endfor
};
static int inmul1_vector[${len(PULP_Nodes_Graph)}] = {\
% for i in range(len(PULP_Nodes_Graph)):
% if PULP_Nodes_Graph[i].inmul1 == 'empty':
0${'' if loop.last else ', '}\
% else:
${PULP_Nodes_Graph[i].inmul1}${'' if loop.last else ', '}\
% endif
% endfor
};
static int inmul2_vector[${len(PULP_Nodes_Graph)}] = {\
% for i in range(len(PULP_Nodes_Graph)):
% if PULP_Nodes_Graph[i].inmul2 == 'empty':
0${'' if loop.last else ', '}\
% else:
${PULP_Nodes_Graph[i].inmul2}${'' if loop.last else ', '}\
% endif
% endfor
};
static int check_activations_out[${len(PULP_Nodes_Graph)}] = {\
% for i in range(len(PULP_Nodes_Graph)):
${PULP_Nodes_Graph[i].check_sum_out}${'' if loop.last else ', '}\
% endfor
};
static int check_activations_out_dimension[${len(PULP_Nodes_Graph)}] = {\
% for i in range(len(PULP_Nodes_Graph)):
${int(PULP_Nodes_Graph[i].output_activation_dimensions)}${'' if loop.last else ', '}\
% endfor
};
static int layer_with_weights[${len(PULP_Nodes_Graph)}] = {\
% for i in range(len(PULP_Nodes_Graph)):
% if 'Gemm' in PULP_Nodes_Graph[i].name or 'Conv' in PULP_Nodes_Graph[i].name or 'MatMul' in PULP_Nodes_Graph[i].name: 
1${'' if loop.last else ', '}\
% else:
0${'' if loop.last else ', '}\
% endif
% endfor
};
% if 'Yes' in performance:
static int NODEs_MACS[${len(PULP_Nodes_Graph)}] = {\
% for i in range(len(PULP_Nodes_Graph)):
${PULP_Nodes_Graph[i].MACs}${'' if loop.last else ', '}\
% endfor
};
% endif

% if verbose_level == 'Check_all+Perf_final':
% if check_layer != 100:
uint8_t act_check[${nof_check*h_out_check*w_out_check}] = {
  ${act_compare}
};

static void check_layer_plus(char *output, int dim) {
  int error_presence = 0;
  for (int k=0; k<${nof_check}; k++) {
    for(int i=0; i<${h_out_check}; i++) {
      for(int j=0; j<${w_out_check}; j++) {
        if(output[i*${nof_check}*${w_out_check}+j*${nof_check}+k] != act_check[i*${nof_check}*${w_out_check}+j*${nof_check}+k]) {
          error_presence = 1;
          printf("(@%08x,%d,%d,%d) %04x instead of %04x\n", (unsigned int) &output[i*${nof_check}*${w_out_check}+j*${nof_check}+k], i,j,k, (output[i*${nof_check}*${w_out_check}+j*${nof_check}+k]) & 0xffff, (act_check[i*${nof_check}*${w_out_check}+j*${nof_check}+k]) & 0xffff);
        }
      }
    }
  }

  if (error_presence == 0)
  {
    printf("\n Test target layer successful: no errors\n\n");
  }
}

% endif
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
float L2_output_mem[1000000], L2_input_add_mem[1000000], L2_input_mem[1000000];
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
  for(int i = 0; i < ${len(PULP_Nodes_Graph)}; i++)
  {
    out_mult = out_mult_vector[i];
    out_shift = out_shift_vector[i];
    inmul1 = inmul1_vector[i];
    inmul2 = inmul2_vector[i];
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
    unsigned int args[9] = {
      L2_input,
      L2_input_add,
      L2_output,
      L2_weights,
      l2_zeros,
      out_mult,
      inmul1,
      inmul2, 
      out_shift};
    switch (i)
    {
% for i in range(len(PULP_Nodes_Graph)):
      case ${i}:
        ${func_name[i]}(args);
        break;
% endfor
    }
    snrt_global_barrier();
% if 'Yes' in performance:
    int MACs = NODEs_MACS[i];
    if (snrt_cluster_compute_core_idx() == 0 && snrt_cluster_idx() == 0)
    {
      printf("[%d] Layer %-3d end: MACs: %-11d,",snrt_cluster_compute_core_idx(), i, MACs); 
      printf(" MACs: %-11d,",MACs );  
    }
% endif

% if verbose_level == 'Check_all+Perf_final':
#ifdef VERBOSE
    if(snrt_cluster_compute_core_idx() == 0 && snrt_cluster_idx() == 0)
    {
      printf("Layer %s %d ended: \n", Layers_name[i], i);
      check_layer(L2_output, check_activations_out[i], check_activations_out_dimension[i]);
      
      if (i==${check_layer})
      {    
        check_layer_plus(L2_output,check_activations_out_dimension[i]);
      }
    }    
    snrt_global_barrier();
#endif 
% elif verbose_level == 'Last+Perf_final':
    if(pi_core_id()==0)
      if (i == ${len(PULP_Nodes_Graph) - 1})
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
  int cid = snrt_cluster_compute_core_idx();    
  int MACs = ${MACs};
  if (cid == 0)
  {
    printf("[%d] : Total MACs: %d\n",cid,MACs ); 
  }
% endif
/* ---------------------------------- */
/* --------- SECTION 3 END ---------- */
/* ---------------------------------- */
}

