/*
 * network.h
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

#ifndef __${prefix.upper()}NETWORK_H__
#define __${prefix.upper()}NETWORK_H__

<%
   l3_supported = DORY_HW_graph[0].HW_description['memory']['levels'] > 2
   single_input = n_inputs==1
%>\
% if not l3_supported:
#include "${prefix}weights_definition.h"
% endif
#include <stddef.h>
#include "pmsis.h"


struct ${prefix}network_run_token {
  struct pi_device cluster_dev;
};


% if l3_supported:
void ${prefix}network_terminate();
void ${prefix}network_initialize();
% endif
void ${prefix}network_run_cluster(void * args);
struct ${prefix}network_run_token ${prefix}network_run_async(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, void **l3_buffer, int exec, int initial_dir${", void *L2_input_h" if not l3_supported else ""});
void network_run_wait(struct ${prefix}network_run_token token);
void ${prefix}network_run(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, void **l3_buffer, int exec, int initial_dir${", void *L2_input_h" if not l3_supported else ""});
void ${prefix}execute_layer_fork(void *arg);

% if l3_supported and not single_input:
static char * ${prefix}Input_names[${n_inputs}] = { \
  % for n in range(n_inputs-1):
  "${f"{prefix}inputs_{n}.hex"}",
  % endfor
  "${f"{prefix}inputs_{n_inputs-1}.hex"}"
};
% endif

#ifdef DEFINE_CONSTANTS
% if l3_supported:
// allocation of buffers with parameters needed by the network execution
static const char * L3_weights_files[] = {
  ${files_list}
};
static int L3_weights_size[${weights_number}];
static int layers_pointers[${len(DORY_HW_graph)}];
% endif
static char * Layers_name[${len(DORY_HW_graph)}] = {\
% for node in DORY_HW_graph:
"${node.prefixed_name}"${'' if loop.last else ', '}\
% endfor
};
% if l3_supported:
static int L3_input_layers[${len(DORY_HW_graph)}] = {\
1,
% for node in DORY_HW_graph[1:]:
% if node.L3_input != 0:
1${'' if loop.last else ', '}\
% else:
0${'' if loop.last else ', '}\
% endif
% endfor
};
static int L3_output_layers[${len(DORY_HW_graph)}] = {\
% for node in DORY_HW_graph:
% if node.tiling_dimensions["L3"]["output_dimensions"] != node.tiling_dimensions["L2"]["output_dimensions"]:
1${'' if loop.last else ', '}\
% else:
0${'' if loop.last else ', '}\
% endif
% endfor
};
static int allocate_layer[${len(DORY_HW_graph)}] = {\
% for node in DORY_HW_graph:
% if node.tiling_dimensions["L3"]["weights_dimensions"] == node.tiling_dimensions["L2"]["weights_dimensions"] and ('FullyConnected' in node.name or 'Conv' in node.name):
1${'' if loop.last else ', '}\
% else:
0${'' if loop.last else ', '}\
% endif
% endfor
};
% else:
static char *Weights_name[${len(DORY_HW_graph)}] = {\
% for i in range(len(DORY_HW_graph)):
% if 'Conv' in DORY_HW_graph[i].name or 'FullyConnected' in DORY_HW_graph[i].name:
Weights_${DORY_HW_graph[i].prefixed_name}${'' if loop.last else ', '}\
% else:
"None"${'' if loop.last else ', '}\
% endif
% endfor
};
% endif
static int branch_input[${len(DORY_HW_graph)}] = {\
% for node in DORY_HW_graph:
% if node.branch_in == 1:
1${'' if loop.last else ', '}\
% else:
0${'' if loop.last else ', '}\
% endif
% endfor
};
static int branch_output[${len(DORY_HW_graph)}] = {\
% for node in DORY_HW_graph:
% if node.branch_out == 1:
1${'' if loop.last else ', '}\
% else:
0${'' if loop.last else ', '}\
% endif
% endfor
};
static int branch_change[${len(DORY_HW_graph)}] = {\
% for node in DORY_HW_graph:
% if node.branch_change == 1:
1${'' if loop.last else ', '}\
% else:
0${'' if loop.last else ', '}\
% endif
% endfor
};
static int weights_checksum[${len(DORY_HW_graph)}] = {\
% for node in DORY_HW_graph:
${node.check_sum_w}${'' if loop.last else ', '}\
% endfor
};
static int weights_size[${len(DORY_HW_graph)}] = {\
% for node in DORY_HW_graph:
% if l3_supported:
${int((node.tiling_dimensions["L2"]["weight_memory"] + node.tiling_dimensions["L2"]["constants_memory"] + node.tiling_dimensions["L2"]["bias_memory"]) * (1 + int(node.tiling_dimensions["L3"]["weights_dimensions"] != node.tiling_dimensions["L2"]["weights_dimensions"])))}${'' if loop.last else ', '}\
% else:
${int(node.tiling_dimensions["L2"]["weight_memory"] + node.tiling_dimensions["L2"]["constants_memory"] + node.tiling_dimensions["L2"]["bias_memory"])}${'' if loop.last else ', '}\
% endif
% endfor
};
static int activations_checksum[${len(DORY_HW_graph)}][${DORY_HW_graph[0].n_test_inputs}] = {\
% for i in range(len(DORY_HW_graph)):
{
  % for j in range(DORY_HW_graph[0].n_test_inputs):
  ${DORY_HW_graph[i].check_sum_in[j]}${", " if j != DORY_HW_graph[0].n_test_inputs-1 else ""}  \
  % endfor
}${"," if i != len(DORY_HW_graph)-1 else ""}
% endfor
};
static int activations_size[${len(DORY_HW_graph)}] = {\
% for node in DORY_HW_graph:
% if l3_supported:
${int(node.tiling_dimensions["L2"]["input_activation_memory"] * (1 + int(node.tiling_dimensions["L3"]["input_dimensions"] != node.tiling_dimensions["L2"]["input_dimensions"])))}${'' if loop.last else ', '}\
% else:
${int(node.tiling_dimensions["L2"]["input_activation_memory"])}${'' if loop.last else ', '}\
% endif
% endfor
};
static int out_mult_vector[${len(DORY_HW_graph)}] = {\
% for node in DORY_HW_graph:
% if hasattr(node, "outmul"):
${node.outmul["value"]}${'' if loop.last else ', '}\
% else:
1${'' if loop.last else ', '}\
% endif
% endfor
};
static int out_shift_vector[${len(DORY_HW_graph)}] = {\
% for node in DORY_HW_graph:
% if hasattr(node, "outshift"):
${node.outshift["value"]}${'' if loop.last else ', '}\
% else:
0${'' if loop.last else ', '}\
% endif
% endfor
};
static int activations_out_checksum[${len(DORY_HW_graph)}][${DORY_HW_graph[0].n_test_inputs}] = {\
% for i in range(len(DORY_HW_graph)):
{
  % for j in range(DORY_HW_graph[0].n_test_inputs):
  ${DORY_HW_graph[i].check_sum_out[j]}${", " if j != DORY_HW_graph[0].n_test_inputs-1 else ""} \
    % endfor
}${"," if i != len(DORY_HW_graph)-1 else ""}
% endfor
};
static int activations_out_size[${len(DORY_HW_graph)}] = {\
% for node in DORY_HW_graph:
% if l3_supported:
${int(node.tiling_dimensions["L2"]["output_activation_memory"] * (1 + int(node.tiling_dimensions["L3"]["output_dimensions"] != node.tiling_dimensions["L2"]["output_dimensions"])))}${'' if loop.last else ', '}\
% else:
${int(node.tiling_dimensions["L2"]["output_activation_memory"])}${'' if loop.last else ', '}\
% endif
% endfor
};
static int layer_with_weights[${len(DORY_HW_graph)}] = {\
% for node in DORY_HW_graph:
% if 'Conv' in node.name or 'FullyConnected' in node.name:
1${'' if loop.last else ', '}\
% else:
0${'' if loop.last else ', '}\
% endif
% endfor
};
% if 'Yes' in performance:
static int NODEs_MACS[${len(DORY_HW_graph)}] = {\
% for node in DORY_HW_graph:
${node.MACs}${'' if loop.last else ', '}\
% endfor
};
% endif
#endif

#endif  // __NETWORK_H__
