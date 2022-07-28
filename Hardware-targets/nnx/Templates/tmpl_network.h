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

#ifndef __NETWORK_H__
#define __NETWORK_H__

% if sdk == 'gap_sdk':
#include "pulp.h"
% endif

#define PAD_TOP    (1 << 3)
#define PAD_RIGHT  (1 << 2)
#define PAD_BOTTOM (1 << 1)
#define PAD_LEFT   (1 << 0)
#define NO_PAD     (0)

typedef struct {
    unsigned int L3_input;
    unsigned int L3_output;
    unsigned int L3_after_weights;
    unsigned int L2_input;
    unsigned int bypass;
    unsigned int L2_output;
    unsigned int L2_weights;
    unsigned int L1_buffer;
    unsigned int ram;
    unsigned int padding;
    unsigned int out_mult;
    unsigned int out_shift;
    unsigned int layer_id;
} layer_args_t;

void network_terminate();
void network_initialize();
void network_run(l2_buffer, l2_buffer_size, l2_final_output);
void execute_layer_fork(void *arg);
void execute_layer(void *arg);

#ifdef DEFINE_CONSTANTS
// allocation of buffers with parameters needed by the network execution
const char * L3_weights_files[] = {
  ${files_list}
};
int L3_weights_size[${weights_number}];
static int layers_pointers[${len(DORY_HW_graph)}];

static char * Layers_name[${len(DORY_HW_graph)}] = {\
% for node in DORY_HW_graph:
"${node.name}"${'' if loop.last else ', '}\
% endfor
};
static int L3_input_layers[${len(DORY_HW_graph)}] = {\
1, \
% for node in DORY_HW_graph[1:]:
% if node.tiling_dimensions["L3"]["input_dimensions"] != node.tiling_dimensions["L2"]["input_dimensions"]:
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
${int((node.tiling_dimensions["L2"]["weight_memory"] + node.tiling_dimensions["L2"]["constants_memory"] + node.tiling_dimensions["L2"]["bias_memory"]) * (1 + int(node.tiling_dimensions["L3"]["weights_dimensions"] != node.tiling_dimensions["L2"]["weights_dimensions"])))}${'' if loop.last else ', '}\
% endfor
};
static int activations_checksum[${len(DORY_HW_graph)}] = {\
% for node in DORY_HW_graph:
${node.check_sum_in}${'' if loop.last else ', '}\
% endfor
};
static int activations_size[${len(DORY_HW_graph)}] = {\
% for node in DORY_HW_graph:
${int(node.tiling_dimensions["L2"]["input_activation_memory"] * (1 + int(node.tiling_dimensions["L3"]["input_dimensions"] != node.tiling_dimensions["L2"]["input_dimensions"])))}${'' if loop.last else ', '}\
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
static int activations_out_checksum[${len(DORY_HW_graph)}] = {\
% for node in DORY_HW_graph:
${node.check_sum_out}${'' if loop.last else ', '}\
% endfor
};
static int activations_out_size[${len(DORY_HW_graph)}] = {\
% for node in DORY_HW_graph:
${int(node.tiling_dimensions["L2"]["output_activation_memory"] * (1 + int(node.tiling_dimensions["L3"]["output_dimensions"] != node.tiling_dimensions["L2"]["output_dimensions"])))}${'' if loop.last else ', '}\
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
