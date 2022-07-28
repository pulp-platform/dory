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
#include "pmsis.h"
#include "bsp/fs.h"
#include "bsp/fs/readfs.h"
#include "bsp/flash.h"
#include "bsp/ram.h"
#include "bsp/flash/hyperflash.h"
#include "bsp/ram/hyperram.h"
#include "mem_controller.h"

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
    unsigned int out_mult;
    unsigned int out_shift;
    unsigned int layer_id;
} layer_args_t;

void network_free();
void network_alloc();
void network_run(char *L2_memory_buffer, int L2_memory_dimension, char *L2_output_to_pass, int begin_end);
void network_run_cluster(void *arg);
void execute_layer_fork(void *arg);

#ifdef DEFINE_CONSTANTS
static int layers_pointers[${len(DORY_HW_graph)}];

static char *Weights_name[${len(DORY_HW_graph)}] = {\
% for i in range(len(DORY_HW_graph)):
% if 'Conv' in DORY_HW_graph[i].name or 'FullyConnected' in DORY_HW_graph[i].name:
Weights_${DORY_HW_graph[i].name}${'' if loop.last else ', '}\
% else:
"None"${'' if loop.last else ', '}\
% endif
% endfor
};
static char * Layers_name[${len(DORY_HW_graph)}] = {\
% for i in range(len(DORY_HW_graph)):
"${DORY_HW_graph[i].name}"${'' if loop.last else ', '}\
% endfor
};
static int branch_input[${len(DORY_HW_graph)}] = {\
% for i in range(len(DORY_HW_graph)):
% if DORY_HW_graph[i].branch_in == 1:
1${'' if loop.last else ', '}\
% else:
0${'' if loop.last else ', '}\
% endif
% endfor
};
static int branch_output[${len(DORY_HW_graph)}] = {\
% for i in range(len(DORY_HW_graph)):
% if DORY_HW_graph[i].branch_out == 1:
1${'' if loop.last else ', '}\
% else:
0${'' if loop.last else ', '}\
% endif
% endfor
};
static int branch_change[${len(DORY_HW_graph)}] = {\
% for i in range(len(DORY_HW_graph)):
% if DORY_HW_graph[i].branch_change == 1:
1${'' if loop.last else ', '}\
% else:
0${'' if loop.last else ', '}\
% endif
% endfor
};
static int check_weights[${len(DORY_HW_graph)}] = {\
% for i in range(len(DORY_HW_graph)):
${DORY_HW_graph[i].check_sum_w}${'' if loop.last else ', '}\
% endfor
};
static int check_weights_dimension[${len(DORY_HW_graph)}] = {\
% for i in range(len(DORY_HW_graph)):
${(DORY_HW_graph[i].tiling_dimensions["L2"]["weight_memory"] + DORY_HW_graph[i].tiling_dimensions["L2"]["constants_memory"] + DORY_HW_graph[i].tiling_dimensions["L2"]["bias_memory"])}${'' if loop.last else ', '}\
% endfor
};
static int check_activations[${len(DORY_HW_graph)}] = {\
% for i in range(len(DORY_HW_graph)):
${DORY_HW_graph[i].check_sum_in}${'' if loop.last else ', '}\
% endfor
};
static int check_activations_dimension[${len(DORY_HW_graph)}] = {\
% for i in range(len(DORY_HW_graph)):
${int(DORY_HW_graph[i].tiling_dimensions["L2"]["input_activation_memory"])}${'' if loop.last else ', '}\
% endfor
};
static int out_mult_vector[${len(DORY_HW_graph)}] = {\
% for i in range(len(DORY_HW_graph)):
% if "outmul" not in DORY_HW_graph[i].__dict__:
1${'' if loop.last else ', '}\
% else:
${DORY_HW_graph[i].outmul["value"]}${'' if loop.last else ', '}\
% endif
% endfor
};
static int out_shift_vector[${len(DORY_HW_graph)}] = {\
% for i in range(len(DORY_HW_graph)):
% if "outshift" not in DORY_HW_graph[i].__dict__:
0${'' if loop.last else ', '}\
% else:
${DORY_HW_graph[i].outshift["value"]}${'' if loop.last else ', '}\
% endif
% endfor
};
static int check_activations_out[${len(DORY_HW_graph)}] = {\
% for i in range(len(DORY_HW_graph)):
${DORY_HW_graph[i].check_sum_out}${'' if loop.last else ', '}\
% endfor
};
static int check_activations_out_dimension[${len(DORY_HW_graph)}] = {\
% for i in range(len(DORY_HW_graph)):
${int(DORY_HW_graph[i].tiling_dimensions["L2"]["output_activation_memory"])}${'' if loop.last else ', '}\
% endfor
};
static int layer_with_weights[${len(DORY_HW_graph)}] = {\
% for i in range(len(DORY_HW_graph)):
% if 'Conv' in DORY_HW_graph[i].name or 'FullyConnected' in DORY_HW_graph[i].name:
1${'' if loop.last else ', '}\
% else:
0${'' if loop.last else ', '}\
% endif
% endfor
};
% if 'Yes' in performance:
static int NODEs_MACS[${len(DORY_HW_graph)}] = {\
% for i in range(len(DORY_HW_graph)):
${DORY_HW_graph[i].MACs}${'' if loop.last else ', '}\
% endfor
};
% endif
#endif

#endif  // __NETWORK_H__
