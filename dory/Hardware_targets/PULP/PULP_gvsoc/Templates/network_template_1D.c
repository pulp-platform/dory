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
% if sdk == 'gap_sdk':
#include "pulp.h"
% endif
#include "dory.h"
% for layer in list_h:
#include "${layer}"
% endfor
#include "pmsis.h"
#include "bsp/fs.h"
#include "bsp/fs/readfs.h"
#include "bsp/flash.h"
#include "bsp/ram.h"
#include "bsp/flash/hyperflash.h"
#include "bsp/ram/hyperram.h"

#define FLASH_BUFF_SIZE 128
% if verbose:
#define VERBOSE 1
% endif

% if sdk == 'pulp_sdk':
unsigned int PMU_set_voltage(unsigned int Voltage, unsigned int CheckFrequencies)
{
  return 0;
}
% endif

// allocation of buffers with parameters needed by the network execution
const char * L3_weights_files[] = {
  ${files_list}
};
int L3_weights_size[${weights_number}];
static int L3_weights;
static int L3_input;
static int bypass_L3_input;
static int L3_output;
static int bypass_L3_output;
static int activations_input;
static int L3_layers[${len(PULP_Nodes_Graph)}] = {\
% for i in range(len(PULP_Nodes_Graph)):
% if 'L3' in func_name[i]: 
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
${int(PULP_Nodes_Graph[i].weights_dimension * BitW / 8.0)}${'' if loop.last else ', '}\
% else:
${int((PULP_Nodes_Graph[i].weights_dimension - PULP_Nodes_Graph[i-1].weights_dimension) * BitW / 8.0)}${'' if loop.last else ', '}\
% endif
% endfor
};