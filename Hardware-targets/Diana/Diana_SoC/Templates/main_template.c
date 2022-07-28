/*
 * test_template.c
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
#include "network.h"

% if verbose:
#define VERBOSE 1
% endif

L2_DATA char L2_memory_buffer[${l2_buffer_size}];
char* L2_input;
char* L2_output;
int rdDone = 0;

int main () {

/*
    Allocating space for input and copying it
*/
  int begin_end = 1;
  L2_input = L2_memory_buffer + (1 - begin_end) * (${l2_buffer_size} - rdDone);
  L2_output = L2_memory_buffer;
/*
    Allocation
*/
    network_alloc();  
/*
    Running of the network
*/
  	network_run(L2_memory_buffer, ${l2_buffer_size}, L2_output, begin_end);
#ifdef VERBOSE
    printf("Network Output: ");
    for(int i = 0; i < ${int(DORY_HW_graph[-1].tiling_dimensions["L2"]["output_activation_memory"])}; i+=4)
    {
      printf("%d ", *(int32_t *)(L2_output + i));
    }
    printf("\n");
#endif
/*
    Deallocation
*/
    network_free();  
}
