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

#define FLASH_BUFF_SIZE 128
% if verbose:
#define VERBOSE 1
% endif

static int activations_input;
static uint8_t flashBuffer[FLASH_BUFF_SIZE];

char* L2_output;

% if sdk == 'pulp-sdk':
unsigned int PMU_set_voltage(unsigned int Voltage, unsigned int CheckFrequencies)
{
  return 0;
}
% endif

int main () {
  char* L2_memory_buffer;
  char* L2_input;
  PMU_set_voltage(1000, 0);
  pi_time_wait_us(10000);
  pi_freq_set(PI_FREQ_DOMAIN_FC, ${fc_frequency});
  pi_time_wait_us(10000);
  pi_freq_set(PI_FREQ_DOMAIN_CL, ${cl_frequency});
  pi_time_wait_us(10000);
% if sdk == 'pulp-sdk':
  #if __PLATFORM__ == ARCHI_PLATFORM_FPGA
    *(int*)(ICACHE_PREFETCH) = 0xFFFF;
  #endif
% endif

/*
    Allocating space for input and copying it
*/
  L2_memory_buffer = pi_l2_malloc((uint32_t) ${l2_buffer_size});
  int begin_end = 1;
  L2_input = L2_memory_buffer + (1 - begin_end) * (${l2_buffer_size} - ${DORY_HW_graph[0].input_dimensions[0] * DORY_HW_graph[0].input_dimensions[1] * DORY_HW_graph[0].input_channels});
  L2_output = L2_memory_buffer;
#ifdef VERBOSE
  printf("\nL2 Buffer alloc initial\t@ 0x%08x:\t%s\n", (unsigned int)L2_memory_buffer, L2_memory_buffer?"Ok":"Failed");
#endif
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
    for(int i = 0; i < ${DORY_HW_graph[-1].tiling_dimensions["L2"]["output_activation_memory"]}; i+=4)
    {
      printf("%d ", *(int32_t *)(L2_output + i));
    }
    printf("\n");
#endif
/*
    Deallocation
*/
    network_free();  
    pi_l2_free(L2_memory_buffer, (uint32_t) ${l2_buffer_size});
}
