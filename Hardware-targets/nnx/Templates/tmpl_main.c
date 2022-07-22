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
#include "mem.h"
#include "pmsis.h"

% if verbose:
#define VERBOSE 1
% endif

% if sdk == 'pulp-sdk':
unsigned int PMU_set_voltage(unsigned int Voltage, unsigned int CheckFrequencies) {
  return 0;
}
% endif


int main () {
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
    Opening of Filesystem and Ram
*/
  mem_init();

  size_t input_size = 1000000;
  void *ram_input = ram_malloc(input_size);
  load_file_to_ram(ram_input, "inputs.hex");
/*
    Allocating space for input and copying it
*/
  void *l2_buffer = pi_l2_malloc(${l2_buffer_size});

  size_t l2_input_size = ${int(DORY_HW_graph[0].tiling_dimensions["L2"]["input_activation_memory"])};
  ram_read(l2_buffer, ram_input, l2_input_size);

#ifdef VERBOSE
  printf("\nL2 Buffer alloc initial\t@ 0x%08x:\t%s\n", (unsigned int)l2_buffer, l2_buffer?"Ok":"Failed");
#endif

  network_initialize();
  network_run(l2_buffer, ${l2_buffer_size}, l2_buffer);

  ram_free(ram_input, input_size);
  network_terminate();
  pi_l2_free(l2_buffer, ${l2_buffer_size});
}
