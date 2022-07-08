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

static struct pi_hyperflash_conf flash_conf;
static struct pi_hyper_conf ram_conf;
static struct pi_device ram;
static int activations_input;
static uint8_t flashBuffer[FLASH_BUFF_SIZE];

char* L2_output;

% if sdk == 'pulp-sdk':
unsigned int PMU_set_voltage(unsigned int Voltage, unsigned int CheckFrequencies)
{
  return 0;
}
% endif

// filesystem management functions
void open_filesystem_and_ram(struct pi_device *flash, struct pi_device *fs)
{
  struct pi_readfs_conf conf;
  struct pi_hyperflash_conf flash_conf;

  /* Init & open flash. */
  pi_hyperflash_conf_init(&flash_conf);
  pi_open_from_conf(flash, &flash_conf);
  if (pi_flash_open(flash))
  {
      printf("Error flash open !\n");
      pmsis_exit(-1);
  }

  /* Open filesystem on flash. */
  pi_readfs_conf_init(&conf);
  conf.fs.flash = flash;
  pi_open_from_conf(fs, &conf);
  if (pi_fs_mount(fs))
  {
      printf("Error FS mounting !\n");
      pmsis_exit(-2);
  }
  pi_task_t task = {0};
  pi_task_block(&task);
  pi_hyperram_conf_init(&ram_conf);
  pi_open_from_conf(&ram, &ram_conf);
  pi_ram_open(&ram);
}

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
    Opening of Filesystem and Ram
*/
  struct pi_device fs;
  struct pi_device flash;
  open_filesystem_and_ram(&flash, &fs);
  pi_ram_alloc(&ram, &activations_input, (uint32_t) 1000000);
  pi_fs_file_t *file;
  file = pi_fs_open(&fs, "inputs.hex", 0);
  if (file == NULL)
  {
    printf("file open failed\n");
    return -1;
  }
/*
    Copying the input file from flash to ram
*/
  int flashBuffSize = FLASH_BUFF_SIZE * sizeof(char);
  int rdDone = 0;
  // loop on chunk in file
  while(rdDone < (${int(DORY_HW_graph[0].tiling_dimensions["L2"]["input_activation_memory"])} / sizeof(char)))
  {
    // read from HyperFlash
    int size = pi_fs_read(file, flashBuffer, flashBuffSize);
    // write to HyperRam
    pi_ram_write(&ram, activations_input+rdDone, flashBuffer, (uint32_t) size);
    rdDone += size / sizeof(char);
  }
/*
    Allocating space for input and copying it
*/
  L2_memory_buffer = pi_l2_malloc((uint32_t) ${l2_buffer_size});
  L2_output = L2_memory_buffer;
  L2_input = L2_memory_buffer;

  // First input read - TODO: is this good?
  pi_ram_read(&ram, activations_input, L2_input, ${int(DORY_HW_graph[0].tiling_dimensions["L2"]["input_activation_memory"])});

#ifdef VERBOSE
  printf("\nL2 Buffer alloc initial\t@ 0x%08x:\t%s\n", (unsigned int)L2_memory_buffer, L2_memory_buffer?"Ok":"Failed");
#endif

  network_initialize(fs, ram);
  network_run(L2_memory_buffer, ${l2_buffer_size}, L2_output, ram);

#if 0
  printf("Network Output: ");
  for(int i = 0; i < ${int(DORY_HW_graph[-1].tiling_dimensions["L2"]["output_activation_memory"] * (1 + int(DORY_HW_graph[-1].tiling_dimensions["L3"]["output_dimensions"] != DORY_HW_graph[-1].tiling_dimensions["L2"]["output_dimensions"]))) }; i+=4)
  {
    printf("%d ", *(int32_t *)(L2_output + i));
  }
  printf("\n");
#endif

  pi_ram_free(&ram, activations_input, ${int(DORY_HW_graph[0].tiling_dimensions["L2"]["input_activation_memory"])});
  network_terminate(ram);
  pi_l2_free(L2_memory_buffer, (uint32_t) ${l2_buffer_size});
}
