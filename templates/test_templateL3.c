/*
 * test_templateL3.c
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
#include "rt/rt_api.h"
#include "${func_nameL3}.h"
#include "pulp.h"
#include "dory.h"
#include "kernels.h"
#include "pmsis.h"
#include "bsp/fs.h"
#include "bsp/fs/readfs.h"
#include "bsp/flash.h"
#include "bsp/ram.h"
#include "bsp/flash/hyperflash.h"
#include "bsp/ram/hyperram.h"

#define FLASH_BUFF_SIZE 128

% if n_tile_x > 1:
RT_L2_DATA uint8_t x_test[${activation_size_in*2}];
% else:
RT_L2_DATA uint8_t x_test[${activation_size_in_full}] = {
  ${x_content}
};
% endif

% if n_tile_W == 1:
RT_L2_DATA char l2_weights[${weight_dim}] = {
  ${W_content}
};
% else:
char * l2_weights;
% endif
% if n_tile_y > 1:
RT_L2_DATA uint8_t y_test[${activation_size_out*2}];
% else:
RT_L2_DATA uint8_t y_test[${activation_size_out * n_tile_W * n_tile_x}];
% endif
int L3_weights_size[1];
static struct pi_hyperflash_conf flash_conf;
static struct pi_hyper_conf ram_conf;
static struct pi_device ram;
const char * L3_weights_files[] = {
  "${file}"
};
static int L3_weights;
static int L3_activations;
static int activations_input;
char * l1_buffer;

unsigned int out_mult_in = ${out_mul};
unsigned int out_shift_in = ${out_shift};
int check_sum_true = ${check_sum};

void open_filesystem(struct pi_device *flash, struct pi_device *fs)
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
}

/* Moves the weights and the biases from hyperflash to hyperram */
int network_setup()
{
  pi_task_t task = {0};
  pi_task_block(&task);
  struct pi_device fs;
  struct pi_device flash;
  pi_hyperram_conf_init(&ram_conf);
  open_filesystem(&flash, &fs);
  pi_open_from_conf(&ram, &ram_conf);
  pi_ram_open(&ram);
  pi_fs_file_t *file;
  pi_ram_alloc(&ram, &L3_weights, (uint32_t) 5000000);
#ifdef VERBOSE
    printf("\nL3 Buffer alloc initial\t@ %d:\t%s\n", (unsigned int)L3_weights, L3_weights?"Ok":"Failed");
#endif
  unsigned int rdDone = 0;
  file = pi_fs_open(&fs, L3_weights_files[0], 0);
  if (file == NULL)
  {
    printf("file open failed\n");
    return -1;
  }
  L3_weights_size[0] = file->size + rdDone;
  int flashBuffSize = FLASH_BUFF_SIZE * sizeof(char);
  uint8_t flashBuffer[FLASH_BUFF_SIZE];
  while(rdDone < (L3_weights_size[0] / sizeof(char))) 
  { 
    int size = pi_fs_read(file, flashBuffer, flashBuffSize);
    pi_ram_write(&ram, L3_weights+rdDone, flashBuffer,size);
    rdDone += size / sizeof(char);
  }
  file = pi_fs_open(&fs, "inputs.hex", 0);
  if (file == NULL)
  {
    printf("file open failed\n");
    return -1;
  }
  activations_input = L3_weights + rdDone;
  rdDone = 0;
  // loop on chunk in file
  while(rdDone < (file->size * ${int(x_data_size_byte / 8.0)} / sizeof(char))) 
  { 
    // read from HyperFlash
    int size = pi_fs_read(file, flashBuffer, flashBuffSize);
    // write to HyperRam
    pi_ram_write(&ram, activations_input+rdDone, flashBuffer, (uint32_t) size);
    rdDone += size / sizeof(char);
  }
  return 1;
}

// on cluster
void cluster_main(void *arg) {
  int *real_arg = (int *) arg;
  unsigned int args[13] = {
    (unsigned int) real_arg[4], 
    (unsigned int) real_arg[3],
    (unsigned int) real_arg[2],
    (unsigned int) real_arg[0],
    0,
    (unsigned int) real_arg[1],
    (unsigned int) l2_weights,
    (unsigned int) l1_buffer,
    (unsigned int) &ram,
    (unsigned int) out_mult_in,
    0, 0,
    (unsigned int) out_shift_in};
  ${func_nameL3}(args
  );
}

void pulp_parallel(void *arg)
{
  printf("Prova fork\n");
% if n_tile_W > 1:
  l2_weights = rt_alloc(RT_ALLOC_L2_CL_DATA, 300000);
% endif
  l1_buffer = rt_alloc(RT_ALLOC_CL_DATA, ${buffer_l1_all});
  rt_team_fork(NUM_CORES, (void *)cluster_main, arg);
}

void network_run_FabricController()
{
  int arg[5];
  arg[0] = (unsigned int) x_test;
  arg[1] = (unsigned int) y_test;
  arg[2] = (unsigned int) L3_weights;
  arg[3] = (unsigned int) L3_activations;
  arg[4] = (unsigned int) activations_input;

  PMU_set_voltage(1000, 0);
  rt_time_wait_us(10000);
  rt_freq_set(RT_FREQ_DOMAIN_FC, 100000000);
  rt_time_wait_us(10000);
  rt_freq_set(RT_FREQ_DOMAIN_CL, 100000000);
  rt_time_wait_us(10000);
  rt_cluster_mount(1, 0, 0, NULL);
  rt_cluster_call(NULL, 0, pulp_parallel, arg, NULL,2048, 0, rt_nb_pe(), NULL);
  rt_cluster_mount(0, 0, 0, NULL);
}

// on fabric controller
int main () {
  network_setup();
  pi_ram_alloc(&ram, &L3_activations, (uint32_t) 2000000);
  network_run_FabricController();
  int checksum = 0;
  uint8_t *ptr = (uint8_t *) y_test;
% if n_tile_y == 1:
  for(int j=0; j<${activation_size_out * n_tile_W  * n_tile_x}; j++) {
% else:
  for(int j=0; j<${activation_size_out}; j++) {
% endif
    checksum += ptr[j];
  }

  if(check_sum_true == checksum)
    printf("Checksum in/out Layer :\tOk\n");
  else 
    printf("Checksum in/out Layer :\tFailed [%u vs. %u]\n", checksum, check_sum_true);

}
