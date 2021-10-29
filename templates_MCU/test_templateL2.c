/*
 * test_templateL2.c
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
#include "${func_name}.h"
#include "pulp.h"
#include "dory.h"
#include "mchan_test.h"

void ${func_name}(
  void *args
);

RT_L2_DATA uint8_t x_test[${activation_size_in}] = {
  ${x_content}
};
RT_L2_DATA uint8_t y_test[${activation_size_out}];
% if conv_order != 'PULP-NN-MAX':
RT_L2_DATA uint8_t W_test[${l2_dim_weights}] = {
  ${W_content}
};
% endif

% if ultra_test == True:
${type} y_expected[${activation_size_out}] = {
  ${y_expected_content}
};
% endif
  unsigned int out_mult_in = ${out_mul};
  unsigned int out_shift_in = ${out_shift};
int8_t * l1_buffer;
int check_sum_true = ${check_sum};
// on cluster
void cluster_main(void *arg) {
  int *real_arg = (int *) arg;
  unsigned int args[13] = {
    0, 0, 0,
    (unsigned int) real_arg[0],
    (unsigned int) 0,
    (unsigned int) real_arg[1],
% if conv_order != 'PULP-NN-MAX':
    (unsigned int) real_arg[2],
% else:
    (unsigned int) 0,
% endif
    (unsigned int) l1_buffer,
    (unsigned int) 0,
    (unsigned int) out_mult_in,
    (unsigned int) 0,
    (unsigned int) 0,
    (unsigned int) out_shift_in};
  ${func_name}(args
  );
}


void pulp_parallel(void *arg)
{
  printf("Prova fork\n");
    l1_buffer = rt_alloc(RT_ALLOC_CL_DATA,${l1_buffer});
  rt_team_fork(NUM_CORES, (void *)cluster_main, arg);
}

// on fabric controller
int main () {

  int arg[3];
  arg[0] = (unsigned int) x_test;
  arg[1] = (unsigned int) y_test;
% if conv_order != 'PULP-NN-MAX':
  arg[2] = (unsigned int) W_test;
% endif
  int error_presence;
  error_presence = 0;
  // rt_freq_set(RT_FREQ_DOMAIN_CL, 100*1000000);
  rt_cluster_mount(1, 0, 0, NULL);
  rt_cluster_call(NULL, 0, pulp_parallel, arg, NULL,1024, 0, rt_nb_pe(), NULL);
  rt_cluster_mount(0, 0, 0, NULL);
  int checksum = 0;
  char *ptr = (char *) y_test;
  for(int j=0; j<${activation_size_out}; j++) {
    checksum += ptr[j];
  }

  if(check_sum_true == checksum)
    printf("Checksum in/out Layer :\tOk\n");
  else 
    printf("Checksum in/out Layer :\tFailed [%u vs. %u]\n", checksum, check_sum_true);

% if ultra_test == True:
  for (int i=0; i<${activation_size_out}; i++) {
        if(y_test[i] != y_expected[i]) {
          error_presence = 1;
          printf(" %04x instead of %04x @ ch %d @ h*w_dim+w %d\n",(y_test[i]) & 0xffff, (y_expected[i]) & 0xffff, (i%${nof}),(i/${nof}));
        }
      }

  if (error_presence == 0)
  {
    printf("\n Test ${func_name} successful: no errors\n\n");
  }
%endif


}
