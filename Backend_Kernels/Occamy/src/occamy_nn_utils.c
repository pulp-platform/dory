/*
 * pulp_nn_utils.h
 * Nazareno Bruschi <nazareno.bruschi@unibo.it>
 * Angelo Garofalo <angelo.garofalo@unibo.it>
 *
 * Copyright (C) 2018-2020 University of Bologna
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
#include "stdint.h"
#include "../../vendor/riscv-opcodes/encoding.h"



float __attribute__((always_inline)) pulp_nn_bn_quant(
  float phi,
  float k,
  float lambda,
  int8_t  d
) {
  double integer_image_phi = ((double) k * phi) + (double) lambda;
  double x = (integer_image_phi) / d;
  float res = 0;
  if (x < 0)
  	res = 0;
  else if (x > 255)
  	res = 255;
  else
  	res = (int) x;
  return res;
}
uint32_t benchmark_get_cycle() { 
  return read_csr(mcycle); 
}