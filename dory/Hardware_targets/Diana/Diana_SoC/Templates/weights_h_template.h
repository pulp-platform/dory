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

#include <hal/pulp.h>
#include "pulp.h"

% for i in range(len(weights_vectors)):
% if weights_dimensions[i] > 0:
% if DORY_HW_graph[i].weight_bits == 2:
L2_DATA uint32_t Weights_${DORY_HW_graph[i].name}[${weights_dimensions[i]}] = {
% else:
L2_DATA uint8_t Weights_${DORY_HW_graph[i].name}[${weights_dimensions[i]}] = {
% endif
${weights_vectors[i]}};
% endif
% endfor