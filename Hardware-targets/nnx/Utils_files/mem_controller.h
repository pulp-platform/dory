/*
 * mem_controller.h
 * Alessio Burrello <alessio.burrello@unibo.it>
 * Luka Macan <luka.macan@unibo.it>
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

#ifndef __MEM_CONTROLLER_H__
#define __MEM_CONTROLLER_H__

void  dory_L2_mem_init(void *begin, int size);
void *dory_L2_alloc(int size, int dir);
void  dory_L2_free(int size, int dir);

#endif  // __MEM_CONTROLLER_H__
