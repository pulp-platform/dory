/*
 * mem_controller.c
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

#include "mem_controller.h"

/* allocation and de-allocation functions for manually manage L2 memory.
   The allocation in L2 is made in a bidirectional way inside an allocator.
   Based on begin_end parameter, the allocation/free happens in begin/end of the buffer.
*/

#define NULL (void*)0

static void *L2_begin = NULL;
static void *L2_end = NULL;

void dory_L2_mem_init(void *begin, int size) {
    L2_begin = begin;
    L2_end = begin + size;
}

void *dory_L2_alloc(int size, int dir) {
    void *retval = NULL;
    if (L2_begin + size < L2_end) {
        if (dir == 1) {
            retval = L2_begin;
            L2_begin += size;
        } else {
            L2_end -= size;
            retval = L2_end;
        }
    }
    return retval;
}

void dory_L2_free(int size, int dir) {
    if (dir == 1)
        L2_begin -= size;
    else
        L2_end += size;
}
