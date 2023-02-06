#ifndef __PERF_H__
#define __PERF_H__

#include "pmsis.h"


#define PERF_INIT() pi_perf_conf(1<<PI_PERF_CYCLES)

#define PERF_RESTART()             \
  do {                             \
    asm volatile("": : :"memory"); \
    pi_perf_stop();                \
    pi_perf_reset();               \
    pi_perf_start();               \
    asm volatile("": : :"memory"); \
  } while (0)

#define PERF_READ(var)                \
  asm volatile("": : :"memory");      \
  var = pi_perf_read(PI_PERF_CYCLES); \
  asm volatile("": : :"memory")

#define PERF_ACCUMULATE(accumulator, var) accumulator += var

#define PERF_ARRAY_DECLARATION(name) static int cycles_ ## name[TOTAL_TILES];

#define PERF_ARRAY_REPORT(name)          \
do {                                     \
  printf(#name " latency:\n");           \
  for (int i = 0; i < TOTAL_TILES; i++)  \
    printf("%d,\n", cycles_ ## name[i]); \
} while (0)


#ifdef MEASURE_LAYER_COMPONENTS

static int cycles_tile_create = 0;
static int cycles_load_input = 0;
static int cycles_load_weights = 0;
static int cycles_acc_init = 0;
static int cycles_tile_status_get_next = 0;
static int cycles_tile_create_next = 0;
static int cycles_load_input_next = 0;
static int cycles_load_weights_next = 0;
static int cycles_mem_wait = 0;
static int cycles_execute = 0;
static int cycles_store = 0;
static int cycles_tile_update = 0;
static int cycles_store_wait = 0;

#define PERF_LAYER_COMPONENT_INIT() \
do { \
    PERF_INIT(); \
    PERF_RESTART(); \
} while (0)

#define PERF_LAYER_COMPONENT_READ(component) \
  PERF_READ(cycles_ ## component);           \
  PERF_RESTART()

#define PERF_LAYER_COMPONENT_REPORT()  \
  do {                                 \
    printf("Measured time:\n" \
           "  - tile_create: %d\n" \
           "  - load_input: %d\n" \
           "  - load_weights: %d\n" \
           "  - acc_init: %d\n" \
           "  - tile_status_get_next: %d\n" \
           "  - tile_create_next: %d\n" \
           "  - load_input_next: %d\n" \
           "  - load_weights_next: %d\n" \
           "  - mem_wait: %d\n" \
           "  - execute: %d\n" \
           "  - store: %d\n" \
           "  - tile_update: %d\n" \
           "  - store_wait: %d\n", \
           cycles_tile_create, \
           cycles_load_input, \
           cycles_load_weights, \
           cycles_acc_init, \
           cycles_tile_status_get_next, \
           cycles_tile_create_next, \
           cycles_load_input_next, \
           cycles_load_weights_next, \
           cycles_mem_wait, \
           cycles_execute, \
           cycles_store, \
           cycles_tile_update, \
           cycles_store_wait); \
  } while (0)

#else  // MEASURE_LAYER_COMPONENTS

#define PERF_LAYER_COMPONENT_INIT()
#define PERF_LAYER_COMPONENT_READ(component)
#define PERF_LAYER_COMPONENT_REPORT()

#endif  // MEASURE_LAYER_COMPONENTS


#ifdef MEASURE_EXECUTION_COMPONENTS

typedef enum {
    exec_component_acquire,
    exec_component_dma_memcpy,
    exec_component_dma_barrier,
    n_exec_component
} exec_component_e;

static int cycles_exec_component[TOTAL_TILES][n_exec_component] = {0};

#define PERF_EXEC_COMPONENT_BEGIN(component)                               \
int cycles_ ## component ## _start = 0, cycles_ ## component ## _stop = 0; \
PERF_READ(cycles_ ## component ## _start)

#define PERF_EXEC_COMPONENT_END(component)                         \
PERF_READ(cycles_ ## component ## _stop);                          \
cycles_exec_component[i_tile][exec_component_ ## component] =      \
    cycles_ ## component ## _stop - cycles_ ## component ## _start

#define PERF_EXEC_COMPONENT_REPORT()              \
do {                                              \
  printf("Execution component report:\n");        \
  printf("acquire,dma_memcpy,dma_barrier\n");     \
  for (int i = 0; i < TOTAL_TILES; i++) {         \
    for (int j = 0; j < n_exec_component; j++) {  \
      printf("%d,", cycles_exec_component[i][j]); \
    }                                             \
    printf("\n");                                 \
  }                                               \
} while (0)

#else  // MEASURE_EXECUTION_COMPONENTS
//
#define PERF_EXEC_COMPONENT_BEGIN(component)
#define PERF_EXEC_COMPONENT_END(component)
#define PERF_EXEC_COMPONENT_REPORT()

#endif  // MEASURE_EXECUTION_COMPONENTS

#endif // __PERF_H__
