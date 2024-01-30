#ifndef __PERF_UTILS_H__
#define __PERF_UTILS_H__
#include <stddef.h>
#include <stdint.h>
#include "monitor.h"

// Padding flags

#define NET_UTILS_PAD_TOP    (1 << 3)
#define NET_UTILS_PAD_RIGHT  (1 << 2)
#define NET_UTILS_PAD_BOTTOM (1 << 1)
#define NET_UTILS_PAD_LEFT   (1 << 0)
#define NET_UTILS_NO_PAD     (0)

typedef struct {
    Monitor input, output, store_conf;
} TaskMonitors;

typedef struct {
  unsigned int L3_input;
  unsigned int L3_output;
  unsigned int L3_after_weights;
  unsigned int L2_input;
  unsigned int bypass;
  unsigned int L2_output;
  unsigned int L2_weights;
  unsigned int L1_buffer;
  unsigned int ram;
  unsigned int padding;
  unsigned int layer_id;
  TaskMonitors *monitor;
} layer_args_t;

void print_perf(const char *name, const int cycles, const int macs);
void checksum(const char *name, const uint8_t *d, size_t size, uint32_t sum_true);
#endif
