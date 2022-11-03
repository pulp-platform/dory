#ifndef __PERF_UTILS_H__
#define __PERF_UTILS_H__
#include <stddef.h>
#include <stdint.h>

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
  unsigned int out_mult;
  unsigned int out_shift;
  unsigned int layer_id;
} layer_args_t;

void print_perf(const char *name, const int cycles, const int macs);
void checksum(const char *name, const uint8_t *d, size_t size, uint32_t sum_true);
#endif
