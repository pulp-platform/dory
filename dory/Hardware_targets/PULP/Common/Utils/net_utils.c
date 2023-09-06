#include "net_utils.h"
#include "pmsis.h"

void print_perf(const char *name, int cycles, int macs) {
  float macs_float = (float) macs;
  float cycles_float = (float) cycles;

  float perf = macs_float / cycles_float;
  printf("\n%s performance:\n", name);
  printf("  - num cycles: %d\n", cycles);
  printf("  - MACs: %d\n", macs );
  printf("  - MAC/cycle: %.3f\n", perf);
  printf("  - n. of Cores: %d\n\n", NUM_CORES);
}

void checksum(const char *name, const uint8_t *d, size_t size, uint32_t sum_true) {
  uint32_t sum = 0;
  for (int i = 0; i < size; i++) sum += d[i];

  printf("Checking %s: Checksum ", name);
  if (sum_true == sum)
    printf("OK\n");
  else
    printf("Failed: true [%u] vs. calculated [%u]\n", sum_true, sum);
}

