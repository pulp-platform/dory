#include "mem.h"
#include "pmsis.h"
#include "bsp/bsp.h"
#include "bsp/fs.h"
#include "bsp/fs/readfs.h"
#include "bsp/flash.h"
#include "bsp/ram.h"

#ifdef USE_HYPERFLASH
#include "bsp/flash/hyperflash.h"
typedef struct pi_hyperflash_conf flash_conf_t;
#define flash_conf_init(conf) pi_hyperflash_conf_init(conf)
#elif defined USE_SPIFLASH
#include "bsp/flash/spiflash.h"
typedef struct pi_mx25u51245g_conf flash_conf_t;
#define flash_conf_init(conf) pi_mx25u51245g_conf_init(conf)
#elif defined USE_MRAM
typedef struct pi_mram_conf flash_conf_t;
#define flash_conf_init(conf) pi_mram_conf_init(conf)
#else
typedef struct pi_default_flash_conf flash_conf_t;
#define flash_conf_init(conf) pi_default_flash_conf_init(conf)
#endif

#ifdef USE_HYPERRAM
#include "bsp/ram/hyperram.h"
typedef struct pi_hyper_conf ram_conf_t;
#define ram_conf_init(conf) pi_hyperram_conf_init(conf)
#else
typedef struct pi_default_ram_conf ram_conf_t;
#define ram_conf_init(conf) pi_default_ram_conf_init(conf)
#endif

#define BUFFER_SIZE 128
static uint8_t buffer[BUFFER_SIZE];

static struct pi_device flash;
static flash_conf_t flash_conf;

static struct pi_device fs;
static struct pi_readfs_conf fs_conf;

static struct pi_device ram;
static ram_conf_t ram_conf;

void mem_init() {
  flash_conf_init(&flash_conf);
#ifdef USE_MRAM
  if (pi_pmu_voltage_get(PI_PMU_VOLTAGE_DOMAIN_CHIP) == 800)
    flash_conf.baudrate = 37000000;
  else
    flash_conf.baudrate = 34000000;
#endif
  pi_open_from_conf(&flash, &flash_conf);
  if (pi_flash_open(&flash)) {
    printf("ERROR: Cannot open flash! Exiting...\n");
    pmsis_exit(-1);
  }

  /* Open filesystem on flash. */
  pi_readfs_conf_init(&fs_conf);
  fs_conf.fs.flash = &flash;
  pi_open_from_conf(&fs, &fs_conf);
  if (pi_fs_mount(&fs)) {
    printf("ERROR: Cannot mount filesystem! Exiting...\n");
    pmsis_exit(-2);
  }

  ram_conf_init(&ram_conf);
  pi_open_from_conf(&ram, &ram_conf);
  if (pi_ram_open(&ram)) {
    printf("ERROR: Cannot open ram! Exiting...\n");
    pmsis_exit(-3);
  }
}

struct pi_device *get_ram_ptr() {
 return &ram;
}

void *ram_malloc(size_t size) {
  void *ptr = NULL;
  pi_ram_alloc(&ram, &ptr, size);
  return ptr;
}

void ram_free(void *ptr, size_t size) {
  pi_ram_free(&ram, ptr, size);
}

void ram_read(void *dest, void *src, const size_t size) {
  if (size > 1 << 19) {
    int n_reads = size >> 19;
    int read_size = 1 << 19;
    int read_rem = size & ((1 << 19) - 1);
    for (int i = 0; i < n_reads; i++) {
      pi_ram_read(&ram, src, dest, 1 << 19);
      src += read_size;
      dest += read_size;
    }
    pi_ram_read(&ram, src, dest, read_rem);
  } else {
    pi_ram_read(&ram, src, dest, size);
  }
}

void ram_write(void *dest, void *src, const size_t size) {
  pi_ram_write(&ram, dest, src, size);
}

size_t load_file_to_ram(void *dest, const char *filename) {
  pi_fs_file_t *fd = pi_fs_open(&fs, filename, 0);
  if (fd == NULL) {
    printf("ERROR: Cannot open file %s! Exiting...", filename);
    pmsis_exit(-4);
  }

  const size_t size = fd->size;

  size_t offset = 0;
  do {
    const size_t read_bytes = pi_fs_read(fd, buffer, BUFFER_SIZE);
    ram_write(dest + offset, buffer, read_bytes);
    offset += read_bytes;
  } while (offset < size);

  return offset;
}
