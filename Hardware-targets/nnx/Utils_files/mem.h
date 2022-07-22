#ifndef __MEM_H__
#define __MEM_H__

#include<stddef.h>

void  mem_init();
struct pi_device *get_ram_ptr();
void *ram_malloc(size_t size);
void  ram_free(ptr, size);
void  ram_read(dest, src, size);
void  ram_write(dest, src, size);
size_t load_file_to_ram(dest, filename, size);

#endif  // __MEM_H__
