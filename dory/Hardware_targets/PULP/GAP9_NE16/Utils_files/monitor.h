#ifndef __MONITOR_H__
#define __MONITOR_H__

#include <stdint.h>


typedef struct Monitor {
    uint32_t empty;
    uint32_t full;
} Monitor;


int monitor_init(Monitor * const monitor, int buffer_size);
void monitor_term(Monitor monitor);
void monitor_produce_begin(Monitor monitor);
void monitor_produce_end(Monitor monitor);
void monitor_consume_begin(Monitor monitor);
void monitor_consume_end(Monitor monitor);

#endif // __MONITOR_H__
