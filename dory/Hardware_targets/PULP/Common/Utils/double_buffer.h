#ifndef __DOUBLEBUFFER_H__
#define __DOUBLEBUFFER_H__

#include "pmsis.h"

typedef struct DoubleBuffer {
    uint32_t addrs[2];
    int index;
} DoubleBuffer;

static inline void double_buffer_increment(DoubleBuffer * db) {
    db->index = (db->index + 1) % 2;
}

static inline uint32_t double_buffer_get_addr(DoubleBuffer db) {
    return db.addrs[db.index];
}

#endif // __DOUBLEBUFFER_H__
