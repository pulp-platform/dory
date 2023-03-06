#ifndef __LAYER_DEBUG_H__
#define __LAYER_DEBUG_H__

#define FEATURE_MAP_CHECKSUM(addr, dims, i_tile)    \
    do {                                            \
        uint8_t *ptr = (uint8_t *)addr;             \
        int sum = 0;                                \
        for (int i = 0; i < dims.height; i++)       \
          for (int j = 0; j < dims.width; j++)      \
            for (int k = 0; k < dims.channel; k++)  \
              sum += *ptr++;                        \
        printf("[%d] Checksum: %d\n", i_tile, sum); \
    } while (0)

#define DMA_CHECKSUM_PRINT(conf, i_tile)                     \
    do {                                                     \
        uint8_t *ptr = (uint8_t *)conf.loc;                  \
        int sum = 0;                                         \
        for (int i = 0; i < conf.number_of_2d_copies; i++)   \
          for (int j = 0; j < conf.number_of_1d_copies; j++) \
            for (int k = 0; k < conf.length_1d_copy; k++)    \
              sum += *ptr++;                                 \
        printf("[%d] Checksum: %d\n", i_tile, sum);          \
    } while (0)

#define FEATURE_MAP_PRINT(name, addr, dims, i_tile) \
    do {                                            \
        printf("[%d] " #name ":\n", i_tile);        \
        uint8_t *ptr = (uint8_t *)addr;             \
        for (int i = 0; i < dims.height; i++)       \
          for (int j = 0; j < dims.width; j++)      \
            for (int k = 0; k < dims.channel; k++)  \
             printf("%d\n", *ptr++);                \
    } while (0)

#define DMA_FEATURE_MAP_PRINT(name, conf, i_tile)            \
    do {                                                     \
        printf("[%d] " #name ":\n", i_tile);                 \
        uint8_t *ptr = (uint8_t *)conf.loc;                  \
        for (int i = 0; i < conf.number_of_2d_copies; i++)   \
          for (int j = 0; j < conf.number_of_1d_copies; j++) \
            for (int k = 0; k < conf.length_1d_copy; k++)    \
             printf("%d\n", *ptr++);                         \
    } while (0)

#endif  // __LAYER_DEBUG_H__
