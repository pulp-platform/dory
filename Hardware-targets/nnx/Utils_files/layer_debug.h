#ifndef __LAYER_DEBUG_H__
#define __LAYER_DEBUG_H__

#ifdef DEBUG_TILE_CHECKSUM
#define TILE_CHECKSUM_PRINT(tile, i_tile)                 \ 
    do {                                                  \
        uint8_t *ptr = (uint8_t *)tile.addr.output;       \
        int sum = 0;                                      \
        for (int i = 0; i < tile.output.height; i++)      \
          for (int j = 0; j < tile.output.width; j++)     \
            for (int k = 0; k < tile.output.channel; k++) \
              sum += *ptr++;                              \
        printf("[%d] Checksum: %d\n", i_tile, sum);       \
    } while (0)
#else
#define TILE_CHECKSUM_PRINT(tile, i_tile)
#endif

#ifdef DEBUG_TILE_OUTPUT
#define TILE_OUTPUT_PRINT(tile, i_tile)                     \ 
    do {                                                    \
        printf("[%d] Output:\n", i_tile);                   \
        uint8_t *ptr = (uint8_t *)tile.addr.output;         \
        for (int i = 0; i < tile.output.height; i++) {      \
          for (int j = 0; j < tile.output.width; j++) {     \
            for (int k = 0; k < tile.output.channel; k++) { \
             printf("%d, ", *ptr++);                        \
            }                                               \
            printf("\n");                                   \
          }                                                 \
          printf("\n");                                     \
        }                                                   \
        printf("\n");                                       \
    } while (0)
#else
#define TILE_OUTPUT_PRINT(tile, i_tile)
#endif

#endif  // __LAYER_DEBUG_H__
