#ifndef __TILE_INDEX_H__
#define __TILE_INDEX_H__

typedef struct TileIndex {
    int height, width, input_channel, output_channel;
} TileIndex;

/*
 * Index order:
 * output_channel -> height -> width -> input_channel
 */
static TileIndex tile_index_get_next(TileIndex index, TileIndex end) {
    index.input_channel += 1;
    if (index.input_channel >= end.input_channel) {
        index.input_channel = 0;
        index.width += 1;
        if (index.width >= end.width) {
            index.width = 0;
            index.height += 1;
            if (index.height >= end.height) {
                index.height = 0;
                index.output_channel += 1;
                if (index.output_channel >= end.output_channel) {
                    index.output_channel = 0;
                }
            }
        }
    }
    return index;
}

/*
 * Index order:
 * input_channel, output_channel -> height -> width
 */
static TileIndex tile_index_get_next_dw(TileIndex index, TileIndex end) {
    index.width += 1;
    if (index.width >= end.width) {
        index.width = 0;
        index.height += 1;
        if (index.height >= end.height) {
            index.height = 0;
            index.output_channel += 1;
            index.input_channel += 1;
            if (index.output_channel >= end.output_channel) {
                index.output_channel = 0;
            }
            if (index.input_channel >= end.input_channel) {
                index.input_channel = 0;
            }
        }
    }
    return index;
}

#endif // __TILE_INDEX_H__
