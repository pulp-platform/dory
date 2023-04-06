#ifndef __TILE_STATUS_H__
#define __TILE_STATUS_H__

#include <stdint.h>

typedef struct Address {
    uint32_t input;
    uint32_t weights;
    uint32_t scale;
    uint32_t bias;
    uint32_t output;
} Address;

typedef struct TileIndex {
    int height, width, output_channel;
} TileIndex;

typedef struct MemoryStatus {
    uint32_t addr_ext;
    int is_transfer;
    int buffer_index;
} MemoryStatus;

typedef struct TileStatus {
    TileIndex index;
    MemoryStatus input;
    MemoryStatus weights;
    MemoryStatus scale;
    MemoryStatus bias;
    MemoryStatus output;
} TileStatus;

static void tile_status_print(TileStatus status) {
    #define MEM_STATUS_ARGS(name) \
        status.name.addr_ext, \
        status.name.is_transfer, \
        status.name.buffer_index

    #define MEM_STATUS_STRING(name) \
        "   - " #name ":\n" \
        "     > addr_ext:%p\n" \
        "     > is_transfer:%d\n" \
        "     > buffer_index:%d\n"

    printf("TileStatus:\n"
           " * index:\n"
           "   - height: %d\n"
           "   - width: %d\n"
           "   - output_channel: %d\n"
           " * memory_status:\n"
           MEM_STATUS_STRING(input)
           MEM_STATUS_STRING(weights)
           MEM_STATUS_STRING(scale)
           MEM_STATUS_STRING(bias)
           MEM_STATUS_STRING(output),
           status.index.height, status.index.width, status.index.output_channel,
           MEM_STATUS_ARGS(input),
           MEM_STATUS_ARGS(weights),
           MEM_STATUS_ARGS(scale),
           MEM_STATUS_ARGS(bias),
           MEM_STATUS_ARGS(output));
}

typedef struct Padding {
    int top, bottom, right, left;
} Padding;

typedef struct Layer {
    Address addr;
    struct {
        int height;
        int width;
        int channel;
    } input, output;
    Padding padding;
} Layer;

static void layer_print(Layer layer) {
    printf("Layer:\n"
           " * addr:\n"
           "   - input: %p\n"
           "   - weights: %p\n"
           "   - scale: %p\n"
           "   - bias: %p\n"
           "   - output: %p\n"
           " * input:\n"
           "   - height: %d\n"
           "   - width: %d\n"
           "   - channel: %d\n"
           " * output:\n"
           "   - height: %d\n"
           "   - width: %d\n"
           "   - channel: %d\n"
           " * padding:\n"
           "   - top: %d\n"
           "   - bottom: %d\n"
           "   - left: %d\n"
           "   - right: %d\n",
           layer.addr.input, layer.addr.weights, layer.addr.scale, layer.addr.bias, layer.addr.output,
           layer.input.height, layer.input.width, layer.input.channel,
           layer.output.height, layer.output.width, layer.output.channel,
           layer.padding.top, layer.padding.bottom, layer.padding.left, layer.padding.right
           );
}

typedef struct Kernel {
    struct {
        int height, width;
    } shape, stride;
    int groups;
} Kernel;

static Layer tile_create(TileIndex index, TileIndex end_index, Layer body, Layer border, Layer layer, Address addr) {
#define SET_DIM(feature_set, dim, index_name) \
    .dim = index.index_name + 1 == end_index.index_name ? border.feature_set.dim : body.feature_set.dim

    return (Layer) {
        .addr = addr,
        .input = {
            SET_DIM(input, height, height),
            SET_DIM(input, width, width),
            SET_DIM(input, channel, output_channel)
        },
        .output = {
            SET_DIM(output, height, height),
            SET_DIM(output, width, width),
            SET_DIM(output, channel, output_channel)
        },
        .padding = {
            .top = index.height == 0 ? layer.padding.top : 0,
            .bottom = index.height + 1 == end_index.height ? layer.padding.bottom : 0,
            .right = index.width + 1 == end_index.width ? layer.padding.right : 0,
            .left = index.width == 0 ? layer.padding.left : 0
        }
    };
}

static TileIndex tile_index_get_next(TileIndex index, TileIndex end) {
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
    return index;
}

static TileIndex tile_index_get_next_reverse(TileIndex index, TileIndex end) {
    index.output_channel += 1;
    if (index.output_channel >= end.output_channel) {
        index.output_channel = 0;
        index.width += 1;
        if (index.width >= end.width) {
            index.width = 0;
            index.height += 1;
            if (index.height >= end.height) {
                index.height = 0;
            }
        }
    }
    return index;
}

static int buffer_index_get_next(int current, int is_transfer_next) {
    return is_transfer_next ? !current : current;
}

/** tile_status_get_next
*
* is_reverse_index - 0 (False): output_channel -> height -> width, 1 (True) height -> width -> output_channel.
*                    Normal (0) is the usual ordering. The reverse has the spatial (height, width) dimensions
*                    in the same order, but looped over first.
*/
static TileStatus tile_status_get_next(TileStatus current, TileIndex end_index, Layer layer, int is_reverse_index, Kernel kernel) {
    TileStatus next = { 0 };

    if (!is_reverse_index) {
        next.index = tile_index_get_next(current.index, end_index);
    } else {
        next.index = tile_index_get_next_reverse(current.index, end_index);
    }

    // Leans on tile loop going from outermost to innermost OUT_CH->H->W
    int is_first_input_again = next.index.height == 0 && next.index.width == 0 && next.index.output_channel == 0;
    int is_change_input = next.index.height != current.index.height || next.index.width != current.index.width
                          || (kernel.groups > 1 && next.index.output_channel != current.index.output_channel);
    next.input.is_transfer = !is_first_input_again && is_change_input;

    int is_last_weights = current.index.output_channel + 1 == end_index.output_channel;
    int is_change_weights = next.index.output_channel != current.index.output_channel;
    next.weights.is_transfer = !(is_last_weights && !is_reverse_index) && is_change_weights;
    next.scale.is_transfer = next.weights.is_transfer;
    next.bias.is_transfer  = next.weights.is_transfer;

    next.output.is_transfer = 1;

    next.input.addr_ext = current.input.addr_ext;
    next.weights.addr_ext = is_last_weights ? layer.addr.weights : current.weights.addr_ext;
    next.scale.addr_ext = is_last_weights ? layer.addr.scale : current.scale.addr_ext;
    next.bias.addr_ext = is_last_weights ? layer.addr.bias : current.bias.addr_ext;
    next.output.addr_ext = current.output.addr_ext;

#define UPDATE_BUFFER_INDEX(name) \
        next.name.buffer_index = \
            buffer_index_get_next(current.name.buffer_index, next.name.is_transfer);

    UPDATE_BUFFER_INDEX(input);
    UPDATE_BUFFER_INDEX(weights);
    UPDATE_BUFFER_INDEX(scale);
    UPDATE_BUFFER_INDEX(bias);
    UPDATE_BUFFER_INDEX(output);

    return next;
}

static Address tile_status_get_addr(TileStatus status, Address * const buffer_addresses) {
#define SET_ADDR(name) .name = buffer_addresses[status.name.buffer_index].name
    return (Address) {
        SET_ADDR(input),
        SET_ADDR(weights),
        SET_ADDR(scale),
        SET_ADDR(bias),
        SET_ADDR(output)
    };
}

#endif
