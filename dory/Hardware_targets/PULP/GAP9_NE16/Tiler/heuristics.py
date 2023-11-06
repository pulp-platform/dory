from .heuristic_util import maximize_divisibility_w_prio, maximize_size_w_prio, maximize_divisibility_or_max_w_prio, minimize_size_w_prio
from .Ne16PerfModel import Ne16PerfModel


TP_IN = 16
TP_OUT = 32
KS = 3
INPUT_BUFFER_H = 5
INPUT_BUFFER_W = 5
OUTPUT_BUFFER_SHAPE = (3, 3, 32)

def div_and_ceil(a, b):
    return ((a - 1) // b) + 1

def calculate_weights_size(channel_out, channel_in, kernel_shape, weight_bits, depthwise):
    if depthwise:
        return div_and_ceil(channel_out, TP_IN) * weight_bits * kernel_shape[0] * kernel_shape[1] * (TP_IN // 8)
    else:
        return channel_out * div_and_ceil(channel_in, TP_IN) * weight_bits * kernel_shape[0] * kernel_shape[1] * (TP_IN // 8)

def heuristic_total_size_l2(total_size, mem_size):
    return [
        # Bigger tile size
        maximize_size_w_prio(total_size, max=mem_size, prio=1)
    ]

def heuristic_tile_shape_l2(layer_in_shape, layer_out_shape,
                            tile_in_shape, tile_out_shape,
                            ks, qw, g, s):

    subtile_out_shape = (2 if s == [2, 2] else OUTPUT_BUFFER_SHAPE[0],
                         2 if s == [2, 2] else OUTPUT_BUFFER_SHAPE[1],
                         OUTPUT_BUFFER_SHAPE[2])

    return [
        # Geometrical shape of tiles
        maximize_divisibility_w_prio(tile_out_shape[2], subtile_out_shape[2], prio=4),
        maximize_divisibility_w_prio(tile_out_shape[0], subtile_out_shape[0], prio=5),
        maximize_size_w_prio(tile_out_shape[2], max=layer_out_shape[2], prio=0.5),
    ]

def heuristic_total_size_l1(total_size, mem_size):
    return [
        # Bigger tile size
        maximize_size_w_prio(total_size, max=mem_size, prio=0.5)
    ]

def heuristic_tile_shape_l1(layer_in_shape, layer_out_shape,
                            tile_in_shape, tile_out_shape,
                            border_tile_in_shape, border_tile_out_shape,
                            ks, qw, g, s):

    ne16_model = Ne16PerfModel('conv', ks, depthwise=g > 1, nq_bias=True)
    ne16_model.set_layer(layer_out_shape + (layer_in_shape[2], ))
    layer_latency = ne16_model.latency
    ne16_model.set_layer(tile_out_shape + (tile_in_shape[2], ))

    def mem_occupancy(in_shape, out_shape):
        def size(shape):
            return shape[0] * shape[1] * shape[2]

        return size(in_shape) + size(out_shape) + \
            calculate_weights_size(out_shape[2], in_shape[2], ks, qw, depthwise=g > 1)

    layer_size = mem_occupancy(layer_in_shape, layer_out_shape)
    tile_size = mem_occupancy(tile_in_shape, tile_out_shape)
    border_tile_size = mem_occupancy(border_tile_in_shape, border_tile_out_shape)

    subtile_out_shape = (2 if s == [2, 2] else OUTPUT_BUFFER_SHAPE[0],
                         2 if s == [2, 2] else OUTPUT_BUFFER_SHAPE[1],
                         OUTPUT_BUFFER_SHAPE[2])

    subtile_in_shape = (INPUT_BUFFER_H, INPUT_BUFFER_W, TP_IN)

    return [
        maximize_divisibility_or_max_w_prio(tile_out_shape[0], subtile_out_shape[0],
                                            max=layer_out_shape[0], prio=5),

        maximize_divisibility_or_max_w_prio(tile_out_shape[1], subtile_out_shape[1],
                                            max=layer_out_shape[1], prio=5),

        # Input channel has to be divisible with subtile shape and has higher priority then output channel
        maximize_divisibility_or_max_w_prio(tile_in_shape[2], subtile_in_shape[2],
                                            max=layer_in_shape[2], prio=3),

        maximize_divisibility_or_max_w_prio(tile_out_shape[2], subtile_out_shape[2],
                                            max=layer_out_shape[2], prio=5),

        # Balance out body and border tile size
        minimize_size_w_prio(tile_size - border_tile_size, max=layer_size, prio=3),

        # Bigger latency -> more time to fetch data
        maximize_size_w_prio(ne16_model.latency, max=layer_latency, prio=2),
    ]

def heuristic_l1(layer_in_shape, layer_out_shape,
                 tile_in_shape, tile_out_shape,
                 border_tile_in_shape, border_tile_out_shape,
                 total_size, mem_size, ks, g, s):
    ne16_model = Ne16PerfModel('conv', ks, depthwise=g>1, nq_bias=True)
    ne16_model.set_layer(layer_out_shape + (layer_in_shape[2], ))
    layer_latency = ne16_model.latency
    ne16_model.set_layer(tile_out_shape + (tile_in_shape[2], ))

    def tile_size(output_shape, cin, ks):
        input_size = output_shape[0] * output_shape[1] * cin
        output_size = output_shape[0] * output_shape[1] * output_shape[2]
        weights_size = ks[0] * ks[1] * cin * output_shape[2]
        return input_size + output_size + weights_size

    subtile_out_shape = (2 if s == [2, 2] else OUTPUT_BUFFER_SHAPE[0],
                         2 if s == [2, 2] else OUTPUT_BUFFER_SHAPE[1],
                         OUTPUT_BUFFER_SHAPE[2])

    subtile_in_shape = (INPUT_BUFFER_H, INPUT_BUFFER_W, TP_IN)

    cin = layer_in_shape[2]

    return [
        # TODO: Add heuristic that prefers more width tiles then height - less switching on borders
        maximize_divisibility_or_max_w_prio(tile_out_shape[0], subtile_out_shape[0],
                                            max=layer_out_shape[0], prio=5),

        maximize_divisibility_or_max_w_prio(tile_out_shape[1], subtile_out_shape[1],
                                            max=layer_out_shape[1], prio=5),

        # Input channel has to be divisible with subtile shape and has higher priority then output channel
        maximize_divisibility_or_max_w_prio(tile_in_shape[2], subtile_in_shape[2],
                                            max=layer_in_shape[2], prio=3),

        maximize_divisibility_or_max_w_prio(tile_out_shape[2], subtile_out_shape[2],
                                            max=layer_out_shape[2], prio=1),

        # Balance out body and border tile size
        minimize_size_w_prio(tile_size(tile_out_shape, cin, ks) - tile_size(border_tile_out_shape, cin, ks),
                             max=tile_size(layer_out_shape, cin, ks), prio=2),

        # Bigger latency -> more time to fetch data
        maximize_size_w_prio(ne16_model.latency, max=layer_latency, prio=5),

        # Bigger tile size
        maximize_size_w_prio(total_size, max=mem_size, prio=0.5)
    ]
