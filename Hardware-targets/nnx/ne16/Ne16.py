import numpy as np
import random
from ..Accelerator import Accelerator
from ..Util import *
from .Ne16PerfModel import Ne16PerfModel


class Ne16(Accelerator):
    TP_IN = 16
    TP_OUT = 32
    KS = 3
    INPUT_BUFFER_H = 5
    INPUT_BUFFER_W = 5
    OUTPUT_BUFFER_SHAPE = (3, 3, 32)

    @property
    def name(self):
        return 'ne16'

    def weights_ko_len(self, ko, dw):
        return div_and_ceil(ko, self.TP_IN) if dw else ko

    def weights_ki_size(self, ki, ks, qw, dw):
        if dw:
            return qw * ks[0] * ks[1] * (self.TP_IN // 8)
        else:
            return div_and_ceil(ki, self.TP_IN) * qw * ks[0] * ks[1] * (self.TP_IN // 8)

    def weights_size(self, ko, ki, ks, qw, dw):
        return self.weights_ko_len(ko, dw) * self.weights_ki_size(ki, ks, qw, dw)

    def heuristic_l2(self,
                     layer_in_shape, layer_out_shape,
                     tile_in_shape, tile_out_shape,
                     total_size, mem_size, ks=None, modifier=1000000):
        heuristics = [
            # Geometrical shape of tiles
            maximize_divisibility_w_prio(tile_out_shape[2], self.OUTPUT_BUFFER_SHAPE[2], prio=1),
            maximize_divisibility_w_prio(tile_out_shape[0], self.OUTPUT_BUFFER_SHAPE[0], prio=1.5),
            maximize_size_w_prio(tile_out_shape[2], max=layer_out_shape[2], prio=0.5),
            # Total dimension of tile
            maximize_size_w_prio(total_size, max=mem_size, prio=1)
        ]

        return sum([int(modifier * h["prio"]) * h["value"] for h in heuristics])

    def constraint_l1(self, layer_out_shape, tile_out_shape):
        return [(tile != layer) * (tile % buffer) == 0 \
            for layer, tile, buffer in zip(layer_out_shape, tile_out_shape, self.OUTPUT_BUFFER_SHAPE)]

    def heuristic_l1(self,
                     layer_in_shape, layer_out_shape,
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

        cin = layer_in_shape[2]

        return [
            # Balance out body and border tile size
            minimize_size_w_prio(tile_size(tile_out_shape, cin, ks) - tile_size(border_tile_out_shape, cin, ks),
                                 max=tile_size(layer_out_shape, cin, ks), prio=2),

            # Bigger latency -> more time to fetch data
            maximize_size_w_prio(ne16_model.latency, max=layer_latency, prio=1.5),

            # Bigger tile size
            maximize_size_w_prio(total_size, max=mem_size, prio=1)
        ]

    # assuming torch shapes, w must already be in uint format!
    # format --> [Ko, KiMajor, Qw, KiMinor] (binary tensor)
    #                          +++++++++++ --> these are *contiguous and packed*
    def conv1x1_unroll(self, w, qw, tp_in=16):
        Ko, Ki, H, W = w.shape
        nb_ki = (Ki // tp_in + (1 if Ki % tp_in != 0 else 0))
        wbytes = np.zeros((Ko * nb_ki * qw, 2), dtype=np.uint8)
        for ko in range(Ko):
            for ki in range(Ki):
                kimaj = ki // tp_in
                kimin = ki % tp_in
                byte = kimin // 8
                shift = kimin % 8
                for q in range(qw):
                    index = ko * nb_ki * qw + kimaj * qw + q
                    wbytes[index, byte] = np.bitwise_or(wbytes[index, byte],
                                                        1 << shift if w[ko, ki, 0, 0] & (1 << q) != 0 else 0)
        wbytes = wbytes.reshape(-1)
        return wbytes

    def conv1x1_roll(self, wbytes, qw, shape, layout='CoutCinK'):
        if layout == 'CoutCinK':
            Ko, Ki, H, W = shape
            w = np.zeros(shape, dtype=np.uint8)
            wv = w
        elif layout == 'CoutKCin':
            Ko, H, W, Ki = shape
            w = np.zeros(shape, dtype=np.uint8)
            wv = w.transpose((0, 3, 1, 2))
        else:
            raise Exception(f'Format {layout} not implemented.')

        nb_ki = (Ki // self.TP_IN + (1 if Ki % self.TP_IN != 0 else 0))
        for ko in range(Ko):
            for kimaj in range(nb_ki):
                for q in range(qw):
                    for kimin in range(self.TP_IN):
                        byte = kimin // 8
                        shift = kimin % 8
                        index = ko * nb_ki * qw * 2 + kimaj * qw * 2 + q * 2 + byte
                        if kimaj * self.TP_IN + kimin < Ki:
                            wv[ko, kimaj * self.TP_IN + kimin, 0, 0] += (1 & (wbytes[index] >> shift)) << q
        return w

    def subtile_bit_extract(self, subtile, bit_idx):
        retval = 0
        for i, el in enumerate(subtile):
            if el.item() & (1 << bit_idx):
                retval |= 1 << i
        return retval

    def conv3x3_unroll(self, w, qw):
        Ko, Ki, H, W = w.shape
        nb_ki = (Ki // self.TP_IN) + (1 if Ki % self.TP_IN != 0 else 0)
        nb_tp_in = self.TP_IN // 8
        wbytes = np.zeros((Ko, nb_ki, qw, H * W, nb_tp_in), dtype=np.uint8)
        for i in range(Ko):
            for j in range(nb_ki):
                tile = w[i, j * self.TP_IN:(j + 1) * self.TP_IN].transpose(1, 2, 0).reshape(H * W, -1)
                for k, subtile in enumerate(tile):
                    for bit in range(qw):
                        subtile_bit = self.subtile_bit_extract(subtile, bit)
                        for l in range(nb_tp_in):
                            wbytes[i, j, bit, k, l] = (subtile_bit >> (l * 8)) & 0xff
        wbytes = wbytes.reshape(-1)
        return wbytes

    def subtile_bit_roll(self, w_subtile, subtile, bit):
        s = 0
        for i, byte in enumerate(subtile):
            s += byte.item() << (i * 8)
        for i in range(w_subtile.size):
            w_subtile[i] += ((s & (1 << i)) >> i) << bit

    def conv3x3_roll(self, wbytes, qw, shape, format="CoutCinK"):
        if format == 'CoutCinK':
            Ko, Ki, H, W = shape
            w = np.zeros(shape, dtype=np.uint8)
            wv = w
        elif format == 'CoutKCin':
            Ko, H, W, Ki = shape
            w = np.zeros(shape, dtype=np.uint8)
            wv = w.transpose((0, 3, 1, 2))
        else:
            raise Exception(f'Format {format} not implemented.')

        nb_ki = (Ki // self.TP_IN) + (1 if Ki % self.TP_IN != 0 else 0)
        wbytes = wbytes.reshape(Ko, nb_ki, qw, H, W, 2)
        for i in range(Ko):
            for j in range(nb_ki):
                for bit in range(qw):
                    for k in range(H):
                        for l in range(W):
                            self.subtile_bit_roll(wv[i, j * self.TP_IN:(j + 1) * self.TP_IN, k, l].reshape(-1),
                                                  wbytes[i, j, bit, k, l], bit)
        return w

    def conv_unroll(self, w, qw, layout='CoutCinK', dw=False):
        if layout == "CoutCinK":
            if dw:
                w = w.transpose(1, 0, 2, 3)  # Swap Cout and Cin
        elif layout == "CoutKCin":
            if dw:
                w = w.transpose(3, 0, 1, 2)
            else:
                w = w.transpose(0, 3, 1, 2)
        elif layout == "CoutCin":
            w = w[:, :, np.newaxis, np.newaxis]
        elif layout == "CinCout":
            w = w.T
            w = w[:, :, np.newaxis, np.newaxis]
        else:
            raise Exception(f'Format {layout} not implemented.')

        fs = w.shape[2]

        if dw:
            assert fs == 3, "Only support filter size of 3 with depthwise convolution"
            assert w.shape[0] == 1, "Assumes that the Cout is equal to 1 in case of depthwise convolution"

        if fs == 1:
            return self.conv1x1_unroll(w, qw)
        elif fs == 3:
            return self.conv3x3_unroll(w, qw)


if __name__ == "__main__":

    def test(name, Ko, Ki, fs, qw):
        print(f'Test {name} shape=({Ko:3}, {Ki:3}, {fs}, {fs}) qw={qw}: ', end='', flush=True)
        shape = (Ko, Ki, fs, fs)
        test_in = np.random.randint(low=0, high=1 << qw, size=shape, dtype=np.uint8)
        test_out = globals()[f'conv{fs}x{fs}_roll'](globals()[f'conv_unroll'](test_in, qw), qw, shape)

        if not np.array_equal(test_in, test_out):
            print(f'Fail!')
            print('Test in:')
            print(test_in)
            print('Test out:')
            print(test_out)
            print(test_in[np.equal(test_in, test_out)])
        else:
            print(f'Success!')


    def test_generator(fs, test_count):
        print(f'Testing {fs}x{fs} convolution:')
        for i in range(test_count):
            Ko = random.randint(1, 128)
            Ki = random.randint(1, 128)
            qw = random.randint(2, 8)
            test(f'[{i}]', Ko, Ki, fs, qw)


    TEST_COUNT = 10

    test_generator(1, TEST_COUNT)
    test_generator(3, TEST_COUNT)
