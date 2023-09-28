from dory.Parsers.HW_node import HW_node


def div_and_ceil(a, b):
    return ((a - 1) // b) + 1


class Ne16_HW_node(HW_node):

    # Ne16's input throughput
    _TP_IN = 16

    def __init__(self, node, HW_description):
        self.weight_memory = self.calculate_weights_size(node.output_channels, node.input_channels, node.kernel_shape, node.weight_bits, node.group > 1)
        super().__init__(node, HW_description)

    def calculate_weights_size(self, channel_out, channel_in, kernel_shape, weight_bits, depthwise):
        if depthwise:
            return div_and_ceil(channel_out, self._TP_IN) * weight_bits * kernel_shape[0] * kernel_shape[1] * (self._TP_IN // 8)
        else:
            return channel_out * div_and_ceil(channel_in, self._TP_IN) * weight_bits * kernel_shape[0] * kernel_shape[1] * (self._TP_IN // 8)

    def calculate_weights_ko_len(self, ko, dw):
        return div_and_ceil(ko, self._TP_IN) if dw else ko

    def calculate_weights_ki_size(self, ki, ks, qw, dw):
        if dw:
            return qw * ks[0] * ks[1] * (self._TP_IN // 8)
        else:
            return div_and_ceil(ki, self._TP_IN) * qw * ks[0] * ks[1] * (self._TP_IN // 8)

    def create_tiling_dimensions(self, previous_node, config_file):
        super().create_tiling_dimensions(previous_node, config_file)
        # Fix weight memory size
        for level in range(1, self.HW_description["memory"]["levels"]):
            output_channels, input_channels = self.tiling_dimensions["L{}".format(level)]["weights_dimensions"]
            self.tiling_dimensions["L{}".format(level)]["weight_memory"] = self.calculate_weights_size(output_channels, input_channels, self.kernel_shape, self.weight_bits, self.group > 1)
