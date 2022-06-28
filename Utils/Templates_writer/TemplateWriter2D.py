from Utils.Templates_writer.TemplateWriter import TemplateWriter
import re
import math


class TemplateWriter2D_L3(TemplateWriter):
    def __init__(self, node):
        super().__init__(node)

        ks = node.kernel_shape
        s = node.strides
        g = node.group
        p = node.pads
        padding_top = p[0];
        padding_left = p[1];
        padding_bottom = p[2];
        padding_right = p[3];
        conv_overlap1 = 2 * (ks[0] // 2) + ks[0] % 2 - 1 - (s[0] - 1)
        conv_overlap2 = 2 * (ks[1] // 2) + ks[1] % 2 - 1 - (s[1] - 1)
        self.flag_DW = 1 if node.group > 1 else 0

        ################## NEED A REWRITING IN THIS TEMPLATE PART ######################
        #### VARIABLE CREATION FOR COMPATIBILITY WITH THE SECTION AFTER ################
        n_in = node.tiling_dimensions["L3"]["input_dimensions"][0]
        h_in = node.tiling_dimensions["L3"]["input_dimensions"][1]
        w_in = node.tiling_dimensions["L3"]["input_dimensions"][2]
        ds_x = node.input_activation_bits

        n_out = node.tiling_dimensions["L3"]["output_dimensions"][0]
        h_out = node.tiling_dimensions["L3"]["output_dimensions"][1]
        w_out = node.tiling_dimensions["L3"]["output_dimensions"][2]
        ds_y = node.output_activation_bits
        ds_act = node.constant_bits

        fs1 = node.kernel_shape[0]
        fs2 = node.kernel_shape[1]
        ds_W = node.weight_bits

        n_in_L2 = node.tiling_dimensions["L2"]["input_dimensions"][0]
        if node.tiling_dimensions["L3"]["output_dimensions"][1] > node.tiling_dimensions["L2"]["output_dimensions"][1]:
            h_in_L2 = node.tiling_dimensions["L2"]["output_dimensions"][1] * s[0] + (ks[0] - 1) - (s[0] - 1)
        else:
            h_in_L2 = node.tiling_dimensions["L2"]["input_dimensions"][1]
        w_in_L2 = node.tiling_dimensions["L2"]["input_dimensions"][2]

        if "Addition" not in node.name and "Pool" not in node.name:
            n_out_L2 = node.tiling_dimensions["L2"]["weights_dimensions"][0]
        else:
            n_out_L2 = node.tiling_dimensions["L2"]["output_dimensions"][0]
        if node.tiling_dimensions["L3"]["input_dimensions"][1] > node.tiling_dimensions["L2"]["input_dimensions"][1]:
            h_out_L2 = int(
                math.floor((node.tiling_dimensions["L2"]["input_dimensions"][1] - (ks[0] - 1) + (s[0] - 1)) / s[0]))
        else:
            h_out_L2 = node.tiling_dimensions["L2"]["output_dimensions"][1]
        w_out_L2 = node.tiling_dimensions["L2"]["output_dimensions"][2]

        ################################################################################

        self.conv_overlap1 = conv_overlap1
        self.conv_overlap2 = conv_overlap2
        self.padding = padding_top
        if (node.tiling_dimensions["L3"]["input_dimensions"] != node.tiling_dimensions["L2"]["input_dimensions"]):
            self.input_L3 = 1
            factor_h_in = int(h_out / h_out_L2)
        else:
            self.input_L3 = 0
            factor_h_in = 1
        factor_h_out = int(
            node.tiling_dimensions["L3"]["output_dimensions"][1] / node.tiling_dimensions["L2"]["output_dimensions"][1])
        if not isinstance(node.tiling_dimensions["L2"]["weights_dimensions"], type(None)):
            factor_ch_out = int(node.tiling_dimensions["L3"]["weights_dimensions"][0] /
                                node.tiling_dimensions["L2"]["weights_dimensions"][0])
        else:
            factor_ch_out = 1
        self.n_tile_W = factor_ch_out
        self.n_tile_x = factor_h_in
        self.n_tile_y = factor_h_out
        self.verbose = False
        if self.padding > 0:
            self.func_name = [node.name + "_L2", node.name + "_L2_p_t", node.name + "_L2_p_b"]
        else:
            self.func_name = [node.name + "_L2"]
        self.func_name_L3 = node.name
        self.BitIn = ds_x
        self.y_data_size_byte = ds_y
        self.x_data_size_byte = ds_x
        self.w_out = w_out_L2
        self.h_out = h_out_L2
        self.n_out = n_out_L2
        self.w_in = w_in_L2
        self.h_in = h_in_L2
        self.n_in = n_in_L2
        self.weight_dim = int(node.tiling_dimensions["L2"]["weight_memory"])
        self.has_bias = int(len([1 for name in node.constant_names if "bias" in name]) > 0)
        if self.has_bias == 1:
            self.bias_dim = node.tiling_dimensions["L2"]["bias_memory"]
        else:
            self.bias_dim = 0
        if not isinstance(node.tiling_dimensions["L2"]["constants_memory"], type(None)):
            self.lambda_dim = int(node.tiling_dimensions["L2"]["constants_memory"] / 2)
            self.k_dim = int(node.tiling_dimensions["L2"]["constants_memory"] / 2)
        else:
            self.lambda_dim = 0
            self.k_dim = 0
        self.dim_out = int(n_out_L2 * w_out_L2 * h_out_L2 * node.output_activation_bits / 8)
        self.dim_in = int(n_in_L2 * w_in_L2 * h_in_L2 * node.input_activation_bits / 8)


class TemplateWriter2D_L2(TemplateWriter):
    def __init__(self, node):
        super().__init__(node)

        ks = node.kernel_shape
        inp_dim = node.tiling_dimensions["L2"]["input_dimensions"][1:]
        out_dim = node.tiling_dimensions["L2"]["output_dimensions"][1:]
        in_ch = node.tiling_dimensions["L2"]["input_dimensions"][0]
        s = node.strides
        g = node.group
        p = node.pads
        conv_overlap_h = 2 * (ks[0] // 2) + ks[0] % 2 - 1 - (s[0] - 1)
        padding_top = p[0]
        padding_left = p[1]
        padding_bottom = p[2]
        padding_right = p[3]
        conv_overlap1 = 2 * (ks[0] // 2) + ks[0] % 2 - 1 - (s[0] - 1)
        conv_overlap2 = 2 * (ks[1] // 2) + ks[1] % 2 - 1 - (s[1] - 1)

        # TODO what is this??
        if re.search('.0', node.name):
            try:
                int(re.search('.0', node.name).group())
                self.first_layer = 0
            except ValueError:
                self.first_layer = 1
        else:
            self.first_layer = 0
        self.sdk = node.HW_description["software development kit"]["name"]
        self.number_of_clusters = node.HW_description[
            "number_of_clusters"] if "number_of_clusters" in node.HW_description.keys() else 1
        #self.optional_type = layer_type
        self.func_name = node.name
        self.flag_DW = 1 if node.group > 1 else 0
        self.optional = node.op_type
        self.FLAG_BATCHNORM = 1 if 'k' in node.constant_names else 0
        self.has_bias = int(len([1 for name in node.constant_names if "bias" in name]) > 0)
        self.FLAG_RELU = 1 if 'outshift' in node.constant_names else 0
        self.type = "char" if node.input_activation_type in ["int", "uint"] else "float"
        self.conv_overlap1 = conv_overlap1
        self.conv_overlap2 = conv_overlap2
        self.padding_top = padding_top
        self.padding_bottom = padding_bottom
        self.padding_left = padding_left
        self.padding_right = padding_right
        self.stride = s[0]

        ################## NEED A REWRITING IN THIS TEMPLATE PART ######################
        #### VARIABLE CREATION FOR COMPATIBILITY WITH THE SECTION AFTER ################
        if self.flag_DW == 0:
            self.g = 1
            self.nif = node.tiling_dimensions["L2"]["input_dimensions"][0]
        else:
            self.g = node.tiling_dimensions["L2"]["input_dimensions"][0]
            self.nif = 1
        n_in = node.tiling_dimensions["L2"]["input_dimensions"][0]
        h_in = node.tiling_dimensions["L2"]["input_dimensions"][1]
        w_in = node.tiling_dimensions["L2"]["input_dimensions"][2]
        tile_n_in = node.tiling_dimensions["L1"]["input_dimensions"][0]
        tile_h_in = node.tiling_dimensions["L1"]["input_dimensions"][1]
        tile_w_in = node.tiling_dimensions["L1"]["input_dimensions"][2]

        if "Addition" not in node.name and "Pool" not in node.name:
            n_out = node.tiling_dimensions["L2"]["weights_dimensions"][0]
        else:
            n_out = node.tiling_dimensions["L2"]["output_dimensions"][0]
        h_out = node.tiling_dimensions["L2"]["output_dimensions"][1]
        w_out = node.tiling_dimensions["L2"]["output_dimensions"][2]
        tile_n_out = node.tiling_dimensions["L1"]["output_dimensions"][0]
        tile_h_out = node.tiling_dimensions["L1"]["output_dimensions"][1]
        tile_w_out = node.tiling_dimensions["L1"]["output_dimensions"][2]

        fs1 = node.kernel_shape[0]
        fs2 = node.kernel_shape[1]

        ds_x = node.input_activation_bits
        ds_y = node.output_activation_bits
        ds_act = node.constant_bits
        ds_W = node.weight_bits
        ds_bias = node.bias_bits

        dt_x = node.input_activation_type
        dt_y = node.output_activation_type
        dt_act = node.constant_type
        dt_W = node.weight_type

        if "Addition" in node.name:
            ds_x2 = node.input_activation_bits
            dt_x2 = node.input_activation_type
            self.data_type_x2 = dt_x2
            self.x_data_size_byte2 = ds_x2
            self.inmul1 = node.inmul1["value"]
            self.inadd1 = node.inadd1["value"]
            self.inshift1 = node.inshift1["value"]
            self.inmul2 = node.inmul2["value"]
            self.inadd2 = node.inadd2["value"]
            self.inshift2 = node.inshift2["value"]
            self.outmul = node.outmul["value"]
            self.outadd = node.outadd["value"]
            self.outshift = node.outshift["value"]

        DW = self.flag_DW
        has_bias = self.has_bias
        number_of_clusters = self.number_of_clusters

        self.data_type_x = dt_x
        self.data_type_y = dt_y
        self.data_type_activations = dt_act
        self.data_type_weights = dt_W
        ################################################################################

        self.nof = n_out
        self.factor = node.tiling_dimensions["L3"]["output_dimensions"][0] / n_out
        # x parameters
        self.x_h = h_in
        self.x_w = w_in
        self.x_data_size_byte = node.input_activation_bits
        self.x_tile_size_nif = tile_n_in
        self.x_tile_size_h = tile_h_in
        self.x_tile_size_w = tile_w_in
        self.x_tile_size_byte = int(math.ceil(ds_x * tile_n_in * tile_h_in * tile_w_in / 8.0))
        self.x_tile_size_nif_byte = int(math.ceil(tile_n_in * ds_x / 8.0))
        self.x_stride_w_byte = int(math.ceil(w_in * n_in * ds_x / 8.0))
        self.x_stride_c_byte = int(math.ceil(n_in * ds_x / 8.0))
        # y parameters
        self.y_h = h_out
        self.y_w = w_out
        self.y_data_size_byte = ds_y
        self.act_dim_bit = ds_act
        self.y_tile_size_nof = tile_n_out if (n_out > tile_n_out) else n_out
        self.y_tile_size_h = tile_h_out if (h_out > tile_h_out) > 0 else h_out
        self.y_tile_size_w = tile_w_out if (w_out > tile_w_out) > 0 else w_out
        self.y_tile_size_byte = int(
            math.ceil(self.y_tile_size_nof * self.y_tile_size_h * self.y_tile_size_w * ds_y / 8.0))
        self.y_stride_w_byte = int(math.ceil(w_out * n_out * self.factor * ds_y / 8.0))
        self.y_stride_c_byte = int(math.ceil(n_out * self.factor * ds_y / 8.0))
        self.y_tile_size_nof_byte = int(math.ceil(tile_n_out * ds_y / 8.0))

        self.tile_dim_h = max(int(math.ceil(float(h_out) / float(self.y_tile_size_h))), 1)
        self.tile_dim_w = max(int(math.ceil(float(w_out) / float(self.y_tile_size_w))), 1)
        self.tile_dim_nof = max(int(math.ceil(float(n_out) / float(self.y_tile_size_nof))), 1)
        self.tile_dim_nif = max(int(math.ceil(float(n_in) / float(tile_n_in))), 1)
        self.tile_n_in_last = n_in % tile_n_in if n_in % tile_n_in > 0 else tile_n_in
        # W parameters
        self.fs1 = fs1
        self.fs2 = fs2
        self.W_data_size = ds_W
        self.W_tile_size_nof = tile_n_out
        if self.has_bias == 1:
            self.b_size_byte = int(math.ceil(n_out * ds_bias / 8.0))
        else:
            self.b_size_byte = 0

        if DW == 0:
            self.W_tile_size_nif = tile_n_in * self.tile_dim_nif
            self.W_tile_size_nif_last = self.tile_n_in_last * self.tile_dim_nif
        else:
            self.W_tile_size_nif = 1
            self.W_tile_size_nif_last = 1
        if "Addition" not in node.name and "Pool" not in node.name:
            self.W_tile_size_byte = int(math.ceil(tile_n_out * self.W_tile_size_nif * fs1 * fs2 * ds_W / 8.0))
            if DW == 0:
                self.W_stride_nof_byte = int(math.ceil(self.nif * fs1 * fs2 * ds_W / 8.0))
            else:
                self.W_stride_nof_byte = int(math.ceil(self.nif * fs1 * fs2 * ds_W / 8.0))
            self.W_stride_hw_byte = int(math.ceil(self.nif * ds_W / 8.0))
            self.W_tile_nif_byte = int(math.ceil(self.W_tile_size_nif * ds_W / 8.0))
            self.W_tile_nif_byte_last = int(math.ceil(self.W_tile_size_nif_last * ds_W / 8.0))
        # l2 parameters
        if self.FLAG_BATCHNORM == 1:
            self.l2_off_k = int(
                math.ceil(self.nof * self.nif * fs1 * fs2 * ds_W / 8.0 + self.b_size_byte))
            self.l2_off_lambda = int(
                math.ceil((self.nof * self.nif * fs1 * fs2 * ds_W + self.nof * ds_act) / 8.0 + self.b_size_byte))
        if has_bias == 1:
            self.l2_off_bias = int(math.ceil(self.nof * self.nif * fs1 * fs2 * ds_W / 8.0))
        if n_in == tile_n_in and w_in == tile_w_in and h_in == tile_h_in:
            x_buffer_size = int(math.ceil(ds_x * tile_n_in * tile_h_in * tile_w_in / 8.0))
        else:
            x_buffer_size = 2 * int(math.ceil(ds_x * tile_n_in * tile_h_in * tile_w_in / 8.0))
            if x_buffer_size % 16 != 0:
                x_buffer_size = x_buffer_size
        if (n_in == (tile_n_in * number_of_clusters) and w_in == tile_w_in and h_in == tile_h_in and n_out == (
                tile_n_out * number_of_clusters) and n_in > number_of_clusters) \
                or (n_in == tile_n_in and w_in == tile_w_in and h_in == tile_h_in and n_out == (
                tile_n_out * number_of_clusters)):
            y_buffer_size = int(math.ceil(ds_y * self.y_tile_size_nof * self.y_tile_size_h * self.y_tile_size_w / 8.0))
            if "Addition" not in node.name and "Pool" not in node.name:
                if DW == 0:
                    W_buffer_size = int(math.ceil(ds_W * self.y_tile_size_nof * self.W_tile_size_nif * fs1 * fs2 / 8.0))
                else:
                    W_buffer_size = int(math.ceil(ds_W * self.y_tile_size_nof * 1 * fs1 * fs2 / 8.0))
            else:
                W_buffer_size = 0
        else:
            y_buffer_size = 2 * int(
                math.ceil(ds_y * self.y_tile_size_nof * self.y_tile_size_h * self.y_tile_size_w / 8.0))
            if "Addition" not in node.name and "Pool" not in node.name:
                if DW == 0:
                    W_buffer_size = 2 * int(
                        math.ceil(ds_W * self.y_tile_size_nof * self.W_tile_size_nif * fs1 * fs2 / 8.0))
                else:
                    W_buffer_size = 2 * int(math.ceil(ds_W * self.y_tile_size_nof * 1 * fs1 * fs2 / 8.0))
            else:
                W_buffer_size = 0
        if self.FLAG_BATCHNORM == 1:
            k_buffer_size = int(n_out * ds_act / 8.0)
            lambd_buffer_size = int(n_out * ds_act / 8.0)
        else:
            k_buffer_size = 0
            lambd_buffer_size = 0

        self.k_tile_size_byte = 0
        self.lambda_tile_size_byte = 0
        self.k_size_byte = 0
        self.lambda_size_byte = 0
        if "Pool" not in node.name:
            if self.FLAG_BATCHNORM == 1:
                self.k_size_byte = k_buffer_size
                self.lambda_size_byte = k_buffer_size
                self.k_tile_size_byte_transfer = int(math.ceil(tile_n_out * ds_act / 8.0))
                self.lambda_tile_size_byte_transfer = int(math.ceil(tile_n_out * ds_act / 8.0))
                if n_in == tile_n_in and w_in == tile_w_in and h_in == tile_h_in and n_out == tile_n_out:
                    self.k_tile_size_byte = int(math.ceil(tile_n_out * ds_act / 8.0))
                    self.lambda_tile_size_byte = int(math.ceil(tile_n_out * ds_act / 8.0))
                else:
                    self.k_tile_size_byte = int(math.ceil(tile_n_out * ds_act / 8.0 * 2))
                    self.lambda_tile_size_byte = int(math.ceil(tile_n_out * ds_act / 8.0 * 2))
            if has_bias == 1:
                self.bias_tile_size_byte = tile_n_out * int(ds_bias / 8.0)
                self.b_size_byte = int(n_out) * int(ds_bias / 8.0)
            else:
                self.bias_tile_size_byte = 0
                self.b_size_byte = 0

        # l1 parameters
        self.l1_x_offset = 0
        self.l1_y_offset = x_buffer_size + 8
        if "Addition" in node.name:
            self.l1_x2_offset = x_buffer_size + 8 + y_buffer_size + 8
        if "Addition" not in node.name and "Pool" not in node.name:
            self.l1_W_offset = x_buffer_size + 8 + y_buffer_size + 8
            if self.FLAG_BATCHNORM == 1:
                self.l1_k_offset = x_buffer_size + 8 + y_buffer_size + 8 + W_buffer_size + 8
                self.l1_lambda_offset = x_buffer_size + 8 + y_buffer_size + 8 + W_buffer_size + 8 + self.k_tile_size_byte + 8
            if has_bias == 1:
                self.l1_b_offset = x_buffer_size + 8 + y_buffer_size + 8 + W_buffer_size + 8 + self.k_tile_size_byte + 8 + \
                                    self.lambda_tile_size_byte + 8

        # W last
        if "Addition" not in node.name and "Pool" not in node.name:
            self.W_tile_size_nof_last = n_out % tile_n_out if (n_out % tile_n_out) > 0 else tile_n_out
            self.W_tile_size_nif_last = self.W_tile_size_nif
            self.W_tile_size_nif_byte_last = int(math.ceil(self.W_tile_size_nif_last * ds_W / 8.0))

        # y last
        self.y_tile_size_nof_last = n_out % tile_n_out if (n_out % tile_n_out) > 0 else tile_n_out
        self.y_tile_size_h_last = h_out % tile_h_out if (h_out % tile_h_out) > 0 else tile_h_out
        self.y_tile_size_w_last = w_out % tile_w_out if (w_out % tile_w_out) > 0 else tile_w_out
        self.y_length_nof_byte_last = int(math.ceil(self.y_tile_size_nof_last * ds_y / 8.0))

        # x last
        self.x_tile_size_nif_last = n_in % tile_n_in if (n_in % tile_n_in) > 0 else tile_n_in
        self.x_tile_size_nif_byte_last = int(math.ceil(self.x_tile_size_nif_last * ds_x / 8.0))
        self.x_tile_size_h_last = self.y_tile_size_h_last * s[0] + ks[0] - s[0] - (
                padding_bottom - ((h_in + padding_bottom + padding_top) - (h_out * s[0] + ks[0] - s[0])))
        self.x_tile_size_w_last = self.y_tile_size_w_last * s[1] + ks[1] - s[1] - (
                padding_right - ((w_in + padding_left + padding_right) - (w_out * s[1] + ks[1] - s[1])))
        ## single tile execution
        if self.x_tile_size_h_last > self.x_tile_size_h:
            self.x_tile_size_h_last = self.x_tile_size_h
        if self.x_tile_size_w_last > self.x_tile_size_w:
            self.x_tile_size_w_last = self.x_tile_size_w

        l = ""
        for k, v in self.__dict__.items():
            try:
                l += "// %s %d\n" % (k.ljust(30), v)
            except TypeError:
                try:
                    l += "// %s %d\n" % (k.ljust(30), v[0])
                except TypeError:
                    l += "// %s %s\n" % (k.ljust(30), v)
        if "Addition" not in node.name and "Pool" not in node.name:
            buffer_l1_all = W_buffer_size + x_buffer_size + y_buffer_size + self.k_tile_size_byte + self.lambda_tile_size_byte + 40 + self.b_size_byte
            self.im2col_dim = (8 * (fs1 * (tile_h_in + padding_bottom + padding_top) + fs1)) * int(
                8 / min(ds_x, ds_y, ds_W))
        elif "Addition" in node.name:
            buffer_l1_all = x_buffer_size * 2 + y_buffer_size + self.k_tile_size_byte + self.lambda_tile_size_byte + 40 + \
                            self.b_size_byte
        elif "Pool" in node.name:
            buffer_l1_all = x_buffer_size + y_buffer_size + self.k_tile_size_byte + self.lambda_tile_size_byte + 40 + self.b_size_byte
        self.buffer_l1_all = buffer_l1_all

        # only used for avg pool layers
        self.out_add = node.outadd["value"] if 'outadd' in node.constant_names else 0
        self.out_mul = node.outmul["value"] if 'outmul' in node.constant_names else 1
        self.out_shift = node.outshift["value"] if 'outshift' in node.constant_names else 0


