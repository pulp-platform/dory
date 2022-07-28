import re
import math

from Utils.Templates_writer.TemplateWriter import TemplateWriter


class TemplateWriter2D_L3(TemplateWriter):
    def __init__(self, node, tmpldir):
        super().__init__(tmpldir)

        ks = node.kernel_shape
        s = node.strides
        g = node.group
        p = node.pads

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

        self.tk['conv_overlap1'] = 2 * (ks[0] // 2) + ks[0] % 2 - 1 - (s[0] - 1)
        self.tk['conv_overlap2'] = 2 * (ks[1] // 2) + ks[1] % 2 - 1 - (s[1] - 1)
        self.tk['padding_top'] = p[0]
        self.tk['padding_left'] = p[1]
        self.tk['padding_bottom'] = p[2]
        self.tk['padding_right'] = p[3]
        if (node.tiling_dimensions["L3"]["input_dimensions"] != node.tiling_dimensions["L2"]["input_dimensions"]):
            self.tk['input_L3'] = 1
            factor_h_in = h_out / h_out_L2
        else:
            self.tk['input_L3'] = 0
            factor_h_in = 1
        factor_h_out = h_out / node.tiling_dimensions["L2"]["output_dimensions"][1]
        if not isinstance(node.tiling_dimensions["L2"]["weights_dimensions"], type(None)):
            factor_ch_out = node.tiling_dimensions["L3"]["weights_dimensions"][0] / \
                            node.tiling_dimensions["L2"]["weights_dimensions"][0]
        else:
            factor_ch_out = 1
        self.tk['n_tile_W'] = math.ceil(factor_ch_out)
        self.tk['n_tile_x'] = math.ceil(factor_h_in)
        self.tk['n_tile_y'] = math.ceil(factor_h_out)
        self.tk['verbose'] = True
        self.tk['func_name'] = node.name
        self.tk['L2_func_names'] = [node.name + "_L2"]
        self.tk['BitIn'] = ds_x
        self.tk['y_data_size_byte'] = ds_y
        self.tk['x_data_size_byte'] = ds_x
        self.tk['w_out'] = w_out_L2
        self.tk['h_out'] = h_out_L2
        self.tk['n_out'] = n_out_L2
        self.tk['w_in'] = w_in_L2
        self.tk['h_in'] = h_in_L2
        self.tk['n_in'] = n_in_L2

        self.tk['has_bias'] = int(len([1 for name in node.constant_names if "bias" in name]) > 0)

        offset = 0
        self.tk['l3_offset_w'] = offset
        offset += node.tiling_dimensions["L3"]["weight_memory"]

        if self.tk['has_bias'] == 1:
            self.tk['l3_offset_b'] = offset
            offset += node.tiling_dimensions["L3"]["bias_memory"]

        if not isinstance(node.tiling_dimensions["L2"]["constants_memory"], type(None)):
            self.tk['l3_offset_k'] = offset
            offset += int(node.tiling_dimensions["L3"]["constants_memory"] / 2)

            self.tk['l3_offset_l'] = offset
            offset += int(node.tiling_dimensions["L3"]["constants_memory"] / 2)

        self.tk['weight_dim'] = int(node.tiling_dimensions["L2"]["weight_memory"])
        if self.tk['has_bias'] == 1:
            self.tk['bias_dim'] = node.tiling_dimensions["L2"]["bias_memory"]
        else:
            self.tk['bias_dim'] = 0
        if not isinstance(node.tiling_dimensions["L2"]["constants_memory"], type(None)):
            self.tk['lambda_dim'] = int(node.tiling_dimensions["L2"]["constants_memory"] / 2)
            self.tk['k_dim'] = int(node.tiling_dimensions["L2"]["constants_memory"] / 2)
        else:
            self.tk['lambda_dim'] = 0
            self.tk['k_dim'] = 0
        self.tk['dim_out'] = int(n_out_L2 * w_out_L2 * h_out_L2 * node.output_activation_bits / 8)
        self.tk['dim_in'] = int(n_in_L2 * w_in_L2 * h_in_L2 * node.input_activation_bits / 8)


class TemplateWriter2D_L2(TemplateWriter):
    def __init__(self, node, tmpldir):
        super().__init__(tmpldir)

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
                self.tk['first_layer'] = 0
            except ValueError:
                self.tk['first_layer'] = 1
        else:
            self.tk['first_layer'] = 0
        self.tk['func_name'] = node.name
        self.tk['sdk'] = node.hw_desc["software development kit"]["name"]
        self.tk['number_of_clusters'] = node.hw_desc[
            "number_of_clusters"] if "number_of_clusters" in node.hw_desc.keys() else 1
        #self.tk['optional_type'] = layer_type
        self.tk['optional'] = node.op_type
        self.tk['flag_DW'] = node.group > 1
        self.tk['FLAG_BATCHNORM'] = 1 if 'k' in node.constant_names else 0
        self.tk['has_bias'] = int(len([1 for name in node.constant_names if "bias" in name]) > 0)
        self.tk['FLAG_RELU'] = 1 if 'outshift' in node.constant_names else 0
        self.tk['type'] = "char" if node.input_activation_type in ["int", "uint"] else "float"
        self.tk['conv_overlap1'] = conv_overlap1
        self.tk['conv_overlap2'] = conv_overlap2
        self.tk['padding_top'] = padding_top
        self.tk['padding_bottom'] = padding_bottom
        self.tk['padding_left'] = padding_left
        self.tk['padding_right'] = padding_right
        self.tk['stride'] = s[0]

        ################## NEED A REWRITING IN THIS TEMPLATE PART ######################
        #### VARIABLE CREATION FOR COMPATIBILITY WITH THE SECTION AFTER ################
        if self.tk['flag_DW'] == 0:
            self.tk['g'] = 1
            self.tk['nif'] = node.tiling_dimensions["L2"]["input_dimensions"][0]
        else:
            self.tk['g'] = node.tiling_dimensions["L2"]["input_dimensions"][0]
            self.tk['nif'] = 1
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
            self.tk['tk']['data_type_x2'] = dt_x2
            self.tk['tk']['x_data_size_byte2'] = ds_x2
            self.tk['tk']['inmul1'] = node.inmul1["value"]
            self.tk['tk']['inadd1'] = node.inadd1["value"]
            self.tk['inshift1'] = node.inshift1["value"]
            self.tk['inmul2'] = node.inmul2["value"]
            self.tk['inadd2'] = node.inadd2["value"]
            self.tk['inshift2'] = node.inshift2["value"]
            self.tk['outmul'] = node.outmul["value"]
            self.tk['outadd'] = node.outadd["value"]
            self.tk['outshift'] = node.outshift["value"]

        DW = self.tk['flag_DW']
        has_bias = self.tk['has_bias']
        number_of_clusters = self.tk['number_of_clusters']

        self.tk['data_type_x'] = dt_x
        self.tk['data_type_y'] = dt_y
        self.tk['data_type_activations'] = dt_act
        self.tk['data_type_weights'] = dt_W
        ################################################################################

        self.tk['nof'] = n_out
        self.tk['factor'] = node.tiling_dimensions["L3"]["output_dimensions"][0] / n_out
        # x parameters
        self.tk['x_h'] = h_in
        self.tk['x_w'] = w_in
        self.tk['x_data_size_byte'] = node.input_activation_bits
        self.tk['x_tile_size_nif'] = tile_n_in
        self.tk['x_tile_size_h'] = tile_h_in
        self.tk['x_tile_size_w'] = tile_w_in
        self.tk['x_tile_size_byte'] = int(math.ceil(ds_x * tile_n_in * tile_h_in * tile_w_in / 8.0))
        self.tk['x_tile_size_nif_byte'] = int(math.ceil(tile_n_in * ds_x / 8.0))
        self.tk['x_stride_w_byte'] = int(math.ceil(w_in * n_in * ds_x / 8.0))
        self.tk['x_stride_c_byte'] = int(math.ceil(n_in * ds_x / 8.0))
        # y parameters
        self.tk['y_h'] = h_out
        self.tk['y_w'] = w_out
        self.tk['y_data_size_byte'] = ds_y
        self.tk['act_dim_bit'] = ds_act
        self.tk['y_tile_size_nof'] = tile_n_out if (n_out > tile_n_out) else n_out
        self.tk['y_tile_size_h'] = tile_h_out if (h_out > tile_h_out) > 0 else h_out
        self.tk['y_tile_size_w'] = tile_w_out if (w_out > tile_w_out) > 0 else w_out
        self.tk['y_tile_size_byte'] = int(
            math.ceil(self.tk['y_tile_size_nof'] * self.tk['y_tile_size_h'] * self.tk['y_tile_size_w'] * ds_y / 8.0))
        self.tk['y_stride_w_byte'] = int(math.ceil(w_out * n_out * self.tk['factor'] * ds_y / 8.0))
        self.tk['y_stride_c_byte'] = int(math.ceil(n_out * self.tk['factor'] * ds_y / 8.0))
        self.tk['y_tile_size_nof_byte'] = int(math.ceil(tile_n_out * ds_y / 8.0))

        self.tk['tile_dim_h'] = max(int(math.ceil(float(h_out) / float(self.tk['y_tile_size_h']))), 1)
        self.tk['tile_dim_w'] = max(int(math.ceil(float(w_out) / float(self.tk['y_tile_size_w']))), 1)
        self.tk['tile_dim_nof'] = max(int(math.ceil(float(n_out) / float(self.tk['y_tile_size_nof']))), 1)
        self.tk['tile_dim_nif'] = max(int(math.ceil(float(n_in) / float(tile_n_in))), 1)
        self.tk['tile_n_in_last'] = n_in % tile_n_in if n_in % tile_n_in > 0 else tile_n_in
        # W parameters
        self.tk['fs1'] = fs1
        self.tk['fs2'] = fs2
        self.tk['W_data_size'] = ds_W
        self.tk['W_tile_size_nof'] = tile_n_out
        if self.tk['has_bias'] == 1:
            self.tk['b_size_byte'] = int(math.ceil(n_out * ds_bias / 8.0))
        else:
            self.tk['b_size_byte'] = 0

        if DW == 0:
            self.tk['W_tile_size_nif'] = tile_n_in * self.tk['tile_dim_nif']
            self.tk['W_tile_size_nif_last'] = self.tk['tile_n_in_last'] * self.tk['tile_dim_nif']
        else:
            self.tk['W_tile_size_nif'] = 1
            self.tk['W_tile_size_nif_last'] = 1
        if "Addition" not in node.name and "Pool" not in node.name:
            self.tk['W_tile_size_byte'] = int(math.ceil(tile_n_out * self.tk['W_tile_size_nif'] * fs1 * fs2 * ds_W / 8.0))
            if DW == 0:
                self.tk['W_stride_nof_byte'] = int(math.ceil(self.tk['nif'] * fs1 * fs2 * ds_W / 8.0))
            else:
                self.tk['W_stride_nof_byte'] = int(math.ceil(self.tk['nif'] * fs1 * fs2 * ds_W / 8.0))
            self.tk['W_stride_hw_byte'] = int(math.ceil(self.tk['nif'] * ds_W / 8.0))
            self.tk['W_tile_nif_byte'] = int(math.ceil(self.tk['W_tile_size_nif'] * ds_W / 8.0))
            self.tk['W_tile_nif_byte_last'] = int(math.ceil(self.tk['W_tile_size_nif_last'] * ds_W / 8.0))
        # l2 parameters
        if self.tk['FLAG_BATCHNORM'] == 1:
            self.tk['l2_off_k'] = int(
                math.ceil(self.tk['nof'] * self.tk['nif'] * fs1 * fs2 * ds_W / 8.0 + self.tk['b_size_byte']))
            self.tk['l2_off_lambda'] = int(
                math.ceil((self.tk['nof'] * self.tk['nif'] * fs1 * fs2 * ds_W + self.tk['nof'] * ds_act) / 8.0 + self.tk['b_size_byte']))
        if has_bias == 1:
            self.tk['l2_off_bias'] = int(math.ceil(self.tk['nof'] * self.tk['nif'] * fs1 * fs2 * ds_W / 8.0))
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
            y_buffer_size = int(math.ceil(ds_y * self.tk['y_tile_size_nof'] * self.tk['y_tile_size_h'] * self.tk['y_tile_size_w'] / 8.0))
            if "Addition" not in node.name and "Pool" not in node.name:
                if DW == 0:
                    W_buffer_size = int(math.ceil(ds_W * self.tk['y_tile_size_nof'] * self.tk['W_tile_size_nif'] * fs1 * fs2 / 8.0))
                else:
                    W_buffer_size = int(math.ceil(ds_W * self.tk['y_tile_size_nof'] * 1 * fs1 * fs2 / 8.0))
            else:
                W_buffer_size = 0
        else:
            y_buffer_size = 2 * int(
                math.ceil(ds_y * self.tk['y_tile_size_nof'] * self.tk['y_tile_size_h'] * self.tk['y_tile_size_w'] / 8.0))
            if "Addition" not in node.name and "Pool" not in node.name:
                if DW == 0:
                    W_buffer_size = 2 * int(
                        math.ceil(ds_W * self.tk['y_tile_size_nof'] * self.tk['W_tile_size_nif'] * fs1 * fs2 / 8.0))
                else:
                    W_buffer_size = 2 * int(math.ceil(ds_W * self.tk['y_tile_size_nof'] * 1 * fs1 * fs2 / 8.0))
            else:
                W_buffer_size = 0
        if self.tk['FLAG_BATCHNORM'] == 1:
            k_buffer_size = int(n_out * ds_act / 8.0)
            lambd_buffer_size = int(n_out * ds_act / 8.0)
        else:
            k_buffer_size = 0
            lambd_buffer_size = 0

        self.tk['k_tile_size_byte'] = 0
        self.tk['lambda_tile_size_byte'] = 0
        self.tk['k_size_byte'] = 0
        self.tk['lambda_size_byte'] = 0
        if "Pool" not in node.name:
            if self.tk['FLAG_BATCHNORM'] == 1:
                self.tk['k_size_byte'] = k_buffer_size
                self.tk['lambda_size_byte'] = k_buffer_size
                self.tk['k_tile_size_byte_transfer'] = int(math.ceil(tile_n_out * ds_act / 8.0))
                self.tk['lambda_tile_size_byte_transfer'] = int(math.ceil(tile_n_out * ds_act / 8.0))
                if n_in == tile_n_in and w_in == tile_w_in and h_in == tile_h_in and n_out == tile_n_out:
                    self.tk['k_tile_size_byte'] = int(math.ceil(tile_n_out * ds_act / 8.0))
                    self.tk['lambda_tile_size_byte'] = int(math.ceil(tile_n_out * ds_act / 8.0))
                else:
                    self.tk['k_tile_size_byte'] = int(math.ceil(tile_n_out * ds_act / 8.0 * 2))
                    self.tk['lambda_tile_size_byte'] = int(math.ceil(tile_n_out * ds_act / 8.0 * 2))
            if has_bias == 1:
                self.tk['bias_tile_size_byte'] = tile_n_out * int(ds_bias / 8.0)
                self.tk['b_size_byte'] = int(n_out) * int(ds_bias / 8.0)
            else:
                self.tk['bias_tile_size_byte'] = 0
                self.tk['b_size_byte'] = 0

        # l1 parameters
        self.tk['l1_x_offset'] = 0
        self.tk['l1_y_offset'] = x_buffer_size + 8
        if "Addition" in node.name:
            self.tk['l1_x2_offset'] = x_buffer_size + 8 + y_buffer_size + 8
        if "Addition" not in node.name and "Pool" not in node.name:
            self.tk['l1_W_offset'] = x_buffer_size + 8 + y_buffer_size + 8
            if self.tk['FLAG_BATCHNORM'] == 1:
                self.tk['l1_k_offset'] = x_buffer_size + 8 + y_buffer_size + 8 + W_buffer_size + 8
                self.tk['l1_lambda_offset'] = x_buffer_size + 8 + y_buffer_size + 8 + W_buffer_size + 8 + self.tk['k_tile_size_byte'] + 8
            if has_bias == 1:
                self.tk['l1_b_offset'] = x_buffer_size + 8 + y_buffer_size + 8 + W_buffer_size + 8 + self.tk['k_tile_size_byte'] + 8 + \
                                    self.tk['lambda_tile_size_byte'] + 8

        # W last
        if "Addition" not in node.name and "Pool" not in node.name:
            self.tk['W_tile_size_nof_last'] = n_out % tile_n_out if (n_out % tile_n_out) > 0 else tile_n_out
            self.tk['W_tile_size_nif_last'] = self.tk['W_tile_size_nif']
            self.tk['W_tile_size_nif_byte_last'] = int(math.ceil(self.tk['W_tile_size_nif_last'] * ds_W / 8.0))

        # y last
        self.tk['y_tile_size_nof_last'] = n_out % tile_n_out if (n_out % tile_n_out) > 0 else tile_n_out
        self.tk['y_tile_size_h_last'] = h_out % tile_h_out if (h_out % tile_h_out) > 0 else tile_h_out
        self.tk['y_tile_size_w_last'] = w_out % tile_w_out if (w_out % tile_w_out) > 0 else tile_w_out
        self.tk['y_length_nof_byte_last'] = int(math.ceil(self.tk['y_tile_size_nof_last'] * ds_y / 8.0))

        # x last
        self.tk['x_tile_size_nif_last'] = n_in % tile_n_in if (n_in % tile_n_in) > 0 else tile_n_in
        self.tk['x_tile_size_nif_byte_last'] = int(math.ceil(self.tk['x_tile_size_nif_last'] * ds_x / 8.0))
        self.tk['x_tile_size_h_last'] = self.tk['y_tile_size_h_last'] * s[0] + ks[0] - s[0] - (
                padding_bottom - ((h_in + padding_bottom + padding_top) - (h_out * s[0] + ks[0] - s[0])))
        self.tk['x_tile_size_w_last'] = self.tk['y_tile_size_w_last'] * s[1] + ks[1] - s[1] - (
                padding_right - ((w_in + padding_left + padding_right) - (w_out * s[1] + ks[1] - s[1])))
        ## single tile execution
        if self.tk['x_tile_size_h_last'] > self.tk['x_tile_size_h']:
            self.tk['x_tile_size_h_last'] = self.tk['x_tile_size_h']
        if self.tk['x_tile_size_w_last'] > self.tk['x_tile_size_w']:
            self.tk['x_tile_size_w_last'] = self.tk['x_tile_size_w']

        self.tk['verbose_log'] = "".join([f'// {k:<30} {v}' for k, v in self.tk.items()])
        if "Addition" not in node.name and "Pool" not in node.name:
            buffer_l1_all = W_buffer_size + x_buffer_size + y_buffer_size + self.tk['k_tile_size_byte'] + self.tk['lambda_tile_size_byte'] + 40 + self.tk['b_size_byte']
            self.tk['im2col_dim'] = (8 * (fs1 * (tile_h_in + padding_bottom + padding_top) + fs1)) * int(
                8 / min(ds_x, ds_y, ds_W))
        elif "Addition" in node.name:
            buffer_l1_all = x_buffer_size * 2 + y_buffer_size + self.tk['k_tile_size_byte'] + self.tk['lambda_tile_size_byte'] + 40 + \
                            self.tk['b_size_byte']
        elif "Pool" in node.name:
            buffer_l1_all = x_buffer_size + y_buffer_size + self.tk['k_tile_size_byte'] + self.tk['lambda_tile_size_byte'] + 40 + self.tk['b_size_byte']
        self.tk['buffer_l1_all'] = buffer_l1_all

        # only used for avg pool layers
        self.tk['out_add'] = node.outadd["value"] if 'outadd' in node.constant_names else 0
        self.tk['out_mul'] = node.outmul["value"] if 'outmul' in node.constant_names else 1
        self.tk['out_shift'] = node.outshift["value"] if 'outshift' in node.constant_names else 0


