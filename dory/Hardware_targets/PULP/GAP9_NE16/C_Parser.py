# C_Parser.py
# Alessio Burrello <alessio.burrello@unibo.it>
#
# Copyright (C) 2019-2020 University of Bologna
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dory.Hardware_targets.PULP.GAP9.C_Parser import C_Parser as C_Parser_gap9
from dory.Hardware_targets.PULP.GAP9_NE16.Ne16_HW_node import Ne16_HW_node
from dory.Utils.Templates_writer.TemplateWriter import TemplateWriter
import dory.Utils.Templates_writer.Layer2D_template_writer as Layer2D_writer
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Backend_Kernels", "pulp-nnx", "test"))


def rem(a, b):
    return ((a - 1) % b) + 1


def div_and_ceil(a, b):
    return ((a - 1) // b) + 1


class C_Parser(C_Parser_gap9):

    def __init__(self, *args, **kwargs):
        super(C_Parser, self).__init__(*args, **kwargs)

    def get_file_path(self):
        return "/".join(os.path.realpath(__file__).split("/")[:-1])

    def node_backend_library(self, node):
        if node.engine == "ne16":
            return "ne16"
        else:
            return super().node_backend_library(node)

    def l2_c_template(self, node, backend_library):
        if "Conv" in node.name and backend_library == "ne16":
            return "layer_L2_c_conv_ne16_multicore_template.c"
        else:
            return super().l2_c_template(node, backend_library)

    def l2_template_keywords(self, node, backend_library):
        tk = super().l2_template_keywords(node, backend_library)
        if isinstance(node, Ne16_HW_node):
            tk = self.__nnx_vars(tk, node)
        return tk

    def mapping_layers_to_C_files(self):
        print("\nMapping the layers files to their templates and copying the kernels associated.")
        n_memory_levels = self.HW_description['memory']['levels']

        for i, node in enumerate(self.HWgraph):
            backend_library = self.node_backend_library(node)
            self.copy_backend_files(node, backend_library)

            #if (node.kernel_shape[0] == 3 and node.group == 96):
                #breakpoint()

            if n_memory_levels > 2 and (node.L3_input != 0 or (node.tiling_dimensions["L3"]["output_dimensions"] != node.tiling_dimensions["L2"]["output_dimensions"]) or (node.tiling_dimensions["L3"]["weights_dimensions"] != node.tiling_dimensions["L2"]["weights_dimensions"])):
                #breakpoint()
                tk = Layer2D_writer.print_template_layer_L3(node)
                TemplateWriter.write(tk, {os.path.join(self.src_dir, node.prefixed_name + ".c"): os.path.join(self.layer_tmpl_dir, "layer_L3_c_template.c"),
                                          os.path.join(self.inc_dir, node.prefixed_name + ".h"): os.path.join(self.layer_tmpl_dir, "layer_L3_h_template.h")})
                if node.tiling_dimensions["L3"]["input_dimensions"][1] > node.tiling_dimensions["L2"]["input_dimensions"][1]:
                    node.tiling_dimensions["L2"]["output_dimensions"][1] = (node.tiling_dimensions["L2"]["input_dimensions"][1] - node.kernel_shape[0] + node.strides[0]) // node.strides[0]
                if node.tiling_dimensions["L3"]["output_dimensions"][1] > node.tiling_dimensions["L2"]["output_dimensions"][1]:
                    node.tiling_dimensions["L2"]["input_dimensions"][1] = node.tiling_dimensions["L2"]["output_dimensions"][1] * node.strides[0] + node.kernel_shape[0] - node.strides[0]
                node.name = node.name + "_L2"
                tk = self.l2_template_keywords(node, backend_library)
                TemplateWriter.write(tk, self.l2_template_mapping(node, backend_library))
                node.name = node.name[:-3]
            else:
                if node.tiling_dimensions["L2"]["input_dimensions"][2] == node.tiling_dimensions["L1"]["input_dimensions"][2]:
                    node.tiling_dimensions["L1"]["output_dimensions"][2] = int((node.tiling_dimensions["L1"]["input_dimensions"][2] + (node.pads[1] + node.pads[3]) - node.kernel_shape[1] + node.strides[1]) / node.strides[1])
                if node.tiling_dimensions["L2"]["input_dimensions"][1] == node.tiling_dimensions["L1"]["input_dimensions"][1]:
                    node.tiling_dimensions["L1"]["output_dimensions"][1] = int((node.tiling_dimensions["L1"]["input_dimensions"][1] + (node.pads[0] + node.pads[2]) - node.kernel_shape[0] + node.strides[0]) / node.strides[0])
                tk = self.l2_template_keywords(node, backend_library)
                TemplateWriter.write(tk, self.l2_template_mapping(node, backend_library))

    def __mem_tmpl_vars(self, tk, node, mem_level):
        mem_name = f'L{mem_level}'
        upper_mem_name = f'L{mem_level + 1}'

        def set_tmpl_var(name, val):
            tk[f'{mem_name.lower()}_{name}'] = val

        flag_depthwise = node.group > 1
        flag_batchnorm = 'k' in node.constant_names or 'k0' in node.constant_names
        flag_bias = len([1 for name in node.constant_names if 'bias' in name]) > 0

        input_tile_shape = node.tiling_dimensions[mem_name]['input_dimensions']
        output_tile_shape = node.tiling_dimensions[mem_name]['output_dimensions']
        weights_tile_shape = node.tiling_dimensions[mem_name]['weights_dimensions']

        weights_tile_ko, weights_tile_ki = weights_tile_shape

        weights_tile_size = node.calculate_weights_size(weights_tile_ko, weights_tile_ki, node.kernel_shape, node.weight_bits, flag_depthwise)
        set_tmpl_var('W_size', weights_tile_size)

        input_el_size = div_and_ceil(node.input_activation_bits, 8)
        output_el_size = div_and_ceil(node.output_activation_bits, 8)

        tile_ko = weights_tile_ki if flag_depthwise else weights_tile_ko
        weights_tile_ko_len = node.calculate_weights_ko_len(tile_ko, flag_depthwise)
        weights_tile_ki_size = node.calculate_weights_ki_size(weights_tile_ki, node.kernel_shape, node.weight_bits, flag_depthwise)

        def feature_len(shape):
            return shape[0] * shape[1] * shape[2]

        n_buffers = {
            'x': node.tiling_dimensions[mem_name]['db_x'],
            'y': node.tiling_dimensions[mem_name]['db_y'],
            'W': node.tiling_dimensions[mem_name]['db_w']
        }

        tile_sizes = {
            'x': feature_len(input_tile_shape) * input_el_size,
            'y': feature_len(output_tile_shape) * output_el_size,
            'W': weights_tile_ko_len * weights_tile_ki_size
        }

        data_arrays = ['x', 'y', 'W']

        if flag_batchnorm:
            activation_el_size = div_and_ceil(node.constant_bits, 8)
            bias_el_size = div_and_ceil(node.bias_bits, 8)
            n_buffers['k'] =      node.tiling_dimensions[mem_name]['db_w']
            n_buffers['lambda'] = node.tiling_dimensions[mem_name]['db_w']
            tile_sizes['k'] =      tile_ko * activation_el_size
            tile_sizes['lambda'] = tile_ko * bias_el_size
            data_arrays.append('k')
            data_arrays.append('lambda')

        if flag_bias:
            n_buffers['b'] = node.tiling_dimensions[mem_name]['db_w']
            tile_sizes['b'] = tile_ko * div_and_ceil(node.weight_bits, 8)
            data_arrays.append('b')

        buffer_sizes = {data_array: tile_sizes[data_array] * n_buffers[data_array] for data_array in data_arrays}

        offset = 0

        for data_array in data_arrays:
            set_tmpl_var(f'{data_array}_offset', offset)
            set_tmpl_var(f'{data_array}_tile_size', tile_sizes[data_array])
            offset += buffer_sizes[data_array]

        if upper_mem_name in node.tiling_dimensions.keys():
            input_shape = node.tiling_dimensions[upper_mem_name]['input_dimensions']
            output_shape = node.tiling_dimensions[upper_mem_name]['output_dimensions']
            weights_shape = node.tiling_dimensions[upper_mem_name]['weights_dimensions']

            input_depth, input_height, input_width = input_shape
            output_depth, output_height, output_width = output_shape
            weights_ko, weights_ki = weights_shape

            ko = weights_ki if flag_depthwise else weights_ko
            weights_ko_len = node.calculate_weights_ko_len(ko, flag_depthwise)

            set_tmpl_var('W_tile_ko_len', weights_tile_ko_len)
            set_tmpl_var('W_tile_ko_len_last', rem(weights_ko_len, weights_tile_ko_len))
            set_tmpl_var('W_tile_ki_size', weights_tile_ki_size)

            x_dma_stride_1d = input_depth * input_el_size
            x_dma_stride_2d = input_width * x_dma_stride_1d
            set_tmpl_var('x_dma_stride_1d', x_dma_stride_1d)
            set_tmpl_var('x_dma_stride_2d', x_dma_stride_2d)

            y_dma_stride_1d = output_depth * output_el_size
            y_dma_stride_2d = output_width * y_dma_stride_1d
            set_tmpl_var('y_dma_stride_1d', y_dma_stride_1d)
            set_tmpl_var('y_dma_stride_2d', y_dma_stride_2d)

    def _pw_dw_pw_mem_tmpl_vars(self, tk, node, mem_level):
        mem_name = f'L{mem_level}'
        upper_mem_name = f'L{mem_level + 1}'

        def set_tmpl_var(name, val):
            tk[f'{mem_name.lower()}_{name}'] = val

        intermediate_ko = node.tiling_dimensions[mem_name]['tile_n_out_pw0']

        input_tile_shape = node.tiling_dimensions[mem_name]['input_dimensions']
        output_tile_shape = node.tiling_dimensions[mem_name]['output_dimensions']
        pw0_output_tile_shape = [intermediate_ko] + input_tile_shape[1:]
        dw_output_tile_shape = [node.output_channels_list[1]] + output_tile_shape[1:]
        weights_tile_shape = node.tiling_dimensions[mem_name]['weights_dimensions']

        weights_tile_ko, weights_tile_ki = weights_tile_shape

        input_el_size = div_and_ceil(node.input_activation_bits, 8)
        output_el_size = div_and_ceil(node.output_activation_bits, 8)

        def feature_len(shape):
            return shape[0] * shape[1] * shape[2]

        n_buffers = {
            'x': node.tiling_dimensions[mem_name]['db_x'],
            'y_pw0': 1,
            'y_dw': 1,
            'y': node.tiling_dimensions[mem_name]['db_y'],
        }

        tile_sizes = {
            'x': feature_len(input_tile_shape) * input_el_size,
            'y_pw0': feature_len(pw0_output_tile_shape) * output_el_size,
            'y_dw': feature_len(dw_output_tile_shape) * output_el_size,
            'y': feature_len(output_tile_shape) * output_el_size,
        }

        data_arrays = ['x', 'y_pw0', 'y_dw', 'y']

        def add_batchnorm_param(name, bits, length):
            if name in node.constant_names:
                el_size = div_and_ceil(bits, 8)
                n_buffers[name] = node.tiling_dimensions[mem_name]['db_w']
                tile_sizes[name] = length * el_size
                data_arrays.append(name)

        def add_weights_param(i, ki, ko, ks, dw):
            n_buffers[f'W{i}'] = node.tiling_dimensions[mem_name]['db_w']
            tile_sizes[f'W{i}'] = node.calculate_weights_size(ko, ki, ks, node.weight_bits, dw)
            data_arrays.append(f'W{i}')
            add_batchnorm_param(f'k{i}', node.constant_bits, ko)
            add_batchnorm_param(f'l{i}', node.bias_bits, ko)

        ki_list = [weights_tile_ki, intermediate_ko, node.input_channels_list[2]]
        ko_list = [intermediate_ko, intermediate_ko, weights_tile_ko]
        ks_list = [[1, 1], node.kernel_shape, [1, 1]]
        dw_list = [False, True, False]

        for i, args in enumerate(zip(ki_list, ko_list, ks_list, dw_list)):
            add_weights_param(i, *args)

        buffer_sizes = {data_array: tile_sizes[data_array] * n_buffers[data_array] for data_array in data_arrays}

        offset = 0

        for data_array in data_arrays:
            set_tmpl_var(f'{data_array}_offset', offset)
            set_tmpl_var(f'{data_array}_tile_size', tile_sizes[data_array])
            offset += buffer_sizes[data_array]

        if upper_mem_name in node.tiling_dimensions.keys():
            def add_weights_dim_params(i, ko_upper, ki, ko, ks, dw):
                weights_ko_len = node.calculate_weights_ko_len(ko_upper, dw=True)
                weights_tile_ko_len = node.calculate_weights_ko_len(ko, dw=True)
                weights_tile_ki_size = node.calculate_weights_ki_size(ki, ks, node.weight_bits, dw)
                set_tmpl_var(f'W{i}_tile_ko_len', weights_tile_ko_len)
                set_tmpl_var(f'W{i}_tile_ko_len_last', rem(weights_ko_len, weights_tile_ko_len))
                set_tmpl_var(f'W{i}_tile_ki_size', weights_tile_ki_size)

            # Using output_channels_list because we for now not tiling L3-L2
            for i, args in enumerate(zip(node.output_channels_list, ki_list,
                                         ko_list, ks_list, dw_list)):
                add_weights_dim_params(i, *args)

            input_shape = node.tiling_dimensions[upper_mem_name]['input_dimensions']
            output_shape = node.tiling_dimensions[upper_mem_name]['output_dimensions']

            input_depth, _, input_width = input_shape
            output_depth, _, output_width = output_shape

            x_dma_stride_1d = input_depth * input_el_size
            x_dma_stride_2d = input_width * x_dma_stride_1d
            set_tmpl_var('x_dma_stride_1d', x_dma_stride_1d)
            set_tmpl_var('x_dma_stride_2d', x_dma_stride_2d)

            y_dma_stride_1d = output_depth * output_el_size
            y_dma_stride_2d = output_width * y_dma_stride_1d
            set_tmpl_var('y_dma_stride_1d', y_dma_stride_1d)
            set_tmpl_var('y_dma_stride_2d', y_dma_stride_2d)

    def _dw_pw_mem_tmpl_vars(self, tk, node, mem_level):
        mem_name = f'L{mem_level}'
        upper_mem_name = f'L{mem_level + 1}'

        def set_tmpl_var(name, val):
            tk[f'{mem_name.lower()}_{name}'] = val

        input_tile_shape = node.tiling_dimensions[mem_name]['input_dimensions']
        output_tile_shape = node.tiling_dimensions[mem_name]['output_dimensions']
        dw_output_tile_shape = [node.input_channels] + output_tile_shape[1:]
        weights_tile_shape = node.tiling_dimensions[mem_name]['weights_dimensions']

        weights_tile_ko, weights_tile_ki = weights_tile_shape

        weights_tile_size = node.calculate_weights_size(weights_tile_ki, weights_tile_ki, node.kernel_shape, node.weight_bits, dw=True) + \
                            node.calculate_weights_size(weights_tile_ko, node.input_channels, [1, 1], node.weight_bits, dw=False)
        set_tmpl_var('W_size', weights_tile_size)

        input_el_size = div_and_ceil(node.input_activation_bits, 8)
        output_el_size = div_and_ceil(node.output_activation_bits, 8)

        dw_weights_tile_ko_len = node.calculate_weights_ko_len(weights_tile_ki, dw=True)
        dw_weights_tile_ki_size = node.calculate_weights_ki_size(weights_tile_ki, node.kernel_shape, node.weight_bits, dw=True)

        pw_weights_tile_ko_len = node.calculate_weights_ko_len(weights_tile_ko, dw=False)
        pw_weights_tile_ki_size = node.calculate_weights_ki_size(node.input_channels, [1, 1], node.weight_bits, dw=False)

        def feature_len(shape):
            return shape[0] * shape[1] * shape[2]

        n_buffers = {
            'x': node.tiling_dimensions[mem_name]['db_x'],
            'y_dw': 1,
            'y': node.tiling_dimensions[mem_name]['db_y'],
        }

        tile_sizes = {
            'x': feature_len(input_tile_shape) * input_el_size,
            'y_dw': feature_len(dw_output_tile_shape) * output_el_size,
            'y': feature_len(output_tile_shape) * output_el_size,
        }

        data_arrays = ['x', 'y_dw', 'y']

        def add_batchnorm_param(name, bits, length):
            if name in node.constant_names:
                el_size = div_and_ceil(bits, 8)
                n_buffers[name] = node.tiling_dimensions[mem_name]['db_w']
                tile_sizes[name] = length * el_size
                data_arrays.append(name)

        n_buffers['W0'] = node.tiling_dimensions[mem_name]['db_w']
        tile_sizes['W0'] = dw_weights_tile_ko_len * dw_weights_tile_ki_size
        data_arrays.append('W0')

        add_batchnorm_param('k0', node.constant_bits, weights_tile_ki)
        add_batchnorm_param('l0', node.bias_bits, weights_tile_ki)

        n_buffers['W1'] = node.tiling_dimensions[mem_name]['db_w']
        tile_sizes['W1'] = pw_weights_tile_ko_len * pw_weights_tile_ki_size
        data_arrays.append('W1')

        add_batchnorm_param('k1', node.constant_bits, weights_tile_ko)
        add_batchnorm_param('l1', node.bias_bits, weights_tile_ko)

        buffer_sizes = {data_array: tile_sizes[data_array] * n_buffers[data_array] for data_array in data_arrays}

        offset = 0

        for data_array in data_arrays:
            set_tmpl_var(f'{data_array}_offset', offset)
            set_tmpl_var(f'{data_array}_tile_size', tile_sizes[data_array])
            offset += buffer_sizes[data_array]

        if upper_mem_name in node.tiling_dimensions.keys():
            input_shape = node.tiling_dimensions[upper_mem_name]['input_dimensions']
            output_shape = node.tiling_dimensions[upper_mem_name]['output_dimensions']
            weights_shape = node.tiling_dimensions[upper_mem_name]['weights_dimensions']

            input_depth, input_height, input_width = input_shape
            output_depth, output_height, output_width = output_shape
            weights_ko, weights_ki = weights_shape

            dw_weights_ko_len = node.calculate_weights_ko_len(weights_ki, dw=True)
            set_tmpl_var('W0_tile_ko_len', dw_weights_tile_ko_len)
            set_tmpl_var('W0_tile_ko_len_last', rem(dw_weights_ko_len, dw_weights_tile_ko_len))
            set_tmpl_var('W0_tile_ki_size', dw_weights_tile_ki_size)

            pw_weights_ko_len = node.calculate_weights_ko_len(weights_ko, dw=False)
            set_tmpl_var('W1_tile_ko_len', pw_weights_tile_ko_len)
            set_tmpl_var('W1_tile_ko_len_last', rem(pw_weights_ko_len, pw_weights_tile_ko_len))
            set_tmpl_var('W1_tile_ki_size', pw_weights_tile_ki_size)

            x_dma_stride_1d = input_depth * input_el_size
            x_dma_stride_2d = input_width * x_dma_stride_1d
            set_tmpl_var('x_dma_stride_1d', x_dma_stride_1d)
            set_tmpl_var('x_dma_stride_2d', x_dma_stride_2d)

            y_dma_stride_1d = output_depth * output_el_size
            y_dma_stride_2d = output_width * y_dma_stride_1d
            set_tmpl_var('y_dma_stride_1d', y_dma_stride_1d)
            set_tmpl_var('y_dma_stride_2d', y_dma_stride_2d)

    def __nnx_vars(self, tk, node):
        def write_l2_offset(name, size):
            tk[f'l2_{name}_offset'] = write_l2_offset.offset
            write_l2_offset.offset += size

        write_l2_offset.offset = 0

        def write_layer_l2_offset(i, ki, ko, ks, dw):
            write_l2_offset(f'W{i}', node.calculate_weights_size(ko, ki, ks, node.weight_bits, dw))

            if f'k{i}' in node.constant_names:
                write_l2_offset(f'k{i}', ko * (node.constant_bits // 8))

            if f'l{i}' in node.constant_names:
                write_l2_offset(f'l{i}', ko * (node.bias_bits // 8))

        if "PointwiseDepthwisePointwise" in node.name:
            self._pw_dw_pw_mem_tmpl_vars(tk, node, mem_level=1)

            ks_list = [[1, 1], node.kernel_shape, [1, 1]]
            dw_list = [False, True, False]

            for i, args in enumerate(zip(node.input_channels_list,
                                         node.output_channels_list,
                                         ks_list, dw_list)):
                write_layer_l2_offset(i, *args)
        elif "DepthwisePointwise" in node.name:
            self._dw_pw_mem_tmpl_vars(tk, node, mem_level=1)

            ko, ki = node.tiling_dimensions['L2']['weights_dimensions']

            ki_list = [ki, ki]
            ko_list = [ki, ko]
            ks_list = [node.kernel_shape, [1, 1]]
            dw_list = [True, False]

            for i, args in enumerate(zip(ki_list,
                                         ko_list,
                                         ks_list, dw_list)):
                write_layer_l2_offset(i, *args)
        else:
            self.__mem_tmpl_vars(tk, node, mem_level=1)

            flag_depthwise = node.group > 1
            flag_batchnorm = 'k' in node.constant_names or 'k0' in node.constant_names

            ko, ki = node.tiling_dimensions['L2']['weights_dimensions']
            weights_size = node.calculate_weights_size(ko, ki, node.kernel_shape, node.weight_bits, flag_depthwise)

            if flag_batchnorm:
                tk['l2_k_offset'] = weights_size
                tile_ko = ki if flag_depthwise else ko
                tk['l2_lambda_offset'] = weights_size + tile_ko * (node.constant_bits // 8)

        return tk
