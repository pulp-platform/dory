# should work even without -*-
# -*- coding: utf-8 -*-
# !/bin/bash
# ONNX_to_DORY_generic.py
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

# Libraries
import json
import os

# DORY modules
from Parsers.Parser_HW_to_C import Parser_HW_to_C
from Utils.Templates_writer.TemplateWriter2D import TemplateWriter2D_L2, TemplateWriter2D_L3
from .Util import div_and_ceil, rem


class nnx_C_Parser(Parser_HW_to_C):
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    def __init__(self, graph, conf, confdir, verbose_level, perf_layer, precision_library, appdir, accelerator, hw_desc_path):
        with open(hw_desc_path, 'r') as f:
            hw_desc = json.load(f)
        super().__init__(graph, conf, os.path.join(confdir, os.path.dirname(conf["onnx_file"])),
                         hw_desc, verbose_level, perf_layer, "Makefile", appdir)

        self.precision_library = precision_library
        self.acc = accelerator

        self.backenddirs = []
        self.backendfiles = []

        pulp_nnx_backenddir = os.path.join(self.targetdir, 'Backend_Kernels', 'pulp-nnx')
        dirs, files = self.__pulp_nnx_backend(pulp_nnx_backenddir, self.acc.name)
        self.backenddirs += dirs
        self.backendfiles += files

        # if precision_library == 'auto':
        #     precision_library = '8bit'

        # bnrelu_bits = conf['BNRelu_bits']

        # if precision_library == '8bit':
        #     pulp_nn_backenddir  = os.path.join(self.targetdir, os.pardir, 'GAP8', 'Backend_Kernels', 'pulp-nn')
        #     dirs, files = self.__pulp_nn_backend(pulp_nn_backenddir, bnrelu_bits)
        # elif 'mixed' in precision_library:
        #     if 'sw' in precision_library:
        #         pulp_nn_mixed_backenddir = os.path.join(self.targetdir, os.pardir, 'GAP8', 'Backend_Kernels', 'pulp-nn-mixed', 'XpulpV2')
        #     elif 'hw' in precision_library:
        #         pulp_nn_mixed_backenddir = os.path.join(self.targetdir, os.pardir, 'GAP8', 'Backend_Kernels', 'pulp-nn-mixed', 'XpulpNN')
        #     dirs, files = self.__pulp_nn_mixed_backend(pulp_nn_mixed_backenddir, bnrelu_bits, graph)
        # self.backenddirs += dirs
        # self.backendfiles += files

    def __pulp_nnx_backend(self, backenddir, accname):
        accdir = os.path.join(backenddir, accname)
        header = os.path.join(backenddir, 'pulp_nnx.h')
        return [accdir], [header]

    def __pulp_nn_backend(self, backenddir, bnrelu_bits):
        incdir = os.path.join(backenddir, f'{bnrelu_bits}bit', 'include')
        srcdir = os.path.join(backenddir, f'{bnrelu_bits}bit', 'src')
        return [incdir, srcdir], []

    def __pulp_nn_mixed_backend(self, backenddir, bnrelu_bits, graph):
        incdir = os.path.join(backenddir, f'{bnrelu_bits}bit', 'include')
        srcdir = os.path.join(backenddir, f'{bnrelu_bits}bit', 'src')
        srcfiles = self.__pulp_nn_mixed_srcfiles(graph, srcdir)
        return [incdir], srcfiles

    def __pulp_nn_mixed_srcfiles(self, graph, srcdir):
        files = set()

        def argtype(type, bits):
            return type[0] + str(bits)

        for node in graph:
            in1 = argtype(node.input_activation_type, node.input_activation_bits)
            out = argtype(node.output_activation_type, node.output_activation_bits)
            file = None

            if 'Addition' in node.name:
                in2 = argtype(node.second_input_activation_type, node.second_input_activation_bits)
                operands = f'{in1}_{in2}_{out}'
                file = os.path.join(srcdir, 'Add', f'pulp_nn_add_{operands}.c')
            elif 'Pool' in node.name:
                operands = f'{in1}_{out}'
                if 'Max' in node.op_type:
                    file = os.path.join(srcdir, 'Pooling', 'MaxPool', f'pulp_nn_maxpool_{operands}.c')
                if 'Avg' in node.op_type or 'Average' in node.op_type:
                    file = os.path.join(srcdir, 'Pooling', 'AvgPool', f'pulp_nn_avgpool_{operands}.c')
            #elif 'FullyConnected' in node.name:
            #    w = argtype(node.weight_type, node.weight_bits)
            #    operands = f'{in1}_{out}_{w}'
            #    if node.output_activation_bits == 32:
            #        file = os.path.join(srcdir, 'LinearNoQuant', f'pulp_nn_linear_{operands}.c')
            #    else:
            #        matmul_in1 = node.input_activation_type[0] + '8'
            #        matmul_operands = f'{matmul_in1}_{out}_{w}'
            #        file = (os.path.join(srcdir, 'LinearQuant', f'pulp_nn_linear_{operands}.c'),
            #                os.path.join(srcdir, 'MatrixMultiplication', f'pulp_nn_matmul_{matmul_operands}.c'))

            if file is not None:
                files.add(file)

        return list(files)

    def map_layers(self):
        print("\nMapping the layers files to their templates and copying the kernels associated.")
        tmpldir = os.path.join(self.targetdir, 'Templates', 'layer_templates')

        for node in self.graph:
            l3_tiling = node.tiling_dimensions["L3"]
            l2_tiling = node.tiling_dimensions["L2"]
            is_l3_access = (l3_tiling["input_dimensions"] != l2_tiling["input_dimensions"]) \
                        or (l3_tiling["output_dimensions"] != l2_tiling["output_dimensions"]) \
                        or (l3_tiling["weights_dimensions"] != l2_tiling["weights_dimensions"])

            if is_l3_access:
                templateWriter = TemplateWriter2D_L3(tmpldir, node)
                tmplfiles = ['tmpl_layer_L3.h', 'tmpl_layer_L3.c']
                outfiles = [f'{node.name}.h', f'{node.name}.c']
                outfiles = [os.path.join(self.destdir(file), file) for file in outfiles]
                templateWriter.write(tmplfiles, outfiles)
                node.name += '_L2'

            templateWriter = TemplateWriter2D_L2(tmpldir, node, self.precision_library)
            if 'Conv' in node.name or 'FullyConnected' in node.name:
                templateWriter = self.__nnx_vars(templateWriter, node)

            tmplfiles = ['tmpl_layer_L2.h']
            if 'Addition' in node.name:
                tmplfiles.append('tmpl_layer_L2_add.c')
            elif 'Pool' in node.name:
                tmplfiles.append('tmpl_layer_L2_pool.c')
            elif 'Conv' in node.name or 'FullyConnected' in node.name:
                tmplfiles.append('tmpl_layer_L2_conv.c')

            outfiles = [f'{node.name}.h', f'{node.name}.c']
            outfiles = [os.path.join(self.destdir(file), file) for file in outfiles]
            templateWriter.write(tmplfiles, outfiles)

            if is_l3_access:
                node.name = node.name[:-3]

    def __mem_tmpl_vars(self, templateWriter, node, mem_level):
        mem_name = f'L{mem_level}'
        upper_mem_name = f'L{mem_level + 1}'

        def set_tmpl_var(name, val):
            templateWriter.set_var(f'{mem_name.lower()}_{name}', val)

        flag_depthwise = node.group > 1
        flag_batchnorm = 'k' in node.constant_names
        flag_bias = len([1 for name in node.constant_names if 'bias' in name]) > 0

        input_tile_shape = node.tiling_dimensions[mem_name]['input_dimensions']
        output_tile_shape = node.tiling_dimensions[mem_name]['output_dimensions']
        weights_tile_shape = node.tiling_dimensions[mem_name]['weights_dimensions']

        weights_tile_ko, weights_tile_ki = weights_tile_shape

        weights_tile_size = self.acc.weights_size(weights_tile_ko, weights_tile_ki, node.kernel_shape, node.weight_bits, flag_depthwise)
        set_tmpl_var('W_size', weights_tile_size)

        input_el_size = div_and_ceil(node.input_activation_bits, 8)
        output_el_size = div_and_ceil(node.output_activation_bits, 8)

        tile_ko = weights_tile_ki if flag_depthwise else weights_tile_ko
        weights_tile_ko_len = self.acc.weights_ko_len(tile_ko, flag_depthwise)
        weights_tile_ki_size = self.acc.weights_ki_size(weights_tile_ki, node.kernel_shape, node.weight_bits, flag_depthwise)

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
            weights_ko_len = self.acc.weights_ko_len(ko, flag_depthwise)

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

    def __nnx_vars(self, templateWriter, node):
        self.__mem_tmpl_vars(templateWriter, node, mem_level=1)
        #self.__mem_tmpl_vars(templateWriter, node, mem_level=2) # TODO: make uniform offsets for L1/L2/Lx mems

        flag_depthwise = node.group > 1
        flag_batchnorm = 'k' in node.constant_names

        ko, ki = node.tiling_dimensions['L2']['weights_dimensions']
        weights_size = self.acc.weights_size(ko, ki, node.kernel_shape, node.weight_bits, flag_depthwise)

        if flag_batchnorm:
            templateWriter.set_var('l2_k_offset', weights_size)
            tile_ko = ki if flag_depthwise else ko
            templateWriter.set_var('l2_lambda_offset', weights_size + tile_ko * (node.constant_bits // 8))

        return templateWriter