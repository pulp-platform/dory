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
    def __init__(self, graph, conf, confdir, verbose_level, perf_layer, precision_library, appdir, accelerator):
        with open(os.path.join(os.path.dirname(__file__), 'HW_description.json'), 'r') as f:
            hw_desc = json.load(f)
        super().__init__(graph, conf, os.path.join(confdir, os.path.dirname(conf["onnx_file"])),
                         hw_desc, verbose_level, perf_layer, "Makefile", appdir)

        self.acc = accelerator
        pulp_nnx_backenddir = os.path.join(self.targetdir, 'Backend_Kernels', 'pulp-nnx')
        self.backenddirs = [os.path.join(pulp_nnx_backenddir, self.acc.name)]
        self.backendfiles = [os.path.join(pulp_nnx_backenddir, 'pulp_nnx.h')]

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
                templateWriter = TemplateWriter2D_L3(node, tmpldir)
                tmplfiles = ['tmpl_layer_L3.h', 'tmpl_layer_L3.c']
                outfiles = [f'{node.name}.h', f'{node.name}.c']
                outfiles = [os.path.join(self.destdir(file), file) for file in outfiles]
                templateWriter.write(tmplfiles, outfiles)
                node.name += '_L2'

            templateWriter = TemplateWriter2D_L2(node, tmpldir)
            templateWriter = self.__nnx_vars(templateWriter, node)
            tmplfiles = ['tmpl_layer_L2.h', 'tmpl_layer_conv_L2.c']
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
        activation_el_size = div_and_ceil(node.constant_bits, 8)

        tile_ko = weights_tile_ki if flag_depthwise else weights_tile_ko
        weights_tile_ko_len = self.acc.weights_ko_len(tile_ko, flag_depthwise)
        weights_tile_ki_size = self.acc.weights_ki_size(weights_tile_ki, node.kernel_shape, node.weight_bits, flag_depthwise)

        def feature_len(shape):
            return shape[0] * shape[1] * shape[2]

        n_buffers = {
            'x': node.tiling_dimensions[mem_name]['db_x'],
            'y': node.tiling_dimensions[mem_name]['db_y'],
            'W': node.tiling_dimensions[mem_name]['db_w'],
            'k': node.tiling_dimensions[mem_name]['db_w'],
            'lambda': node.tiling_dimensions[mem_name]['db_w'],
            'b': node.tiling_dimensions[mem_name]['db_w']
        }

        tile_sizes = {
            'x': feature_len(input_tile_shape) * input_el_size,
            'y': feature_len(output_tile_shape) * output_el_size,
            'W': weights_tile_ko_len * weights_tile_ki_size,
            'k': tile_ko * activation_el_size,
            'lambda': tile_ko * activation_el_size,
            'b': tile_ko * div_and_ceil(node.weight_bits, 8)
        }

        buffer_sizes = {key: tile_sizes[key] * n_buffers[key] for key in tile_sizes.keys()}

        data_arrays = ['x', 'y', 'W']

        if flag_batchnorm:
            data_arrays.append('k')
            data_arrays.append('lambda')

        if flag_bias:
            data_arrays.append('b')

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
        ko, ki = node.tiling_dimensions['L2']['weights_dimensions']
        weights_size = self.acc.weights_size(ko, ki, node.kernel_shape, node.weight_bits, flag_depthwise)
        templateWriter.set_var('l2_k_offset', weights_size)
        tile_ko = ki if flag_depthwise else ko
        templateWriter.set_var('l2_lambda_offset', weights_size + tile_ko * (node.constant_bits // 8))

        return templateWriter
