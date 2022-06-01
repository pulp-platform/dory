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
import shutil

# DORY modules
from Parsers.Parser_HW_to_C import Parser_HW_to_C
from Utils.Templates_writer import Layer2D_template_writer as Layer2D_writer
from Utils.Templates_writer.TemplateWriter2D import TemplateWriter2D_L2
from .ne16 import weights_size, weights_ki_size, div_and_ceil, rem, TP_IN


class C_Parser(Parser_HW_to_C):
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    def __init__(self, graph, config_file, config_file_dir, verbose_level, perf_layer, precision_library, app_directory):
        with open(os.path.join(os.path.dirname(__file__), 'HW_description.json'), 'r') as f:
            HW_description = json.load(f)
        self.precision_library = precision_library
        self.source_Constant_bits_library = config_file["BNRelu_bits"]
        self.config_file = config_file
        super().__init__(graph, os.path.join(config_file_dir, os.path.dirname(config_file["onnx_file"])),
                         HW_description, verbose_level, perf_layer, "Makefile", app_directory)

    def copy_backend_files(self, node):
        out_dir = os.path.join(self.app_directory, 'DORY_network')
        out_inc_dir = os.path.join(out_dir, 'inc')
        out_src_dir = os.path.join(out_dir, 'src')
        backend_dir = os.path.join(os.path.dirname(__file__), 'Backend_Kernels', 'pulp-nnx')
        backend_inc_dir = os.path.join(backend_dir, 'include')
        backend_src_dir = os.path.join(backend_dir, 'src')
        accelerator = 'ne16'

        def cp_files(src_dir, dest_dir, files):
            for file in files:
                shutil.copy(os.path.join(src_dir, file), os.path.join(dest_dir, file))

        def cp_dir_files(src_dir, dest_dir, accelerator):
            for (dirpath, dirnames, filenames) in os.walk(src_dir):
                cp_files(dirpath, dest_dir, filenames)
                if accelerator in dirnames:
                    for (dirpath, dirnames, filenames) in os.walk(os.path.join(dirpath, accelerator)):
                        cp_files(dirpath, dest_dir, filenames)
                        break
                break

        cp_dir_files(backend_inc_dir, out_inc_dir, accelerator)
        cp_dir_files(backend_src_dir, out_src_dir, accelerator)

    def mapping_layers_to_C_files(self):
        print("\nMapping the layers files to their templates and copying the kernels associated.")
        tmpl_dir = os.path.join(os.path.dirname(__file__), 'Templates', 'layer_templates')
        out_dir = os.path.join(self.app_directory, 'DORY_network')

        for node in self.HWgraph:
            self.copy_backend_files(node)
            tmpl_writer = TemplateWriter2D_L2(node)
            tmpl_writer = nnx_vars(tmpl_writer, node)
            tmpl_files = ['layer_L2_h_template.h', 'layer_L2_c_conv_template.c']
            tmpl_files = [os.path.join(tmpl_dir, tmpl_file) for tmpl_file in tmpl_files]
            tmpl_writer.write(tmpl_files, out_dir)


def mem_tmpl_vars(tmpl_writer, node, mem_level):
    mem_name = f'L{mem_level}'
    upper_mem_name = f'L{mem_level + 1}'

    def set_tmpl_var(name, val):
        prefix = f'{mem_name.lower()}_'

        def attr(var_name):
            return f'{prefix}{var_name}'

        setattr(tmpl_writer, attr(name), val)

    input_tile_shape = node.tiling_dimensions[mem_name]['input_dimensions']
    output_tile_shape = node.tiling_dimensions[mem_name]['output_dimensions']
    weights_tile_shape = node.tiling_dimensions[mem_name]['weights_dimensions']

    weights_tile_ko, weights_tile_ki = weights_tile_shape

    weights_tile_size = weights_size(weights_tile_ko, weights_tile_ki, node.kernel_shape, node.weight_bits)
    set_tmpl_var('W_size', weights_tile_size)

    flag_depthwise = node.group > 1
    flag_batchnorm = 'k' in node.constant_names
    flag_bias = len([1 for name in node.constant_names if 'bias' in name]) > 0

    input_el_size = div_and_ceil(node.input_activation_bits, 8)
    output_el_size = div_and_ceil(node.output_activation_bits, 8)
    activation_el_size = div_and_ceil(node.constant_bits, 8)

    if not flag_depthwise:
        tile_ko = weights_tile_ko
        weights_tile_ko_len = tile_ko
    else:
        tile_ko = weights_tile_ki
        weights_tile_ko_len = div_and_ceil(tile_ko, TP_IN)

    weights_tile_ki_size = weights_ki_size(weights_tile_ki, node.kernel_shape, node.weight_bits, flag_depthwise)

    def feature_len(shape):
        return shape[0] * shape[1] * shape[2]

    if upper_mem_name in node.tiling_dimensions.keys():
        input_shape = node.tiling_dimensions[upper_mem_name]['input_dimensions']
        output_shape = node.tiling_dimensions[upper_mem_name]['output_dimensions']
        weights_shape = node.tiling_dimensions[upper_mem_name]['weights_dimensions']

        n_buffers = {
            'x': 1 if input_shape == input_tile_shape else 2,
            'y': 1 if output_shape == output_tile_shape else 2,
            'W': 1 if weights_shape == weights_tile_shape else 2,
            'k': 1 if weights_shape == weights_tile_shape else 2,
            'lambda': 1 if weights_shape == weights_tile_shape else 2,
            'b': 1 if weights_shape == weights_tile_shape else 2,
        }
    else:
        n_buffers = {'x': 1, 'y': 1, 'W': 1, 'k': 1, 'lambda': 1, 'b': 1}

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

        weights_ko_len = div_and_ceil(weights_ki, TP_IN) if flag_depthwise else weights_ko

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


def nnx_vars(tmpl_writer, node):
    mem_tmpl_vars(tmpl_writer, node, mem_level=1)
    mem_tmpl_vars(tmpl_writer, node, mem_level=2)
    return tmpl_writer
