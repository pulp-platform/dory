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
from _00_Parsers.Parser_HW_to_C import Parser_HW_to_C
from _01_Utils.Templates_writer import Layer2D_template_writer as Layer2D_writer


class C_Parser(Parser_HW_to_C):
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    def __init__(self, graph, conf, conf_dir, app_dir, verbose_level, perf_layer, precision_library):
        with open("HW_description.json", 'r') as f:
            hw_desc = json.load(f)
        self.precision_library = precision_library
        self.app_dir = app_dir
        super().__init__(graph, os.path.join(conf_dir, os.path.dirname(conf["onnx_file"])),
                         hw_desc, verbose_level, perf_layer, "Makefile")

    def copy_backend_files(self, node):
        out_dir = os.path.join(self.app_dir, 'DORY_network')
        out_inc_dir = os.path.join(out_dir, 'inc')
        out_src_dir = os.path.join(out_dir, 'src')
        backend_dir = "Backend_Kernels/pulp-nnx"
        backend_inc_dir = os.path.join(backend_dir, 'include')
        backend_src_dir = os.path.join(backend_dir, 'src')
        accelerator = "ne16"

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
        tmpl_dir = os.path.join(os.path.dirname(__file__), 'Templates/layer_templates')
        out_dir = os.path.join(self.app_dir, 'DORY_network')
        for i, node in enumerate(self.HWgraph):
            self.copy_backend_files(node)
            if (node.tiling_dimensions["L3"]["input_dimensions"] != node.tiling_dimensions["L2"]["input_dimensions"]) \
                    or (node.tiling_dimensions["L3"]["output_dimensions"] != node.tiling_dimensions["L2"]["output_dimensions"]) \
                    or (node.tiling_dimensions["L3"]["weights_dimensions"] != node.tiling_dimensions["L2"]["weights_dimensions"]):
                Layer2D_writer.print_template_layer_L3(node, tmpl_dir, out_dir)
                node.name = node.name + "_L2"
                padding = node.pads
                node.pads = [0, padding[1], 0, padding[3]]
                Layer2D_writer.print_template_layer(node, tmpl_dir, out_dir)
                node.name = node.name[:-3]
                if padding[0] > 0:
                    node.name = node.name + "_L2_p_t"
                    node.pads = [padding[0], padding[1], 0, padding[3]]
                    Layer2D_writer.print_template_layer(node, tmpl_dir, out_dir)
                    node.name = node.name[:-1] + "b"
                    node.pads = [0, padding[1], padding[2], padding[3]]
                    # pad_bot = padding[2] - ((node.tiling_dimensions["L3"]["input_dimensions"][1] - node.tiling_dimensions["L3"]["input_dimensions"][2]) % (node.tiling_dimensions["L3"]["input_dimensions"][2] - (2 * (node.kernel_shape[0] // 2) + node.kernel_shape[0] % 2 - 1 - (node.strides[0] - 1)) -padding[0]))
                    # node.pads = [0, padding[1], pad_bot, padding[3]]
                    Layer2D_writer.print_template_layer(node, tmpl_dir, out_dir)
                    node.name = node.name[:-7]
            else:
                Layer2D_writer.print_template_layer(node, tmpl_dir, out_dir)
