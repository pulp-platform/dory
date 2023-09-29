     # should work even without -*-
# -*- coding: utf-8 -*-
#!/bin/bash
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
import numpy as np

# DORY modules
from dory.Parsers.Parser_HW_to_C import Parser_HW_to_C
import dory.Utils.Templates_writer.Layer2D_template_writer as Layer2D_writer
from dory.Utils.Templates_writer.TemplateWriter import TemplateWriter

# Directory
file_path = "/".join(os.path.realpath(__file__).split("/")[:-1])


class C_Parser(Parser_HW_to_C):
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    def __init__(self, graph, config_file, config_file_dir, verbose_level, perf_layer, precision_library, app_directory):
        f = open(os.path.join(file_path, "HW_description.json"))
        HW_description = json.load(f)
        self.precision_library = precision_library
        self.source_Constant_bits_library = config_file["BNRelu_bits"]
        self.config_file = config_file
        super().__init__(graph, os.path.join(config_file_dir, os.path.dirname(config_file["onnx_file"])), HW_description, verbose_level, perf_layer, "Makefile", app_directory)

    def copy_backend_files(self, node):
        root = os.path.dirname(__file__)
        files = os.path.join(root, "Backend_Kernels/")
        if os.listdir(os.path.join(files, "include"))[0] not in os.listdir(os.path.join(self.app_directory, "DORY_network/inc")):
            for file in os.listdir(os.path.join(files, "include")):
                file_to_copy = os.path.join(files, "include", file)
                os.system('cp "{}" {}'.format(file_to_copy, os.path.join(self.app_directory, 'DORY_network/inc')))
        if os.listdir(os.path.join(files, "src"))[0] not in os.listdir(os.path.join(self.app_directory, "DORY_network/src")):
            for file in os.listdir(os.path.join(files, "src")):
                file_to_copy = os.path.join(files, "src", file)
                os.system('cp "{}" {}'.format(file_to_copy, os.path.join(self.app_directory, 'DORY_network/src')))

    def mapping_layers_to_C_files(self):
        print("\nMapping the layers files to their templates and copying the kernels associated.")
        tmpl_dir = os.path.join(os.path.dirname(__file__), 'Templates/layer_templates')
        out_dir = '{}/DORY_network'.format(self.app_directory)
        for i, node in enumerate(self.HWgraph):
            self.copy_backend_files(node)
            if node.tiling_dimensions["L2"]["input_dimensions"][2] == node.tiling_dimensions["L1"]["input_dimensions"][2]:
                node.tiling_dimensions["L1"]["output_dimensions"][2] = int((node.tiling_dimensions["L1"]["input_dimensions"][2] + (node.pads[1] + node.pads[3]) - node.kernel_shape[1] + node.strides[1]) / node.strides[1])
            if node.tiling_dimensions["L2"]["input_dimensions"][1] == node.tiling_dimensions["L1"]["input_dimensions"][1]:
                node.tiling_dimensions["L1"]["output_dimensions"][1] = int((node.tiling_dimensions["L1"]["input_dimensions"][1] + (node.pads[0] + node.pads[2]) - node.kernel_shape[0] + node.strides[0]) / node.strides[0])
            tk = Layer2D_writer.print_template_layer(node, self.precision_library)
            tm = self.l2_template_mapping(node, self.precision_library)
            TemplateWriter.write(tk, tm)


