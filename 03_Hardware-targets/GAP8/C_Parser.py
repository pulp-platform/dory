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
import sys
import onnx 
import numpy as np
import json
import os 

# Directory
file_path = "/".join(os.path.realpath(__file__).split("/")[:-1])

## DORY modules
from Parser_HW_to_C import Parser_HW_to_C
import Layer2D_template_writer as Layer2D_writer

class C_Parser(Parser_HW_to_C):
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    def __init__(self,  graph, json_configuration_file, verbose_level, perf_layer, precision_library):
        f = open(os.path.join(file_path, "HW_description.json"))
        HW_description = json.load(f)
        self.precision_library = precision_library
        super().__init__(graph, os.path.dirname(json_configuration_file["onnx_file"]), HW_description, verbose_level, perf_layer, "Makefile")

    def copy_backend_files(self, node):
        if self.precision_library == 'auto':
            self.precision_library = '8bit'
            if "Addition" not in node.name and "Pool" not in node.name:
                if node.get_parameter('output_activation_bits') < 8 or node.get_parameter('input_activation_bits') < 8 or node.get_parameter('weight_bits') < 8:
                    self.precision_library = 'mixed-sw'
            else:
                if node.get_parameter('output_activation_bits') < 8 or node.get_parameter('input_activation_bits') < 8:
                    self.precision_library = 'mixed-sw'

        root = '/'.join(os.getcwd().split('/')[:-1])
        if self.precision_library == "8bit":
            files = os.path.join(root, "03_Hardware-targets", self.HW_description["name"], "Backend_Kernels/pulp-nn/")
        elif self.precision_library == "mixed-sw":
            files = os.path.join(root, "03_Hardware-targets", self.HW_description["name"], "Backend_Kernels/pulp-nn-mixed/XpulpV2/")
        elif self.precision_library == "mixed-hw":
            files = os.path.join(root, "03_Hardware-targets", self.HW_description["name"], "Backend_Kernels/pulp-nn-mixed/XpulpNN/")
        if isinstance(node.constant_bits, type(None)):
            node.constant_bits = 32
        for file in os.listdir(os.path.join(files, "{}bit/include".format(node.constant_bits))):
            file_to_copy = os.path.join(files, "{}bit/include".format(node.constant_bits), file)
            os.system('cp "{}" application/DORY_network/inc'.format(file_to_copy))
        if self.precision_library == "8bit":
            if os.listdir(os.path.join(files, "{}bit/src".format(node.constant_bits)))[0] not in os.listdir("application/DORY_network/src"):
                for file in os.listdir(os.path.join(files, "{}bit/src".format(node.constant_bits))):
                    file_to_copy = os.path.join(files, "{}bit/src".format(node.constant_bits), file)
                    os.system('cp "{}" application/DORY_network/src'.format(file_to_copy))
        else:
            print("WARNING: Still to implement the mixed part\n")

    def mapping_layers_to_C_files(self):
        print("\nMapping the layers files to their templates and copying the kernels associated.")
        for i, node in enumerate(self.HWgraph):
            self.copy_backend_files(node)
            if (node.tiling_dimensions["L3"]["input_dimensions"] != node.tiling_dimensions["L2"]["input_dimensions"]) or (node.tiling_dimensions["L3"]["output_dimensions"] != node.tiling_dimensions["L2"]["output_dimensions"]) or (node.tiling_dimensions["L3"]["weights_dimensions"] != node.tiling_dimensions["L2"]["weights_dimensions"]):
                Layer2D_writer.print_template_layer_L3(node)
                node.name = node.name + "_L2"
                padding = node.pads
                node.pads = [0, 0, 0, 0]
                Layer2D_writer.print_template_layer(node)
                node.name = node.name[:-3]
                if padding[0] > 0:
                    node.name = node.name + "_L2_p_t"
                    node.pads = [padding[0], padding[1], 0, padding[3]]
                    Layer2D_writer.print_template_layer(node)
                    node.name = node.name[:-1] + "b"
                    node.pads = [0, padding[1], padding[2], padding[3]]
                    # h_in_last = h_in
                    # h_out_last = int(np.floor((h_in_last + p_bottom - (fs1 - 1) + (s - 1)) / s))
                    # #### CHECK WELL especially second nested if
                    # if factor_h_in > 2 or factor_h_out > 2:
                    #     if ((self.x_shape[-2] - h_in - h_in + conv_overlap_h + p_top) % (h_in - conv_overlap_h )) != 0:
                    #         h_in_last = ((self.x_shape[-2] - h_in - h_in + conv_overlap_h + p_top) % (h_in - conv_overlap_h )) + conv_overlap_h
                    #         h_out_last = int(np.floor((h_in_last + p_bottom - (fs1 - 1) + (s - 1)) / s))
                    #     elif (h_in - conv_overlap_h ) == 1:
                    #         h_in_last = h_in - 1
                    #         h_out_last = int(np.floor((h_in_last + p_bottom - (fs1 - 1) + (s - 1)) / s))
                    #     pad_bot = p_bottom - ((self.x_shape[-2] - h_in - h_in + conv_overlap_h + p_top + p_bottom) % (h_in - conv_overlap_h ))
                    # elif factor_h_in > 1 or factor_h_out > 1:
                    #     if ((self.x_shape[-2] - h_in) % (h_in - conv_overlap_h -p_top)) != 0:
                    #         h_in_last = ((self.x_shape[-2] - h_in) % (h_in - conv_overlap_h -p_top)) + conv_overlap_h + p_bottom
                    #         h_out_last = int(np.floor((h_in_last + p_bottom - (fs1 - 1) + (s - 1)) / s))
                    #     elif (h_in - conv_overlap_h ) == 1:
                    #         h_in_last = h_in - 1
                    #         h_out_last = int(np.floor((h_in_last + p_bottom - (fs1 - 1) + (s - 1)) / s))
                    #     pad_bot = p_bottom - ((self.x_shape[-2] - h_in) % (h_in - conv_overlap_h -p_top))
                    Layer2D_writer.print_template_layer(node)
                    node.name = node.name[:-7]
            else:
                Layer2D_writer.print_template_layer(node)



