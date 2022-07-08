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
from Parsers.Parser_HW_to_C import Parser_HW_to_C
import Utils.Templates_writer.Layer2D_template_writer as Layer2D_writer

# Directory
file_path = "/".join(os.path.realpath(__file__).split("/")[:-1])


class C_Parser(Parser_HW_to_C):
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    def __init__(self, graph, config_file, config_file_dir, verbose_level, perf_layer, precision_library, appdir):
        f = open(os.path.join(file_path, "HW_description.json"))
        HW_description = json.load(f)
        self.precision_library = precision_library
        self.source_Constant_bits_library = config_file["BNRelu_bits"]
        self.config_file = config_file
        super().__init__(graph, os.path.join(config_file_dir, os.path.dirname(config_file["onnx_file"])), HW_description, verbose_level, perf_layer, "Makefile", appdir)

    def copy_backend_files(self, node):
        if self.precision_library == 'auto':
            self.precision_library = '8bit'
            if "Addition" not in node.name and "Pool" not in node.name:
                if node.get_parameter('output_activation_bits') < 8 or node.get_parameter('input_activation_bits') < 8 or node.get_parameter('weight_bits') < 8:
                    self.precision_library = 'mixed-sw'
            else:
                if node.get_parameter('output_activation_bits') < 8 or node.get_parameter('input_activation_bits') < 8:
                    self.precision_library = 'mixed-sw'

        root = os.path.dirname(__file__)
        if self.precision_library == "8bit":
            files = os.path.join(root, "../Backend_Kernels/pulp-nn/")
        elif self.precision_library == "mixed-sw":
            files = os.path.join(root, "../Backend_Kernels/pulp-nn-mixed/XpulpV2/")
        elif self.precision_library == "mixed-hw":
            files = os.path.join(root, "../Backend_Kernels/pulp-nn-mixed/XpulpNN/")
        if os.listdir(os.path.join(files, "{}bit/include".format(self.source_Constant_bits_library)))[0] not in os.listdir(os.path.join(self.appdir, "DORY_network/inc")):
            for file in os.listdir(os.path.join(files, "{}bit/include".format(self.source_Constant_bits_library))):
                file_to_copy = os.path.join(files, "{}bit/include".format(self.source_Constant_bits_library), file)
                os.system('cp "{}" {}'.format(file_to_copy, os.path.join(self.appdir, 'DORY_network/inc')))
        if self.precision_library == "8bit":
            if os.listdir(os.path.join(files, "{}bit/src".format(self.source_Constant_bits_library)))[0] not in os.listdir(os.path.join(self.appdir, "DORY_network/src")):
                for file in os.listdir(os.path.join(files, "{}bit/src".format(self.source_Constant_bits_library))):
                    file_to_copy = os.path.join(files, "{}bit/src".format(self.source_Constant_bits_library), file)
                    os.system('cp "{}" {}'.format(file_to_copy, os.path.join(self.appdir, 'DORY_network/src')))
        elif self.precision_library == "mixed-sw":
            Input_bits = str(node.get_parameter('input_activation_bits'))
            Output_bits = str(node.get_parameter('output_activation_bits'))
            Input_type = node.get_parameter('input_activation_type')[0]
            Output_type = node.get_parameter('output_activation_type')[0]
            in_out = "_" + Input_type + Input_bits + "_" + Output_type + Output_bits
            if "Addition" in node.name:
                in1_in2_out = "_" + Input_type + Input_bits + "_" + node.get_parameter('second_input_activation_type')[0] + str(node.get_parameter('second_input_activation_bits')) + "_" + Output_type + Output_bits
                file = 'Add/pulp_nn_add{}.c'.format(in1_in2_out)
            elif "Pool" in node.name and "Max" in node.op_type:
                file = 'Pooling/MaxPool/pulp_nn_maxpool{}.c'.format(in_out)
            elif "Pool" in node.name and ("Avg" in node.op_type or "Average" in node.op_type):
                file = 'Pooling/AvgPool/pulp_nn_avgpool{}.c'.format(in_out)

            in_out_weights = "_" + Input_type + Input_bits + "_" + Output_type + Output_bits + "_" + node.get_parameter('weight_type')[0] + str(node.get_parameter('weight_bits'))
            if "Conv" in node.name and node.group > 1:
                file = 'Depthwise/pulp_nn_depthwise{}.c'.format(in_out_weights)
            elif "Conv" in node.name and node.group == 1:
                file = 'Convolution/pulp_nn_conv{}.c'.format(in_out_weights)
            elif "FullyConnected" in node.name and node.output_activation_bits == 32: 
                file = 'LinearNoQuant/pulp_nn_linear{}.c'.format(in_out_weights)
            elif "FullyConnected" in node.name:     
                file = 'LinearQuant/pulp_nn_linear{}.c'.format(in_out_weights)
            file_to_copy = os.path.join(files, "{}bit/src".format(self.source_Constant_bits_library), file)
            os.system('cp "{}" {}'.format(file_to_copy, os.path.join(self.appdir, 'DORY_network/src')))
            if ("Conv" in node.name or "FullyConnected" in node.name) and node.get_parameter('output_activation_bits') != 32:
                in_out_weights = "_" + Input_type + "8" + "_" + Output_type + Output_bits + "_" + node.get_parameter('weight_type')[0] + str(node.get_parameter('weight_bits'))
                file = 'MatrixMultiplication/pulp_nn_matmul{}.c'.format(in_out_weights)
                file_to_copy = os.path.join(files, "{}bit/src".format(self.source_Constant_bits_library), file)
                os.system('cp "{}" {}'.format(file_to_copy, os.path.join(self.appdir, 'DORY_network/src')))

    def mapping_layers_to_C_files(self):
        print("\nMapping the layers files to their templates and copying the kernels associated.")
        tmpl_dir = os.path.join(os.path.dirname(__file__), 'Templates/layer_templates')
        out_dir = '{}/DORY_network'.format(self.appdir)
        for i, node in enumerate(self.graph):
            self.copy_backend_files(node)
            if (node.tiling_dimensions["L3"]["input_dimensions"] != node.tiling_dimensions["L2"]["input_dimensions"]) or (node.tiling_dimensions["L3"]["output_dimensions"] != node.tiling_dimensions["L2"]["output_dimensions"]) or (node.tiling_dimensions["L3"]["weights_dimensions"] != node.tiling_dimensions["L2"]["weights_dimensions"]):
                Layer2D_writer.print_template_layer_L3(node, tmpl_dir, out_dir)
                if node.tiling_dimensions["L3"]["input_dimensions"][1] > node.tiling_dimensions["L2"]["input_dimensions"][1]:
                    node.tiling_dimensions["L2"]["output_dimensions"][1]  = int(np.floor((node.tiling_dimensions["L2"]["input_dimensions"][1] - node.kernel_shape[0] + node.strides[0]) / node.strides[0]))
                if node.tiling_dimensions["L3"]["output_dimensions"][1] > node.tiling_dimensions["L2"]["output_dimensions"][1]:
                    node.tiling_dimensions["L2"]["input_dimensions"][1]   = node.tiling_dimensions["L2"]["output_dimensions"][1] * node.strides[0] + node.kernel_shape[0] - node.strides[0]
                node.name = node.name + "_L2"
                padding = node.pads
                node.pads = [0, padding[1], 0, padding[3]]
                Layer2D_writer.print_template_layer(node, self.precision_library, tmpl_dir, out_dir, double_buffering = 1)
                node.name = node.name[:-3]
                if padding[0] > 0:
                    node.name = node.name + "_L2_p_t"
                    node.pads = [padding[0], padding[1], 0, padding[3]]
                    Layer2D_writer.print_template_layer(node, self.precision_library, tmpl_dir, out_dir, double_buffering = 1)
                    node.name = node.name[:-1] + "b"
                    node.pads = [0, padding[1], padding[2], padding[3]]
                    node.tiling_dimensions["L2"]["input_dimensions"][1] -= (padding[2] - ((node.tiling_dimensions["L3"]["input_dimensions"][1] + padding[0] + padding[2]) - (node.tiling_dimensions["L3"]["output_dimensions"][1]* node.strides[0] + node.kernel_shape[0] - node.strides[0])))
                    if node.tiling_dimensions["L1"]["input_dimensions"][1] > node.tiling_dimensions["L2"]["input_dimensions"][1]:
                        node.tiling_dimensions["L1"]["input_dimensions"][1] = node.tiling_dimensions["L2"]["input_dimensions"][1]
                    if node.tiling_dimensions["L1"]["output_dimensions"][1] > node.tiling_dimensions["L2"]["output_dimensions"][1]:
                        node.tiling_dimensions["L1"]["output_dimensions"][1] = node.tiling_dimensions["L2"]["output_dimensions"][1]
                    Layer2D_writer.print_template_layer(node, self.precision_library, tmpl_dir, out_dir, double_buffering = 1)
                    node.name = node.name[:-7]
            else:
                if node.tiling_dimensions["L2"]["input_dimensions"][2] == node.tiling_dimensions["L1"]["input_dimensions"][2]:
                    node.tiling_dimensions["L1"]["output_dimensions"][2] = int((node.tiling_dimensions["L1"]["input_dimensions"][2] + (node.pads[1] + node.pads[3]) - node.kernel_shape[1] + node.strides[1]) / node.strides[1])
                if node.tiling_dimensions["L2"]["input_dimensions"][1] == node.tiling_dimensions["L1"]["input_dimensions"][1]:
                    node.tiling_dimensions["L1"]["output_dimensions"][1] = int((node.tiling_dimensions["L1"]["input_dimensions"][1] + (node.pads[0] + node.pads[2]) - node.kernel_shape[0] + node.strides[0]) / node.strides[0])
                Layer2D_writer.print_template_layer(node, self.precision_library, tmpl_dir, out_dir, double_buffering = 1)


