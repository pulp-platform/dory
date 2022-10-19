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
from collections import OrderedDict
from mako.template import Template
import copy 

# DORY modules
from dory.Parsers.Parser_HW_to_C import Parser_HW_to_C
import dory.Utils.Templates_writer.Layer2D_template_writer as Layer2D_writer
import dory.Utils.Templates_writer.writer_utils as utils
from dory.Parsers.HW_node import HW_node

# Directory
file_path = "/".join(os.path.realpath(__file__).split("/")[:-1])


class C_Parser(Parser_HW_to_C):
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    def __init__(self, graph, config_file, config_file_dir, verbose_level, perf_layer, precision_library, app_directory, n_inputs = 1):
        with open(os.path.join(file_path, "HW_description.json")) as f:
            HW_description = json.load(f)
        self.precision_library = precision_library
        self.source_Constant_bits_library = config_file["BNRelu_bits"]
        self.config_file = config_file
        super().__init__(graph, os.path.join(config_file_dir, os.path.dirname(config_file["onnx_file"])), HW_description, verbose_level, perf_layer, "Makefile", app_directory, n_inputs)

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
        if os.listdir(os.path.join(files, "{}bit/include".format(self.source_Constant_bits_library)))[0] not in os.listdir(self.inc_dir):
            for file in os.listdir(os.path.join(files, "{}bit/include".format(self.source_Constant_bits_library))):
                file_to_copy = os.path.join(files, "{}bit/include".format(self.source_Constant_bits_library), file)
                os.system('cp "{}" {}'.format(file_to_copy, self.inc_dir))
        if self.precision_library == "8bit":
            if os.listdir(os.path.join(files, "{}bit/src".format(self.source_Constant_bits_library)))[0] not in os.listdir(self.src_dir):
                for file in os.listdir(os.path.join(files, "{}bit/src".format(self.source_Constant_bits_library))):
                    file_to_copy = os.path.join(files, "{}bit/src".format(self.source_Constant_bits_library), file)
                    os.system('cp "{}" {}'.format(file_to_copy, self.src_dir))
        elif self.precision_library in ["mixed-sw", "mixed-hw"]:
            Input_bits = str(node.get_parameter('input_activation_bits'))
            Output_bits = str(node.get_parameter('output_activation_bits'))
            Input_type = node.get_parameter('input_activation_type')[0]
            Output_type = node.get_parameter('output_activation_type')[0]
            out = "_" + Output_type + Output_bits
            in_out = "_" + Input_type + Input_bits + out
            maybe_x = 'x' if self.precision_library == "mixed-hw" else ''
            if "Addition" in node.name:
                in1_in2_out = "_" + Input_type + Input_bits + "_" + node.get_parameter('second_input_activation_type')[0] + str(node.get_parameter('second_input_activation_bits')) + "_" + Output_type + Output_bits
                file = f'Add/{maybe_x}pulp_nn_add{in1_in2_out}.c'
            elif "Pool" in node.name and "Max" in node.op_type:
                file = f'Pooling/MaxPool/{maybe_x}pulp_nn_maxpool{out}.c'
            elif "Pool" in node.name and ("Avg" in node.op_type or "Average" in node.op_type):
                file = f'Pooling/AvgPool/{maybe_x}pulp_nn_avgpool{in_out}.c'

            in_out_weights = "_" + Input_type + Input_bits + "_" + Output_type + Output_bits + "_" + node.get_parameter('weight_type')[0] + str(node.get_parameter('weight_bits'))
            if "Conv" in node.name and node.group > 1:
                file = f'Depthwise/{maybe_x}pulp_nn_depthwise{in_out_weights}.c'
            elif "Conv" in node.name and node.group == 1:
                if node.conv1d and self.precision_library == 'mixed-hw':
                    file = f'Convolution/xpulp_nn_conv1d{in_out_weights}.c'
                else:
                    file = f'Convolution/{maybe_x}pulp_nn_conv{in_out_weights}.c'
            elif "FullyConnected" in node.name and node.output_activation_bits == 32:
                file = f'LinearNoQuant/{maybe_x}pulp_nn_linear{in_out_weights}.c'
            elif "FullyConnected" in node.name:
                file = f'LinearQuant/{maybe_x}pulp_nn_linear{in_out_weights}.c'
            file_to_copy = os.path.join(files, "{}bit/src".format(self.source_Constant_bits_library), file)
            os.system('cp "{}" {}'.format(file_to_copy, self.src_dir))
            if ("Conv" in node.name or "FullyConnected" in node.name) and node.get_parameter('output_activation_bits') != 32:
                in_bits_matmul = "8" if self.precision_library == "mixed-sw" else str(Input_bits)
                in_out_weights = "_" + Input_type + in_bits_matmul + "_" + Output_type + Output_bits + "_" + node.get_parameter('weight_type')[0] + str(node.get_parameter('weight_bits'))
                file = f'MatrixMultiplication/{maybe_x}pulp_nn_matmul{in_out_weights}.c'
                file_to_copy = os.path.join(files, "{}bit/src".format(self.source_Constant_bits_library), file)
                os.system('cp "{}" {}'.format(file_to_copy, self.src_dir))


    def mapping_layers_to_C_files(self):
        print("\nMapping the layers files to their templates and copying the kernels associated.")
        tmpl_dir = os.path.join(os.path.dirname(__file__), 'Templates/layer_templates')
        out_dir = self.app_directory
        for i, node in enumerate(self.HWgraph):
            self.copy_backend_files(node)
            Layer2D_writer.print_template_layer(node, self.precision_library, tmpl_dir, out_dir, double_buffering = 1)

    def create_hex_weights_files(self):
        print("\nGenerating .h weight files.")
        weights_vectors = []
        weights_dimensions = []
        for i, node in enumerate(self.HWgraph):
            constants = [0, 0, 0, 0]
            for name in node.constant_names:
                if "weight" in name:
                    constants[0] = name
                elif "bias" in name:
                    constants[1] = name
                elif "k" == name:
                    constants[2] = name
                elif "l" == name:
                    constants[3] = name
            weights = np.asarray([])
            for i in np.arange(4):
                if constants[i]!= 0:
                    weights = np.concatenate((weights,node.__dict__[constants[i]]["value"]))
            while len(weights) % 4 != 0:
                weights = np.concatenate((weights, np.asarray([0])))
            ww, ww_dim = utils.print_test_vector(weights, 'char'), weights.shape[0]
            weights_vectors.append(ww)
            weights_dimensions.append(ww_dim)
        tk = OrderedDict([])
        tk['weights_vectors'] = weights_vectors
        tk['weights_dimensions'] = weights_dimensions
        tk['DORY_HW_graph'] = self.HWgraph
        tk['sdk'] = node.HW_description["software development kit"]["name"]
        root = os.path.dirname(__file__)
        tmpl = Template(filename=os.path.join(root, "Templates/weights_h_template.h"))
        s = tmpl.render(**tk)
        save_string = os.path.join(self.inc_dir, 'weights.h') 
        with open(save_string, "w") as f:
            f.write(s)
        tmpl = Template(filename=os.path.join(root, "Templates/weights_definition_h_template.h"))
        s = tmpl.render(**tk)
        save_string = os.path.join(self.inc_dir, 'weights_definition.h') 
        with open(save_string, "w") as f:
            f.write(s)

    def create_hex_input(self):
        print("\nGenerating .h input file.")
        x_in_l = []
        for in_idx in range(self.n_inputs):
            infile = 'input.txt' if self.n_inputs == 1 else f'input_{in_idx}.txt'
            try:
                x_in = np.loadtxt(os.path.join(self.network_directory, infile), delimiter=',', dtype=np.uint8, usecols=[0])
            except FileNotFoundError:
                print(f"========= WARNING ==========\nInput file {os.path.join(self.network_directory, 'input.txt')} not found; generating random inputs!")
                x_in = np.random.randint(low=0, high=2*8,
                                         size=self.group * self.input_channels * self.input_dimensions[0] * self.input_dimensions[1],
                                         dtype=np.uint8)
            x_in_l.append(x_in.flatten())

        x_in = np.concatenate(x_in_l)
        in_node = self.HWgraph[0]
        in_bits = in_node.input_activation_bits
        if in_bits != 8:
            x_in = HW_node._compress(x_in, in_bits)

        temp = x_in
        input_values = utils.print_test_vector(temp.flatten(), 'char')
        tk = OrderedDict([])
        tk['input_values'] = input_values
        tk['dimension'] = len(x_in)
        tk['sdk'] = self.HW_description["software development kit"]["name"]
        root = os.path.dirname(__file__)
        tmpl = Template(filename=os.path.join(root, "Templates/input_h_template.h"))
        s = tmpl.render(**tk)
        save_string = os.path.join(self.inc_dir, 'input.h') 
        with open(save_string, "w") as f:
            f.write(s)
