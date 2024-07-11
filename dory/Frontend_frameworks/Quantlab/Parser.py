     # should work even without -*-
# -*- coding: utf-8 -*-
#!/bin/bash
# ONNX_management.py
# Alessio Burrello <alessio.burrello@unibo.it>
# Thorir Mar Ingolfsson <thoriri@iis.ee.ethz.ch>
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
from dory.Frontend_frameworks.Quantlab.Pattern_rewriter import Pattern_rewriter
from dory.Parsers.Parser_ONNX_to_DORY import Parser_ONNX_to_DORY

# Directory
file_path = "/".join(os.path.realpath(__file__).split("/")[:-1])


class onnx_manager(Parser_ONNX_to_DORY):
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.

    def __init__(self, onnx, config_file, net_prefix="", n_trainable_layers=0):
        layers_accepted = ['Conv', 'Pad', 'Mul', 'Add', 'Div', 'Constant', 'AveragePool', 'GlobalAveragePool', 'MaxPool', 'Cast', 'Clip', 'Floor', 'Flatten', 'Gemm', 'MatMul', 'Shape', 'Gather', 'Unsqueeze', 'Squeeze', 'Concat', 'Reshape', 'Sigmoid', 'LogSoftmax']
        layers_neglected = ['Cast', 'Floor', 'Flatten', 'Shape', 'Gather', 'Unsqueeze', 'Concat', 'Reshape', 'Sigmoid', 'LogSoftmax', 'Squeeze']
        layers_to_node = ['AveragePool', 'MaxPool', 'Conv', 'Gemm', 'MatMul', 'GlobalAveragePool']
        with open(os.path.join(file_path, "rules.json"), 'r') as f:
            rules = json.load(f)
        self.BNRelu_bits = config_file["BNRelu_bits"]
        try:
            self.input_signed = config_file["input_signed"]
            self.input_bits = config_file["input_bits"]
        except KeyError:
            print("'input_signed' and/or 'input_bits' not found in network configuration JSON - setting input format to uint8!")
            self.input_signed = False
            self.input_bits = 8
        try:
            self.n_test_inputs = config_file["n_inputs"]
        except KeyError:
            self.n_test_inputs = 1
        super().__init__(onnx, rules, layers_accepted, layers_neglected, layers_to_node, net_prefix, n_trainable_layers)

    def frontend_mapping_to_DORY_nodes(self):
        print("\nQuantlab Frontend: Matching patterns from generated ONNX to DORY.")
        for node in self.DORY_Graph:
            node.add_existing_parameter('n_test_inputs', self.n_test_inputs)
        for i, node in enumerate(self.DORY_Graph):
            string_matching, indexes = self.pattern_matching(node, i)
            if isinstance(string_matching, str):
                self.DORY_Graph = Pattern_rewriter(self.DORY_Graph).execute(string_matching, indexes)
        print("\nQuantlab Frontend: Updating Add nodes with constants, adding 'conv1d' flags and extending 1D parameters to 2D.")
        for i, node in enumerate(self.DORY_Graph):
            if "Conv" in node.name:
                if len(node.kernel_shape) == 1:
                    node.conv1d = 1
                    node.kernel_shape = [1] + node.kernel_shape
                    node.dilations = [1] + node.dilations
                    node.strides = [1] + node.strides
                else:
                    node.conv1d = 0
            if "Addition" in node.name:
                ## output parameters
                node.outshift = {}
                node.outshift["value"] = node.out_shift
                node.outshift["layout"] = ""
                node.constant_names.append("outshift")
                delattr(node, 'out_shift')
                node.outmul = {}
                node.outmul["value"] = node.out_mul
                node.outmul["layout"] = ""
                node.constant_names.append("outmul")
                delattr(node, 'out_mul')
                node.outadd = {}
                node.outadd["value"] = node.out_add
                node.outadd["layout"] = ""
                node.constant_names.append("outadd")
                delattr(node, 'out_add')
                # input 1 parameters
                #### Look at the order of inputs in Onnx. If the lowest index
                #is not the first argument, revert the order inmul1 and inmul2
                def pos_in_schedule(node: str) -> int:
                    for idx, node_iterating in enumerate(list(self.graph.graph.node)):
                        if node_iterating.output[0] == node:
                            return idx
                if pos_in_schedule(node.input_indexes[0]) > pos_in_schedule(node.input_indexes[1]):
                    temp_shift = node.in1_shift
                    temp_mul   = node.in1_mul
                    temp_add   = node.in1_add
                    node.in1_shift = node.in2_shift
                    node.in1_mul   = node.in2_mul
                    node.in1_add   = node.in2_add
                    node.in2_shift = temp_shift
                    node.in2_mul   = temp_mul
                    node.in2_add   = temp_add
                node.inshift1 = {}
                node.inshift1["value"] = node.in1_shift
                node.inshift1["layout"] = ""
                node.constant_names.append("inshift1")
                delattr(node, 'in1_shift')
                node.inmul1 = {}
                node.inmul1["value"] = node.in1_mul
                node.inmul1["layout"] = ""
                node.constant_names.append("inmul1")
                delattr(node, 'in1_mul')
                node.inadd1 = {}
                node.inadd1["value"] = node.in1_add
                node.inadd1["layout"] = ""
                node.constant_names.append("inadd1")
                delattr(node, 'in1_add')
                # input 2 parameters
                node.inshift2 = {}
                node.inshift2["value"] = node.in2_shift
                node.inshift2["layout"] = ""
                node.constant_names.append("inshift2")
                delattr(node, 'in2_shift')
                node.inmul2 = {}
                node.inmul2["value"] = node.in2_mul
                node.inmul2["layout"] = ""
                node.constant_names.append("inmul2")
                delattr(node, 'in2_mul')
                node.inadd2 = {}
                node.inadd2["value"] = node.in2_add
                node.inadd2["layout"] = ""
                node.constant_names.append("inadd2")
                delattr(node, 'in2_add')
                # removing irrelevant parameters
                delattr(node, 'in2_n_levels')
                delattr(node, 'in2_rq')
                #delattr(node, 'out_n_levels')
                delattr(node, 'out_rq')
                delattr(node, 'in1_n_levels')
                delattr(node, 'in1_rq')


    def add_nodes_precision(self):
        print("\nQuantlab Frontend: Adding Bit and Types to Nodes.")
        # Right now, the precision is fixed. We can extract it from either the original onnx graph or from a json.
        for i, node in enumerate(self.DORY_Graph):
            node.add_existing_parameter("weight_type", "int")
            node.add_existing_parameter("constant_type", "int")
            node.add_existing_parameter("bias_bits", 32)
            if i == 0:
                ty = "int" if self.input_signed else "uint"
                node.add_existing_parameter("input_activation_type", ty)
                node.add_existing_parameter("input_activation_bits", self.input_bits)
            else:
                for previous_nodes in self.DORY_Graph:
                    if previous_nodes.output_index == self.DORY_Graph[i].input_indexes[0]:
                        node.add_existing_parameter("input_activation_type", previous_nodes.output_activation_type)
                        node.add_existing_parameter("input_activation_bits", previous_nodes.output_activation_bits)
                if len(self.DORY_Graph[i].input_indexes) == 2:
                    for previous_nodes in self.DORY_Graph:
                        if previous_nodes.output_index == self.DORY_Graph[i].input_indexes[1]:
                            node.second_input_activation_type = previous_nodes.output_activation_type
                            node.second_input_activation_bits = previous_nodes.output_activation_bits
            if node.name in ["Addition"]:
                node.add_existing_parameter("output_activation_bits", int(np.log2(node.out_n_levels)))
                delattr(node, "add_bits")
                delattr(node, "out_n_levels")
                # if the node has the "out_signed" attribute, use it to set the
                # output act type. Otherwise, infer it from the input types 
                if hasattr(node, "out_signed"):
                    out_type = "int" if node.out_signed else "uint"
                    delattr(node, "out_signed")
                else:
                    out_type = "uint" if node.input_activation_type == "uint" and node.second_input_activation_type == "uint" else "int"
                
                node.add_existing_parameter("output_activation_type", out_type)
            if node.name in ["Convolution", "FullyConnected"]:
                node.add_existing_parameter("output_activation_bits", 32)
                # before merging with activations, conv/FC always have 32b int outputs
                node.add_existing_parameter("output_activation_type", "int")
            if node.name in ["QAddition", "Relu", "BNRelu", "Clip", "Mul", "Add", "Div", "Shift"]:
                node.add_existing_parameter("constant_bits", self.BNRelu_bits)
            if node.name in ["Pooling"]:
                node.add_existing_parameter("output_activation_bits", self.DORY_Graph[i].input_activation_bits)
                node.add_existing_parameter("output_activation_type", self.DORY_Graph[i].input_activation_type)
        #node.add_existing_parameter("output_activation_type", "int") # last
        #node is NOT always int!

    def add_data_layout(self):
        print("\nQuantlab Frontend: Adding Data Layout.")
        for i, node in enumerate(self.DORY_Graph):
            for name in node.constant_names:
                if name not in ["l","k","outshift","outmul","outadd", "inmul1", "inmul2", "inshift1", "inshift2", "inadd1", "inadd2"]:
                    if "bias" not in name:
                        weights_name = name
            if weights_name in node.__dict__:
                if node.name in "FullyConnected":
                    if node.__dict__[weights_name]["value"].shape[0] == node.input_channels:
                        node.__dict__[weights_name]["layout"] = "CinCout"
                    else:
                        node.__dict__[weights_name]["layout"] = "CoutCin"
                else:
                    node.__dict__[weights_name]["layout"] = "CoutCinK"
            node.layout = "CHW"
