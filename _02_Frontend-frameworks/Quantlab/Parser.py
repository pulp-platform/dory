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

# DORY modules
from _00_Parsers.Parser_ONNX_to_DORY import Parser_ONNX_to_DORY
from .Pattern_rewriter import Pattern_rewriter

# Directory
file_path = "/".join(os.path.realpath(__file__).split("/")[:-1])


class onnx_manager(Parser_ONNX_to_DORY):
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.

    def __init__(self, onnx, config_file):
        layers_accepted = ['Conv', 'Pad', 'Mul', 'Add', 'Div', 'Constant', 'AveragePool', 'GlobalAveragePool', 'MaxPool', 'Cast', 'Clip', 'Floor', 'Flatten', 'Gemm', 'MatMul', 'Shape', 'Gather', 'Unsqueeze', 'Concat', 'Reshape', 'Sigmoid', 'LogSoftmax']
        layers_neglected = ['Cast', 'Floor', 'Flatten', 'Shape', 'Gather', 'Unsqueeze', 'Concat', 'Reshape', 'Sigmoid', 'LogSoftmax']
        layers_to_node = ['AveragePool', 'MaxPool', 'Conv', 'Gemm', 'MatMul', 'GlobalAveragePool']
        f = open(os.path.join(file_path, "rules.json"))
        rules = json.load(f)
        self.BNRelu_bits = config_file["BNRelu_bits"]
        super().__init__(onnx, rules, layers_accepted, layers_neglected, layers_to_node)

    def frontend_mapping_to_DORY_nodes(self):
        print("\nQuantlab Frontend: Matching patterns from generated ONNX to DORY.")
        for i, node in enumerate(self.DORY_Graph):
            string_matching, indexes = self.pattern_matching(node, i)
            if isinstance(string_matching, str):
                self.DORY_Graph = Pattern_rewriter(self.DORY_Graph).execute(string_matching, indexes)
        
    def add_nodes_precision(self):
        print("\nNEMO Frontend: Adding Bit and Types to Nodes.")
        # Right now, the precision is fixed. We can extract it from either the original onnx graph or from a json.
        for i, node in enumerate(self.DORY_Graph):
            node.add_existing_parameter("weight_type", "int")
            node.add_existing_parameter("constant_type", "int")
            node.add_existing_parameter("output_activation_type", "int")
            node.add_existing_parameter("input_activation_type", "int")
            if node.name in ["Addition"]:
                node.add_existing_parameter("output_activation_bits", node.add_bits)
            if node.name in ["Convolution", "FullyConnected"]:
                node.add_existing_parameter("output_activation_bits", 32)
            if i == 0:
                node.add_existing_parameter("input_activation_bits", 8)
            elif i > 0:
                node.add_existing_parameter("input_activation_bits", self.DORY_Graph[i-1].output_activation_bits)
            if node.name in ["QAddition", "Relu", "BNRelu", "Clip", "Mul", "Add", "Div", "Shift"]:
                node.add_existing_parameter("constant_bits", self.BNRelu_bits)
            if node.name in ["Pooling"]:
                node.add_existing_parameter("output_activation_bits", self.DORY_Graph[i].input_activation_bits)

    def add_data_layout(self):
        print("\nNEMO Frontend: Adding Data Layout.")
        for i, node in enumerate(self.DORY_Graph):
            for name in node.constant_names:
                if name not in ["l","k","outshift","outmul", "inmul1", "inmul2"]:
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



