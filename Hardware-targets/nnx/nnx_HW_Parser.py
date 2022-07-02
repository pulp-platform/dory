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
import os
import json

# DORY modules
from Parsers.HW_node import HW_node
from Parsers.Layer_node import Layer_node
from Parsers.Parser_DORY_to_HW import Parser_DORY_to_HW
from .HW_Pattern_rewriter import Pattern_rewriter
from .Tiler.tiler import Tiler


class nnx_HW_Parser(Parser_DORY_to_HW):
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    def __init__(self, graph, conf, confdir, accelerator):
        supported_layers = ["Convolution", "ReluConvolution", "BNReluConvolution"]
        self.nnxdir = os.path.dirname(__file__)
        with open(os.path.join(self.nnxdir, "pattern_rules.json")) as f:
            rules = json.load(f)
        with open(os.path.join(self.nnxdir, "HW_description.json")) as f:
            hw_description = json.load(f)
        self.acc = accelerator
        weights_size = self.acc.weights_size
        Tiler.acc = self.acc
        super().__init__(graph, rules, Pattern_rewriter, supported_layers, hw_description,
                         os.path.join(confdir, os.path.dirname(conf["onnx_file"])), conf, Tiler,
                         weights_size=lambda self, dim:
                         weights_size(dim[0], dim[1], self.kernel_shape, self.weight_bits, self.group > 1))

    def adjust_data_layout(self):
        print("\nNNX Backend: Adjusting Feature Data Layout to HWC and Weights Data Layout to accelerator specific")
        for i, node in enumerate(self.DORY_Graph):
            if "Convolution" in node.name:
                for name in node.constant_names:
                    if name not in ["l", "k", "outshift", "outmult"] and "bias" not in name:
                        weights_name = name
                weights = getattr(node, weights_name)
                weights["value"] = self.acc.conv_unroll(weights["value"], node.weight_bits, weights["layout"],
                                                        node.group > 1)
            # Todo elif "Fullyconnected"

    def check_parameters(self):
        warning_count = 0

        def warning(msg):
            print(f'WARNING: DORY Backend. Attribute {attr} of Node {node.name} is {msg}.')
            nonlocal warning_count
            warning_count += 1

        vanilla_attrs = list(Layer_node().__dict__.keys()) + \
                        list(HW_node(Layer_node(), self.HW_description).__dict__.keys())

        for node in self.DORY_Graph:
            for attr, value in node.__dict__.items():
                if attr not in vanilla_attrs and attr not in node.constant_names:
                    warning('not inside the predefined parameters for DORY nodes')
                if value is None:
                    warning('not initialized')
                elif isinstance(value, list) and len(value) == 0:
                    warning('an empty list')

        print(f"\nDORY checking of the attribute of the graph: {warning_count} warnings\n")
