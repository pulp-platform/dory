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
import numpy as np

# DORY modules
from Parsers.HW_node import HW_node
from Parsers.Layer_node import Layer_node
from Parsers.Parser_DORY_to_HW import Parser_DORY_to_HW
from .HW_Pattern_rewriter import Pattern_rewriter
from .Tiler.tiler import Tiler


class nnx_HW_Parser(Parser_DORY_to_HW):
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    def __init__(self, graph, conf, confdir, accelerator, hw_desc_path):
        supported_layers = ["Convolution", "ReluConvolution", "BNReluConvolution", "RequantPooling", "Addition", "FullyConnected",
                            "BNReluConvolutionDepthwisePointwise"]
        self.nnxdir = os.path.dirname(__file__)
        with open(os.path.join(self.nnxdir, "pattern_rules.json")) as f:
            rules = json.load(f)
        with open(hw_desc_path, 'r') as f:
            hw_desc = json.load(f)
        self.acc = accelerator
        acc_weights_size = self.acc.weights_size
        def weights_size(self, dim):
            if "DepthwisePointwise" in self.name:
                return acc_weights_size(dim[1], dim[1], self.kernel_shape, self.weight_bits, dw=True) + \
                       acc_weights_size(dim[0], dim[1], [1, 1], self.weight_bits, dw=False)
            else:
                return acc_weights_size(dim[0], dim[1], self.kernel_shape, self.weight_bits, self.group > 1)
        Tiler.acc = self.acc
        super().__init__(graph, rules, Pattern_rewriter, supported_layers, hw_desc,
                         os.path.join(confdir, os.path.dirname(conf["onnx_file"])), conf, Tiler,
                         weights_size=weights_size)

    def adjust_data_layout(self):
        print("\nNNX Backend: Adjusting Feature Data Layout to HWC and Weights Data Layout to accelerator specific")
        for i, node in enumerate(self.DORY_Graph):
            if 'DepthwisePointwise' in node.name:
                weights_name_dw = node.weights_names[0]
                weights_dw = getattr(node, weights_name_dw)
                weights_dw["value"] = self.acc.conv_unroll(weights_dw["value"].astype(np.int32),
                                                           node.weight_bits, weights_dw["layout"],
                                                           dw=True)

                weights_name_pw = node.weights_names[1]
                weights_pw = getattr(node, weights_name_pw)
                weights_pw["value"] = self.acc.conv_unroll(weights_pw["value"].astype(np.int32),
                                                           node.weight_bits, weights_pw["layout"],
                                                           dw=False)

            elif 'Convolution' in node.name or 'FullyConnected' in node.name:
                for name in node.constant_names:
                    if name not in ["l", "k", "outshift", "outmult"] and "bias" not in name:
                        weights_name = name
                weights = getattr(node, weights_name)

                if 'FullyConnected' in node.name and i > 0:
                    if weights["layout"] == "CinCout":
                        weights["value"] = weights["value"].T
                        weights["layout"] = "CoutCin"
                    prev = self.DORY_Graph[i - 1]
                    if prev.layout == "CHW":
                        temp = weights["value"]
                        temp = temp.reshape(node.output_channels, prev.output_channels, prev.output_dimensions[0], prev.output_dimensions[1])
                        temp = np.transpose(temp, (0, 2, 3, 1))
                        temp = temp.reshape(node.output_channels, -1)
                        weights["value"] = temp
                        # needed to compute final checksum for <8b layers

                weights["value"] = self.acc.conv_unroll(weights["value"].astype(np.int32), node.weight_bits, weights["layout"],
                                                        node.group > 1)

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
