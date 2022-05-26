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
import numpy as np
import json
import os

# DORY modules
from Parsers import HW_node, Layer_node
from Parsers.Parser_DORY_to_HW import Parser_DORY_to_HW
from .HW_Pattern_rewriter import Pattern_rewriter
from .Tiler.tiler import Tiler
from .ne16 import conv_unroll


class onnx_manager(Parser_DORY_to_HW):
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    def __init__(self, graph, conf_file, conf_file_dir):
        supported_layers = ["Convolution", "ReluConvolution", "BNReluConvolution"]
        mod_dir = os.path.dirname(__file__)
        with open(os.path.join(mod_dir, "pattern_rules.json")) as f:
            rules = json.load(f)
        with open(os.path.join(mod_dir, "HW_description.json")) as f:
            hw_description = json.load(f)
        super().__init__(graph, rules, Pattern_rewriter, supported_layers, hw_description,
                         os.path.join(conf_file_dir, os.path.dirname(conf_file["onnx_file"])), conf_file, Tiler)

    def adjust_data_layout(self):
        print("\nNNX Backend: Adjusting Feature Data Layout to HWC and Weights Data Layout to accelerator specific")
        for i, node in enumerate(self.DORY_Graph):
            if "Convolution" in node.name:
                for name in node.constant_names:
                    if name not in ["l", "k", "outshift", "outmult"] and "bias" not in name:
                        weights_name = name
                qw = node.weight_bits
                layout = node.__dict__[weights_name]["layout"]
                dw = node.group > 1
                w = node.__dict__[weights_name]["value"].astype(np.uint8)
                node.__dict__[weights_name]["value"] = conv_unroll(w, qw, layout, dw)
            # Todo elif "Fullyconnected"

    def check_parameters(self):
        warning_count = 0

        def warning(attr, node, msg):
            print(f'WARNING: DORY Backend. Attribute {attr} of Node {node} {msg}')

        for node in self.DORY_Graph:
            for key, value in node.__dict__.items():
                if key not in HW_node.HW_node(Layer_node.Layer_node(), self.HW_description).__dict__.keys() and \
                        key not in Layer_node.Layer_node().__dict__.keys() and \
                        key not in node.constant_names:
                    warning(key, node.name, 'is not inside the predefined parameters for DORY nodes.')
                    warning_count += 1
                if value is None:
                    warning_count += 1
                    warning(key, node.name, 'is still not initialized.')
                elif isinstance(value, list) and len(value) == 0:
                    warning_count += 1
                    warning(key, node.name, 'is an empty list.')
        print(f"\nDORY checking of the attribute of the graph: {warning_count} warnings\n")
