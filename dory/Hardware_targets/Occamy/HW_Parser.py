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
from dory.Parsers import HW_node, Layer_node
from dory.Parsers.Parser_DORY_to_HW import Parser_DORY_to_HW
from .HW_Pattern_rewriter import Pattern_rewriter
from .Tiler.tiler import Tiler

# Directory
file_path = "/".join(os.path.realpath(__file__).split("/")[:-1])


class onnx_manager(Parser_DORY_to_HW):
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    def __init__(self, graph, config_file, config_file_dir):
        layers_supported_by_HW_Backend_IR = ["Convolution", "Pooling", "FullyConnected", "Addition", "QAddition"]
        layers_supported_by_HW_Backend_IR+= ["ReluConvolution", "ReluPooling", "ReluFullyConnected", "ReluAddition", "ReluQAddition"]
        layers_supported_by_HW_Backend_IR+= ["BNReluConvolution", "RequantPooling", "BNReluFullyConnected", "BNReluAddition", "BNReluQAddition"]
        with open(os.path.join(file_path, "pattern_rules.json")) as f:
            rules = json.load(f)
        with open(os.path.join(file_path, "HW_description.json")) as f:
            HW_description = json.load(f)
        super().__init__(graph, rules, Pattern_rewriter, layers_supported_by_HW_Backend_IR, HW_description,
                         os.path.join(config_file_dir, os.path.dirname(config_file["onnx_file"])), config_file, Tiler)

    def adjust_data_layout(self):
        print("\nPULP Backend: Adjusting Data Layout to HWC and CoutKCin.")
        for i, node in enumerate(self.DORY_Graph):
            if "FullyConnected" in node.name:
                for name in node.constant_names:
                    if name not in ["l","k","outshift","outmul"]:
                        if "bias" not in name:
                            weights_name = name
                if node.__dict__[weights_name]["layout"] == "CinCout":
                    node.__dict__[weights_name]["value"] = node.__dict__[weights_name]["value"].T
                    node.__dict__[weights_name]["layout"] = "CoutCin"
                if i != 0 and self.DORY_Graph[i-1].layout == "CHW":
                    temp = node.__dict__[weights_name]["value"]
                    temp = temp.reshape(node.output_channels, self.DORY_Graph[i-1].output_channels, self.DORY_Graph[i-1].get_parameter('output_dimensions')[0], self.DORY_Graph[i-1].get_parameter('output_dimensions')[1])
                    temp = np.transpose(temp, (0, 2, 3, 1))
                    temp = temp.flatten()
                    node.__dict__[weights_name]["value"] = temp
                    # needed to compute final checksum for <8b layers
            elif "Convolution" in node.name:
                for name in node.constant_names:
                    if name not in ["l","k","outshift","outmul"]:
                        if "bias" not in name:
                            weights_name = name
                if node.__dict__[weights_name]["layout"] == "CoutCinK":
                    node.__dict__[weights_name]["value"] = np.transpose(node.__dict__[weights_name]["value"], (0, 2, 3, 1))
                    node.__dict__[weights_name]["layout"] = "CoutKCin"

    def check_parameters(self):
        WARNINGS =0
        for node in self.DORY_Graph:
            for key, value in node.__dict__.items():
                if key not in HW_node.HW_node(Layer_node.Layer_node(), self.HW_description).__dict__.keys() and key not in Layer_node.Layer_node().__dict__.keys():
                    if key not in node.constant_names:
                        print("WARNING: DORY Backend. Attribute {} of Node {} is not inside the predefined parameters for DORY nodes.".format(key, node.name))
                        WARNINGS +=1
                if isinstance(value,list):
                    if len(value) == 0:
                        WARNINGS +=1
                        print("WARNING: DORY Backend. Attribute {} of Node {} is an empty list.".format(key, node.name))
                if isinstance(value,type(None)):
                    WARNINGS +=1
                    print("WARNING: DORY Backend. Attribute {} of Node {} is still not initialized.".format(key, node.name))
        print("\nDORY checking of the attribute of the graph: {} WARNINGS\n".format(WARNINGS))
