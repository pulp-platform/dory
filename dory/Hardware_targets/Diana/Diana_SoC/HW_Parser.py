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
    def __init__(self, graph, config_file, config_file_dir, n_inputs=1):
        layers_supported_by_HW_Backend_IR = ["Convolution", "Pooling", "FullyConnected", "Addition", "QAddition"]
        layers_supported_by_HW_Backend_IR+= ["ReluConvolution", "ReluPooling", "ReluFullyConnected", "ReluAddition", "ReluQAddition"]
        layers_supported_by_HW_Backend_IR+= ["BNReluConvolution", "RequantPooling", "BNReluFullyConnected", "BNReluAddition", "BNReluQAddition"]
        with open(os.path.join(file_path, "pattern_rules.json")) as f:
            rules = json.load(f)
        with open(os.path.join(file_path, "HW_description.json")) as f:
            HW_description = json.load(f)
        super().__init__(graph, rules, Pattern_rewriter, layers_supported_by_HW_Backend_IR, HW_description,
                         os.path.join(config_file_dir, os.path.dirname(config_file["onnx_file"])), config_file, Tiler)

    def adjust_dimensions(self):
        for i, node in enumerate(self.DORY_Graph):
            if "FullyConnected" not in node.name and node.weight_bits == 8:
                node.input_dimensions[1] = int((node.input_dimensions[1] + 15) / 16) * 16
                node.output_dimensions[1] = int((node.output_dimensions[1] + 15) / 16) * 16
                print("\nFind One other solution, It will not work for real networks with multiple strides = 2")
            
    def adjust_data_layout(self):
        self.adjust_dimensions()
        print("\nDiana Backend: Adjusting Data Layout to HWC and CoutKCin.")
        for i, node in enumerate(self.DORY_Graph):
            if "FullyConnected" in node.name:
                for name in node.constant_names:
                    if name not in ["l","k","outshift","outmul"]:
                        if "bias" not in name:
                            weights_name = name
                if node.__dict__[weights_name]["layout"] == "CinCout":
                    node.__dict__[weights_name]["value"] = node.__dict__[weights_name]["value"].T
                    node.__dict__[weights_name]["layout"] = "CoutCin"
                npad = ((0, (16 - (node.__dict__[weights_name]["value"].shape[0] % 16)) % 16), (0, (16 - (node.__dict__[weights_name]["value"].shape[1] % 16)) % 16), (0, 0), (0,0))
                temp = np.pad(node.__dict__[weights_name]["value"], pad_width=npad, mode='constant', constant_values=0)
                temp = np.transpose(temp, (0, 2, 3, 1))
                temp = temp.reshape(int(temp.shape[0]/16), 16,temp.shape[1],temp.shape[2],int(temp.shape[3]/16), 16)
                temp = np.transpose(temp, (0, 4, 1, 2, 3, 5 ))
                node.__dict__[weights_name]["value"] = temp
                node.__dict__[weights_name]["layout"] = "Cout2Cin2Cout1KCin1"
                for name in node.constant_names:
                    if name not in ["l","k","outshift","outmul"]:
                        if "bias" in name:
                            weights_name = name
                            npad = ((0, (16 - (node.__dict__[weights_name]["value"].shape[0] % 16)) % 16))
                            node.__dict__[weights_name]["value"] = np.pad(node.__dict__[weights_name]["value"], pad_width=npad, mode='constant', constant_values=0)
            elif "Convolution" in node.name:
                for name in node.constant_names:
                    if name not in ["l","k","outshift","outmul"]:
                        if "bias" not in name:
                            weights_name = name
                if node.__dict__[weights_name]["layout"] == "CoutCinK":
                    if node.get_parameter('weight_bits') < 8:
                        node.__dict__[weights_name]["value"] = np.transpose(node.__dict__[weights_name]["value"], (1, 2, 3, 0))
                        node.__dict__[weights_name]["layout"] = "CinKCout"
                    else:
                        if node.group == 1:
                            npad = ((0, (16 - (node.__dict__[weights_name]["value"].shape[0] % 16)) % 16), (0, 0), (0, 0), (0,0))
                            temp = np.pad(node.__dict__[weights_name]["value"], pad_width=npad, mode='constant', constant_values=0)
                            temp = np.transpose(temp, (1, 2, 3, 0))
                            temp = temp.reshape(temp.shape[0],temp.shape[1],temp.shape[2],int(temp.shape[3]/16), 16)
                            temp = np.transpose(temp, (3, 0, 1, 2, 4))
                        else:
                            for ch in np.arange(node.__dict__[weights_name]["value"].shape[0]):
                                if ch == 0:
                                    temp = np.concatenate((node.__dict__[weights_name]["value"][ch,:,:,:].reshape(1,node.__dict__[weights_name]["value"].shape[1],node.__dict__[weights_name]["value"].shape[2],node.__dict__[weights_name]["value"].shape[3]), np.zeros((3, 1, node.__dict__[weights_name]["value"].shape[2], node.__dict__[weights_name]["value"].shape[3]))), axis = 0)
                                else:
                                    temp1 = np.concatenate((node.__dict__[weights_name]["value"][ch,:,:,:].reshape(1,node.__dict__[weights_name]["value"].shape[1],node.__dict__[weights_name]["value"].shape[2],node.__dict__[weights_name]["value"].shape[3]), np.zeros((3, 1, node.__dict__[weights_name]["value"].shape[2], node.__dict__[weights_name]["value"].shape[3]))), axis = 0)
                                    temp = np.concatenate((temp, temp1), axis=0)
                            temp = np.transpose(temp, (1, 2, 3, 0))
                            temp = temp.reshape(temp.shape[0],temp.shape[1],temp.shape[2],int(temp.shape[3]/16), 16)
                            temp = np.transpose(temp, (3, 0, 1, 2, 4))

                            # for ch in np.arange(node.__dict__[weights_name]["value"].shape[0]):
                            #     if ch == 0:
                            #         temp = node.__dict__[weights_name]["value"][ch,:,:,:].reshape(1,node.__dict__[weights_name]["value"].shape[1],node.__dict__[weights_name]["value"].shape[2],node.__dict__[weights_name]["value"].shape[3])
                            #     else:
                            #         temp1 = node.__dict__[weights_name]["value"][ch,:,:,:].reshape(1,node.__dict__[weights_name]["value"].shape[1],node.__dict__[weights_name]["value"].shape[2],node.__dict__[weights_name]["value"].shape[3])
                            #         temp = np.concatenate((temp, temp1), axis=0)
                            # temp = np.transpose(temp, (1, 2, 3, 0))
                            # temp = temp.reshape(temp.shape[0],temp.shape[1],temp.shape[2],int(temp.shape[3]/16), 16)
                            # temp = np.transpose(temp, (3, 0, 1, 2, 4))

                        node.__dict__[weights_name]["value"] = temp
                        node.__dict__[weights_name]["layout"] = "Cout2CinKCout1"
                for name in node.constant_names:
                    if name not in ["l","k","outshift","outmul"]:
                        if "bias" in name:
                            weights_name = name
                            if node.group == 1:
                                npad = ((0, (16 - (node.__dict__[weights_name]["value"].shape[0] % 16)) % 16))
                                node.__dict__[weights_name]["value"] = np.pad(node.__dict__[weights_name]["value"], pad_width=npad, mode='constant', constant_values=0)
                            else:
                                npad = ((0, node.__dict__[weights_name]["value"].shape[0] * 3))
                                for ch in np.arange(node.__dict__[weights_name]["value"].shape[0]):
                                    if ch == 0:
                                        temp = np.concatenate((node.__dict__[weights_name]["value"][ch].reshape(1), np.zeros(3)), axis = 0)
                                    else:
                                        temp1 = np.concatenate((node.__dict__[weights_name]["value"][ch].reshape(1), np.zeros(3)), axis = 0)
                                        temp = np.concatenate((temp, temp1), axis=0)
                                node.__dict__[weights_name]["value"] = np.asarray(temp)
                                # npad = ((0, node.__dict__[weights_name]["value"].shape[0] * 15))
                                # for ch in np.arange(node.__dict__[weights_name]["value"].shape[0]):
                                #     if ch == 0:
                                #         temp = node.__dict__[weights_name]["value"][ch].reshape(1)
                                #     else:
                                #         temp1 = node.__dict__[weights_name]["value"][ch].reshape(1)
                                #         temp = np.concatenate((temp, temp1), axis=0)
                                # node.__dict__[weights_name]["value"] = np.asarray(temp)

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

    def formatting_constant_parameters_tensors_and_activations(self):
        print("\nDiana Backend: Formatting constants and adding checksums")
        for i, node in enumerate(self.DORY_Graph):         
            if node.get_parameter('weight_bits') >= 8:  
                node.add_checksum_w_integer()           
                node.add_checksum_activations_integer(self.network_directory, i)
            else:
                weight_name = ""
                if "Convolution" in node.name or "FullyConnected" in node.name:
                    for name in node.constant_names:
                        if name not in ["l","k","outshift","outmul","outadd"]:
                            if "bias" not in name:
                                weight_name = name
                if weight_name in node.__dict__:
                    node.__dict__[weight_name]["value"] = node.__dict__[weight_name]["value"].flatten().tolist()
                node.check_sum_w = 0
                node.add_checksum_activations_integer(self.network_directory, i)
