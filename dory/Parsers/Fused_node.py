# should work even without -*-
# -*- coding: utf-8 -*-
#!/bin/bash
# PULP_node.py
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

import numpy as np
from .Layer_node import Layer_node
from .DORY_node import DORY_node

class Fused_node(Layer_node):
    # A node allocated in the PULP_Graph
    def __init__(self):
        super().__init__()
        self.node0 = False
        self.node1 = False 

    def add_memory_and_MACs(self):
        for node in ["node0", "node1"]:
            self.__dict__[node].add_existing_parameter("MACs", int(np.prod(self.__dict__[node].output_dimensions)*self.__dict__[node].output_channels*self.__dict__[node].input_channels*np.prod(self.__dict__[node].kernel_shape)/self.__dict__[node].group))
            self.__dict__[node].add_existing_parameter("weight_memory", int(self.__dict__[node].output_channels*self.__dict__[node].input_channels*np.prod(self.__dict__[node].kernel_shape)/self.__dict__[node].group*self.__dict__[node].weight_bits/8))
            self.__dict__[node].add_existing_parameter("input_activation_memory", int(np.prod(self.__dict__[node].input_dimensions)*self.__dict__[node].input_channels*self.__dict__[node].input_activation_bits/8))
            self.__dict__[node].add_existing_parameter("output_activation_memory", int(np.prod(self.__dict__[node].output_dimensions)*self.__dict__[node].output_channels*self.__dict__[node].output_activation_bits/8))
            constants_memory = 0
            bias_memory = 0
            for name in self.__dict__[node].constant_names:
                if name in ["l","k"]:
                    constants_memory+=self.__dict__[node].output_channels*self.__dict__[node].constant_bits/8
                if "bias" in name:
                    bias_memory+=self.__dict__[node].output_channels*self.__dict__[node].bias_bits/8
                self.__dict__[node].add_existing_parameter("bias_memory", int(bias_memory))
                self.__dict__[node].add_existing_parameter("constants_memory", int(constants_memory))
        self.add_existing_parameter("MACs", self.node0.MACs + self.node1.MACs)
        self.add_existing_parameter("weight_memory", self.node0.MACs + self.node1.MACs)
        self.add_existing_parameter("input_activation_memory", self.node0.input_activation_memory)
        self.add_existing_parameter("output_activation_memory", self.node1.output_activation_memory)
        self.add_existing_parameter("bias_memory", self.node0.bias_memory + self.node1.bias_memory)
        self.add_existing_parameter("constants_memory", self.node0.constants_memory + self.node1.constants_memory)
            

    def export_to_dict(self):
        node_dict = {}
        node_dict["name"] = self.name
        node_dict["Node1"] = {}
        node_dict["Node2"] = {}
        node_dict["Node1"]["DORY_node_parameters"] = {}
        node_dict["Node1"]["Layer_node_parameters"] = {}
        node_dict["Node1"]["Weights"] = {}
        node_dict["Node2"]["DORY_node_parameters"] = {}
        node_dict["Node2"]["Layer_node_parameters"] = {}
        node_dict["Node2"]["Weights"] = {}
        for key_ext, value_ext in self.__dict__.items():
            if key_ext == "node0":
                for key, value in self.__dict__[key_ext].__dict__.items():
                    if not isinstance(value, dict) and key != "name" and key in DORY_node().__dict__.keys():
                        node_dict["Node1"]["DORY_node_parameters"][key] = value
                    elif not isinstance(value, dict) and key != "name":
                        node_dict["Node1"]["Layer_node_parameters"][key] = value
                    elif key != "name":
                        node_dict["Node1"]["Weights"][key] = {}
                        node_dict["Node1"]["Weights"][key]["Present"] = 'Yes'
                        node_dict["Node1"]["Weights"][key]["Layout"] = value["layout"]
            elif key_ext == "node1":
                for key, value in self.__dict__[key_ext].__dict__.items():
                    if not isinstance(value, dict) and key != "name" and key in DORY_node().__dict__.keys():
                        node_dict["Node2"]["DORY_node_parameters"][key] = value
                    elif not isinstance(value, dict) and key != "name":
                        node_dict["Node2"]["Layer_node_parameters"][key] = value
                    elif key != "name":
                        node_dict["Node2"]["Weights"][key] = {}
                        node_dict["Node2"]["Weights"][key]["Present"] = 'Yes'
                        node_dict["Node2"]["Weights"][key]["Layout"] = value["layout"]
            else:
                if not isinstance(value_ext, dict) and key_ext != "name":
                    node_dict[key_ext] = value_ext
        return node_dict
