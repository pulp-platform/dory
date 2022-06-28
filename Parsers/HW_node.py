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

# Libraries
import numpy as np
import copy
import os

# DORY modules
from Parsers.DORY_node import DORY_node
from Parsers.Layer_node import Layer_node


class HW_node(DORY_node):
    # A self allocated in the PULP_Graph

    # Class attributes
    Tiler = None
    weights_size = lambda self, dim: np.prod(dim) / self.group * np.prod(self.kernel_shape) * self.weight_bits / 8

    def __init__(self, node, HW_description):
        super().__init__()
        self.__dict__ = node.__dict__

        self.tiling_dimensions = {}
        for level in range(HW_description["memory"]["levels"]):
            self.tiling_dimensions["L{}".format(level+1)] = {}
            self.tiling_dimensions["L{}".format(level+1)]["weights_dimensions"] = None
            self.tiling_dimensions["L{}".format(level+1)]["input_dimensions"] = None
            self.tiling_dimensions["L{}".format(level+1)]["output_dimensions"] = None
            self.tiling_dimensions["L{}".format(level+1)]["weight_memory"] = None
            self.tiling_dimensions["L{}".format(level+1)]["bias_memory"] = None
            self.tiling_dimensions["L{}".format(level+1)]["constants_memory"] = None
            self.tiling_dimensions["L{}".format(level+1)]["input_activation_memory"] = None
            self.tiling_dimensions["L{}".format(level+1)]["output_activation_memory"] = None
        if not isinstance(self.name, type(None)):
            if "Convolution" in self.name or "FullyConnected" in self.name:
                self.tiling_dimensions["L{}".format(level+1)]["weights_dimensions"] = [self.output_channels, self.input_channels]
        self.tiling_dimensions["L{}".format(level+1)]["input_dimensions"] = [self.input_channels] + self.input_dimensions
        self.tiling_dimensions["L{}".format(level+1)]["output_dimensions"] = [self.output_channels] + self.output_dimensions
        self.tiling_dimensions["L{}".format(level+1)]["weight_memory"] = self.weight_memory
        self.tiling_dimensions["L{}".format(level+1)]["bias_memory"] = self.bias_memory
        self.tiling_dimensions["L{}".format(level+1)]["constants_memory"] = self.constants_memory
        self.tiling_dimensions["L{}".format(level+1)]["input_activation_memory"] = self.input_activation_memory
        self.tiling_dimensions["L{}".format(level+1)]["output_activation_memory"] = self.output_activation_memory
        self.HW_description = HW_description
        self.check_sum_w = None
        self.check_sum_in = None
        self.check_sum_out = None

    def create_tiling_dimensions(self, prev_node, config):
        #  ATTENTION MEMORY L3 --> TILE MEMORY DIMENSION --> Decide how to set. Re-init the whole memory?
        for level in range(self.HW_description["memory"]["levels"], 1, -1):
            mem = f'L{level-1}'
            (weights_dim, input_dims, output_dims) = self.Tiler(self, prev_node, config['code reserved space']).get_tiling(level)
            self.tiling_dimensions[mem]["input_dimensions"] = input_dims
            self.tiling_dimensions[mem]["output_dimensions"] = output_dims
            if "Convolution" in self.name or "FullyConnected" in self.name:
                self.tiling_dimensions[mem]["weights_dimensions"] = weights_dim
                self.tiling_dimensions[mem]["weight_memory"] = self.weights_size(weights_dim)
            else:
                self.tiling_dimensions[mem]["weight_memory"] = 0

            constants_memory = 0
            bias_memory = 0
            for name in self.constant_names:
                if name in ["l", "k"]:
                    constants_memory += weights_dim[0]*self.constant_bits/8
                if "bias" in name:
                    bias_memory += weights_dim[0]*self.bias_bits/8
            self.tiling_dimensions[mem]["bias_memory"] = int(bias_memory)
            self.tiling_dimensions[mem]["constants_memory"] = int(constants_memory)
            self.tiling_dimensions[mem]["input_activation_memory"] = np.prod(self.tiling_dimensions[mem]["input_dimensions"])*self.input_activation_bits/8
            self.tiling_dimensions[mem]["output_activation_memory"] = np.prod(self.tiling_dimensions[mem]["output_dimensions"])*self.output_activation_bits/8

    def rename_weights(self):
        weight_name = ""
        if "Convolution" in self.name or "FullyConnected" in self.name:
            for i, name in enumerate(self.constant_names):
                if name not in ["l","k","outshift","outmul","outadd"]:
                    if "bias" not in name:
                        if len(self.__dict__[name]["value"].flatten()) > self.output_channels:
                            self.__dict__["weights"] = self.__dict__.pop(name)
                            self.constant_names[i] = "weights"

    def _compress(self, x, bits):
        compressed = []
        n_elements_in_byte = 8 // bits
        i_element_in_byte = 0
        for el in x:
            if i_element_in_byte == 0:
                compressed.append(el.item())
            else:
                compressed[-1] += el.item() << i_element_in_byte * bits

            i_element_in_byte += 1
            if i_element_in_byte == n_elements_in_byte:
                i_element_in_byte = 0
        return np.asarray(compressed, dtype=np.uint8)

    def add_checksum_w_integer(self):
        self.check_sum_w = 0
        bias_name = ""
        weight_name = ""

        if "Convolution" in self.name or "FullyConnected" in self.name:
            for name in self.constant_names:
                if name not in ["l", "k", "outshift", "outmul", "outadd"]:
                    if "bias" in name:
                        bias_name = name
                    else:
                        weight_name = name
        else:
            return

        if hasattr(self, weight_name):
            weight = getattr(self, weight_name)
            weight["value"] = self._compress(weight["value"].ravel(), self.weight_bits)
            self.check_sum_w += sum(weight["value"])

        def to_byte(x, bits):
            x = x.ravel().astype(np.int64 if bits > 32 else np.int32)
            return [np.uint8((el >> shift) & 255) for el in x for shift in range(0, bits, 8)]

        if hasattr(self, bias_name):
            bias = getattr(self, bias_name)
            bias["value"] = to_byte(bias["value"], self.bias_bits)
            self.check_sum_w += sum(bias["value"])

        if hasattr(self, 'k'):
            self.k["value"] = to_byte(self.k["value"], self.constant_bits)
            self.check_sum_w += sum(self.k["value"])

        if hasattr(self, 'l'):
            self.l["value"] = to_byte(self.l["value"], self.constant_bits)
            self.check_sum_w += sum(self.l["value"])

    def add_checksum_activations_integer(self, load_directory, node_number):
        ###########################################################################
        ###### SECTION 4: GENERATE CHECKSUM BY USING OUT_LAYER{i}.TXT FILES  ######
        ###########################################################################

        def load(filename, dtype):
            return np.loadtxt(os.path.join(load_directory, filename), delimiter=',', dtype=dtype, usecols=[0])

        filename = 'input.txt' if node_number == 0 else f'out_layer{node_number-1}.txt'

        try:
            x = load(filename, dtype=np.int64)
        except ValueError:
            x = load(filename, dtype=np.float)

        if self.input_activation_bits <= 8:
            x = self._compress(x.ravel(), self.input_activation_bits)

        self.check_sum_in = int(x.sum())

        try:
            y = load(f'out_layer{node_number}.txt', dtype=np.int64)
        except ValueError:
            y = load(f'out_layer{node_number}.txt', dtype=np.float)

        if self.output_activation_bits <= 8:
            y = self._compress(y.ravel(), self.output_activation_bits)

        self.check_sum_out = int(y.sum())

    def export_to_dict(self):
        node_dict = {}
        node_dict["name"] = self.name
        node_dict["DORY_node_parameters"] = {}
        node_dict["Layer_node_parameters"] = {}
        node_dict["Weights"] = {}
        for key, value in self.__dict__.items():
            if not isinstance(value, dict) and key != "name" and key in DORY_node().__dict__.keys():
                node_dict["DORY_node_parameters"][key] = value
            elif not isinstance(value, dict) and key != "name" and key in Layer_node().__dict__.keys():
                node_dict["Layer_node_parameters"][key] = value
            elif key == "tiling_dimensions":
                node_dict["Tiling_parameters"] = {}
                for key1, value1 in value.items():
                    node_dict["Tiling_parameters"][key1] = {}
                    for key2, value2 in value1.items():
                        node_dict["Tiling_parameters"][key1][key2] = value2
            elif key in self.constant_names:
                node_dict["Weights"][key] = {}
                node_dict["Weights"][key]["Present"] = 'Yes'
                node_dict["Weights"][key]["Layout"] = value["layout"]
        return node_dict
