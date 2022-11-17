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
import os
import sys
import numpy as np

# DORY modules
from Parsers.DORY_node import DORY_node
from Parsers.Layer_node import Layer_node
from Utils.DORY_utils import loadtxt


class HW_node(DORY_node):
    # A self allocated in the PULP_Graph

    # Class attributes
    Tiler = None
    weights_size = lambda self, dim: \
        np.prod(dim) / (self.group if self.group < dim[0] else dim[0]) \
        * np.prod(self.kernel_shape) * self.weight_bits / 8

    def __init__(self, node, hw_desc):
        super().__init__()
        self.__dict__ = node.__dict__

        self.tiling_dimensions = {}
        for level in range(hw_desc["memory"]["levels"]):
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
                weights_dim = [self.output_channels, self.input_channels]
                self.tiling_dimensions["L{}".format(level+1)]["weights_dimensions"] = weights_dim
                self.weight_memory = self.weights_size(weights_dim)
        self.tiling_dimensions["L{}".format(level+1)]["input_dimensions"] = [self.input_channels] + self.input_dimensions
        self.tiling_dimensions["L{}".format(level+1)]["output_dimensions"] = [self.output_channels] + self.output_dimensions
        self.tiling_dimensions["L{}".format(level+1)]["weight_memory"] = self.weight_memory
        self.tiling_dimensions["L{}".format(level+1)]["bias_memory"] = self.bias_memory
        self.tiling_dimensions["L{}".format(level+1)]["constants_memory"] = self.constants_memory
        self.tiling_dimensions["L{}".format(level+1)]["input_activation_memory"] = self.input_activation_memory
        self.tiling_dimensions["L{}".format(level+1)]["output_activation_memory"] = self.output_activation_memory
        self.hw_desc = hw_desc
        self.check_sum_w = None
        self.check_sum_in = None
        self.check_sum_out = None

    def create_tiling_dimensions(self, prev_node, config):
        #  ATTENTION MEMORY L3 --> TILE MEMORY DIMENSION --> Decide how to set. Re-init the whole memory?
        for level in range(self.hw_desc["memory"]["levels"], 1, -1):
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
                if name == 'k':
                    constants_memory += weights_dim[0] * self.constant_bits / 8
                if name == 'l':
                    constants_memory += weights_dim[0] * self.bias_bits / 8
                if "bias" in name:
                    bias_memory += weights_dim[0] * self.bias_bits / 8
            self.tiling_dimensions[mem]["bias_memory"] = int(bias_memory)
            self.tiling_dimensions[mem]["constants_memory"] = int(constants_memory)
            self.tiling_dimensions[mem]["input_activation_memory"] = np.prod(self.tiling_dimensions[mem]["input_dimensions"])*self.input_activation_bits/8
            self.tiling_dimensions[mem]["output_activation_memory"] = np.prod(self.tiling_dimensions[mem]["output_dimensions"])*self.output_activation_bits/8

    def rename_weights(self):
        weight_name = ""
        if "Convolution" in self.name or "FullyConnected" in self.name:
            for i, name in enumerate(self.constant_names):
                if name not in ["l", "k", "outshift", "outmul", "outadd"]:
                    if "bias" not in name:
                        if self.__dict__[name]["value"].size > self.output_channels:
                            self.__dict__["weights"] = self.__dict__.pop(name)
                            self.constant_names[i] = "weights"

    def __compress(self, x, bits):
        compressed = []
        n_elements_in_byte = 8 // bits
        i_element_in_byte = 0
        for el in x:
            data = el.item() & (2**bits - 1)

            if i_element_in_byte == 0:
                compressed.append(data)
            else:
                compressed[-1] += data << i_element_in_byte * bits

            i_element_in_byte += 1
            if i_element_in_byte == n_elements_in_byte:
                i_element_in_byte = 0
        return np.asarray(compressed, dtype=np.uint8)

    def add_checksum_w_integer(self):
        self.check_sum_w = 0
        bias_name = ""
        weights_name = ""

        if "Convolution" in self.name or "FullyConnected" in self.name:
            for name in self.constant_names:
                if name not in ["l", "k", "outshift", "outmul", "outadd"]:
                    if "bias" in name:
                        bias_name = name
                    else:
                        weights_name = name
        else:
            return

        if hasattr(self, weights_name):
            weights = getattr(self, weights_name)
            if self.weight_bits < 8 and self.group > 1:
                ko = weights["value"].shape[0]
                shape = (ko // 2, 2) + weights["value"].shape[1:]
                weights["value"] = weights["value"].reshape(shape).transpose(0, 2, 3, 1, 4)
            weights["value"] = self.__compress(weights["value"].ravel().astype(np.uint8), self.weight_bits)
            self.check_sum_w += sum(weights["value"])

        def to_byte(x, bits):
            x = x.ravel().astype(np.int64 if bits > 32 else np.int32)
            #### TO CHECK ORDER OF BIASES
            byte = [(el >> shift) & 255 for el in x for shift in range(0, bits, 8)]
            return np.asarray(byte, dtype=np.uint8)

        if hasattr(self, bias_name):
            bias = getattr(self, bias_name)
            bias["value"] = to_byte(bias["value"], self.bias_bits)
            self.check_sum_w += sum(bias["value"])

        if hasattr(self, 'k'):
            self.k["value"] = to_byte(self.k["value"], self.constant_bits)
            self.check_sum_w += sum(self.k["value"])

        if hasattr(self, 'l'):
            self.l["value"] = to_byte(self.l["value"], self.bias_bits)
            self.check_sum_w += sum(self.l["value"])

    def add_checksum_activations_integer(self, load_directory, i_node):
        ###########################################################################
        ###### SECTION 4: GENERATE CHECKSUM BY USING OUT_LAYER{i}.TXT FILES  ######
        ###########################################################################

        def file_checksum(filename, bits):
            filepath = os.path.join(load_directory, filename)
            try:
                data = loadtxt(filepath, dtype=np.int64)
            except ValueError:
                data = loadtxt(filepath, dtype=np.float)
            except FileNotFoundError:
                print(f"File {filename} doesn't exist. Exiting DORY...")
                sys.exit(-1)

            if bits <= 8:
                data = self.__compress(data.ravel(), bits)

            return data.sum().item()

        self.check_sum_in = file_checksum('input.txt' if i_node == 0 else f'out_layer{i_node - 1}.txt', self.input_activation_bits)
        self.check_sum_out = file_checksum(f'out_layer{i_node}.txt', self.output_activation_bits)

    def export_to_dict(self):
        node_dict = {"name": self.name, "DORY_node_parameters": {}, "Layer_node_parameters": {}, "Weights": {}}
        for key, value in self.__dict__.items():
            if not isinstance(value, dict) and key != "name" and key in DORY_node().__dict__.keys():
                node_dict["DORY_node_parameters"][key] = value
            elif not isinstance(value, dict) and key != "name" and key in Layer_node().__dict__.keys():
                node_dict["Layer_node_parameters"][key] = value
            elif key == "tiling_dimensions":
                node_dict["Tiling_parameters"] = value
            elif key in self.constant_names:
                node_dict["Weights"][key] = {"Present": 'Yes', "Layout": value["layout"]}
        return node_dict
