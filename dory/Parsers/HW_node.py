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
from .DORY_node import DORY_node
from .Layer_node import Layer_node


class HW_node(DORY_node):
    # A self allocated in the PULP_Graph

    # Class attributes
    Tiler = None

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
        self.L3_input = 0
        try:
            self.split_ints = HW_description['split_ints']
        except KeyError:
            self.split_ints = False

    def create_tiling_dimensions(self, previous_node, config_file):
        #  ATTENTION MEMORY L3 --> TILE MEMORY DIMENSION --> Decide how to set. Re-init the whole memory?
        for level in np.arange(self.HW_description["memory"]["levels"],1, -1):
            (weights_dim, input_dims, output_dims) = self.Tiler(self, previous_node, config_file["code reserved space"]).get_tiling(level)
            assert all(dim > 0 for dim in weights_dim + input_dims + output_dims)
            self.tiling_dimensions["L{}".format(level-1)]["input_dimensions"] = input_dims
            self.tiling_dimensions["L{}".format(level-1)]["output_dimensions"] = output_dims
            if "Convolution" in self.name or "FullyConnected" in self.name:
                self.tiling_dimensions["L{}".format(level-1)]["weights_dimensions"] = weights_dim
                #groups = self.group if self.group < weights_dim[0] else
                #weights_dim[0] # not really correct: If we tile a grouped
                #conv, the effective number of groups is the higher of the two
                #channel numbers
                groups = self.group if all(self.group <= d for d in weights_dim) else max(weights_dim)

                self.tiling_dimensions["L{}".format(level-1)]["weight_memory"] = np.prod(weights_dim)/groups*np.prod(self.kernel_shape)*self.weight_bits/8
            else:
                self.tiling_dimensions["L{}".format(level-1)]["weight_memory"] = 0
            constants_memory = 0
            bias_memory = 0
            for name in self.constant_names:
                if name in ["l","k"]:
                    constants_memory+=weights_dim[0]*self.constant_bits/8
                if "bias" in name:
                    if groups == 1:
                        bias_memory+=weights_dim[0]*self.bias_bits/8
                    else:
                        bias_memory+=weights_dim[0]*self.bias_bits/8*16

            self.tiling_dimensions["L{}".format(level-1)]["bias_memory"] = int(bias_memory)
            self.tiling_dimensions["L{}".format(level-1)]["constants_memory"] = int(constants_memory)
            self.tiling_dimensions["L{}".format(level-1)]["input_activation_memory"] = int(np.prod(self.tiling_dimensions["L{}".format(level-1)]["input_dimensions"])*self.input_activation_bits/8)
            self.tiling_dimensions["L{}".format(level-1)]["output_activation_memory"] = int(np.prod(self.tiling_dimensions["L{}".format(level-1)]["output_dimensions"])*self.output_activation_bits/8)

    def rename_weights(self):
        weight_name = ""
        if "Convolution" in self.name or "FullyConnected" in self.name:
            for i, name in enumerate(self.constant_names):
                if name not in ["l","k","outshift","outmul","outadd"]:
                    if "bias" not in name:
                        if len(self.__dict__[name]["value"].flatten()) > self.output_channels:
                            self.__dict__["weights"] = self.__dict__.pop(name)
                            self.constant_names[i] = "weights"

    @staticmethod
    def _compress(x, bits):
        compressed = []
        n_elements_in_byte = 8 // bits
        i_element_in_byte = 0
        x_masked = x & ((2**bits) - 1)
        x_reshaped_masked = x_masked.reshape((-1, n_elements_in_byte))
        po2 = 2**(np.arange(n_elements_in_byte) * bits)
        po2 = np.tile(po2, (x_reshaped_masked.shape[0], 1))
        x_reshaped_masked_scaled = x_reshaped_masked * po2
        x_out = np.sum(x_reshaped_masked_scaled, axis=1).flatten().astype(np.uint8)
        return x_out

    @staticmethod
    def _to_uint8(x, bits):
        #import ipdb; ipdb.set_trace()
        n_mult = bits//8
        x = np.tile(x[:, None], (1, n_mult))
        shifts = np.tile(8 * np.arange(n_mult), (x.shape[0], 1))
        x_shift_masked = (x >> shifts) & 255
        x_flat = x_shift_masked.ravel().astype(np.uint8)
        return x_flat

    def add_checksum_w_integer(self):
        self.check_sum_w = 0

        weight_name = ""
        if "Convolution" in self.name or "FullyConnected" in self.name:
            for name in self.constant_names:
                if name not in ["l","k","outshift","outmul","outadd"]:
                    if "bias" not in name:
                        weight_name = name
        if weight_name in self.__dict__:
            if self.weight_bits < 8 and self.group > 1:
                self.__dict__[weight_name]["value"] = np.asarray(self.__dict__[weight_name]["value"])
                self.__dict__[weight_name]["value"] = self.__dict__[weight_name]["value"].reshape(self.__dict__[weight_name]["value"].shape[0]//2,2,self.__dict__[weight_name]["value"].shape[1],self.__dict__[weight_name]["value"].shape[2],self.__dict__[weight_name]["value"].shape[3]).transpose(0,2,3,1,4).flatten()
            else:
                self.__dict__[weight_name]["value"] = self.__dict__[weight_name]["value"].flatten()
            # self.__dict__[weight_name+"_raw"] = self.__dict__[weight_name]
            self.__dict__[weight_name]["value"] = self.__dict__[weight_name]["value"].astype(np.uint8)
            if self.weight_bits != 8:
                self.__dict__[weight_name]["value"] = self._compress(self.__dict__[weight_name]["value"], self.weight_bits)
            self.check_sum_w += sum(self.__dict__[weight_name]["value"])

        bias_name = ""
        if "Convolution" in self.name or "FullyConnected" in self.name:
            for name in self.constant_names:
                if name not in ["l","k","outshift","outmul","outadd"]:
                    if "bias" in name:
                        bias_name = name

        def to_byte(x, bits):
            x = x.ravel().astype(np.int64 if bits > 32 else np.int32)
            #### TO CHECK ORDER OF BIASES
            return [np.uint8((el >> shift) & 255) for el in x for shift in range(0, bits, 8)]

        if bias_name in self.__dict__:
            self.__dict__[bias_name]["value"] = self._to_uint8(self.__dict__[bias_name]['value'].astype(np.int64).ravel(), self.bias_bits)
            self.check_sum_w += sum(self.__dict__[bias_name]["value"])

        if 'k' in self.__dict__:
            self.k["value"] = self._to_uint8(self.k['value'].astype(np.int64).ravel(), self.constant_bits)
            self.check_sum_w += sum(self.k["value"])

        if 'l' in self.__dict__:
            self.l["value"] = self._to_uint8(self.l['value'].astype(np.int64).ravel(), self.constant_bits)
            self.check_sum_w += sum(self.l["value"])

    def add_checksum_activations_integer(self, load_directory, node_number, n_inputs=1):
        ###########################################################################
        ###### SECTION 4: GENERATE CHECKSUM BY USING OUT_LAYER{i}.TXT FILES  ######
        ###########################################################################
        self.check_sum_in = []
        self.check_sum_out = []
        for in_idx in range(n_inputs):
            if node_number == 0:
                infile = 'input.txt' if n_inputs == 1 else f'input_{in_idx}.txt'
                try:
                    try:
                        x = np.loadtxt(os.path.join(load_directory, infile), delimiter=',', dtype=np.uint8, usecols=[0])
                    except ValueError:
                        x = np.loadtxt(os.path.join(load_directory, infile), delimiter=',', dtype=np.float, usecols=[0]).astype(np.int64)
                    x = x.ravel()
                    if self.input_activation_bits <= 8:
                        x = self._compress(x, self.input_activation_bits)
                except FileNotFoundError:
                    print("========= WARNING ==========")
                    print(f"Input file {os.path.join(load_directory, 'input.txt')} not found; generating random inputs!")
                    x = np.random.randint(low=0, high=2**8 - 1,
                                             size=self.input_channels * self.input_dimensions[0] * self.input_dimensions[1],
                                             dtype=np.uint8)
            else:
                infile = f'out_layer{node_number-1}.txt' if n_inputs == 1 else f'out_{in_idx}_layer{node_number-1}.txt'
                try:
                    x = np.loadtxt(os.path.join(load_directory, infile), delimiter=',', dtype=np.int64, usecols=[0])
                except ValueError:
                    x = np.loadtxt(os.path.join(load_directory, infile), delimiter=',', dtype=np.float, usecols=[0]).astype(np.int64)
                if self.input_activation_bits <= 8:
                    x = self._compress(x.ravel(), self.input_activation_bits)

            self.check_sum_in.append(int(sum(x)))
            outfile = f'out_layer{node_number}.txt' if n_inputs == 1 else f'out_{in_idx}_layer{node_number}.txt'
            # quantlib hack
            if not os.path.isfile(os.path.join(load_directory, outfile)):
                outfile = "output.txt"
            try:
                y = np.loadtxt(os.path.join(load_directory, outfile), delimiter=',', dtype=np.int64, usecols=[0])
            except ValueError:
                y = np.loadtxt(os.path.join(load_directory, outfile), delimiter=',', dtype=np.float, usecols=[0]).astype(np.int64)
            if self.output_activation_bits <= 8:
                y = self._compress(y.ravel(), self.output_activation_bits)
            elif self.split_ints and self.output_activation_bits > 8:
                y = self._to_uint8(y.ravel(), self.output_activation_bits)

            self.check_sum_out.append(int(y.sum()))

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
