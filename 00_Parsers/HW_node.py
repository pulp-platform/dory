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
import pandas as pd 
import torch

## DORY modules
from DORY_node import DORY_node
from Layer_node import Layer_node
from tiler import Tiler

class HW_node(DORY_node):
    # A self allocated in the PULP_Graph
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

    def create_tiling_dimensions(self, previous_node):
        #  ATTENTION MEMORY L3 --> TILE MEMORY DIMENSION --> Decide how to set. Re-init the whole memory?
        for level in np.arange(self.HW_description["memory"]["levels"],1, -1):
            (weights_dim, input_dims, output_dims) = Tiler(self, previous_node).get_tiling(level)
            self.tiling_dimensions["L{}".format(level-1)]["input_dimensions"] = input_dims
            self.tiling_dimensions["L{}".format(level-1)]["output_dimensions"] = output_dims
            weight_name = ""
            for name in self.constant_names:
                if name not in ["l","k","outshift","outmul"]:
                    if "bias" not in name:
                        weight_name = name
            if weight_name != "":
                self.tiling_dimensions["L{}".format(level-1)]["weights_dimensions"] = weights_dim
                groups = self.group if self.group < weights_dim[0] else weights_dim[0]
                self.tiling_dimensions["L{}".format(level-1)]["weight_memory"] = np.prod(weights_dim)/groups*np.prod(self.kernel_shape)*self.weight_bits/8
            else:
                self.tiling_dimensions["L{}".format(level-1)]["weight_memory"] = 0
            constants_memory = 0
            bias_memory = 0
            for name in self.constant_names:
                if name in ["l","k"]:
                    constants_memory+=weights_dim[0]*self.constant_bits/8
                if "bias" in name:
                    bias_memory+=weights_dim[0]*self.weight_bits/8
            self.tiling_dimensions["L{}".format(level-1)]["bias_memory"] = int(bias_memory)
            self.tiling_dimensions["L{}".format(level-1)]["constants_memory"] = int(constants_memory)
            self.tiling_dimensions["L{}".format(level-1)]["input_activation_memory"] = np.prod(self.tiling_dimensions["L{}".format(level-1)]["input_dimensions"])*self.input_activation_bits/8
            self.tiling_dimensions["L{}".format(level-1)]["output_activation_memory"] = np.prod(self.tiling_dimensions["L{}".format(level-1)]["output_dimensions"])*self.output_activation_bits/8

    def add_checksum_w_integer(self):
        self.check_sum_w = 0

        weight_name = ""
        for name in self.constant_names:
            if name not in ["l","k","outshift","outmul"]:
                if "bias" not in name:
                    weight_name = name
        if weight_name in self.__dict__:
            self.__dict__[weight_name]["value"] = self.__dict__[weight_name]["value"].flatten().tolist()
            # self.__dict__[weight_name+"_raw"] = self.__dict__[weight_name]
            if self.weight_bits < 8 and self.group > 1:
                self.__dict__[weight_name]["value"] = self.__dict__[weight_name]["value"].reshape(int(self.__dict__[weight_name]["value"].shape[0]/2),2,self.__dict__[weight_name]["value"].shape[1],self.__dict__[weight_name]["value"].shape[2],self.__dict__[weight_name]["value"].shape[3]).transpose(0,2,3,1,4).flatten().tolist()
            temp = []
            z = 0
            for i_w, _ in enumerate(self.__dict__[weight_name]["value"]):
                self.__dict__[weight_name]["value"][i_w] = np.uint8(self.__dict__[weight_name]["value"][i_w])
            for i_x, _ in enumerate(self.__dict__[weight_name]["value"]):
                if z % int(8 / self.weight_bits) == 0:
                    temp.append(self.__dict__[weight_name]["value"][i_x] & (2**self.weight_bits-1))
                else:
                    temp[-1] += (self.__dict__[weight_name]["value"][i_x]& (2**self.weight_bits-1)) << self.weight_bits * (z % int(8 / self.weight_bits))
                z+=1
            self.__dict__[weight_name]["value"] = temp
            self.check_sum_w += sum(self.__dict__[weight_name]["value"])

        bias_name = ""
        for name in self.constant_names:
            if name not in ["l","k","outshift","outmul"]:
                if "weight" not in name:
                    bias_name = name
        if bias_name in self.__dict__:
            self.__dict__[bias_name]["value"] = self.__dict__[bias_name]["value"].flatten().tolist()
            for i_w, _ in enumerate(self.__dict__[bias_name]["value"]):
                self.__dict__[bias_name]["value"][i_w] = np.uint8(self.__dict__[bias_name]["value"][i_w])
            self.check_sum_w += sum(self.__dict__[bias_name]["value"])

        if 'k' in self.__dict__:
            k_byte = []
            for i_k, _ in enumerate(self.k["value"].flatten()):
                if self.constant_bits == 64:
                    val = np.int64(self.k["value"].flatten()[i_k])
                else:
                    val = np.int32(self.k["value"].flatten()[i_k])
                for shift in np.arange(0,self.constant_bits,8):
                    k_byte.append(np.uint8((val >> shift)  & 255))
            self.k["value"] = k_byte
            self.check_sum_w += sum(self.k["value"])

        if 'l' in self.__dict__:
            lambd = np.float64(self.l["value"].flatten())
            lambd_byte = []
            for i_l, _ in enumerate(self.l["value"].flatten()):
                if self.constant_bits == 64:
                    val = np.int64(lambd[i_l])
                else:
                    val = np.int32(lambd[i_l])
                for shift in np.arange(0,self.constant_bits,8):
                    lambd_byte.append(np.uint8((val >> shift)  & 255))
            self.l["value"] = lambd_byte
            self.check_sum_w += sum(self.l["value"])

    def add_checksum_activations_integer(self, load_directory, node_number):
        ######################################################################################
        ###### SECTION 4: GENERATE CHECKSUM BY USING WEIGHT AND OUT_LAYER{i}.TXT FILES  ######
        ######################################################################################
        if node_number == 0:        
            try:
                x_in = pd.read_csv(os.path.join(load_directory, 'input.txt'))
                x_in = x_in.values[:, 0].astype(int)
            except FileNotFoundError:
                print(f"========= WARNING ==========\nInput file {os.path.join(load_directory, 'input.txt')} not found; generating random inputs!")
                x_in = torch.Tensor(1, self.group, self.input_channels, self.input_dimensions[0], self.input_dimensions[1]).uniform_(0, (2**(9)))
                x_in[x_in > (2**8 - 1)] = 0
                x_in = torch.round(x_in)
                x_in = x_in.flatten().numpy().astype(int)
            for i, _ in enumerate(x_in):
                x_in[i] = np.uint8(x_in[i])
            self.check_sum_in = sum(x_in)
        else:
            x_in = pd.read_csv(os.path.join(load_directory, 'out_layer{}.txt'.format(node_number-1)))
            x_in = x_in.values[:, 0].astype(int)
            for i, _ in enumerate(x_in):
                x_in[i] = np.uint8(x_in[i])
            in_compressed = []
            z = 0
            Loop_over = copy.deepcopy(x_in)
            for _, i_x in enumerate(Loop_over):
                if (z % int(8 / self.input_activation_bits)) == 0:
                    in_compressed.append(int(i_x.item()))
                else:
                    in_compressed[-1] += int(i_x.item()) << (self.input_activation_bits * (z % int(8 / self.input_activation_bits)))
                z += 1
            self.check_sum_in = sum(in_compressed)

        x_out = pd.read_csv(os.path.join(load_directory, 'out_layer{}.txt'.format(node_number)))
        x_out = x_out.values[:, 0].astype(int)
        if self.output_activation_bits <= 8:
            for i, _ in enumerate(x_out):
                x_out[i] = np.uint8(x_out[i])
            out_compressed = []
            z = 0
            Loop_over = copy.deepcopy(x_out)
            for _, i_x in enumerate(Loop_over):
                if (z % int(8 / self.output_activation_bits)) == 0:
                    out_compressed.append(int(i_x.item()))
                else:
                    out_compressed[-1] += int(i_x.item()) << (self.output_activation_bits * (z % int(8 / self.output_activation_bits)))
                z += 1
        else:
            out_compressed = x_out
        self.check_sum_out = sum(out_compressed)

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
