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
from collections import namedtuple

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
        self.hex_weights_size = None

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

            if "DepthwisePointwise" in self.name:
                if 'k0' in self.constant_names:
                    constants_memory += weights_dim[1] * self.constant_bits / 8
                if 'l0' in self.constant_names:
                    constants_memory += weights_dim[1] * self.bias_bits / 8
                if 'k1' in self.constant_names:
                    constants_memory += weights_dim[0] * self.constant_bits / 8
                if 'l1' in self.constant_names:
                    constants_memory += weights_dim[0] * self.bias_bits / 8
            else:
                if 'k' in self.constant_names:
                    constants_memory += weights_dim[0] * self.constant_bits / 8
                if 'l' in self.constant_names:
                    constants_memory += weights_dim[0] * self.bias_bits / 8
                if 'bias' in self.constant_names:
                    bias_memory += weights_dim[0] * self.bias_bits / 8

            self.tiling_dimensions[mem]["bias_memory"] = int(bias_memory)
            self.tiling_dimensions[mem]["constants_memory"] = int(constants_memory)
            self.tiling_dimensions[mem]["input_activation_memory"] = int(np.prod(self.tiling_dimensions[mem]["input_dimensions"])*self.input_activation_bits/8)
            self.tiling_dimensions[mem]["output_activation_memory"] = int(np.prod(self.tiling_dimensions[mem]["output_dimensions"])*self.output_activation_bits/8)

    def rename_weights(self):
        if (not "DepthwisePointwise" in self.name and "Convolution" in self.name) or "FullyConnected" in self.name:
            for i, name in enumerate(self.constant_names):
                if name not in ["l", "k", "outshift", "outmul", "outadd"]:
                    if "bias" not in name:
                        if self.__dict__[name]["value"].size > self.output_channels:
                            self.__dict__["weights"] = self.__dict__.pop(name)
                            self.constant_names[i] = "weights"

    def _compress(self, x, bits):
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

    def _get_weights_into_shape(self, weights):
        if self.weight_bits < 8 and self.group > 1:
            ko = weights["value"].shape[0]
            shape = (ko // 2, 2) + weights["value"].shape[1:]
            weights["value"] = weights["value"].reshape(shape).transpose(0, 2, 3, 1, 4)
        weights["value"] = self._compress(weights["value"].ravel().astype(np.uint8), self.weight_bits)

    def _to_byte(self, x, bits):
        x = x.ravel().astype(np.int64 if bits > 32 else np.int32)
        #### TO CHECK ORDER OF BIASES
        byte = [(el >> shift) & 255 for el in x for shift in range(0, bits, 8)]
        return np.asarray(byte, dtype=np.uint8)

    def add_checksum_w_integer(self):
        self.check_sum_w = 0
        bias_name = ""
        weights_name = ""

        # Hack DepthwisePointwise into this as a separate case
        if "DepthwisePointwise" in self.name:
            for const_name in self.constant_names:
                if "out" in const_name:
                    continue
                elif "weights" in const_name:
                    weights = getattr(self, const_name)
                    self._get_weights_into_shape(weights)
                    self.check_sum_w += sum(weights["value"])
                else:
                    const = getattr(self, const_name)
                    const["value"] = self._to_byte(const["value"], self.constant_bits if "k" in const_name
                                                             else self.bias_bits)
                    self.check_sum_w += sum(const["value"])
            return

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
            self._get_weights_into_shape(weights)
            self.check_sum_w += sum(weights["value"])

        if hasattr(self, bias_name):
            bias = getattr(self, bias_name)
            bias["value"] = self._to_byte(bias["value"], self.bias_bits)
            self.check_sum_w += sum(bias["value"])

        if hasattr(self, 'k'):
            self.k["value"] = self._to_byte(self.k["value"], self.constant_bits)
            self.check_sum_w += sum(self.k["value"])

        if hasattr(self, 'l'):
            self.l["value"] = self._to_byte(self.l["value"], self.bias_bits)
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
                data = self._compress(data.ravel(), bits)

            return data.sum().item()

        input_index = self.input_indexes[0] if len(self.input_indexes) == 1 else None

        self.check_sum_in = file_checksum('input.txt' if i_node == 0 else f'out_layer{input_index}.txt', self.input_activation_bits)
        self.check_sum_out = file_checksum(f'out_layer{self.output_index}.txt', self.output_activation_bits)

    Dim = namedtuple("Dim", "start size tile_size")

    def _tile_checksum(self, data, h: Dim, w: Dim, c: Dim, padding, kernel_shape, stride=(1, 1)):
        overlap = (kernel_shape[0] - stride[0], kernel_shape[1] - stride[1])
        i_tile = 0
        for c_tile_start in range(c.start, c.start + c.size, c.tile_size):
            for h_tile_start in range(h.start, h.start + h.size - overlap[0], h.tile_size - overlap[0]):
                if h_tile_start > 0:
                    h_tile_start -= padding[0]
                for w_tile_start in range(w.start, w.start + w.size - overlap[1], w.tile_size - overlap[1]):
                    if w_tile_start > 0:
                        w_tile_start -= padding[3]
                    data_tile = data[h_tile_start:h_tile_start + h.tile_size,
                                     w_tile_start:w_tile_start + w.tile_size,
                                     c_tile_start:c_tile_start + c.tile_size]
                    checksum_tile = data_tile.sum()
                    print(f'[{i_tile}] Checksum: {checksum_tile}')
                    i_tile += 1

                    #if stride[0] != 1 or stride[1] != 1:
                    #    self._tile_checksum(data,
                    #                        Dim(h_tile_start, h.tile_size, stride[0]),
                    #                        Dim(w_tile_start, w.tile_size, stride[1]),
                    #                        Dim(c_tile_start, c.tile_size, c.tile_size))

    def _load_data(self, networkdir, filename):
        filepath = os.path.join(networkdir, filename)

        try:
            return loadtxt(filepath, dtype=np.int64)
        except ValueError:
            return loadtxt(filepath, dtype=np.float)
        except FileNotFoundError:
            print(f"File {filename} doesn't exist. Exiting DORY...")
            sys.exit(-1)

    def tile_input_checksum(self, networkdir, index):
        filename = f'out_layer{index-1}.txt' if index > 0 else 'input.txt'
        data = self._load_data(networkdir, filename)

        if self.output_activation_bits <= 8:
            data = self._compress(data.ravel(), self.output_activation_bits)

        data = data.reshape((self.input_dimensions[0], self.input_dimensions[1], self.input_channels))

        # TODO: extend to L2 mem, for now only L1, so when there is L3-L2 tiling it will probably be wrong

        h, w, c = self.input_dimensions + [self.input_channels]
        c_tile, h_tile, w_tile = self.tiling_dimensions["L1"]["input_dimensions"]

        self._tile_checksum(data,
                            self.Dim(0, h, h_tile),
                            self.Dim(0, w, w_tile),
                            self.Dim(0, c, c_tile),
                            padding=self.pads,
                            kernel_shape=self.kernel_shape,
                            stride=self.strides)

    def tile_output_checksum(self, networkdir, index):
        filename = f'out_layer{index}.txt'
        filepath = os.path.join(networkdir, filename)

        try:
            data = loadtxt(filepath, dtype=np.int64)
        except ValueError:
            data = loadtxt(filepath, dtype=np.float)
        except FileNotFoundError:
            print(f"File {filename} doesn't exist. Exiting DORY...")
            sys.exit(-1)

        if self.output_activation_bits <= 8:
            data = self._compress(data.ravel(), self.output_activation_bits)

        data = data.reshape((self.output_dimensions[0], self.output_dimensions[1], self.output_channels))

        # TODO: extend to L2 mem, for now only L1, so when there is L3-L2 tiling it will probably be wrong

        h, w, c = self.output_dimensions + [self.output_channels]
        c_tile, h_tile, w_tile = self.tiling_dimensions["L1"]["output_dimensions"]

        Dim = namedtuple("Dim", "start size tile_size")

        def tile_checksum(h: Dim, w: Dim, c: Dim, stride=(1, 1)):
            i_tile = 0
            for c_tile_start in range(c.start, c.start + c.size, c.tile_size):
                for h_tile_start in range(h.start, h.start + h.size, h.tile_size):
                    for w_tile_start in range(w.start, w.start + w.size, w.tile_size):
                        data_tile = data[h_tile_start:h_tile_start + h.tile_size,
                                         w_tile_start:w_tile_start + w.tile_size,
                                         c_tile_start:c_tile_start + c.tile_size]
                        checksum_tile = data_tile.sum()
                        print(f'[{i_tile}] Checksum: {checksum_tile}')
                        i_tile += 1

                        #if stride[0] != 1 or stride[1] != 1:
                        #    tile_checksum(Dim(h_tile_start, h.tile_size, stride[0]),
                        #                  Dim(w_tile_start, w.tile_size, stride[1]),
                        #                  Dim(c_tile_start, c.tile_size, c.tile_size))

        tile_checksum(Dim(0, h, h_tile), Dim(0, w, w_tile), Dim(0, c, c_tile), stride=self.strides)

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
