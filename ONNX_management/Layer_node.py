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

from DORY_node import DORY_node
import numpy as np
import pdb

class Layer_node(DORY_node):
    # A node allocated in the PULP_Graph
    def __init__(self):
        DORY_node.__init__(self)
        self.kernel_shape = [] # fH x fW
        self.dilations = [] 
        self.group = None
        self.strides = [] 
        self.input_channels = None 
        self.output_channels = None
        self.input_dimensions = [] # H x W
        self.output_dimensions = [] # H x W
        self.pads    = [] # Top, Left, Bottom, Right
        self.weight_bits = None
        self.output_activation_bits = None
        self.input_activation_bits = None

    def update_input_dimensions(self, activation_tensor, Layer_parameters):
        if activation_tensor.name in self.input_indexes:
            ## Batch, C, H, W
            dimension = activation_tensor.type.tensor_type.shape.dim
            Layer_parameters["input_channels"] = dimension[1].dim_value
            Layer_parameters["input_dimensions"] = [e.dim_value for e in dimension[2:]]
        return Layer_parameters

    def update_output_dimensions(self, activation_tensor, Layer_parameters):
        if activation_tensor.name == self.output_index:
            ## Batch, C, H, W
            dimension = activation_tensor.type.tensor_type.shape.dim
            Layer_parameters["output_channels"] = dimension[1].dim_value
            Layer_parameters["output_dimensions"] = [e.dim_value for e in dimension[2:]]
        return Layer_parameters

    def populate_Layer_node(self, node_iterating, graph):
        self.populate_DORY_node(node_iterating,graph)
        Layer_parameters = {}
        ## kernel_shape, dilations, group, strides, pads DEFAULTS
        if 'Global' in node_iterating.name:
            Layer_parameters['kernel_shape'] = Layer_parameters['input_dimensions']
        if self.name in ['Fully-Connected', 'Addition']:
            Layer_parameters['kernel_shape'] = [1, 1]
            Layer_parameters['dilations'] = [1, 1]
            Layer_parameters['strides'] = [1, 1]
            Layer_parameters['group'] = 1
        if self.name in ['Pooling']:
            Layer_parameters['dilations'] = [1, 1]
            Layer_parameters['group'] = 1
        Layer_parameters['pads'] = [0, 0, 0, 0]

        ## 'kernel_shape', 'dilations', 'group', 'strides', 'pads'
        for attribute in node_iterating.attribute:
            if attribute.name in ['kernel_shape', 'dilations', 'group', 'strides', 'pads']:
                if bool(attribute.i):
                    Layer_parameters[attribute.name] = attribute.i
                if bool(attribute.ints):
                    Layer_parameters[attribute.name] = list(attribute.ints)

        ## adding dimensions
        for activation_tensor in graph.graph.input:
            Layer_parameters = self.update_input_dimensions(activation_tensor, Layer_parameters)
        for activation_tensor in graph.graph.value_info:
            Layer_parameters = self.update_input_dimensions(activation_tensor, Layer_parameters)
            Layer_parameters = self.update_output_dimensions(activation_tensor, Layer_parameters)
        for activation_tensor in graph.graph.output:
            Layer_parameters = self.update_output_dimensions(activation_tensor, Layer_parameters)
        self.add_existing_dict_parameter(Layer_parameters)


