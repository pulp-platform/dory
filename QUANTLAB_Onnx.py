# -*- coding: future_fstrings -*-     # should work even without -*-
# -*- coding: utf-8 -*-
#!/bin/bash
# ONNX_management.py
# Alessio Burrello <alessio.burrello@unibo.it>
# Thorir Mar Ingolfsson <thoriri@iis.ee.ethz.ch>
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

import onnx
from onnx import numpy_helper
from onnx import helper, shape_inference
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
from collections import OrderedDict
import logging
import PULP_node as pulp
from ONNX_management import ONNX_management

class Quantlab_onnx(ONNX_management):
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.

    def __init__(self, onnx, platform):
        layers_accepted = ['Conv', 'Pad', 'Mul', 'Add', 'Div', 'Constant', 'AveragePool', 'GlobalAveragePool', 'MaxPool', 'Cast', 'Clip', 'Floor', 'Flatten', 'Gemm', 'MatMul', 'Shape', 'Gather', 'Unsqueeze', 'Concat', 'Reshape', 'Sigmoid', 'LogSoftmax']
        layers_neglected = ['Cast', 'Floor', 'Flatten', 'Shape', 'Gather', 'Unsqueeze', 'Concat', 'Reshape', 'Sigmoid', 'LogSoftmax']
        layers_to_node = ['AveragePool', 'MaxPool', 'Conv', 'Gemm', 'MatMul', 'GlobalAveragePool']
        backend = ['ConvBNRelu', 'ConvRelu', 'ConvDWBNRelu', 'ConvDWRelu', 'AveragePool', 'GlobalAveragePool', 'MaxPool', 'LinearBNRelu', 'GemmRelu', 'Gemm', 'QAddRelu']
        rules = {}
        rules['Relu'] = 'Mul-Div-Floor-Clip'
        rules['BNRelu'] = 'Mul-Add-Div-Floor-Clip'
        rules['QAdd'] = 'Mul-Add-Div-Floor-Clip-Add'
        ONNX_management.__init__(self, onnx, platform, backend, rules, layers_accepted, layers_neglected, layers_to_node)

    def apply_rule(self, node, rule):
        pulp_node = pulp.node_element()
        out = node.output[0]
        nodes_to_search = rule.split('-')
        blocks_to_search = len(nodes_to_search)
        i = 0
        for key, value in self.rules.items():
            if value == rule:
                break
        pulp_node.add_parameter('name', key)
        if rule in [self.rules['Relu'], self.rules['BNRelu']]:
            for node_iterating in (self.model.graph.node):
                if (out == node_iterating.output[0] or i > 0) and node_iterating.op_type == nodes_to_search[i] and i < blocks_to_search:
                    if i == 0:
                        pulp_node.add_parameter('input_index',[input_i for input_i in node_iterating.input if 'weight' not in input_i][0])
                    elif i == (blocks_to_search-1):
                        pulp_node.add_parameter('output_index',node_iterating.output[0])
                    if node_iterating.op_type in ['Mul', 'Add', 'Div']:
                        const = self.search_constant(node_iterating.input[1], self.model)
                        if isinstance(const, str):
                            const = self.search_constant(node_iterating.input[0], self.model)
                        assert (not(isinstance(const, str))), f"Error in searching BNRelu parameters"
                        if node_iterating.op_type == 'Mul' and rule == self.rules['BNRelu']:
                            pulp_node.add_parameter('k', const)
                            pulp_node.add_parameter('outmul', 1)
                        elif node_iterating.op_type == 'Mul' and rule == self.rules['Relu']:
                            pulp_node.add_parameter('outmul', const)

                        elif node_iterating.op_type == 'Add':
                            pulp_node.add_parameter('lambda', const)
                        elif node_iterating.op_type == 'Div':
                            try:
                                const[0]
                                pulp_node.add_parameter('outshift',round(np.log2(const[0])))
                            except:
                                pulp_node.add_parameter('outshift',round(np.log2(const)))
                    elif node_iterating.op_type in ['Clip']:
                        attributes_names = [attribute.name for attribute in node_iterating.attribute]
                        for attribute in node_iterating.attribute:
                            if attribute.name == 'out_bits':
                                pulp_node.add_parameter('out_activation_bits', attribute.i)
                    i+=1
                if i >= blocks_to_search:
                    break
        elif rule == rules['QAdd']:
            pass
        return pulp_node

            