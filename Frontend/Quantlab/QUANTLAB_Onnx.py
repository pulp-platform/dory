     # should work even without -*-
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
        layers_to_node = ['AveragePool', 'MaxPool', 'Conv', 'Gemm', 'MatMul', 'GlobalAveragePool', 'Add']
        backend = ['ConvBNRelu', 'ConvRelu', 'ConvDWBNRelu', 'ConvDWRelu', 'AveragePool', 'GlobalAveragePool', 'MaxPool', 'LinearBNRelu', 'GemmRelu', 'Gemm', 'MatMulRelu', 'MatMul', 'Add', 'AddBNRelu', 'BNReluAddBNRelu',
                    'PadConvBNRelu', 'PadConvRelu', 'PadConvDWBNRelu', 'PadConvDWRelu', 'PadAveragePool', 'PadGlobalAveragePool', 'PadMaxPool', 'PadLinearBNRelu', 'PadGemmRelu', 'PadGemm', 'PadMatMulRelu', 'PadMatMul', 
                    'PadAdd', 'PadAddBNRelu', 'PadBNReluAddBNRelu']
        rules = {}
        rules['Relu'] = 'Mul-Div-Floor-Clip'
        rules['BNRelu'] = 'Mul-Add-Div-Floor-Clip'
        rules['Pad'] = 'Pad'
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
        elif rule == self.rules['Pad']:
            pulp_node.add_parameter('name', key)
            for node_iterating in (self.model.graph.node):
                if out == node_iterating.output[0] and node_iterating.op_type == nodes_to_search[i] and i < blocks_to_search:
                    inp = []
                    for input_i in node_iterating.input:
                        if 'weight' not in input_i:
                            if input_i not in [node.output[0] for node in self.model.graph.node if node.op_type in 'Constant']:
                                inp.append(input_i)
                    pulp_node.add_parameter('input_index', inp[0])
                    pulp_node.add_parameter('output_index',node_iterating.output[0])
                    if np.array(node_iterating.attribute[1].ints).shape[0] == 8:
                        pulp_node.add_parameter('pads',[node_iterating.attribute[1].ints[2],node_iterating.attribute[1].ints[3],node_iterating.attribute[1].ints[6],node_iterating.attribute[1].ints[7]])
                    elif np.array(node_iterating.attribute[1].ints).shape[0] == 6:
                        pulp_node.add_parameter('pads',[0, node_iterating.attribute[1].ints[2], 0, node_iterating.attribute[1].ints[5]])
                    break

        return pulp_node

    def fuse_graph(self):
        # Logging function to report exported graph of PULP
        while True:
            PULP_Nodes_Graph_fused = []
            skip = 0
            not_fused = 1
            fuse_at_least_1 = 0
            for node_1, node_2 in zip(self.PULP_Nodes_Graph[:-1], self.PULP_Nodes_Graph[1:]):
                last_node = 0
                if node_1.name+node_2.name in '.'.join([*self.backend]) and skip == 0:
                    PULP_Nodes_Graph_fused.append(self.fuse_nodes(node_1, node_2))
                    skip = 1
                    not_fused = 0
                    fuse_at_least_1 = 1
                elif skip == 0:
                    PULP_Nodes_Graph_fused.append(node_1)
                    not_fused = 1
                else:
                    skip = 0
                    last_node = 1
            if not_fused == 1 or last_node == 1:
                PULP_Nodes_Graph_fused.append(node_2)
            self.PULP_Nodes_Graph = PULP_Nodes_Graph_fused
            if fuse_at_least_1 == 0:
                break
        self.fuse_graph_BNReluADD()

    def fuse_nodes(self, node_1, node_2):
        assert (node_1.get_parameter('output_index') == node_2.get_parameter('input_index')), f"Error in fusion of near nodes with different indexes"
        node_1.add_parameter('name', node_1.get_parameter('name')+node_2.get_parameter('name') )
        for key, value in node_2.__dict__.items():
            if (isinstance(value,str)):
                if value == 'Not-initialized':
                    pass
                elif key not in ['name', 'input_index']:
                    node_1.add_parameter(key,value)
            elif key in ['pads']:
                node_1_pads = node_1.get_parameter('pads')
                node_2_pads = node_2.get_parameter('pads')
                for i in range(len(node_2_pads)):
                    node_1_pads[i] += node_2_pads[i]
                node_1.add_parameter('pads', node_1_pads)
            elif key in ['branch_in']:
                node_1.add_parameter('branch_in', node_1.get_parameter('branch_in') + node_2.get_parameter('branch_in'))
            elif key in ['branch_out']:
                node_1.add_parameter('branch_out', node_1.get_parameter('branch_out') + node_2.get_parameter('branch_out'))
            elif key in ['branch_change']:
                node_1.add_parameter('branch_change', node_1.get_parameter('branch_change') + node_2.get_parameter('branch_change'))
            elif key in ['branch_last']:
                node_1.add_parameter('branch_last', node_1.get_parameter('branch_last') + node_2.get_parameter('branch_last'))
            elif key not in ['name', 'input_index', 'input_dim', 'weight_bits']:
                node_1.add_parameter(key,value)
            elif key in ['input_dim']:
                if 'input' not in node_1.get_parameter('input_index') and '0' != node_1.get_parameter('input_index'):
                    value[0] = value[0]-node_1.get_parameter('pads')[0]-node_1.get_parameter('pads')[2]
                    value[1] = value[1]-node_1.get_parameter('pads')[1]-node_1.get_parameter('pads')[3]
                node_1.add_parameter(key,value)

        return node_1

    def fuse_Add(self, node_1, node_2):
        assert (node_1.get_parameter('output_index') == node_2.get_parameter('input_index')), f"Error in fusion of near nodes with different indexes"
        node_2.add_parameter('name', node_1.get_parameter('name')+node_2.get_parameter('name') )
        node_2.add_parameter('input_index', node_1.get_parameter('input_index'))
        return node_2

    def fuse_graph_BNReluADD(self):
        BNRelu_fused = []                
        for j, node_1 in enumerate(self.PULP_Nodes_Graph):
            if node_1.name == 'BNRelu':
                for i, node_2 in enumerate(self.PULP_Nodes_Graph):
                    if node_1.name+node_2.name in '.'.join([*self.backend]) and node_1.output_index == node_2.input_index:
                        self.PULP_Nodes_Graph[i] = self.fuse_Add(node_1, node_2)
                        BNRelu_fused.append(j)
                        break
        PULP_Nodes_Graph_fused = []
        for j, node in enumerate(self.PULP_Nodes_Graph):
            if j not in BNRelu_fused:
                PULP_Nodes_Graph_fused.append(node)
        self.PULP_Nodes_Graph = PULP_Nodes_Graph_fused
