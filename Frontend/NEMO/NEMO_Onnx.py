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
import sys
import pandas as pd
from collections import OrderedDict
import logging
import PULP_node as pulp
from ONNX_management import ONNX_management

class NEMO_onnx(ONNX_management):
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.

    def __init__(self, onnx, platform):
        layers_accepted = ['Conv', 'Pad', 'Mul', 'Add', 'Div', 'Constant', 'AveragePool', 'GlobalAveragePool', 'MaxPool', 'Cast', 'Clip', 'Floor', 'Flatten', 'Gemm', 'MatMul', 'Shape', 'Gather', 'Unsqueeze', 'Concat', 'Reshape', 'Sigmoid', 'LogSoftmax']
        layers_neglected = ['Cast', 'Flatten', 'Shape', 'Gather', 'Unsqueeze', 'Concat', 'Reshape', 'Sigmoid', 'LogSoftmax', 'Clip']
        layers_to_node = ['AveragePool', 'MaxPool', 'Conv', 'Gemm', 'MatMul', 'GlobalAveragePool']
        backend = ['ConvBNRelu', 'ConvRelu', 'ConvDWBNRelu', 'ConvDWRelu', 'AveragePool', 'GlobalAveragePool', 'MaxPool', 'MatMulBNRelu', 'GemmRelu', 'Gemm', 'MatMul', 'Add','QAddRelu',\
        'PadConvBNRelu', 'PadConvDWBNRelu', 'PadAveragePool', 'PadGlobalAveragePool', 'PadMaxPool', 'PadMatMulBNRelu', 'PadGemmRelu', 'PadGemm', 'PadQAddRelu',\
        'PadPadConvBNRelu', 'PadPadConvDWBNRelu', 'PadPadAveragePool', 'PadPadGlobalAveragePool', 'PadPadMaxPool', 'PadPadMatMulBNRelu', 'PadPadGemmRelu', 'PadPadGemm', 'PadPadQAddRelu']
        rules = {}
        rules['Relu'] = 'To define'#'Mul-Div-Floor-Clip'
        rules['BNRelu_1'] = 'Mul-Add-Mul-Div-Floor-Clip'
        rules['BNRelu_2'] = 'Mul-Add-Cast-Mul-Div-Cast-Clip'
        rules['BNRelu_3'] = 'Mul-Add-Cast-Mul-Div-Cast-Cast-Cast-Clip'
        rules['QAdd'] = 'Mul-Add-Mul-Div-Floor'
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
        pulp_node.add_parameter('name', key.split('_')[0])
        if rule in [self.rules['Relu'], self.rules['BNRelu_1'], self.rules['BNRelu_2'], self.rules['BNRelu_3']]:
            for node_iterating in (self.model.graph.node):
                if (out == node_iterating.output[0] or i > 0) and node_iterating.op_type == nodes_to_search[i] and i < blocks_to_search:
                    if i == 0:
                        inp = []
                        for input_i in node_iterating.input:
                            if 'weight' not in input_i and 'bn' not in input_i and 'BN' not in input_i and 'kappa' not in input_i and 'lambda' not in input_i:
                                if input_i not in [node.output[0] for node in self.model.graph.node if node.op_type in 'Constant']:
                                    inp.append(input_i)
                        pulp_node.add_parameter('input_index', inp[0])
                    elif i == (blocks_to_search-1):
                        pulp_node.add_parameter('output_index',node_iterating.output[0])
                    if node_iterating.op_type in ['Mul', 'Add', 'Div']:
                        const = self.search_constant(node_iterating.input[1], self.model)
                        if isinstance(const, str):
                            const = self.search_constant(node_iterating.input[0], self.model)
                        assert (not(isinstance(const, str))), f"Error in searching BNRelu parameters"
                        if node_iterating.op_type == 'Mul' and rule in [self.rules['BNRelu_1'], self.rules['BNRelu_2'], self.rules['BNRelu_3']] and i == 0:
                            k = const
                        elif node_iterating.op_type == 'Mul' and rule in [self.rules['BNRelu_1'], self.rules['BNRelu_2'], self.rules['BNRelu_3']]:
                            outmul = const
                            pulp_node.add_parameter('k', k*outmul)
                            pulp_node.add_parameter('outmul', 1)
                            pulp_node.add_parameter('lambda', l*outmul)
                        elif node_iterating.op_type == 'Mul' and rule == self.rules['Relu']:
                            pulp_node.add_parameter('outmul', const)
                        elif node_iterating.op_type == 'Add':
                            l = const
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
                    if node_iterating.op_type in '.'.join([*self.rules.values()]):
                        i+=1
                if i >= blocks_to_search:
                    break
        elif rule == self.rules['QAdd']:
            first_node_found = 0
            for node_iterating in (self.model.graph.node):
                if out == node_iterating.output[0] and node_iterating.op_type == nodes_to_search[i] and i < blocks_to_search:
                    inp = []
                    for input_i in node_iterating.input:
                        if 'weight' not in input_i and 'bn' not in input_i and 'BN' not in input_i and 'kappa' not in input_i and 'lambda' not in input_i:
                            if input_i not in [node.output[0] for node in self.model.graph.node if node.op_type in 'Constant']:
                                inp.append(input_i)
                    input_index = inp[0]
                    i+=1
                    first_node_found = 1              
                    const = self.search_constant(node_iterating.input[1], self.model)
                    if isinstance(const, str):
                        const = self.search_constant(node_iterating.input[0], self.model)
                    assert (not(isinstance(const, str))), f"Error in searching Inmul1"
                    try:
                        const = const[0]
                    except:
                        pass
                    inmul1 = const
                elif node_iterating.op_type == 'Add' and i < blocks_to_search and first_node_found == 1:
                    pulp_node = self.create_node(pulp.node_element(), 0, node_iterating, self.model)
                    pulp_node.add_parameter('input_index', inp[0])
                    pulp_node.add_parameter('inmul1', const)
                    i+=2
                elif node_iterating.op_type == nodes_to_search[i] and i < blocks_to_search and first_node_found == 1:
                    i+=1
                    if node_iterating.op_type == 'Div':                
                        const = self.search_constant(node_iterating.input[1], self.model)
                        if isinstance(const, str):
                            const = self.search_constant(node_iterating.input[0], self.model)
                        assert (not(isinstance(const, str))), f"Error in searching BNRelu parameters"
                        try:
                            const[0]
                            pulp_node.add_parameter('outshift',round(np.log2(const[0])))
                        except:
                            pulp_node.add_parameter('outshift',round(np.log2(const)))
                    if i == blocks_to_search:
                        pulp_node.add_parameter('output_index',node_iterating.output[0])
                        break
            for node_iterating in (self.model.graph.node):
                if pulp_node.get_parameter('input_index_add') == node_iterating.output[0]:
                    inp = []
                    for input_i in node_iterating.input:
                        if 'weight' not in input_i and 'bn' not in input_i and 'BN' not in input_i and 'kappa' not in input_i and 'lambda' not in input_i:
                            if input_i not in [node.output[0] for node in self.model.graph.node if node.op_type in 'Constant']:
                                inp.append(input_i)
                    input_index = inp[0]
                    i+=1
                    first_node_found = 1              
                    const = self.search_constant(node_iterating.input[1], self.model)
                    if isinstance(const, str):
                        const = self.search_constant(node_iterating.input[0], self.model)
                    assert (not(isinstance(const, str))), f"Error in searching Inmul2"
                    try:
                        const = const[0]
                    except:
                        pass
                    inmul2 = const
                    pulp_node.add_parameter('input_index_add', inp[0])
                    pulp_node.add_parameter('inmul2', const)
                    if int(pulp_node.get_parameter('input_index_add')) < int(pulp_node.get_parameter('input_index')):
                        input_index = pulp_node.get_parameter('input_index')
                        pulp_node.add_parameter('input_index', pulp_node.get_parameter('input_index_add'))
                        pulp_node.add_parameter('input_index_add', input_index)
                        inmul2 = pulp_node.get_parameter('inmul2')
                        pulp_node.add_parameter('inmul2', pulp_node.get_parameter('inmul1'))
                        pulp_node.add_parameter('inmul1', inmul2)
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

    def update_precisions_graph(self):
        # adding input bit precision
        for i, nodes in enumerate(self.PULP_Nodes_Graph):
            if i == 0:
                nodes.add_parameter('input_activation_bits', 8)
                nodes.add_parameter('out_activation_bits', 8)
            else:
                for j, precedent_nodes in enumerate(self.PULP_Nodes_Graph[:i]):
                    if precedent_nodes.output_index == nodes.input_index:
                        nodes.add_parameter('input_activation_bits', 8)
                        if 'Add' in nodes.name:
                            nodes.add_parameter('out_activation_bits', 8)
                nodes.add_parameter('out_activation_bits', 8)
            if i == (len(self.PULP_Nodes_Graph)-1):
                nodes.add_parameter('out_activation_bits', 32)
            if 'Pool' in nodes.name:
                nodes.add_parameter('out_activation_bits', nodes.get_parameter('input_activation_bits'))
        for i, nodes in enumerate(self.PULP_Nodes_Graph):
            if nodes.get_parameter('input_activation_bits') != 8 or nodes.get_parameter('out_activation_bits') != 8 or nodes.get_parameter('weight_bits') != 8:
                multiple = 8/min(nodes.get_parameter('input_activation_bits'), nodes.get_parameter('out_activation_bits'), nodes.get_parameter('weight_bits'))
                if nodes.get_parameter('ch_in')%multiple !=0 or nodes.get_parameter('ch_out')%multiple !=0:
                    sys.exit("ERROR 01. Channels of a layer not multiple of 2 (int4 precision layers) or 4 (int2 precision layers). Exiting...")

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
            elif key not in ['name', 'input_index', 'input_dim']:
                node_1.add_parameter(key,value)
            elif key in ['input_dim']:
                if 'input' not in node_1.get_parameter('input_index') and '0' != node_1.get_parameter('input_index'):
                    value[0] = value[0]-node_1.get_parameter('pads')[0]-node_1.get_parameter('pads')[2]
                    value[1] = value[1]-node_1.get_parameter('pads')[1]-node_1.get_parameter('pads')[3]
                node_1.add_parameter(key,value)

        return node_1

    def check_rules(self, node):
        out = node.output[0] 
        string_rule = node.op_type
        QAdd_possible = 0
        if string_rule in [*self.rules.values()]:
            return string_rule
        for node_i in (self.model.graph.node):
            try:
                input_i = [input_i for input_i in node_i.input if ('bn' not in input_i and 'BN' not in input_i and 'kappa' not in input_i and 'lambda' not in input_i)][0]
            except:
                continue
            if input_i == out:
                out = node_i.output[0]
                if node_i.op_type not in 'Constant':
                    string_rule = string_rule + '-' + node_i.op_type
                if string_rule not in '.'.join([*self.rules.values()]):
                    return False
                elif string_rule in '.'.join([*self.rules.values()]):
                    if string_rule in [*self.rules.values()]:
                        if QAdd_possible == 1 or (QAdd_possible == 0 and string_rule !=self.rules['QAdd']):
                            return string_rule
                if node_i.op_type in 'Add':
                    try:
                        input_2 = [input_i for input_i in node_i.input if ('bn' not in input_i and 'BN' not in input_i and 'kappa' not in input_i and 'lambda' not in input_i)][1]
                    except:
                        continue
                    for node_j in (self.model.graph.node):
                        out_inner = node_j.output[0]
                        if input_2 == out_inner:
                            if node_j.op_type not in 'Constant':
                                string_rule = string_rule + '-' + node_j.op_type
                                QAdd_possible = 1
                            if string_rule not in '.'.join([*self.rules.values()]):
                                return False
                            elif string_rule in '.'.join([*self.rules.values()]):
                                if string_rule in [*self.rules.values()]:
                                    return string_rule