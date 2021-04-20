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

class node_element(nn.Module):
    # A node allocated in the PULP_Graph
    def __init__(self):
        self.name = 'empty'
        self.filter_size_h = 1
        self.filter_size_w = 1
        self.input_channels = 0
        self.output_channels = 0
        self.padding_top    = 0
        self.padding_left   = 0
        self.padding_bottom = 0
        self.padding_right  = 0
        self.stride = 1
        self.groups = 1
        self.weights = 'empty'
        self.k = 'empty'
        self.lambd = 'empty'
        self.outmul = 'empty'
        self.inmul1 = 'empty'
        self.inmul2 = 'empty'
        self.outshift = 'empty'
        self.outshift2 = 'empty'
        self.bias = 'empty'
        self.input_index = 0
        self.input_index_add = 0
        self.output_index = 0
        self.input_h = 0
        self.input_w = 0
        self.output_h = 0
        self.output_w = 0
        self.L3_allocation = 0
        self.input_activation_dimensions = 0
        self.input_activation_dimensions_L3 = 0
        self.output_activation_dimensions = 0
        self.output_activation_dimensions_L3 = 0
        self.check_sum_in = 0
        self.check_sum_out = 0
        self.check_sum_w = 0
        self.l1_dimensions = 0
        self.branch_out = 0
        self.branch_in = 0
        self.weights_dimension = 0
        self.weights_dimension_L3 = 0
        self.MACs = 0
        self.branch_last = 0
        self.branch_change = 0
        self.conv_1d = 0
        self.dilation = 1


class ONNX_management():
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.

    def __init__(self, platform, chip, network="model_to_convert.onnx"):
        self.network = network
        self.platform = platform
        self.chip = chip

    def create_node_add(self, new_node, first_node, node_iterating, model, PULP_Nodes_Graph):
        # Allocation of an Addition (for residuals) layer
        new_node.input_index = [input_i for input_i in node_iterating.input if 'weight' not in input_i][0]
        new_node.input_index_add = [input_i for input_i in node_iterating.input if 'weight' not in input_i][1]
        for node_iterating_mul in (model.graph.node):
            if 'Mul' == node_iterating_mul.op_type:
                flag_mul = 0
                if new_node.input_index == node_iterating_mul.output[0]:
                    new_node.input_index = node_iterating_mul.input[0]
                    inputs = [input_i for input_i in node_iterating_mul.input if 'weight' not in input_i]
                    index = [input_search for input_search in inputs if input_search][0]
                    const = self.search_constant(index, model)
                    if const == 'empty':
                        index = [input_search for input_search in inputs if input_search][1]
                        const = self.search_constant(index, model)
                    try:
                        new_node.inmul1 = const[0]
                    except:
                        new_node.inmul1 = const
                if new_node.input_index_add == node_iterating_mul.output[0]:
                    new_node.input_index_add = node_iterating_mul.input[0]
                    inputs = [input_i for input_i in node_iterating_mul.input if 'weight' not in input_i]
                    index = [input_search for input_search in inputs if input_search][0]
                    const = self.search_constant(index, model)
                    if const == 'empty':
                        index = [input_search for input_search in inputs if input_search][1]
                        const = self.search_constant(index, model)
                    try:
                        new_node.inmul2 = const[0]
                    except:
                        new_node.inmul2 = const
        new_node.output_index = node_iterating.output[0]
        for nodes in PULP_Nodes_Graph:
            if nodes.output_index == new_node.input_index or nodes.output_index == new_node.input_index_add:
                new_node.input_h = nodes.output_h
                new_node.input_w = nodes.output_w
                new_node.input_channels = nodes.output_channels
                new_node.output_channels = nodes.output_channels
                new_node.output_h = nodes.output_h
                new_node.output_w = nodes.output_w
        new_node.name = node_iterating.op_type
        new_node.branch_in = 1
        return new_node



    def create_node(self, new_node, first_node, node_iterating, model, PULP_Nodes_Graph):
        # Allocation of a Node, Convolution, Pooling or Linear
        new_node.input_index = [input_i for input_i in node_iterating.input if 'weight' not in input_i][0]
        new_node.output_index = node_iterating.output[0]
        new_node.padding_top    = 0
        new_node.padding_left   = 0
        new_node.padding_bottom = 0
        new_node.padding_right  = 0
        if 'Conv' in node_iterating.op_type:
            if(len(node_iterating.attribute[0].ints) == 1):
                new_node.conv_1d = 1
        if 'Pool' in node_iterating.op_type:
            if(len(node_iterating.attribute[0].ints) == 1):
                new_node.conv_1d = 1
        # scan 2 successive Pad layers
        for node in model.graph.node:
            if node.output[0] in new_node.input_index and node.op_type == 'Pad':
                new_node.input_index = node.input[0]
                if(new_node.conv_1d == 1):
                    if int(sum(node.attribute[1].ints)>0):
                        new_node.padding_left+=node.attribute[1].ints[2]
                        new_node.padding_right+=node.attribute[1].ints[5]
                else:        
                    if int(sum(node.attribute[1].ints)>0):
                        new_node.padding_top+=node.attribute[1].ints[2]
                        new_node.padding_left+=node.attribute[1].ints[3]
                        new_node.padding_bottom+=node.attribute[1].ints[6]
                        new_node.padding_right+=node.attribute[1].ints[7]
                for pad_2 in model.graph.node:
                    if pad_2.output[0] in node.input and pad_2.op_type == 'Pad':
                        new_node.input_index = pad_2.input[0]
                        if(new_node.conv_1d == 1):
                            new_node.padding_left+=pad_2.attribute[1].ints[2]
                            new_node.padding_right+=pad_2.attribute[1].ints[5]
                        else:
                            if int(sum(pad_2.attribute[1].ints)>0):
                                new_node.padding_top+=pad_2.attribute[1].ints[2]
                                new_node.padding_left+=pad_2.attribute[1].ints[3]
                                new_node.padding_bottom+=pad_2.attribute[1].ints[6]
                                new_node.padding_right+=pad_2.attribute[1].ints[7]
        if first_node == 1:
            new_node.input_h = model.graph.input[0].type.tensor_type.shape.dim[-2].dim_value
            new_node.input_w = model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
        else:
            if 'Gemm' not in node_iterating.op_type and 'MatMul' not in node_iterating.op_type:
                for nodes in PULP_Nodes_Graph:
                    if nodes.output_index == new_node.input_index:
                        new_node.input_h = nodes.output_h
                        new_node.input_w = nodes.output_w
            else:
                new_node.input_h = 1
                new_node.input_w = 1
        new_node.name = node_iterating.op_type
        try:
            if 'MatMul' == node_iterating.op_type:
            	weight_name = [input_i for input_i in node_iterating.input][1]
            else:
            	weight_name = [input_i for input_i in node_iterating.input if 'weight' in input_i][0]
        except:
            weight_name = 'NotFound'
        try:
            bias_name = [input_i for input_i in node_iterating.input if 'bias' in input_i][0]
        except:
            bias_name = 'NotFound'
        if 'Conv' in node_iterating.op_type:
            if(len(node_iterating.attribute[0].ints) == 1):
                new_node.conv_1d = 1
                new_node.name = 'Conv1D'
                new_node.dilation = node_iterating.attribute[0].ints[0]
            if(new_node.conv_1d == 1):
                for weight in model.graph.initializer:
                    if weight.name == weight_name:
                        new_node.weights = np.transpose(numpy_helper.to_array(weight), (0, 2, 1))
                        new_node.input_channels = weight.dims[1]
                        new_node.output_channels = weight.dims[0]
            else:
                for weight in model.graph.initializer:
                    if weight.name == weight_name:
                        new_node.weights = np.transpose(numpy_helper.to_array(weight), (0, 2, 3, 1))
                        new_node.input_channels = weight.dims[1]
                        new_node.output_channels = weight.dims[0]
        elif 'Gemm' in node_iterating.op_type or 'MatMul' in node_iterating.op_type:
            for weight in model.graph.initializer:
                if weight.name == weight_name:
                    temp = numpy_helper.to_array(weight)
                    if 'MatMul' in node_iterating.op_type:
                        temp = temp.T
                        temp = temp.reshape(temp.shape[0], PULP_Nodes_Graph[-1].output_channels, PULP_Nodes_Graph[-1].output_h, PULP_Nodes_Graph[-1].output_w)
                    else:
                        temp = temp.reshape(temp.shape[0], PULP_Nodes_Graph[-1].output_channels, PULP_Nodes_Graph[-1].output_h, PULP_Nodes_Graph[-1].output_w)
                    if 'MatMul' in node_iterating.op_type:
                        temp = np.transpose(temp, (0, 2, 3, 1))
                    else:
                        temp = np.transpose(temp, (0, 2, 3, 1))
                    temp = temp.flatten()
                    new_node.weights = temp
                    if 'MatMul' in node_iterating.op_type:
                        new_node.input_channels = weight.dims[0]
                        new_node.output_channels = weight.dims[1]
                    else:
                        new_node.input_channels = weight.dims[1]
                        new_node.output_channels = weight.dims[0]
                if weight.name == bias_name:
                    new_node.bias = numpy_helper.to_array(weight)
        elif 'Pool' in node_iterating.op_type:
            new_node.input_channels = PULP_Nodes_Graph[-1].output_channels
            new_node.output_channels = PULP_Nodes_Graph[-1].output_channels
        if 'Gemm' not in node_iterating.op_type and 'MatMul' not in node_iterating.op_type:
            if(new_node.conv_1d == 0):
                for field in node_iterating.attribute:
                    if field.name == 'kernel_shape':
                        new_node.filter_size_h = field.ints[0]
                        new_node.filter_size_w = field.ints[1]
                    if field.name == 'pads':
                        new_node.padding_top    += field.ints[0]
                        new_node.padding_left   += field.ints[1]
                        new_node.padding_bottom += field.ints[2]
                        new_node.padding_right  += field.ints[3]
                    if field.name == 'strides':
                        new_node.stride = field.ints[0]
                        if field.ints[0]==1 and field.ints[1]!=1:
                            new_node.stride = field.ints[1]
                    if field.name == 'group':
                        new_node.groups = field.i
            else:
                for field in node_iterating.attribute:
                    if field.name == 'kernel_shape':
                        new_node.filter_size_h = 1
                        new_node.filter_size_w = field.ints[0]
                    if field.name == 'pads':
                        new_node.padding_top    += 0
                        new_node.padding_left   += field.ints[0]
                        new_node.padding_bottom += 0
                        new_node.padding_right  += field.ints[1]
                    if field.name == 'strides':
                        new_node.stride = field.ints[0]
                    if field.name == 'group':
                        new_node.groups = field.i
            if new_node.groups > 1:
                new_node.name = new_node.name + 'DW'
        elif 'Gemm' in node_iterating.op_type or 'MatMul' in node_iterating.op_type:
            new_node.filter_size_h = 1
            new_node.filter_size_w = 1
            new_node.padding_top    = 0
            new_node.padding_left   = 0
            new_node.padding_bottom = 0
            new_node.padding_right  = 0
            new_node.stride = 1
        if 'Gemm' not in node_iterating.op_type and 'MatMul' not in node_iterating.op_type:
            new_node.output_h = int(np.ceil((new_node.input_h - (new_node.filter_size_h - 1) + new_node.padding_top + new_node.padding_bottom) / new_node.stride))
            if(new_node.conv_1d == 0):
                new_node.output_w = int(np.ceil((new_node.input_w - (new_node.filter_size_w - 1) + new_node.padding_left + new_node.padding_right) / new_node.stride))
            else:
                new_node.output_w = int(np.ceil((new_node.input_w - ((new_node.filter_size_w - 1)*new_node.dilation) + new_node.padding_left + new_node.padding_right) / new_node.stride))
        else:
            new_node.output_h = 1
            new_node.output_w = 1
        return new_node

    def search_constant(self, index, model):
        ## searching for the parameters of BN abd Relu
        constant = 'empty'
        
        for node_iterating in (model.graph.initializer):
            if node_iterating.name == index:
                constant = numpy_helper.to_array(node_iterating)
        for node_iterating in (model.graph.node):
            if node_iterating.op_type == 'Constant' and node_iterating.output[0] == index:
                constant = numpy_helper.to_array(node_iterating.attribute[0].t)
        return constant
    def update_node(self, PULP_node, out_index, const, op_type):
        # Add BN and Relu to the nodes.
        PULP_node.output_index = out_index
        if str(const) != 'empty':
            if op_type == 'Add':
                PULP_node.lambd = const
                PULP_node.name = PULP_node.name + 'BN'
            elif op_type == 'Div':
                if(PULP_node.name == 'AddRelu'):
                    try:
                        const[0]
                        PULP_node.outshift2 = round(np.log2(const[0]))
                    except:
                        PULP_node.outshift2 = round(np.log2(const))
                if 'Relu' not in PULP_node.name:
                    try:
                        const[0]
                        PULP_node.outshift = round(np.log2(const[0]))
                    except:
                        PULP_node.outshift = round(np.log2(const))
                    PULP_node.name = PULP_node.name + 'Relu'
            elif op_type == 'Mul':
                try:
                    const.shape[0]
                    ### TO FIX FOR SINGLE CHANNEL OUTPUT
                    if len(const.flatten())!=1 or np.asarray(const.shape).shape[0]>2:
                        PULP_node.k = const
                    else:
                        if str(PULP_node.outmul) == 'empty':
                            PULP_node.outmul = const[0]
                        else:
                            PULP_node.outmul = const[0] * PULP_node.outmul                        
                except:
                    if str(PULP_node.outmul) == 'empty':
                        PULP_node.outmul = const
                    else:
                        PULP_node.outmul = const * PULP_node.outmul
        return PULP_node

    def print_PULP_graph(self, PULP_Nodes_Graph):
        # Logging function to report exported graph of PULP
        print("Creating annotated graph in Network_annotated_graph.log")
        os.system('rm -rf logs')
        os.system('mkdir logs')
        logging.basicConfig(filename='logs/Network_annotated_graph.log',
                            format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)
        for nodes in PULP_Nodes_Graph:
            logging.debug(f'New node_iterating: {nodes.name}')
            if 'Conv' in nodes.name or 'Gemm' in nodes.name or 'MatMul' in nodes.name:
                logging.debug(f'Filter Dimension i_ch,fs1,fs2,o_ch: [{nodes.input_channels},{nodes.filter_size_h},{nodes.filter_size_w},{nodes.output_channels}]')
            logging.debug(f'Stride: {nodes.stride}')
            logging.debug(f'Padding: {nodes.padding_top}, {nodes.padding_left}, {nodes.padding_bottom}, {nodes.padding_right}')
            logging.debug(f'Groups {nodes.groups}')
            logging.debug(f'MACs {nodes.MACs}')
            logging.debug(f'In-Out dimensions: [{nodes.input_h},{nodes.input_w}], [{nodes.output_h},{nodes.output_w}]')
            if str(nodes.weights) != 'empty':
                logging.debug(f'Weigths: present ')
            else:
                logging.debug(f'Weights: empty')
            if str(nodes.k) != 'empty':
                logging.debug(f'k: present ')
            else:
                logging.debug(f'k: empty')
            if str(nodes.lambd) != 'empty':
                logging.debug(f'lambd: present ')
            else:
                logging.debug(f'lambd: empty')
            if str(nodes.outmul) != 'empty':
                logging.debug(f'outmul: present ')
            else:
                logging.debug(f'outmul: empty')
            if 'Add' in nodes.name:
                if str(nodes.inmul1) != 'empty':
                    logging.debug(f'inmul1: present ')
                else:
                    logging.debug(f'inmul1: empty')
                if str(nodes.inmul2) != 'empty':
                    logging.debug(f'inmul2: present ')
                else:
                    logging.debug(f'inmul2: empty')
            if str(nodes.outshift) != 'empty':
                logging.debug(f'outshift: present ')
            else:
                logging.debug(f'outshift: empty')
            logging.debug(f'Input branch: {nodes.branch_in}')
            logging.debug(f'Output branch: {nodes.branch_out}')
            logging.debug(f'Input: {nodes.input_index}')
            if 'Add' in nodes.name:
                logging.debug(f'     : {nodes.input_index_add}')
            logging.debug(f'Output: {nodes.output_index}')
            logging.debug(f' ')

    def check_add(self, node_iterating, model):
        not_real_add = 0
        inputs = [input_i for input_i in node_iterating.input if 'weight' not in input_i]
        for inp in inputs:
            if 'lamda' in inp:
                not_real_add = 1
        for node_const in (model.graph.node):
            for inp in inputs:
                if inp == node_const.output[0] and 'Const' in node_const.op_type:
                    not_real_add = 1
        return not_real_add

    def check_Mul_position(self, model, node_iterating, maxL):
        mul_for_add = 0
        flag_mul = 0
        for node_bn_relu in (model.graph.node[:(maxL * 15)]):
            inputs_bn_relu = [input_i for input_i in node_bn_relu.input if 'weight' not in input_i]
            for inp_bn_relu in inputs_bn_relu:
                if inp_bn_relu == node_iterating.output[0]:
                    if ('Div' not in node_bn_relu.op_type and 'Add' not in node_bn_relu.op_type and 'Mul' not in node_bn_relu.op_type):
                        print("Mul not followed by Div or Add or one other Mul. Exiting...")
                        os._exit(0)
                    if  'Add' in node_bn_relu.op_type:
                        inputs_add = [input_i for input_i in node_bn_relu.input if 'weight' not in input_i]
                        for inp_add in inputs_add:
                            const = self.search_constant(inp_add, model)
                            if str(const) != 'empty':
                                flag_mul = 1
                            if 'lamda' in inp_add:
                                flag_mul = 1
                        if flag_mul == 0:
                            mul_for_add = 1
        return mul_for_add

    def parameters_from_onnx(self, maxL):
        # Load all parameters from the onnx model.
        layers_accepted = ['Conv', 'Pad', 'Mul', 'Add', 'Div', 'Constant', 'AveragePool', 'MaxPool', 'Cast', 'Clip', 'Floor', 'Flatten', 'Gemm', 'MatMul', 'Shape', 'Gather', 'Unsqueeze', 'Concat', 'Reshape', 'Sigmoid', 'LogSoftmax']
        layers_neglected = ['Cast', 'Clip', 'Floor', 'Flatten', 'Shape', 'Gather', 'Unsqueeze', 'Concat', 'Reshape', 'Sigmoid', 'LogSoftmax']
        layers_to_node = ['AveragePool', 'MaxPool', 'Conv', 'Gemm', 'MatMul']
        model = onnx.load(self.network)
        PULP_Nodes_Graph = []
        first_node = 1
        for node_iterating in (model.graph.node[:(maxL * 15)]):
            assert (node_iterating.op_type in layers_accepted), f"{node_iterating.op_type} not supported by DORY"
            # Adding a new Conv, Pool or Linear layer
            if node_iterating.op_type in layers_to_node:
                new_node = self.create_node(node_element(), first_node, node_iterating, model, PULP_Nodes_Graph)
                PULP_Nodes_Graph.append(new_node)
                first_node = 0
                continue
            # Adding an addition layer
            if 'Add' in node_iterating.op_type:
                if self.check_add(node_iterating, model)==0:
                    new_node = self.create_node_add(node_element(), first_node, node_iterating, model, PULP_Nodes_Graph)
                    PULP_Nodes_Graph.append(new_node)
                    continue
            if node_iterating.op_type in layers_neglected:
                PULP_node.output_index = node_iterating.output[0]
                continue
            if node_iterating.op_type in 'Pad':
                continue
            inputs = [input_i for input_i in node_iterating.input if 'weight' not in input_i]
            for PULP_node in PULP_Nodes_Graph:
                for inp in inputs:
                    if inp == PULP_node.output_index:
                        # insert BN and/or Relu. Note that you need Mul-Add-Mul-Div
                        if 'Mul' == node_iterating.op_type:
                            if self.check_Mul_position(model, node_iterating, maxL):
                                break
                        index = [input_search for input_search in inputs if input_search != inp][0]
                        const = self.search_constant(index, model)
                        PULP_node = self.update_node(PULP_node, node_iterating.output[0], const, node_iterating.op_type)
                        break
        # updating branch in/out connections
        for i, nodes in enumerate(PULP_Nodes_Graph):
            counter = 0
            for nodes_scan in PULP_Nodes_Graph:
                if nodes.output_index == nodes_scan.input_index:
                    counter += 1
                if 'Add' in nodes_scan.name:
                    if nodes.output_index == nodes_scan.input_index_add:
                        counter += 1
            if counter > 1:
                PULP_Nodes_Graph[i].branch_out = 1
        branch_change = [0] * len(PULP_Nodes_Graph)
        branch_last = [0] * len(PULP_Nodes_Graph)
        index_of_first_add = 0
        index_of_second_add = 0
        for i, node in enumerate(PULP_Nodes_Graph):
            if('Add' in node.name):
                first_add = node.input_index
                second_add = node.input_index_add
                for j, node_two in enumerate(PULP_Nodes_Graph):
                    if node_two.output_index == first_add:
                          index_of_first_add = j
                    elif node_two.output_index == second_add:
                        index_of_second_add = j
                if(PULP_Nodes_Graph[index_of_first_add].branch_out != 1 and PULP_Nodes_Graph[index_of_second_add].branch_out != 1):
                    if(index_of_first_add > index_of_second_add):
                        branch_change[index_of_second_add] = 1
                    else:
                        branch_change[index_of_first_add] = 1
            if 'Add' in node.name:
                second_add_last = node.input_index_add
                first_add_last = node.input_index
                if int(first_add_last) > int(second_add_last):
                    second_add_last = first_add_last
                for p, node_two_last in enumerate(PULP_Nodes_Graph):
                    if node_two_last.output_index == second_add:
                        branch_last[p] = 1    
        for i, node in enumerate(PULP_Nodes_Graph):
            PULP_Nodes_Graph[i].branch_change = branch_change[i]
            PULP_Nodes_Graph[i].branch_last = branch_last[i]
        os.system('rm -rf logs/*log')
        # computing MACs per layer
        for i, nodes in enumerate(PULP_Nodes_Graph):
            PULP_Nodes_Graph[i].MACs = nodes.filter_size_h * nodes.filter_size_w * \
                nodes.output_channels * nodes.input_channels * nodes.output_h * nodes.output_w
        # printing graph
        self.print_PULP_graph(PULP_Nodes_Graph)
        return PULP_Nodes_Graph

    