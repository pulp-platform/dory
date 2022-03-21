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
import sys
import pandas as pd
from collections import OrderedDict
import logging
import PULP_node as pulp
from onnx import shape_inference

class ONNX_management():
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    def __init__(self, network, platform, backend, rules, layers_accepted, layers_neglected, layers_to_node):
        self.model = onnx.load(network)
        self.platform = platform
        self.layers_accepted = layers_accepted
        self.layers_neglected = layers_neglected
        self.layers_to_node = layers_to_node
        self.backend = backend
        self.rules = rules
        self.PULP_Nodes_Graph = []

    def create_node(self, new_node, first_node, node_iterating, model):
        # Allocation of a Node, Convolution, Pooling or Linear
        new_parameters = {}
        new_parameters['input_index'] = [input_i for input_i in node_iterating.input if 'weight' not in input_i][0]
        if 'Add' == node_iterating.op_type:
            new_parameters['input_index_add'] = [input_i for input_i in node_iterating.input if 'weight' not in input_i][1]
            new_parameters['branch_in'] = 1
        new_parameters['output_index'] = node_iterating.output[0] 
        new_parameters['pads'] = [0, 0, 0, 0]
        ## attributes
        for attribute in node_iterating.attribute:
        	if bool(attribute.i):
        		new_parameters[attribute.name] = attribute.i
        	if bool(attribute.ints):
        		new_parameters[attribute.name] = attribute.ints
        if 'kernel_shape' in new_parameters.keys():
            if np.asarray(new_parameters['kernel_shape']).shape[0] == 1:
                new_parameters['pads'] = [0, 0]
                new_parameters['kernel_shape'] = [1, new_parameters['kernel_shape'][0]]
        ###### NOT SUPPORTING H AND W DIFFERENT STRIDES #####
        if 'strides' in new_parameters.keys():
            if len(new_parameters['strides'])>1:
                if new_parameters['strides'][0] > 1:
                    new_parameters['strides'] = new_parameters['strides'][0]
                else:
                    new_parameters['strides'] = new_parameters['strides'][1]
            else:
                new_parameters['strides'] = new_parameters['strides'][0]
        if 'dilations' in new_parameters.keys():
                new_parameters['dilations'] = new_parameters['dilations'][0]
        if 'strides' not in new_parameters.keys():
            new_parameters['strides'] = 1
        if 'group' not in new_parameters.keys():
            new_parameters['group'] = 1
        inferred_model = shape_inference.infer_shapes(self.model)
        shapes_info = inferred_model.graph.value_info
        for shape_i in shapes_info:
            if shape_i.name == new_parameters['output_index']:
                new_parameters['ch_out'] = shape_i.type.tensor_type.shape.dim[1].dim_value
                try:
                    if np.array(shape_i.type.tensor_type.shape.dim).shape[0] == 3:
                        new_parameters['output_dim'] = [1, shape_i.type.tensor_type.shape.dim[2].dim_value]
                    elif np.array(shape_i.type.tensor_type.shape.dim).shape[0] == 4: 
                        new_parameters['output_dim'] = [shape_i.type.tensor_type.shape.dim[2].dim_value, shape_i.type.tensor_type.shape.dim[3].dim_value]
                    else:
                        shape_i.type.tensor_type.shape.dim[3]
                except:
                    ### SETTING LAST NODE DIMENSIONS TO [1, 1]
                    new_parameters['output_dim'] = [1, 1]
            if shape_i.name == new_parameters['input_index']:
                try:
                    new_parameters['ch_in'] = shape_i.type.tensor_type.shape.dim[1].dim_value
                    ch_in_temp = new_parameters['ch_in']
                    if np.array(shape_i.type.tensor_type.shape.dim).shape[0] == 3:
                        new_parameters['input_dim'] = [1, shape_i.type.tensor_type.shape.dim[2].dim_value]
                    elif np.array(shape_i.type.tensor_type.shape.dim).shape[0] == 4: 
                        new_parameters['input_dim'] = [shape_i.type.tensor_type.shape.dim[2].dim_value, shape_i.type.tensor_type.shape.dim[3].dim_value]
                    else:
                        shape_i.type.tensor_type.shape.dim[3]
                except:
                    print(f'Infering Shapes of Input {shape_i.name} of Node {node_iterating.op_type} from previous Node for Graph Incompatibility')
                    if self.PULP_Nodes_Graph[-1].get_parameter('ch_out')=='Not-initialized':
                        node = -2
                    else:
                        node = -1
                    if node_iterating.op_type in ['MatMul', 'Gemm']:
                        new_parameters['ch_in'] = self.PULP_Nodes_Graph[node].get_parameter('ch_out')
                        ch_in_temp = new_parameters['ch_in']
                        for out in self.PULP_Nodes_Graph[node].get_parameter('output_dim'): 
                            new_parameters['ch_in'] *=out
                        new_parameters['input_dim'] = [1, 1]
                    else:
                        new_parameters['ch_in'] = self.PULP_Nodes_Graph[node].get_parameter('ch_out')
                        ch_in_temp = new_parameters['ch_in']
                        new_parameters['input_dim'] = self.PULP_Nodes_Graph[node].get_parameter('output_dim')
        if 'ch_out' not in new_parameters.keys():
            new_parameters['ch_out'] = model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value
            new_parameters['output_dim'] = model.graph.output[0].type.tensor_type.shape.dim[-2].dim_value
        if first_node == 1:
            dim = []
            ######## FOR 1D NOT WORKING ###########
            if np.array(model.graph.input[0].type.tensor_type.shape.dim).shape[0] == 4: 
                dim.append(model.graph.input[0].type.tensor_type.shape.dim[-2].dim_value)
                dim.append(model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value)
                new_parameters['ch_in'] = model.graph.input[0].type.tensor_type.shape.dim[-3].dim_value
            elif np.array(model.graph.input[0].type.tensor_type.shape.dim).shape[0] == 3: 
                dim.append(1)
                dim.append(model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value)
                new_parameters['ch_in'] = model.graph.input[0].type.tensor_type.shape.dim[-2].dim_value
            new_parameters['input_dim'] = dim
        new_parameters['ch_in'] = int ( new_parameters['ch_in'] / new_parameters['group'] )
        new_parameters['name'] = node_iterating.op_type
        if 'Global' in new_parameters['name']:
            new_parameters['kernel_shape'] = new_parameters['input_dim']
        if new_parameters['name'] in ['MatMul', 'Gemm']:
            new_parameters['kernel_shape'] = [1, 1]
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
        flag = '2D'
        if len(new_parameters['pads']) == 2:
            flag = '1D'
            new_parameters['pads'] =[0, new_parameters['pads'][0], 0, new_parameters['pads'][1]]
        if 'Conv' in new_parameters['name']:
            if new_parameters['group'] > 1:
                new_parameters['name'] = new_parameters['name'] + 'DW'
        for weight in model.graph.initializer:
            if weight.name == weight_name:
                if '1D' in flag:
                    new_parameters['weights'] = np.transpose(numpy_helper.to_array(weight), (0, 2, 1))
                elif 'Conv' in new_parameters['name']:
                    new_parameters['weights'] = np.transpose(numpy_helper.to_array(weight), (0, 2, 3, 1))
                elif node_iterating.op_type in ['Gemm', 'MatMul']:
                    temp = numpy_helper.to_array(weight)
                    if 'MatMul' in node_iterating.op_type:
                        temp = temp.T
                    try:
                        temp = temp.reshape(new_parameters['ch_out'], ch_in_temp, self.PULP_Nodes_Graph[-1].get_parameter('output_dim')[0], self.PULP_Nodes_Graph[-1].get_parameter('output_dim')[1])
                    except:
                        temp = temp.reshape(new_parameters['ch_out'], ch_in_temp, self.PULP_Nodes_Graph[-2].get_parameter('output_dim')[0], self.PULP_Nodes_Graph[-2].get_parameter('output_dim')[1])
                    temp = np.transpose(temp, (0, 2, 3, 1))
                    temp = temp.flatten()
                    new_parameters['weights'] = temp
            if weight.name == bias_name:
                new_parameters['bias'] = numpy_helper.to_array(weight)
        if 'Add' not in new_parameters['name']:
            try:
                new_parameters['MACs'] = new_parameters['kernel_shape'][0] * new_parameters['kernel_shape'][1] * new_parameters['ch_in'] * new_parameters['ch_out'] * new_parameters['output_dim'][0] * new_parameters['output_dim'][1]
            except:
                new_parameters['MACs'] = new_parameters['kernel_shape'][0] * new_parameters['kernel_shape'][1] * new_parameters['ch_in'] * new_parameters['ch_out'] * new_parameters['output_dim']
        else:
            new_parameters['MACs'] = new_parameters['ch_in'] * new_parameters['ch_out']* new_parameters['output_dim'][0] * new_parameters['output_dim'][1]
        new_node.add_dict_parameter(new_parameters)
        return new_node

    def fuse_nodes(self, node_1, node_2):
    	assert (node_1.get_parameter('output_index') == node_2.get_parameter('input_index')), f"Error in fusion of near nodes with different indexes"
    	node_1.add_parameter('name', node_1.get_parameter('name')+node_2.get_parameter('name') )
    	for key, value in node_2.__dict__.items():
            if (isinstance(value,str)):
                if value == 'Not-initialized':
                    pass
                elif key not in ['name', 'input_index']:
                    node_1.add_parameter(key,value)
            elif key not in ['name', 'input_index', 'weight_bits']:
                node_1.add_parameter(key,value)
    	return node_1

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

    def check_rules(self, node):
        out = node.output[0] 
        string_rule = node.op_type
        if string_rule in [*self.rules.values()]:
            return string_rule
        for node_i in (self.model.graph.node):
            try:
                input_i = [input_i for input_i in node_i.input if 'weight' not in input_i][0]
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
                        return string_rule

    def apply_rule(self, node, rule):
        pass

    def print_PULP_graph(self, name_file):
        # Logging function to report exported graph of PULP
        logging.basicConfig(filename='logs/'+name_file+'.log',
                            format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)
        print("Creating annotated graph in " + name_file + ".log")
        for i,node in enumerate(self.PULP_Nodes_Graph):
            logging.debug(f'\nNode: {i}')
            node.log_parameters()
        log = logging.getLogger() 
        for hdlr in log.handlers[:]:
            hdlr.close()
            log.removeHandler(hdlr)

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



    def update_precisions_graph(self):
        # adding input bit precision
        for i, nodes in enumerate(self.PULP_Nodes_Graph):
            if i == 0:
                nodes.add_parameter('input_activation_bits', 8)
            else:
                for j, precedent_nodes in enumerate(self.PULP_Nodes_Graph[:i]):
                    if precedent_nodes.output_index == nodes.input_index:
                        nodes.add_parameter('input_activation_bits', precedent_nodes.get_parameter('out_activation_bits'))
                        if 'Add' in nodes.name:
                            nodes.add_parameter('out_activation_bits', nodes.get_parameter('input_activation_bits'))
            if i == (len(self.PULP_Nodes_Graph)-1):
                nodes.add_parameter('out_activation_bits', 32)
            if 'Pool' in nodes.name:
                nodes.add_parameter('out_activation_bits', nodes.get_parameter('input_activation_bits'))
        for i, nodes in enumerate(self.PULP_Nodes_Graph):
            if nodes.get_parameter('input_activation_bits') != 8 or nodes.get_parameter('out_activation_bits') != 8 or nodes.get_parameter('weight_bits') != 8:
                multiple = 8/min(nodes.get_parameter('input_activation_bits'), nodes.get_parameter('out_activation_bits'), nodes.get_parameter('weight_bits'))
                if (nodes.get_parameter('ch_in')*nodes.get_parameter('group'))%multiple !=0 or nodes.get_parameter('ch_out')%multiple !=0:
                    sys.exit("ERROR 01. Channels of a layer not multiple of 2 (int4 precision layers) or 4 (int2 precision layers). Exiting...")

    def update_branches_graph(self):
        # updating branch in/out connections
        for i, nodes in enumerate(self.PULP_Nodes_Graph):
            counter = 0
            for nodes_scan in self.PULP_Nodes_Graph:
                if nodes.output_index == nodes_scan.input_index:
                    counter += 1
                if 'Add' in nodes_scan.name:
                    if nodes.output_index == nodes_scan.input_index_add:
                        counter += 1
            if counter > 1:
                self.PULP_Nodes_Graph[i].add_parameter('branch_out', 1)
        index_of_first_add = 0
        index_of_second_add = 0
        for i, node in enumerate(self.PULP_Nodes_Graph):
            if 'Add' in node.name:
                input1_add = node.input_index
                input2_add = node.input_index_add
                for j, node_two in enumerate(self.PULP_Nodes_Graph):
                    if node_two.output_index == input1_add:
                        PULP_graph_index_of_input1_add_node = j
                    elif node_two.output_index == input2_add:
                        PULP_graph_index_of_input2_add_node = j
                if self.PULP_Nodes_Graph[PULP_graph_index_of_input1_add_node].get_parameter('branch_out') != 1 and self.PULP_Nodes_Graph[PULP_graph_index_of_input2_add_node].get_parameter('branch_out') != 1:
                    if(PULP_graph_index_of_input1_add_node > PULP_graph_index_of_input2_add_node):
                        self.PULP_Nodes_Graph[PULP_graph_index_of_input2_add_node].add_parameter('branch_change', 1) 
                    else:
                        self.PULP_Nodes_Graph[PULP_graph_index_of_input1_add_node].add_parameter('branch_change', 1) 
                if int(input1_add) > int(input2_add):
                    input2_add = input1_add
                for node_two in self.PULP_Nodes_Graph:
                    if node_two.output_index == input2_add:
                        node_two.add_parameter('branch_last', 1)  

    def check_graph(self):
        # Logging function to report exported graph of PULP
        for node in self.PULP_Nodes_Graph:
            if node.name not in self.backend:
                sys.exit(f"ERROR 02. Node {node.name} inside the graph not supported by the backend. Exiting...")

    def onnx_to_PULP(self):
        # Load all parameters from the onnx model.
        first_node = 1
        skips = 0
        os.system('rm -rf logs')
        os.system('mkdir logs')
        ######### CREATING NODES ###########
        for node_iterating in (self.model.graph.node):
        	### check if the node is supported
            assert (node_iterating.op_type in self.layers_accepted), f"{node_iterating.op_type} not supported by DORY"
            ### check for rules
            if skips > 0:
            	skips -=1
            	continue
            if self.check_rules(node_iterating):
                node = self.apply_rule(node_iterating, self.check_rules(node_iterating))
                self.PULP_Nodes_Graph.append(node)
                skips = self.check_rules(node_iterating).count('-')
                continue
            # Adding a new Conv, Pool or Linear layer
            if node_iterating.op_type in self.layers_to_node:
                new_node = self.create_node(pulp.node_element(), first_node, node_iterating, self.model)
                self.PULP_Nodes_Graph.append(new_node)
                first_node = 0
                continue
            if node_iterating.op_type in self.layers_neglected and int(node_iterating.output[0]) > int(self.PULP_Nodes_Graph[-1].get_parameter('output_index')):
                self.PULP_Nodes_Graph[-1].add_parameter('output_index', node_iterating.output[0]) 
                continue
        self.print_PULP_graph("PULP_Raw_Graph")
        self.fuse_graph()
        self.print_PULP_graph("PULP_Fused_Graph")
        self.update_precisions_graph()
        self.update_branches_graph()
        self.print_PULP_graph("PULP_Final_Graph")
        self.check_graph()
        return self.PULP_Nodes_Graph

    