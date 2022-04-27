     # should work even without -*-
# -*- coding: utf-8 -*-
#!/bin/bash
# ONNX_to_DORY_generic.py
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
import onnx
from onnx import numpy_helper
from onnx import helper, shape_inference
import numpy as np
import os
import sys
import json

## DORY modules
import Layer_node 
import DORY_node

class ONNX_management():
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    def __init__(self, network, rules, layers_accepted, layers_neglected, layers_to_node):
        self.graph = onnx.load(network)
        self.graph = shape_inference.infer_shapes(self.graph)
        self.layers_accepted = layers_accepted
        self.layers_neglected = layers_neglected
        self.layers_to_node = layers_to_node
        self.rules = rules
        self.DORY_Graph = []

    def print_DORY_graph(self, name_file):
        # Logging function to report exported graph of PULP
        with open("logs/{}.json".format(name_file), "w") as outfile:
            for nodes in self.DORY_Graph:
              json.dump(nodes.export_to_dict(), outfile)
              outfile.write('\n')
        print("Creating {}.json in logs/". format(name_file))

    def create_node(self, node_iterating, graph):
        '''
        Creation of a node between Add, Convolution, Pooling and Fully Connected
        '''
        new_node = DORY_node.DORY_node()
        new_node.populate_DORY_node(node_iterating,graph)
        if new_node.name in ['Fully-Connected', 'Addition', 'Convolution', 'Pooling']:
            new_node = Layer_node.Layer_node()
            new_node.populate_Layer_node(node_iterating,graph)
        return new_node

    def remove_Constants(self):
        removed = 1
        while removed:
            for node in self.DORY_Graph:
                if node.name == 'Constant':
                    self.DORY_Graph.remove(node)
                    break
                if node == self.DORY_Graph[-1]:
                    removed = 0

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


    def check_rules(self, node):
        out = node.output[0] 
        string_rule = node.op_type
        if string_rule in [*self.rules.values()]:
            return string_rule
        for node_i in (self.graph.graph.node):
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


    def fuse_graph(self):
        # Logging function to report exported graph of PULP
        while True:
            DORY_Graph_fused = []
            skip = 0
            not_fused = 1
            fuse_at_least_1 = 0
            for node_1, node_2 in zip(self.DORY_Graph[:-1], self.DORY_Graph[1:]):
                last_node = 0
                if node_1.name+node_2.name in '.'.join([*self.backend]) and skip == 0:
                    DORY_Graph_fused.append(self.fuse_nodes(node_1, node_2))
                    skip = 1
                    not_fused = 0
                    fuse_at_least_1 = 1
                elif skip == 0:
                    DORY_Graph_fused.append(node_1)
                    not_fused = 1
                else:
                    skip = 0
                    last_node = 1
            if not_fused == 1 or last_node == 1:
                DORY_Graph_fused.append(node_2)
            self.DORY_Graph = DORY_Graph_fused
            if fuse_at_least_1 == 0:
                break

    def update_precisions_graph(self):
        # adding input bit precision
        for i, nodes in enumerate(self.DORY_Graph):
            if i == 0:
                nodes.add_parameter('input_activation_bits', 8)
            else:
                for j, precedent_nodes in enumerate(self.DORY_Graph[:i]):
                    if precedent_nodes.output_index == nodes.input_index:
                        nodes.add_parameter('input_activation_bits', precedent_nodes.get_parameter('out_activation_bits'))
                        if 'Add' in nodes.name:
                            nodes.add_parameter('out_activation_bits', nodes.get_parameter('input_activation_bits'))
            if i == (len(self.DORY_Graph)-1):
                nodes.add_parameter('out_activation_bits', 32)
            if 'Pool' in nodes.name and not hasattr(nodes, "requant_pool"):
                nodes.add_parameter('out_activation_bits', nodes.get_parameter('input_activation_bits'))
        for i, nodes in enumerate(self.DORY_Graph):
            if nodes.get_parameter('input_activation_bits') != 8 or nodes.get_parameter('out_activation_bits') != 8 or nodes.get_parameter('weight_bits') != 8:
                multiple = 8/min(nodes.get_parameter('input_activation_bits'), nodes.get_parameter('out_activation_bits'), nodes.get_parameter('weight_bits'))
                if (nodes.get_parameter('ch_in')*nodes.get_parameter('group'))%multiple !=0 or nodes.get_parameter('ch_out')%multiple !=0:
                    sys.exit("ERROR 01. Channels of a layer not multiple of 2 (int4 precision layers) or 4 (int2 precision layers). Exiting...")

    def update_branches_graph(self):
        # updating branch in/out connections
        for i, nodes in enumerate(self.DORY_Graph):
            counter = 0
            for nodes_scan in self.DORY_Graph:
                if nodes.output_index == nodes_scan.input_index:
                    counter += 1
                if 'Add' in nodes_scan.name:
                    if nodes.output_index == nodes_scan.input_index_add:
                        counter += 1
            if counter > 1:
                self.DORY_Graph[i].add_parameter('branch_out', 1)
        index_of_first_add = 0
        index_of_second_add = 0
        for i, node in enumerate(self.DORY_Graph):
            if 'Add' in node.name:
                input1_add = node.input_index
                input2_add = node.input_index_add
                for j, node_two in enumerate(self.DORY_Graph):
                    if node_two.output_index == input1_add:
                        PULP_graph_index_of_input1_add_node = j
                    elif node_two.output_index == input2_add:
                        PULP_graph_index_of_input2_add_node = j
                if self.DORY_Graph[PULP_graph_index_of_input1_add_node].get_parameter('branch_out') != 1 and self.DORY_Graph[PULP_graph_index_of_input2_add_node].get_parameter('branch_out') != 1:
                    if(PULP_graph_index_of_input1_add_node > PULP_graph_index_of_input2_add_node):
                        self.DORY_Graph[PULP_graph_index_of_input2_add_node].add_parameter('branch_change', 1) 
                    else:
                        self.DORY_Graph[PULP_graph_index_of_input1_add_node].add_parameter('branch_change', 1) 
                if int(input1_add) > int(input2_add):
                    input2_add = input1_add
                for node_two in self.DORY_Graph:
                    if node_two.output_index == input2_add:
                        node_two.add_parameter('branch_last', 1)  

    def check_graph(self):
        # Logging function to report exported graph of PULP
        for node in self.DORY_Graph:
            if node.name not in self.backend:
                sys.exit(f"ERROR 02. Node {node.name} inside the graph not supported by the backend. Exiting...")

    def onnx_to_DORY(self):
        # Load all parameters from the onnx graph.
        os.system('rm -rf logs')
        os.system('mkdir logs')
        ######### CREATING NODES ###########
        for node_iterating in (self.graph.graph.node):
            ### check if the node is supported
            assert (node_iterating.op_type in self.layers_accepted), f"{node_iterating.op_type} not supported by DORY"
            ### Neglecting some nodes since they are not translated to any operation on any backend
            if node_iterating.op_type in self.layers_neglected and int(node_iterating.output[0]) > int(self.DORY_Graph[-1].get_parameter('output_index')):
                self.DORY_Graph[-1].add_existing_parameter('output_index', node_iterating.output[0]) 
                continue
            # Adding a new Conv, Pool or Linear layer
            if node_iterating.op_type in self.layers_accepted:
                new_node = self.create_node(node_iterating, self.graph)
                self.DORY_Graph.append(new_node)
                continue

            ### check for rules
            # if skips > 0:
            #     skips -=1
            #     continue
            # if self.check_rules(node_iterating):
            #     node = self.apply_rule(node_iterating, self.check_rules(node_iterating))
            #     self.DORY_Graph.append(node)
            #     skips = self.check_rules(node_iterating).count('-')
            #     continue
        self.print_DORY_graph("DORY_raw_graph")
        self.remove_Constants()
        self.print_DORY_graph("DORY_graph_constants_removed")
        self.fuse_graph()
        self.print_PULP_graph("PULP_Fused_Graph")
        self.update_precisions_graph()
        self.update_branches_graph()
        self.print_PULP_graph("PULP_Final_Graph")
        self.check_graph()
        return self.DORY_Graph
