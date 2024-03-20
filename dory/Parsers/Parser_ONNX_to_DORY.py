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
from onnx import shape_inference
import sys
import copy

# DORY modules
from . import Layer_node
from . import DORY_node
from dory.Utils.DORY_utils import Printer


class Parser_ONNX_to_DORY:
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    def __init__(self, network, rules, layers_accepted, layers_neglected, layers_to_node, net_prefix="", n_trainable_layers=0):
        self.graph = onnx.load(network)
        self.Printer_Frontend = Printer("logs/Frontend")
        self.Printer_Frontend.print_onnx("Original_graph", self.graph)
        self.Printer_Frontend.print_json("Original_graph", self.graph)
        self.graph = shape_inference.infer_shapes(self.graph)
        self.layers_accepted = layers_accepted
        self.layers_neglected = layers_neglected
        self.layers_to_node = layers_to_node
        self.layers_supported_by_DORY_Frontend_IR = ["Convolution", "Pooling", "FullyConnected", "Addition", "QAddition", "Relu", "BNRelu", "Requant"]
        self.rules = rules
        self.net_prefix = net_prefix
        self.n_trainable_layers = n_trainable_layers
        print(f"onnx_to_dory net prefix: {net_prefix}")
        self.DORY_Graph = []

    def create_node(self, node_iterating, graph):
        '''
        Creation of a Layer node between Add, Convolution, Pooling and Fully Connected.
        As an alternative, create a DORY node (Mul, Shift, Div, Clip, etc...).
        '''
        new_node = DORY_node.DORY_node()
        new_node.populate_DORY_node(node_iterating,graph, self.net_prefix)
        if new_node.name in ['FullyConnected', 'Addition', 'Convolution', 'Pooling']:
            new_node = Layer_node.Layer_node()
            new_node.populate_Layer_node(node_iterating,graph, self.net_prefix)
        return new_node

    def ONNXtoDORY(self):
        def pos_in_schedule(node: str) -> int:
            for idx, node_iterating in enumerate(list(self.graph.graph.node)):
                if node_iterating.output[0] == node:
                    return idx
        ######### CREATING NODES ###########
        print("\nParsing ONNX Graph to create DORY graph.")
        for node_iterating in (self.graph.graph.node):
            ### check if the node is supported
            assert (node_iterating.op_type in self.layers_accepted), f"{node_iterating.op_type} not supported by DORY"
            ### Neglecting some nodes since they are not translated to any operation on any backend
            if node_iterating.op_type in self.layers_neglected:
                for node in self.DORY_Graph[::-1]:
                    node_iter_output = pos_in_schedule(node_iterating.output[0])
                    node_output_index = pos_in_schedule(node.get_parameter('output_index'))
                    if node_iter_output > node_output_index and node.get_parameter("name") != "Constant":
                        node.add_existing_parameter('output_index', node_iterating.output[0]) 
                        break
            # Adding a new layer
            elif node_iterating.op_type in self.layers_accepted:
                new_node = self.create_node(node_iterating, self.graph)
                self.DORY_Graph.append(new_node)
            else:
                sys.exit("DORY Frontend. Node not parsed.")

    def remove_Constants(self):
        print("\nEmbedding constant nodes inside nodes to which the tensors belong.")
        removed = 1
        while removed:
            for node in self.DORY_Graph:
                if node.name == 'Constant':
                    self.DORY_Graph.remove(node)
                    break
                if node == self.DORY_Graph[-1]:
                    removed = 0

    def frontend_mapping_to_DORY_nodes(self):
        print("\nTo be implemented in the target backend")

    def check_graph(self):
        for node in self.DORY_Graph:
            if node.name not in self.layers_supported_by_DORY_Frontend_IR:
                sys.exit("\nDORY Frontend Check. Node {} is not accepted inside the DORY Frontend IR.\n".format(node.name))
        print("\nDORY checking of the graph: OK\n")

    def pattern_matching(self, input_node, input_index):
        number_of_nodes = 0
        rule_found = False
        DORY_node_indexes_to_export = []
        for key, rule in self.rules.items():
            DORY_node_indexes = []
            DORY_node_indexes.append(input_index)
            index_in_pattern = []
            if rule["number_of_nodes"] == 1 and input_node.name in rule["nodes_name"]:
                rule_found = key
            elif input_node.name in rule["nodes_name"]:
                node = input_node
                match = 1
                nodes = copy.deepcopy(rule["nodes_name"])
                index = nodes.index(node.name)
                nodes[index] = "Match"
                while match == 1:
                    match = 0
                    inputs = rule["dependencies"][str(index)]["inputs"]
                    outputs = rule["dependencies"][str(index)]["outputs"]
                    for j, nodes_index in enumerate(inputs):
                        int_index = node.input_indexes
                        node_to_search = nodes[int(nodes_index)]
                        for i,node_i in enumerate(self.DORY_Graph):
                            if node_i.output_index in int_index and node_i.name == node_to_search and node_i.output_index not in index_in_pattern:
                                nodes[int(nodes_index)] = "Match"
                                match = 1
                                DORY_node_indexes.append(i)
                                index_in_pattern.append(node_i.output_index)
                    for nodes_index in outputs:
                        out_index = node.output_index
                        node_to_search = nodes[int(nodes_index)]
                        for i,node_i in enumerate(self.DORY_Graph):
                            if out_index in node_i.input_indexes and node_i.name == node_to_search:
                                nodes[int(nodes_index)] = "Match"
                                match = 1
                                DORY_node_indexes.append(i)
                                node = node_i
                                index = int(nodes_index) 
                                index_in_pattern.append(out_index)
                if sum(x=="Match" for x in nodes) == len(nodes):
                    if number_of_nodes < rule["number_of_nodes"]:
                        rule_found = key
                        number_of_nodes = rule["number_of_nodes"]
                        DORY_node_indexes_to_export = DORY_node_indexes
        return rule_found, DORY_node_indexes_to_export

    def add_nodes_precision(self):
        print("\nTo be implemented in the target backend")

    def update_branches_graph(self):
        print("\nDORY generic Frontend. Updating branches pointers.")
        # updating branch in/out connections
        for i, node in enumerate(self.DORY_Graph):
            if  len(node.input_indexes)>1:
                node.add_existing_parameter("branch_in", 1)
            else:
                node.add_existing_parameter("branch_in", 0)
            node_out = 0
            for nodes_scan in self.DORY_Graph:
                if node.output_index in nodes_scan.input_indexes:
                    node_out+=1
            if node_out > 1:
                node.add_existing_parameter("branch_out", 1)
            else:
                node.add_existing_parameter("branch_out", 0)
            node.add_existing_parameter("branch_change", 0)
            node.add_existing_parameter("branch_last", 0)
            for nodes_scan in self.DORY_Graph:
                if node.output_index in nodes_scan.input_indexes and len(nodes_scan.input_indexes)>1:
                    for j, nodes_scan_2 in enumerate(self.DORY_Graph):
                        if nodes_scan_2.output_index in nodes_scan.input_indexes and nodes_scan_2.output_index != node.output_index:
                            if nodes_scan_2.branch_out != 1 and node.branch_out != 1:
                                if(i < j):
                                    node.add_existing_parameter("branch_change", 1)
                                    nodes_scan_2.add_existing_parameter("branch_last", 0)
                                    break
                                else:
                                    nodes_scan_2.add_existing_parameter("branch_change", 1)  
                                    node.add_existing_parameter("branch_last", 1)   
                                    break
                            else:
                                if(i < j):
                                    node.add_existing_parameter("branch_last", 1)
                                else:
                                    nodes_scan_2.add_existing_parameter("branch_last", 1)  

        # Remove trainable layers, handled externally
        self.DORY_Graph = self.DORY_Graph[:len(self.DORY_Graph)-self.n_trainable_layers]

    def add_data_layout(self):
        print("\nTo be implemented in the target backend")

    def full_graph_parsing(self):
        print("")
        print("##################################")
        print("## DORY GENERAL PARSING OF ONNX ##")
        print("## FINAL RAPRESENTATION:DORY IR ##")
        print("##################################")
        self.ONNXtoDORY()
        self.Printer_Frontend.print_json_from_DORY_graph("00_DORY_raw_graph", self.DORY_Graph)
        self.Printer_Frontend.print_onnx_from_DORY_graph("00_DORY_raw_graph", self.DORY_Graph)
        self.remove_Constants()
        self.Printer_Frontend.print_json_from_DORY_graph("01_DORY_graph_constants_removed", self.DORY_Graph)
        self.Printer_Frontend.print_onnx_from_DORY_graph("01_DORY_graph_constants_removed", self.DORY_Graph)
        self.frontend_mapping_to_DORY_nodes()
        self.Printer_Frontend.print_json_from_DORY_graph("02_DORY_mapped_graph", self.DORY_Graph)
        self.Printer_Frontend.print_onnx_from_DORY_graph("02_DORY_mapped_graph", self.DORY_Graph)
        self.add_nodes_precision()
        self.Printer_Frontend.print_json_from_DORY_graph("03_DORY_graph_with_node_precision", self.DORY_Graph)
        self.Printer_Frontend.print_onnx_from_DORY_graph("03_DORY_graph_with_node_precision", self.DORY_Graph)
        self.update_branches_graph()
        self.Printer_Frontend.print_json_from_DORY_graph("04_DORY_updated_branches", self.DORY_Graph)
        self.Printer_Frontend.print_onnx_from_DORY_graph("04_DORY_updated_branches", self.DORY_Graph)
        self.add_data_layout()
        self.Printer_Frontend.print_json_from_DORY_graph("05_DORY_Frontend_final_graph", self.DORY_Graph)
        self.Printer_Frontend.print_onnx_from_DORY_graph("05_DORY_Frontend_final_graph", self.DORY_Graph)
        self.check_graph()
        return self.DORY_Graph

