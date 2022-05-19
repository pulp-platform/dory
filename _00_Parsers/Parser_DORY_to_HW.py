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
import numpy as np
import sys
import copy

# DORY modules
from _00_Parsers.HW_node import HW_node
from _01_Utils.DORY_utils import Printer


class Parser_DORY_to_HW:
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    def __init__(self, graph, rules, Pattern_rewriter, supported_nodes, HW_description, network_directory, Tiler):
        self.supported_nodes = supported_nodes
        self.DORY_Graph = graph
        self.Printer_Frontend = Printer("logs/HW_related")
        self.Pattern_rewriter = Pattern_rewriter
        self.rules = rules
        self.HW_description = HW_description
        self.network_directory = network_directory
        HW_node.Tiler = Tiler

    def mapping_to_HW_nodes(self):
        print("\nGAP8 Backend: Matching patterns from generated DORY ONNX to HW Nodes.")
        for i, node in enumerate(self.DORY_Graph):
            string_matching, indexes = self.pattern_matching(node, i)
            if isinstance(string_matching, str):
                self.DORY_Graph = self.Pattern_rewriter(self.DORY_Graph).execute(string_matching, indexes)
        
    def check_graph(self):
        for node in self.DORY_Graph:
            if node.name not in self.supported_nodes:
                sys.exit("\nDORY Backend Check. Node {} is not accepted inside the HW Frontend IR.\n".format(node.name))
        print("\nDORY checking of the graph: OK\n")

    def check_parameters(self):
        print("\nTo be implemented in the target backend")

    def pattern_matching(self, input_node, input_index):
        number_of_nodes = 0
        rule_found = False
        DORY_node_indexes_to_export = []
        for key, rule in self.rules.items():
            DORY_node_indexes = []
            DORY_node_indexes.append(input_index)
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
                    for nodes_index in inputs:
                        int_index = node.input_indexes
                        node_to_search = nodes[int(nodes_index)]
                        for i,node_i in enumerate(self.DORY_Graph):
                            if node_i.output_index in int_index and node_i.name == node_to_search:
                                nodes[int(nodes_index)] = "Match"
                                match = 1
                                DORY_node_indexes.append(i)
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
                if sum(x=="Match" for x in nodes) == len(nodes):
                    if number_of_nodes < rule["number_of_nodes"]:
                        rule_found = key
                        number_of_nodes = rule["number_of_nodes"]
                        DORY_node_indexes_to_export = DORY_node_indexes
        return rule_found, DORY_node_indexes_to_export

    def update_branches_graph(self):
        print("\nDORY generic Frontend. Updating branches pointers.")
        # updating branch in/out connections
        for i, node in enumerate(self.DORY_Graph):
            if len(node.input_indexes)>1:
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

    def update_dimensions_graph(self):
        print("\nUpdating dimensions of vectors inside the graph, if they do not match among nodes")
        for i, node in enumerate(self.DORY_Graph):
            if i > 0:
                if isinstance(self.DORY_Graph[i].input_channels, type(None)):
                    if "FullyConnected" in self.DORY_Graph[i].name:
                        self.DORY_Graph[i].input_channels = int(self.DORY_Graph[i-1].output_channels*np.prod(self.DORY_Graph[i-1].output_dimensions))
                    else:
                        self.DORY_Graph[i].input_channels = self.DORY_Graph[i-1].output_channels
                if len(self.DORY_Graph[i].input_dimensions)==0:
                    self.DORY_Graph[i].input_dimensions = self.DORY_Graph[i-1].output_dimensions

    def add_tensors_memory_occupation_and_MACs(self):
        print("\nUpdating memory occupation and MACs of tensors in layers")
        for i, node in enumerate(self.DORY_Graph):
            if "Convolution" in node.name or "FullyConnected" in node.name or "Addition" in node.name or "Pooling" in node.name:
                node.add_memory_and_MACs()

    def adjust_data_layout(self):
        print("\nTo be implemented in the target backend")

    def tiling(self):
        ####################################################################################
        ###### SECTION 3: PARSING OF EACH LAYER INDEPENDENT. TILING + LAYER CREATION  ######
        ####################################################################################
        print("\nInsert tiling parameters per layer inside graph nodes")
        for i, node_to_tile in enumerate(self.DORY_Graph):            
            ######################## NEED A  FIX ####################################################
            #### OTHERWISE ONLY WEIGHT < L2/2 GO in L2 --> much more L3 tiling not needed############
            #########################################################################################
            New_HW_node = HW_node(node_to_tile, self.HW_description)
            if i > 0:
                New_HW_node.create_tiling_dimensions(previous_node)
            else:
                New_HW_node.create_tiling_dimensions(New_HW_node)
            previous_node = New_HW_node
            self.DORY_Graph[i] = New_HW_node

    def renaming_weights(self):
        print("\nGAP8 Backend: Renaming Weights tensors.")
        for i, node in enumerate(self.DORY_Graph):            
            node.rename_weights()           

    def formatting_constant_parameters_tensors_and_activations(self):
        print("\nGAP8 Backend: Formatting constants and adding checksums")
        for i, node in enumerate(self.DORY_Graph):            
            node.add_checksum_w_integer()           
            node.add_checksum_activations_integer(self.network_directory, i)

    def full_graph_parsing(self):
        print("#####################################################")
        print("## DORY GENERAL PARSING FROM DORY IR TO DORY HW IR ##")
        print("## FINAL RAPRESENTATION: DORY HW IR                ##")
        print("#####################################################")
        self.mapping_to_HW_nodes()
        self.Printer_Frontend.print_json_from_DORY_graph("01_DORY_HW_graph_raw", self.DORY_Graph)
        self.Printer_Frontend.print_onnx_from_DORY_graph("01_DORY_HW_graph_raw", self.DORY_Graph)
        self.update_branches_graph()
        self.Printer_Frontend.print_json_from_DORY_graph("02_DORY_HW_graph_fixed_branches", self.DORY_Graph)
        self.Printer_Frontend.print_onnx_from_DORY_graph("02_DORY_HW_graph_fixed_branches", self.DORY_Graph)
        self.update_dimensions_graph()
        self.Printer_Frontend.print_json_from_DORY_graph("03_DORY_HW_graph_fixed_dimensions", self.DORY_Graph)
        self.Printer_Frontend.print_onnx_from_DORY_graph("03_DORY_HW_graph_fixed_dimensions", self.DORY_Graph)
        self.add_tensors_memory_occupation_and_MACs()
        self.Printer_Frontend.print_json_from_DORY_graph("04_DORY_HW_graph_added_tensors_dim", self.DORY_Graph)
        self.Printer_Frontend.print_onnx_from_DORY_graph("04_DORY_HW_graph_added_tensors_dim", self.DORY_Graph)
        self.adjust_data_layout()
        self.Printer_Frontend.print_json_from_DORY_graph("05_DORY_HW_adjusted_data_layout", self.DORY_Graph)
        self.Printer_Frontend.print_onnx_from_DORY_graph("05_DORY_HW_adjusted_data_layout", self.DORY_Graph)
        self.tiling()
        self.Printer_Frontend.print_json_from_DORY_graph("06_DORY_HW_tiled_graph", self.DORY_Graph)
        self.Printer_Frontend.print_onnx_from_DORY_graph("06_DORY_HW_tiled_graph", self.DORY_Graph)
        self.renaming_weights()
        self.formatting_constant_parameters_tensors_and_activations()
        self.Printer_Frontend.print_json_from_DORY_graph("07_DORY_HW_with_checksums", self.DORY_Graph)
        self.Printer_Frontend.print_onnx_from_DORY_graph("07_DORY_HW_with_checksums", self.DORY_Graph)
        self.check_graph()
        self.check_parameters()
        return self.DORY_Graph

