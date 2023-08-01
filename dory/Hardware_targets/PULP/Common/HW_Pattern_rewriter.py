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

from dory.Parsers import Fused_node

class Pattern_rewriter_PULP:
    def __init__(self, graph):
        self.graph = graph

    def execute(self, rule, i):
        if rule in ["ConvolutionBNRelu", "FullyConnectedBNRelu", "AdditionBNRelu", "QAdditionBNRelu", "PoolingBNRelu"]:
            self.NodeBNRelu_pattern_rewriter(i)
        if rule in ["PoolingRequant"]:
            self.NodeRequant_pattern_rewriter(i)
        if rule in ["ConvolutionRelu", "FullyConnectedRelu", "AdditionRelu", "QAdditionRelu", "PoolingRelu"]:
            self.NodeRelu_pattern_rewriter(i)
        if rule in ["DW-PW-Fused"]:
            self.Double_node_BNRelu_pattern_rewriter(i, rule)
        return self.graph

    def NodeBNRelu_pattern_rewriter(self, i):
        DORY_BNRelu_node = self.graph[i[0]]
        DORY_BNRelu_node.constant_bits = self.graph[i[1]].constant_bits
        DORY_BNRelu_node.name = "BNRelu"+self.graph[i[0]].name
        DORY_BNRelu_node.op_type = "BNRelu"+self.graph[i[0]].op_type
        DORY_BNRelu_node.output_index = self.graph[i[1]].output_index
        DORY_BNRelu_node.k = self.graph[i[1]].k
        DORY_BNRelu_node.l = self.graph[i[1]].l
        DORY_BNRelu_node.outshift = self.graph[i[1]].outshift
        DORY_BNRelu_node.min = self.graph[i[1]].min
        DORY_BNRelu_node.max = self.graph[i[1]].max
        DORY_BNRelu_node.constant_names.append("k")
        DORY_BNRelu_node.constant_names.append("l")
        DORY_BNRelu_node.constant_names.append("outshift")
        DORY_BNRelu_node.output_activation_bits = self.graph[i[1]].output_activation_bits
        DORY_BNRelu_node.output_activation_type = self.graph[i[1]].output_activation_type
        for ele in sorted(i, reverse = True):
            del self.graph[ele]
        self.graph.insert(i[0], DORY_BNRelu_node)

    def Double_node_BNRelu_pattern_rewriter(self, i, rule):
        DORY_BNRelu_node = Fused_node.Fused_node()
        for node in [0, 1]:
            DORY_BNRelu_node.__dict__[f"node{str(node)}"] = self.graph[i[2*node]]
            DORY_BNRelu_node.__dict__[f"node{str(node)}"].constant_bits = self.graph[i[2*node+1]].constant_bits
            DORY_BNRelu_node.__dict__[f"node{str(node)}"].k = self.graph[i[2*node+1]].k
            DORY_BNRelu_node.__dict__[f"node{str(node)}"].l = self.graph[i[2*node+1]].l
            DORY_BNRelu_node.__dict__[f"node{str(node)}"].outshift = self.graph[i[2*node+1]].outshift
            DORY_BNRelu_node.__dict__[f"node{str(node)}"].min = self.graph[i[2*node+1]].min
            DORY_BNRelu_node.__dict__[f"node{str(node)}"].max = self.graph[i[2*node+1]].max
            DORY_BNRelu_node.__dict__[f"node{str(node)}"].constant_names.append("k")
            DORY_BNRelu_node.__dict__[f"node{str(node)}"].constant_names.append("l")
            DORY_BNRelu_node.__dict__[f"node{str(node)}"].constant_names.append("outshift")
            DORY_BNRelu_node.__dict__[f"node{str(node)}"].output_activation_bits = self.graph[i[2*node+1]].output_activation_bits
            DORY_BNRelu_node.__dict__[f"node{str(node)}"].output_activation_type = self.graph[i[2*node+1]].output_activation_type
        DORY_BNRelu_node.name = rule
        DORY_BNRelu_node.op_type = rule
        DORY_BNRelu_node.input_dimensions = DORY_BNRelu_node.node0.input_dimensions
        DORY_BNRelu_node.input_activation_bits = DORY_BNRelu_node.node0.input_activation_bits
        DORY_BNRelu_node.output_activation_bits = DORY_BNRelu_node.node1.output_activation_bits
        DORY_BNRelu_node.input_indexes = self.graph[i[0]].input_indexes
        DORY_BNRelu_node.output_index = self.graph[i[3]].output_index
        DORY_BNRelu_node.output_channels = DORY_BNRelu_node.node1.output_channels
        DORY_BNRelu_node.output_dimensions = DORY_BNRelu_node.node1.output_dimensions
        DORY_BNRelu_node.constant_names = DORY_BNRelu_node.node0.constant_names + DORY_BNRelu_node.node1.constant_names
        for ele in sorted(i, reverse = True):
            del self.graph[ele]
        self.graph.insert(i[0], DORY_BNRelu_node)


    def NodeRequant_pattern_rewriter(self, i):
        DORY_BNRelu_node = self.graph[i[0]]
        DORY_BNRelu_node.constant_bits = self.graph[i[1]].constant_bits
        DORY_BNRelu_node.name = "Requant"+self.graph[i[0]].name
        DORY_BNRelu_node.op_type = "Requant"+self.graph[i[0]].op_type
        DORY_BNRelu_node.output_index = self.graph[i[1]].output_index
        DORY_BNRelu_node.outmul = self.graph[i[1]].outmul
        DORY_BNRelu_node.outadd = self.graph[i[1]].outadd
        DORY_BNRelu_node.outshift = self.graph[i[1]].outshift
        DORY_BNRelu_node.min = self.graph[i[1]].min
        DORY_BNRelu_node.max = self.graph[i[1]].max
        DORY_BNRelu_node.constant_names.append("outmul")
        DORY_BNRelu_node.constant_names.append("outadd")
        DORY_BNRelu_node.constant_names.append("outshift")
        DORY_BNRelu_node.output_activation_bits = self.graph[i[1]].output_activation_bits
        DORY_BNRelu_node.output_activation_type = self.graph[i[1]].output_activation_type
        for ele in sorted(i, reverse = True):
            del self.graph[ele]
        self.graph.insert(i[0], DORY_BNRelu_node)

    def NodeRelu_pattern_rewriter(self, i):
        DORY_Relu_node = self.graph[i[0]]
        DORY_Relu_node.name = "Relu"+self.graph[i[0]].name
        DORY_Relu_node.op_type = "Relu"+self.graph[i[0]].op_type
        DORY_Relu_node.output_index = self.graph[i[1]].output_index
        DORY_Relu_node.outmul = self.graph[i[1]].outmul
        DORY_Relu_node.outshift = self.graph[i[1]].outshift
        DORY_Relu_node.min = self.graph[i[1]].min
        DORY_Relu_node.max = self.graph[i[1]].max
        DORY_Relu_node.constant_names.append("outmul")
        DORY_Relu_node.constant_names.append("outshift")
        DORY_Relu_node.output_activation_bits = self.graph[i[1]].output_activation_bits
        DORY_Relu_node.output_activation_type = self.graph[i[1]].output_activation_type
        for ele in sorted(i, reverse = True):
            del self.graph[ele]
        self.graph.insert(i[0], DORY_Relu_node)

