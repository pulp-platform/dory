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

# Libraries
import numpy as np

# DORY modules
from _00_Parsers import DORY_node


class Pattern_rewriter:
    def __init__(self, graph):
        self.graph = graph

    def execute(self, rule, i):
        if rule == "BNRelu":
            self.BNRelu_pattern_rewriter(i)
        if rule == "Relu":
            self.Relu_pattern_rewriter(i)
        if rule == "PadConvolution" or rule == "PadPooling":
            self.PADNode_pattern_rewriter(i)
        # if rule == "QAdd":
        #     self.QAdd_pattern_rewriter(i)
        return self.graph

    def BNRelu_pattern_rewriter(self, i):
        DORY_BNRelu_node = DORY_node.DORY_node()
        DORY_BNRelu_node.name = "BNRelu"
        DORY_BNRelu_node.op_type = "BNRelu"
        DORY_BNRelu_node.input_indexes = self.graph[i[0]].input_indexes
        DORY_BNRelu_node.output_index = self.graph[i[-1]].output_index
        DORY_BNRelu_node.number_of_input_nodes = self.graph[i[0]].number_of_input_nodes
        DORY_BNRelu_node.number_of_input_constants = sum(self.graph[x].number_of_input_constants for x in i)
        DORY_BNRelu_node.output_activation_bits = self.graph[i[-1]].out_bits
        DORY_BNRelu_node.branch_out = None
        DORY_BNRelu_node.branch_in = None
        DORY_BNRelu_node.branch_change = None
        DORY_BNRelu_node.branch_last = None

        ### k ###
        for key, value in self.graph[i[0]].__dict__.items():
            if isinstance(value, dict):
                k = value["value"]
        ### l ###
        for key, value in self.graph[i[1]].__dict__.items():
            if isinstance(value, dict):
                l = value["value"]
        ### outshift ###
        for key, value in self.graph[i[2]].__dict__.items():
            if isinstance(value, dict):
                outshift = (value["value"][0] if isinstance(value["value"].tolist(),list) else value["value"])
        DORY_BNRelu_node.k = {}
        DORY_BNRelu_node.k["value"] = k
        DORY_BNRelu_node.k["layout"] = ""
        DORY_BNRelu_node.l = {}
        DORY_BNRelu_node.l["value"] = l
        DORY_BNRelu_node.l["layout"] = ""
        DORY_BNRelu_node.outshift = {}
        DORY_BNRelu_node.outshift["value"] = round(np.log2(outshift))
        DORY_BNRelu_node.outshift["layout"] = ""
        DORY_BNRelu_node.min = self.graph[i[-1]].min
        DORY_BNRelu_node.max = self.graph[i[-1]].max
        DORY_BNRelu_node.constant_names = ["k", "l", "outshift"]
        for ele in sorted(i, reverse = True):
            del self.graph[ele]
        self.graph.insert(i[0], DORY_BNRelu_node)

    def Relu_pattern_rewriter(self, i):
        DORY_Relu_node = DORY_node.DORY_node()
        DORY_Relu_node.name = "Relu"
        DORY_Relu_node.op_type = "Relu"
        DORY_Relu_node.input_indexes = self.graph[i[0]].input_indexes
        DORY_Relu_node.output_index = self.graph[i[-1]].output_index
        DORY_Relu_node.number_of_input_nodes = self.graph[i[0]].number_of_input_nodes
        DORY_Relu_node.number_of_input_constants = 2
        DORY_Relu_node.output_activation_bits = self.graph[i[-1]].out_bits
        DORY_Relu_node.branch_out = None
        DORY_Relu_node.branch_in = None
        DORY_Relu_node.branch_change = None
        DORY_Relu_node.branch_last = None

        ### outmul ###
        for key, value in self.graph[i[0]].__dict__.items():
            if isinstance(value, dict):
                outmul = value["value"]
        DORY_Relu_node.outmul = {}
        DORY_Relu_node.outmul["value"] = int(outmul)
        DORY_Relu_node.outmul["layout"] = ""
        ### outshift ###
        for key, value in self.graph[i[1]].__dict__.items():
            if isinstance(value, dict):
                outshift = (value["value"][0] if isinstance(value["value"].tolist(),list) else value["value"])
        DORY_Relu_node.outshift = {}
        DORY_Relu_node.outshift["value"] = round(np.log2(outshift))
        DORY_Relu_node.outshift["layout"] = ""
        DORY_Relu_node.min = self.graph[i[-1]].min
        DORY_Relu_node.max = self.graph[i[-1]].max
        DORY_Relu_node.constant_names = ["outmul", "outshift"]
        for ele in sorted(i, reverse = True):
            del self.graph[ele]
        self.graph.insert(i[0], DORY_Relu_node)

    def PADNode_pattern_rewriter(self, i):
        DORY_Pad_node = self.graph[i[-1]]
        DORY_Pad_node.input_indexes = self.graph[i[0]].input_indexes
        ### 2D ###
        for j in np.arange(len(i)-1):
            if len(DORY_Pad_node.pads) == 4:
                DORY_Pad_node.pads[0] += self.graph[i[j]].pads[2]
                DORY_Pad_node.pads[1] += self.graph[i[j]].pads[3]
                DORY_Pad_node.pads[2] += self.graph[i[j]].pads[6]
                DORY_Pad_node.pads[3] += self.graph[i[j]].pads[7]
                DORY_Pad_node.input_dimensions[0] -= (self.graph[i[j]].pads[2] + self.graph[i[j]].pads[6])
                DORY_Pad_node.input_dimensions[1] -= (self.graph[i[j]].pads[3] + self.graph[i[j]].pads[7])
            else:
                DORY_Pad_node.pads[0] += self.graph[i[j]].pads[2]
                DORY_Pad_node.pads[1] += self.graph[i[j]].pads[5]
                DORY_Pad_node.input_dimensions[0] -= (self.graph[i[j]].pads[2] + self.graph[i[j]].pads[5])
        for ele in sorted(i, reverse = True):
            del self.graph[ele]
        self.graph.insert(i[0], DORY_Pad_node)

