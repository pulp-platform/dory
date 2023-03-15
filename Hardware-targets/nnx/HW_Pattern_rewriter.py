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


from copy import deepcopy


class Pattern_rewriter:
    def __init__(self, graph):
        self.graph = graph

    def execute(self, rule, i):
        if rule in ["ConvolutionBNRelu", "FullyConnectedBNRelu", "AdditionBNRelu", "QAdditionBNRelu", "PoolingBNRelu"]:
            self.NodeBNRelu_pattern_rewriter(i)
        if rule in ["PoolingRequant"]:
            self.NodeRequant_pattern_rewriter(i)
        if rule in ["ConvolutionRelu", "FullyConnectedRelu", "AdditionRelu", "QAdditionRelu", "PoolingRelu"]:
            self.NodeRelu_pattern_rewriter(i)
        if rule in ["DepthwisePointwise"]:
            self.NodeDepthwisePointwise_pattern_rewriter(i)
        return self.graph

    def NodeDepthwisePointwise_pattern_rewriter(self, indexes):
        convs = [self.graph[indexes[0]], self.graph[indexes[2]]]
        bnrelus = [self.graph[indexes[1]], self.graph[indexes[3]]]
        node = deepcopy(self.graph[indexes[0]])

        assert convs[1].strides == [1, 1]
        assert convs[1].pads == [0, 0, 0, 0]
        assert convs[1].group == 1
        assert convs[0].weight_bits == convs[1].weight_bits

        node.name = "BNRelu" + convs[0].name + "DepthwisePointwise"
        node.op_type = "BNRelu" + convs[0].op_type
        node.output_index = self.graph[indexes[3]].output_index
        node.output_channels = convs[1].output_channels

        # delete consts
        for const in node.constant_names:
            delattr(node, const)
        node.constant_names = []

        # add renamed consts
        for node_group in [convs, bnrelus]:
            for i, old_node in enumerate(node_group):
                for const in old_node.constant_names:
                    new_const = const + str(i)
                    node.constant_names.append(new_const)
                    setattr(node, new_const, getattr(old_node, const))

        # add weight names
        node.weights_names = []
        for i, conv in enumerate(convs):
            for const in conv.constant_names:
                node.weights_names.append(const + str(i))

        def copy_attr(attr):
            assert getattr(bnrelus[0], attr) == getattr(bnrelus[1], attr)
            setattr(node, attr, getattr(bnrelus[0], attr))

        copy_attr("constant_bits")
        copy_attr("min")
        copy_attr("max")
        copy_attr("output_activation_bits")
        copy_attr("output_activation_type")

        for i in sorted(indexes, reverse = True):
            del self.graph[i]

        self.graph.insert(indexes[0], node)

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

