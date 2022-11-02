# should work even without -*-
# -*- coding: utf-8 -*-
#!/bin/bash
# PULP_node.py
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
import sys
from onnx import numpy_helper
import numpy as np


class DORY_node:
    """
    This class define a generic node in the DORY graph, with name, connections, and general information.
    The class also define the methods to manage these nodes.
    """
    def __init__(self):
        self.name = None
        self.op_type = None
        self.input_indexes = []
        self.output_index = None
        self.constant_names = []
        self.number_of_input_nodes = None
        self.number_of_input_constants = None
        self.branch_out = None
        self.branch_in = None
        self.branch_change = None
        self.branch_last = None
        self.constant_bits = None
        self.weight_bits = None
        self.bias_bits = None
        self.output_activation_bits = None
        self.input_activation_bits = None
        self.second_input_activation_bits = None
        self.weight_type = None
        self.constant_type = None
        self.output_activation_type = None
        self.input_activation_type = None
        self.second_input_activation_type = None
        self.n_test_inputs = None
        self.conv1d = None
        self.min = None
        self.max = None
        self.prefix = None

    @property
    def prefixed_name(self):
        if self.name and self.prefix:
            return self.prefix + self.name

        return self.name

    def print_parameters(self):
        for parameter in self.__dict__:
            if parameter not in ['weights', 'k', 'lambda']:
                print(parameter + ': ' + str(self.__dict__[parameter]))
            else:
                print(parameter + ': Present')

    def add_existing_parameter(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            sys.exit("Adding {} parameter to a graph node. This parameter does not exist. ".format(key))

    def add_existing_dict_parameter(self, dict_parameters):
        for key, value in dict_parameters.items():
            self.add_existing_parameter(key, value)

    def get_parameter(self, name):
        return self.__dict__[name]

    def check_uninitialized_parameters(self):
        for key, value in self.__dict__.items():
            if isinstance(value, type(None)) or (isinstance(value, list) and len(value) == 0):
                sys.exit("DORY FRONTEND error. Missing some Node initialization. Stopping at argument {}".format(key))

    def populate_DORY_node(self, node_iterating, graph, prefix=""):
        DORY_parameters = {}
        #### Names: Convolution, Addition, FullyConnected, Pooling
        mapping_names = {'AveragePool': 'Pooling', 
                         'MaxPool': 'Pooling', 
                         'Conv': 'Convolution', 
                         'Gemm': 'FullyConnected', 
                         'MatMul': 'FullyConnected', 
                         'GlobalAveragePool': 'Pooling', 
                         'Add': 'Addition'}
        if node_iterating.op_type in mapping_names.keys():
            DORY_parameters['name'] = mapping_names[node_iterating.op_type]
        else:
            DORY_parameters['name'] = node_iterating.op_type
        DORY_parameters['op_type'] = node_iterating.op_type
        DORY_parameters['input_indexes'] = []
        DORY_parameters['constant_names'] = []
        for iterating_input in node_iterating.input:
            is_constant = len([t for t in graph.graph.initializer if t.name == iterating_input])
            if is_constant == 1:
                DORY_parameters['constant_names'].append(iterating_input)
            elif is_constant == 0:
                DORY_parameters['input_indexes'].append(iterating_input)
        DORY_parameters['output_index'] = node_iterating.output[0]
        DORY_parameters['number_of_input_nodes'] = len(DORY_parameters['input_indexes'])
        DORY_parameters['number_of_input_constants'] = len(DORY_parameters['constant_names'])
        DORY_parameters['prefix'] = prefix
        self.add_existing_dict_parameter(DORY_parameters)

        self.add_constants(node_iterating, graph)

        if self.name == 'Addition' and len(self.input_indexes) == 1:
            self.name = node_iterating.op_type

        self.add_special_attributes(node_iterating)

    def add_constants(self, node_iterating, graph):
        '''
        two ways for identifying constants:
        - in the graph.graph.initializer
        - in graph.graph.node as Constant nodes
        '''
        for weight_tensor in graph.graph.initializer:
            if weight_tensor.name in self.constant_names:
                self.__dict__[weight_tensor.name] = {}
                self.__dict__[weight_tensor.name]["value"] = numpy_helper.to_array(weight_tensor)
                self.__dict__[weight_tensor.name]["layout"] = None

        for weight_tensor in graph.graph.node:
            if weight_tensor.op_type == 'Constant' and weight_tensor.output[0] in self.input_indexes:
                self.__dict__["constant"] = {}
                self.__dict__["constant"]["value"] = numpy_helper.to_array(weight_tensor.attribute[0].t)
                self.__dict__["constant"]["layout"] = None
                self.constant_names.append(weight_tensor.output[0])
                self.input_indexes.remove(weight_tensor.output[0])
                self.number_of_input_nodes -= 1
                self.number_of_input_constants += 1


    def add_special_attributes(self, node_iterating):
        '''
        adding not expected and custom attributes (e.g., min and max in clip)
        '''
        for attribute in node_iterating.attribute:
            if attribute.name not in ['kernel_shape', 'dilations', 'group', 'strides', 'pads'] or self.name == "Pad":
                if bool(attribute.i):
                    self.__dict__[attribute.name] = int(attribute.i)
                elif bool(attribute.f):
                    self.__dict__[attribute.name] = int(attribute.f)
                elif bool(attribute.ints):
                    self.__dict__[attribute.name] = list(attribute.ints)
                elif attribute.i == 0:
                    self.__dict__[attribute.name] = 0
                else:
                    sys.exit("DORY FRONTEND error. DORY does not find any values for the attribute {}".format(attribute.name))

    def export_to_dict(self):
        node_dict = {}
        node_dict["name"] = self.name
        node_dict["DORY_node_parameters"] = {}
        node_dict["Weights"] = {}
        for key, value in self.__dict__.items():
            if not isinstance(value, dict) and key != "name":
                node_dict["DORY_node_parameters"][key] = value
            elif key != "name":
                node_dict["Weights"][key] = {}
                node_dict["Weights"][key]["Present"] = 'Yes'
                node_dict["Weights"][key]["Layout"] = value["layout"]
        return node_dict

    def export_to_onnx(self):
        node_dict = {}
        node_dict["name"] = self.name
        node_dict["input"] = self.input_indexes + self.constant_names
        node_dict["output"] = [self.output_index]
        node_dict["op_type"] = self.op_type
        node_dict["attribute"] = []
        added_parameters = ["name", "input_indexes", "constant_names", "output_index", "op_type"]
        for key, value in self.__dict__.items():
            if not isinstance(value, np.ndarray) and not isinstance(value,str) and not isinstance(value,dict) and not isinstance(value,type(None)) and key not in added_parameters and not isinstance(value, bool):
                node_dict["attribute"].append({"name": key, "ints": ([str(value)] if not isinstance(value,list) else [str(v) for v in value])})
        return node_dict

