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

import sys
import onnx 
import numpy as np

class DORY_node():
    '''
    This class define a generic node in the DORY graph, with name, connections, and general informations.
    The class also define the methods to manage these nodes.
    '''
    def __init__(self):
        self.name = None
        self.input_indexes = []
        self.output_index = None
        self.constant_names = []
        self.number_of_input_nodes = None
        self.number_of_input_constants = None
        self.branch_out = None
        self.branch_in = None
        self.branch_change = None
        self.branch_last = None

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
            sys.exit("Adding {} parameter to a graph node that does not exist. ".format(key))

    def add_existing_dict_parameter(self, dict_parameters):
        for key, value in dict_parameters.items():
            self.add_existing_parameter(key, value)

    def get_parameter(self, name):
        return self.__dict__[name]

    def check_uninitialized_parameters(self):
        for key, value in self.__dict__.items():
            if isinstance(value, type(None)) or (isinstance(value, list) and len(value) == 0):
                sys.exit("DORY FRONTEND error. Missing some Node initialization. Stopping at argument {}".format(key))

    def populate_DORY_node(self, node_iterating, graph):
        DORY_parameters = {}
        #### Names: Convolution, Addition, Fully-Connected, Pooling
        mapping_names = {'AveragePool': 'Pooling', 
                         'MaxPool': 'Pooling', 
                         'Conv': 'Convolution', 
                         'Gemm': 'Fully-Connected', 
                         'MatMul': 'Fully-Connected', 
                         'GlobalAveragePool': 'Pooling', 
                         'Add': 'Addition'}
        if node_iterating.op_type in mapping_names.keys():
            DORY_parameters['name'] = mapping_names[node_iterating.op_type]
        else:
            DORY_parameters['name'] = node_iterating.op_type
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
        self.add_existing_dict_parameter(DORY_parameters)

        self.add_constants(node_iterating, graph)

        if self.name == 'Addition' and len(self.input_indexes):
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
                self.__dict__[weight_tensor.name] = onnx.numpy_helper.to_array(weight_tensor)
                #elif node_iterating.op_type in ['Gemm', 'MatMul']:
                #    temp = numpy_helper.to_array(weight)
                #    if 'MatMul' in node_iterating.op_type:
                #        temp = temp.T
                #    try:
                #        temp = temp.reshape(new_parameters['ch_out'], ch_in_temp, self.DORY_Graph[-1].get_parameter('output_dim')[0], self.DORY_Graph[-1].get_parameter('output_dim')[1])
                #    except:
                #        temp = temp.reshape(new_parameters['ch_out'], ch_in_temp, self.DORY_Graph[-2].get_parameter('output_dim')[0], self.DORY_Graph[-2].get_parameter('output_dim')[1])
                #    temp = np.transpose(temp, (0, 2, 3, 1))
                #    temp = temp.flatten()
                #    new_parameters['weights'] = temp
                #    # needed to compute final checksum for <8b layers
                #    new_parameters['weights_raw'] = temp
        for weight_tensor in graph.graph.node:
            if weight_tensor.op_type == 'Constant' and weight_tensor.output[0] in self.input_indexes:
                self.constant = onnx.numpy_helper.to_array(weight_tensor.attribute[0].t)
                self.input_indexes.remove(weight_tensor.output[0])
                self.number_of_input_nodes -= 1
                self.number_of_input_constants += 1


    def add_special_attributes(self, node_iterating):
        '''
        adding not expected and custom attributes (e.g., min and max in clip)
        '''
        for attribute in node_iterating.attribute:
            if attribute.name not in ['kernel_shape', 'dilations', 'group', 'strides', 'pads']:
                if bool(attribute.i):
                    self.__dict__[attribute.name] = int(attribute.i)
                elif bool(attribute.f):
                    self.__dict__[attribute.name] = int(attribute.f)
                elif bool(attribute.ints):
                    self.__dict__[attribute.name] = list(attribute.ints)
                elif attribute.name == 'min':
                    self.__dict__[attribute.name] = 0
                elif bool(attribute.t):
                    self.__dict__[attribute.name] = 'tensor'
                else:
                    sys.exit("DORY FRONTEND error. DORY does not find any values for the attribute {}".format(attribute.name))

    def export_to_dict(self):
        layer_dict = {}
        for key, value in self.__dict__.items():
            if not isinstance(value, np.ndarray):
                layer_dict[key] = value
            else:
                layer_dict[key] = 'Present'
        return layer_dict


