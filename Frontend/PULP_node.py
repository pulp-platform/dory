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

import logging

class node_element():
    # A node allocated in the PULP_Graph
    def __init__(self):
        self.name = 'Not-initialized'
        self.kernel_shape = 'Not-initialized' # fH x fW
        self.ch_in = 'Not-initialized' 
        self.ch_out = 'Not-initialized' 
        self.input_index = 'Not-initialized'
        self.output_index = 'Not-initialized'
        self.input_dim = 'Not-initialized' # H x W
        self.output_dim = 'Not-initialized' # H x W
        self.pads    = 'Not-initialized' # Top, Left, Bottom, Right
        self.branch_out = 0
        self.branch_in = 0
        self.branch_change = 0
        self.branch_last = 0
        self.input_activation_dimensions_L3 = 0
        self.output_activation_dimensions_L3 = 0
        self.inmul1 = 'empty'
        self.inmul2 = 'empty'
        self.weight_bits = 8
        self.out_activation_bits = 8
        self.input_activation_bits = 8
        self.outshift = 0

    def log_parameters(self):
        for parameter in self.__dict__:
            if parameter not in ['weights', 'k', 'lambda']:
                logging.debug(parameter + ': ' + str(self.__dict__[parameter]))
            else:
                logging.debug(parameter + ': Present')

    def print_parameters(self):
        for parameter in self.__dict__:
            if parameter not in ['weights', 'k', 'lambda']:
                print(parameter + ': ' + str(self.__dict__[parameter]))
            else:
                print(parameter + ': Present')

    def add_parameter(self, name, value):
        self.__dict__[name] = value

    def add_dict_parameter(self, dict_parameters):
        for key, value in dict_parameters.items():
            self.__dict__[key] = value

    def get_parameter(self, name):
        return self.__dict__[name]



