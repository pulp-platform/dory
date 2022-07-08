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
import json
import os
import numpy as np
from mako.template import Template
from collections import OrderedDict
import copy 

# DORY modules
from Parsers.Parser_HW_to_C import Parser_HW_to_C
import Utils.Templates_writer.Layer2D_template_writer as Layer2D_writer
import Utils.Templates_writer.writer_utils as utils

# Directory
file_path = "/".join(os.path.realpath(__file__).split("/")[:-1])


class C_Parser(Parser_HW_to_C):
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    def __init__(self, graph, config_file, config_file_dir, verbose_level, perf_layer, precision_library, appdir):
        f = open(os.path.join(file_path, "HW_description.json"))
        HW_description = json.load(f)
        self.precision_library = precision_library
        self.source_Constant_bits_library = config_file["BNRelu_bits"]
        self.config_file = config_file
        super().__init__(graph, os.path.join(config_file_dir, os.path.dirname(config_file["onnx_file"])), HW_description, verbose_level, perf_layer, "Makefile", appdir)

    def copy_backend_files(self, node):
        root = os.path.dirname(__file__)
        files = os.path.join(root, "Backend_Kernels/dory-hal/")
        if os.listdir(os.path.join(files, "include".format(self.source_Constant_bits_library)))[0] not in os.listdir(os.path.join(self.appdir, "DORY_network/inc")):
            for file in os.listdir(os.path.join(files, "include".format(self.source_Constant_bits_library))):
                file_to_copy = os.path.join(files, "include".format(self.source_Constant_bits_library), file)
                os.system('cp "{}" {}'.format(file_to_copy, os.path.join(self.appdir, 'DORY_network/inc')))
        if os.listdir(os.path.join(files, "src".format(self.source_Constant_bits_library)))[0] not in os.listdir(os.path.join(self.appdir, "DORY_network/src")):
            for file in os.listdir(os.path.join(files, "src".format(self.source_Constant_bits_library))):
                file_to_copy = os.path.join(files, "src".format(self.source_Constant_bits_library), file)
                os.system('cp "{}" {}'.format(file_to_copy, os.path.join(self.appdir, 'DORY_network/src')))
        

    def mapping_layers_to_C_files(self):
        print("\nMapping the layers files to their templates and copying the kernels associated.")
        tmpl_dir = os.path.join(os.path.dirname(__file__), 'Templates/layer_templates')
        out_dir = '{}/DORY_network'.format(self.appdir)
        for i, node in enumerate(self.graph):
            self.copy_backend_files(node)
            Layer2D_writer.print_template_layer(node, self.precision_library, tmpl_dir, out_dir)

    def create_hex_weights_files(self):
        print("\nGenerating .h weight files.")
        weights_vectors = []
        weights_dimensions = []
        for i, node in enumerate(self.graph):
            constants = [0, 0, 0, 0]
            for name in node.constant_names:
                if "weight" in name:
                    constants[0] = name
                elif "bias" in name:
                    constants[1] = name
                elif "k" == name:
                    constants[2] = name
                elif "l" == name:
                    constants[3] = name
            weights = np.asarray([])
            save_vector = 0
            for i in np.arange(4):
                if constants[i]!= 0:
                    if i==0:
                        temp = np.asarray(node.__dict__[constants[i]]["value"])
                        temp = temp.reshape(int(temp.shape[0]/4), 4)
                        temp1 = copy.deepcopy(temp)
                        temp[:,0] = temp1[:,3] 
                        temp[:,1] = temp1[:,2] 
                        temp[:,2] = temp1[:,1] 
                        temp[:,3] = temp1[:,0]
                        node.__dict__[constants[i]]["value"] = temp.flatten()
                    if i==1:
                        new_weights = []
                        for pos in range(4):
                            for ch in range(int(np.asarray(node.__dict__[constants[i]]["value"]).shape[0]/16)):
                                for pos_in in [3,2,1,0]:
                                    new_weights.append(node.__dict__[constants[i]]["value"][pos+4*(ch*4+pos_in)])
                        node.__dict__[constants[i]]["value"] = new_weights
                    weights = np.concatenate((weights,node.__dict__[constants[i]]["value"]))
                    save_vector = 1
            while len(weights) % 4 != 0:
                weights = np.concatenate((weights, np.asarray([0])))
            if save_vector == 1:
                weights_vectors.append(utils.print_test_vector(weights, 'char'))
                weights_dimensions.append(weights.shape[0])
            else:
                weights_vectors.append(['0'])
                weights_dimensions.append(0)
        tk = OrderedDict([])
        tk['weights_vectors'] = weights_vectors
        tk['weights_dimensions'] = weights_dimensions
        tk['DORY_HW_graph'] = self.graph
        root = os.path.dirname(__file__)
        tmpl = Template(filename=os.path.join(root, "Templates/weights_h_template.h"))
        s = tmpl.render(**tk)
        save_string = os.path.join(self.appdir, 'DORY_network/inc/weights.h')
        with open(save_string, "w") as f:
            f.write(s)
        tmpl = Template(filename=os.path.join(root, "Templates/weights_definition_h_template.h"))
        s = tmpl.render(**tk)
        save_string = os.path.join(self.appdir, 'DORY_network/inc/weights_definition.h')
        with open(save_string, "w") as f:
            f.write(s)

    def create_hex_input(self):    
        print("\nGenerating .h input file.")
        try:
            x_in = np.loadtxt(os.path.join(self.netdir, 'input.txt'), delimiter=',', dtype=np.uint8, usecols=[0])
            x_in_w = int(x_in.shape[0] / (self.graph[0].input_channels * self.graph[0].input_dimensions[0]))
            x_in = x_in.reshape(self.graph[0].input_channels, self.graph[0].input_dimensions[0], x_in_w)
            npad = ((0, 0), (0,0), (0, (16 - (x_in_w % 16)) % 16))
            temp = np.pad(x_in, pad_width=npad, mode='constant', constant_values=0)
            x_in = temp.flatten()
        except FileNotFoundError:
            print(f"========= WARNING ==========\nInput file {os.path.join(self.netdir, 'input.txt')} not found; generating random inputs!")
            x_in = np.random.randint(low=0, high=2*8 - 1,
                                     size=self.group * self.input_channels * self.input_dimensions[0] * self.input_dimensions[1],
                                     dtype=np.uint8)
        temp = x_in.reshape(int(x_in.shape[0]/4), 4)
        temp1 = copy.deepcopy(temp)
        temp[:,0] = temp1[:,3] 
        temp[:,1] = temp1[:,2] 
        temp[:,2] = temp1[:,1] 
        temp[:,3] = temp1[:,0] 
        input_values = utils.print_test_vector(temp.flatten(), 'char')
        tk = OrderedDict([])
        tk['input_values'] = input_values
        tk['dimension'] = len(x_in)
        root = os.path.dirname(__file__)
        tmpl = Template(filename=os.path.join(root, "Templates/input_h_template.h"))
        s = tmpl.render(**tk)
        save_string = os.path.join(self.appdir, 'DORY_network/inc/input.h')
        with open(save_string, "w") as f:
            f.write(s)