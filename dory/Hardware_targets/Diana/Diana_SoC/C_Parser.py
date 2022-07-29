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
from dory.Parsers.Parser_HW_to_C import Parser_HW_to_C
import dory.Utils.Templates_writer.Layer2D_template_writer as Layer2D_writer
import dory.Utils.Templates_writer.writer_utils as utils
import dory.Hardware_targets.Diana.Diana_SoC.weights_encoder_analog as ana_enc

# Directory
file_path = "/".join(os.path.realpath(__file__).split("/")[:-1])


class C_Parser(Parser_HW_to_C):
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    def __init__(self, graph, config_file, config_file_dir, verbose_level, perf_layer, precision_library, app_directory):
        f = open(os.path.join(file_path, "HW_description.json"))
        HW_description = json.load(f)
        self.precision_library = precision_library
        self.source_Constant_bits_library = config_file["BNRelu_bits"]
        self.config_file = config_file
        super().__init__(graph, os.path.join(config_file_dir, os.path.dirname(config_file["onnx_file"])), HW_description, verbose_level, perf_layer, "Makefile", app_directory)

    def copy_backend_files(self, node):
        root = os.path.dirname(__file__)
        files = os.path.join(root, "../Backend_Kernels/dory-hal/")
        if os.listdir(os.path.join(files, "include".format(self.source_Constant_bits_library)))[0] not in os.listdir(os.path.join(self.app_directory, "DORY_network/inc")):
            for file in os.listdir(os.path.join(files, "include".format(self.source_Constant_bits_library))):
                file_to_copy = os.path.join(files, "include".format(self.source_Constant_bits_library), file)
                os.system('cp "{}" {}'.format(file_to_copy, os.path.join(self.app_directory, 'DORY_network/inc')))
        if os.listdir(os.path.join(files, "src".format(self.source_Constant_bits_library)))[0] not in os.listdir(os.path.join(self.app_directory, "DORY_network/src")):
            for file in os.listdir(os.path.join(files, "src".format(self.source_Constant_bits_library))):
                file_to_copy = os.path.join(files, "src".format(self.source_Constant_bits_library), file)
                os.system('cp "{}" {}'.format(file_to_copy, os.path.join(self.app_directory, 'DORY_network/src')))

    def adding_numbers_to_layers(self):
        for i, node in enumerate(self.HWgraph):
            if node.weight_bits == 8:
                node.name = node.name + "Digital" + str(i)        
            elif node.weight_bits == 2:
                node.name = node.name + "Analog" + str(i)        
            else:
                node.name = node.name + str(i) 



    def mapping_layers_to_C_files(self):
        print("\nMapping the layers files to their templates and copying the kernels associated.")
        tmpl_dir = os.path.join(os.path.dirname(__file__), 'Templates/layer_templates')
        out_dir = '{}/DORY_network'.format(self.app_directory)
        precision_library = self.precision_library
        h_files = []; c_files = []
        for i, node in enumerate(self.HWgraph):
            if self.precision_library == 'auto':
                precision_library = '8bit'
                if "Addition" not in node.name:
                    if node.get_parameter('weight_bits') < 8:
                        precision_library = 'ternary'
            self.copy_backend_files(node)
            h_layer, c_layer = Layer2D_writer.print_template_layer(node, precision_library, tmpl_dir, out_dir)
            h_files.append(h_layer)
            c_files.append(c_layer)
        return h_files, c_files

    def create_hex_weights_files(self):
        print("\nGenerating .h weight files.")
        weights_vectors = []
        weights_dimensions = []
        for i, node in enumerate(self.HWgraph):
            if node.get_parameter('weight_bits') < 8:
                ww, ww_dim = self.create_analog_weights(node)
            else:
                ww, ww_dim = self.create_digital_weights(node)
            weights_vectors.append(ww)
            weights_dimensions.append(ww_dim)
        tk = OrderedDict([])
        tk['weights_vectors'] = weights_vectors
        tk['weights_dimensions'] = weights_dimensions
        tk['DORY_HW_graph'] = self.HWgraph
        root = os.path.dirname(__file__)
        tmpl = Template(filename=os.path.join(root, "Templates/weights_h_template.h"))
        s_h = tmpl.render(**tk)
        save_string = os.path.join(self.app_directory, 'DORY_network/inc/weights.h') 
        with open(save_string, "w") as f:
            f.write(s_h)
        tmpl = Template(filename=os.path.join(root, "Templates/weights_definition_h_template.h"))
        s_def = tmpl.render(**tk)
        save_string = os.path.join(self.app_directory, 'DORY_network/inc/weights_definition.h') 
        with open(save_string, "w") as f:
            f.write(s_def)
        return s_h, s_def

    def create_hex_input(self):    
        print("\nGenerating .h input file.")
        try:
            x_in = np.loadtxt(os.path.join(self.network_directory, 'input.txt'), delimiter=',', dtype=np.uint8, usecols=[0])
        except FileNotFoundError:
            print(f"========= WARNING ==========\nInput file {os.path.join(self.network_directory, 'input.txt')} not found; generating random inputs!")
            x_in = np.random.randint(low=0, high=2*8 - 1,
                                     size=self.group * self.input_channels * self.input_dimensions[0] * self.input_dimensions[1],
                                     dtype=np.uint8)
        if self.HWgraph[0].weight_bits == 2:
            x_in = x_in.flatten() 
            temp = x_in
        else:
            x_in_w = int(x_in.shape[0]/(self.HWgraph[0].input_channels*self.HWgraph[0].input_dimensions[0]))
            x_in = x_in.reshape(self.HWgraph[0].input_channels, self.HWgraph[0].input_dimensions[0], x_in_w)
            npad = ((0, 0), (0,0), (0, (16 - (x_in_w % 16)) % 16))
            temp = np.pad(x_in, pad_width=npad, mode='constant', constant_values=0)
            x_in = temp.flatten()
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
        save_string = os.path.join(self.app_directory, 'DORY_network/inc/input.h') 
        with open(save_string, "w") as f:
            f.write(s)

    def create_digital_weights(self, node):
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
        for batch in np.arange(0, int(np.floor((getattr(node, 'output_channels')+15)/16))):
            for i in [0, 1]:
                if constants[i]!= 0:
                    if i==0:  
                        dim = getattr(node, 'input_channels') * 16 * np.prod(getattr(node, 'kernel_shape'))
                        weights = np.concatenate((weights,node.__dict__[constants[i]]["value"][(batch*dim):((batch+1)*dim)]))
                    if i==1:  
                        weights = np.concatenate((weights,node.__dict__[constants[i]]["value"][(batch*16*4):((batch+1)*16*4)]))
                    save_vector = 1
        for i in [2, 3]:
            if constants[i]!= 0:
                weights = np.concatenate((weights,node.__dict__[constants[i]]["value"]))

        while len(weights) % 4 != 0:
            weights = np.concatenate((weights, np.asarray([0])))
        if save_vector == 1:
            return utils.print_test_vector(weights, 'char'), weights.shape[0]
        else:
            return ["0"], 0


    def _compress(self, x, bits):
        compressed = []
        n_elements_in_byte = 8 // bits
        i_element_in_byte = 0
        for el in x:
            if i_element_in_byte == 0:
                compressed.append(el)
            else:
                compressed[-1] += el << i_element_in_byte * bits

            i_element_in_byte += 1
            if i_element_in_byte == n_elements_in_byte:
                i_element_in_byte = 0
        return np.asarray(compressed, dtype=np.uint8)

    def create_analog_weights(self, node):
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
                    node.__dict__[constants[i]]["value"] = np.asarray(node.__dict__[constants[i]]["value"]).reshape(1,node.input_channels*np.prod(node.kernel_shape),node.output_channels)
                    w_list = ana_enc.pad(node.__dict__[constants[i]]["value"], True, True)
                    w_list = ana_enc.mirror_rows(w_list)
                    w_list = ana_enc.flip_weights(w_list, False)
                    w_list = ana_enc.map_weights(w_list)
                    w_list = ana_enc.flatten_list(w_list)
                    w_list_compressed = self._compress(w_list, 1)
                    node.__dict__[constants[i]]["value"] = w_list_compressed
                    weights = np.concatenate((weights,node.__dict__[constants[i]]["value"]))
                    save_vector = 1
        for i in [2, 3]:
            if constants[i]!= 0:
                weights = np.concatenate((weights,node.__dict__[constants[i]]["value"]))

        if save_vector == 1:
            return utils.print_test_vector(weights, 'char'), weights.shape[0]
        else:
            return ["0"], 0
