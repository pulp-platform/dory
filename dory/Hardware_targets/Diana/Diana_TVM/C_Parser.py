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
import dory.Hardware_targets.Diana.Diana_TVM.Layer2D_template_writer as Layer2D_writer
import dory.Utils.Templates_writer.writer_utils as utils
import dory.Hardware_targets.Diana.Diana_TVM.weights_encoder_analog as ana_enc

# Directory
file_path = "/".join(os.path.realpath(__file__).split("/")[:-1])


class C_Parser(Parser_HW_to_C):
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    def __init__(self, graph, config_file, config_file_dir, verbose_level, perf_layer, precision_library, app_directory, n_inputs=1):
        with open(os.path.join(file_path, "HW_description.json")) as f:
            HW_description = json.load(f)
        self.precision_library = precision_library
        self.source_Constant_bits_library = config_file["BNRelu_bits"]
        self.config_file = config_file
        super().__init__(graph, os.path.join(config_file_dir, os.path.dirname(config_file["onnx_file"])), HW_description, verbose_level, perf_layer, "Makefile", app_directory)

    def copy_backend_files(self, node):
        root = os.path.dirname(__file__)
        files = os.path.join(root, "../Backend_Kernels/dory-hal/")
        if os.listdir(os.path.join(files, "include".format(self.source_Constant_bits_library)))[0] not in os.listdir(os.path.join(self.app_directory, "inc")):
            for file in os.listdir(os.path.join(files, "include".format(self.source_Constant_bits_library))):
                file_to_copy = os.path.join(files, "include".format(self.source_Constant_bits_library), file)
                os.system('cp "{}" {}'.format(file_to_copy, os.path.join(self.app_directory, 'inc')))
        if os.listdir(os.path.join(files, "src".format(self.source_Constant_bits_library)))[0] not in os.listdir(os.path.join(self.app_directory, "src")):
            for file in os.listdir(os.path.join(files, "src".format(self.source_Constant_bits_library))):
                file_to_copy = os.path.join(files, "src".format(self.source_Constant_bits_library), file)
                os.system('cp "{}" {}'.format(file_to_copy, os.path.join(self.app_directory, 'src')))

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
        out_dir = '{}'.format(self.app_directory)
        precision_library = self.precision_library
        c_files = []
        for i, node in enumerate(self.HWgraph):
            if self.precision_library == 'auto':
                precision_library = '8bit'
                if "Add" not in node.op_type:
                    if node.get_parameter('weight_bits') < 8:
                        precision_library = 'ternary'
            node.skip_L2_L1 = False
            tk = self.create_hex_weights_files(node)
            c_layer = Layer2D_writer.print_template_layer(tk, node, precision_library, tmpl_dir, out_dir)
            c_files.append(c_layer)
        return c_files

    def create_hex_weights_files(self, node):
        print(f"\nGenerating weight string for {node.name}.")
        if node.get_parameter('weight_bits') < 8:
            ww, ww_dim = self.create_analog_weights(node)
        else:
            if 'Gemm' in node.op_type:
                ww, ww_dim = self.create_digital_FC_weights(node)
            else:
                ww, ww_dim = self.create_digital_weights(node)
        tk = OrderedDict([])
        tk['weights_vectors'] = ww
        tk['weights_dimensions'] = ww_dim
        return tk

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
                    if node.group == 1:
                        new_weights = []
                        for pos in range(4):
                            for ch in range(int(np.asarray(node.__dict__[constants[i]]["value"]).shape[0]/16)):
                                for pos_in in [3,2,1,0]:
                                    new_weights.append(node.__dict__[constants[i]]["value"][pos+4*(ch*4+pos_in)])
                        final_weights = []
                        for ch in range(int((node.output_channels+15)/16)):
                            for byte in range(4):
                                final_weights.append(new_weights[(16*byte + ch*16):(16*byte + ch*16 + 16)])
                    else:
                        new_weights = []
                        for pos in range(4):
                            for ch in range(int(np.asarray(node.__dict__[constants[i]]["value"]).shape[0]/16)):
                                for pos_in in [3,2,1,0]:
                                    new_weights.append(node.__dict__[constants[i]]["value"][pos+4*(ch*4+pos_in)])
                        final_weights = []
                        for ch in range(int((node.output_channels * 4 +15)/16)):
                            for byte in range(4):
                                final_weights.append(new_weights[(node.output_channels * 4 * byte + ch*16):(node.output_channels * 4 * byte + ch*16 + 16)])
                    node.__dict__[constants[i]]["value"] = np.asarray(final_weights).flatten().tolist()
        if node.group == 1:
            for batch in np.arange(0, int(np.floor((getattr(node, 'output_channels')+15)/16))):
                for i in [0, 1]:
                    if constants[i]!= 0:
                        if i==0:  
                            dim = (getattr(node, 'input_channels')+15)//16*16 * 16 * np.prod(getattr(node, 'kernel_shape'))
                            weights = np.concatenate((weights,node.__dict__[constants[i]]["value"][(batch*dim):((batch+1)*dim)]))
                        if i==1:  
                            weights = np.concatenate((weights,node.__dict__[constants[i]]["value"][(batch*16*int(node.bias_bits/8)):((batch+1)*16*int(node.bias_bits/8))]))
                        save_vector = 1
        else:
            for batch in np.arange(0, getattr(node, 'output_channels')):
                for i in [0, 1]:
                    if constants[i]!= 0:
                        if i==0:  
                            dim =  16 * np.prod(getattr(node, 'kernel_shape'))
                            weights = np.concatenate((weights,node.__dict__[constants[i]]["value"][(batch*dim):((batch+1)*dim)]))
                        if i==1:  
                            weights = np.concatenate((weights,node.__dict__[constants[i]]["value"][(batch*16*int(node.bias_bits/8)):((batch+1)*16*int(node.bias_bits/8))]))
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

    def create_digital_FC_weights(self, node):
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
                    temp = np.asarray(node.__dict__[constants[i]]["value"])
                    temp = temp.reshape(int(temp.shape[0]/4), 4)
                    temp1 = copy.deepcopy(temp)
                    temp[:,0] = temp1[:,3] 
                    temp[:,1] = temp1[:,2] 
                    temp[:,2] = temp1[:,1] 
                    temp[:,3] = temp1[:,0]
                    node.__dict__[constants[i]]["value"] = temp.flatten()
                    '''
                    Bias e' su 32bit:
                    Si impacchettano i 32bit in 8 valori hex, che noi chiameremo BH[0:7], con MSB alla posizione 0
                    I valori vengono poi riordinati come: BH[6] BH[7] BH[4] BH[5] BH[2] BH[3] BH[0] BH[1]
                    Il modo che sono scritti nell'header file segue BH[6] BH[7] BH[4] BH[5] BH[2] BH[3] BH[0] BH[1]
                    '''
                    pass
        for batch in np.arange(0, int(np.floor((getattr(node, 'output_channels')+15)/16))):
            for i in [0, 1]:
                if constants[i]!= 0:
                    if i==0:  
                        dim = (getattr(node, 'input_channels')+15)//16*16 * 16 * np.prod(getattr(node, 'kernel_shape'))
                        weights = np.concatenate((weights,node.__dict__[constants[i]]["value"][(batch*dim):((batch+1)*dim)]))
                    if i==1:  
                        weights = np.concatenate((weights,node.__dict__[constants[i]]["value"][(batch*16*int(node.bias_bits/8)):((batch+1)*16*int(node.bias_bits/8))]))
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

    def _compress_analog(self, x, bits):
        compressed = []
        n_elements_in_byte = 32 // bits
        i_element_in_byte = 0
        for el in x:
            if i_element_in_byte == 0:
                compressed.append(el << (n_elements_in_byte - i_element_in_byte - 1) * bits)
            else:
                compressed[-1] += el << (n_elements_in_byte - i_element_in_byte - 1) * bits

            i_element_in_byte += 1
            if i_element_in_byte == n_elements_in_byte:
                i_element_in_byte = 0
        return np.asarray(compressed, dtype=np.uint32)

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
                    w_list_compressed = self._compress_analog(w_list, 1)
                    node.__dict__[constants[i]]["value"] = w_list_compressed
                    weights = np.concatenate((weights,node.__dict__[constants[i]]["value"]))
                    save_vector = 1
        for i in [2, 3]:
            if constants[i]!= 0:
                weights = np.concatenate((weights,node.__dict__[constants[i]]["value"]))

        if save_vector == 1:
            return utils.print_test_vector(weights, 'uint32_t'), weights.shape[0]
        else:
            return ["0"], 0

    def full_graph_parsing(self):
        print("#####################################################")
        print("## DORY GENERAL PARSING FROM DORY HW IR TO C FILES ##")
        print("## FINAL RAPRESENTATION: COMPILABLE C PROJECT      ##")
        print("#####################################################")
        self.adding_numbers_to_layers()
        os.system('rm -rf {}'.format(self.app_directory))
        os.system('mkdir {}'.format(self.app_directory))
        os.system('mkdir {}/DORY_network'.format(self.app_directory))
        os.system('mkdir {}/DORY_network/inc'.format(self.app_directory))
        os.system('mkdir {}/DORY_network/src'.format(self.app_directory))
        layer_string = self.mapping_layers_to_C_files()
        self.copy_utils_files()
        return layer_string
