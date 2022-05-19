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
import os
import torch
import pandas as pd

# DORY modules
import _01_Utils.Templates_writer.Network_template_writer as Network_writer
import _01_Utils.Templates_writer.Makefile_template_writer as Makefile_writer


class Parser_HW_to_C:
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    def __init__(self, graph, network_directory, HW_description, verbose_level, perf_layer, save_string):
        self.HWgraph = graph
        self.HW_description = HW_description
        self.verbose_level = verbose_level
        self.perf_layer = perf_layer
        self.save_string = save_string
        self.network_directory = network_directory

    def adding_numbers_to_layers(self):
        for i, node in enumerate(self.HWgraph):
            node.name = node.name + str(i)

    def mapping_network_to_C_file(self):
        print("\nGenerating the .c file of the network.")
        Network_writer.print_template_network(
            self.HWgraph,
            self.HW_description,
            self.verbose_level,
            self.perf_layer)

    def mapping_makefile(self):
        print("\nGenerating the Makefile.")
        Makefile_writer.print_template_Makefile(
            self.HWgraph,
            self.HW_description,
            self.save_string)

    def mapping_layers_to_C_files(self):
        print("\nTo be implemented in the target backend")

    def copy_backend_files(self, node):
        print("\nTo be implemented in the target backend")

    def copy_utils_files(self):
        print("\nCopying Utils.")
        utils_files_dir = os.path.join(os.path.dirname(__file__), '../_03_Hardware-targets', self.HW_description["name"], 'Utils_files')
        for file in os.listdir(utils_files_dir):
            file_to_copy = os.path.join(utils_files_dir, file)
            if file_to_copy[-1] == 'c':
                os.system('cp "{}" application/DORY_network/src'.format(file_to_copy))
            elif file_to_copy[-1] == 'h': 
                os.system('cp "{}" application/DORY_network/inc'.format(file_to_copy))

    def create_hex_weights_files(self):
        print("\nGenerating .hex weight files.")
        for i, node in enumerate(self.HWgraph):
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
            for i in np.arange(4):
                if constants[i]!= 0:
                    weights = np.concatenate((weights,node.__dict__[constants[i]]["value"]))
            while len(weights) % 4 != 0:
                weights = np.concatenate((weights, np.asarray([0])))
            if weights.shape[0] != 0:
                string_layer = node.name + "_weights.hex"
                save_s = './application/DORY_network/' + string_layer
                with open(save_s, 'wb') as f:
                    for l in weights.astype('uint8').flatten():
                        f.write(bytes((l,)))

    def create_hex_input(self):    
        print("\nGenerating .hex input file.")  
        try:
            x_in = pd.read_csv(os.path.join(self.network_directory, 'input.txt'))
            x_in = x_in.values[:, 0].astype(int)
        except FileNotFoundError:
            print(f"========= WARNING ==========\nInput file {os.path.join(self.network_directory, 'input.txt')} not found; generating random inputs!")
            x_in = torch.Tensor(1, self.HWgraph[0].group, self.HWgraph[0].input_channels, self.HWgraph[0].input_dimensions[0], self.HWgraph[0].input_dimensions[1]).uniform_(0, (2**(9)))
            x_in[x_in > (2**8 - 1)] = 0
            x_in = torch.round(x_in)
            x_in = x_in.flatten().numpy().astype(int)
        for i, _ in enumerate(x_in):
            x_in[i] = np.uint8(x_in[i])

        string_layer = "inputs.hex"
        save_s = './application/DORY_network/' + string_layer
        with open(save_s, 'wb') as f:
            for i in x_in.astype('uint8').flatten():
                f.write(bytes((i,)))

    def full_graph_parsing(self):
        print("#####################################################")
        print("## DORY GENERAL PARSING FROM DORY HW IR TO C FILES ##")
        print("## FINAL RAPRESENTATION: COMPILABLE C PROJECT      ##")
        print("#####################################################")
        os.system('rm -rf application')
        os.system('mkdir application')
        os.system('mkdir application/DORY_network')
        os.system('mkdir application/DORY_network/inc')
        os.system('mkdir application/DORY_network/src')
        self.adding_numbers_to_layers()
        self.mapping_network_to_C_file()
        self.mapping_makefile()
        self.mapping_layers_to_C_files()
        self.copy_utils_files()
        self.create_hex_weights_files()
        self.create_hex_input()

