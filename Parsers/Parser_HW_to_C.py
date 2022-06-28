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
import os
import sys
import numpy as np

# DORY modules
import Utils.Templates_writer.Network_template_writer as Network_writer
import Utils.Templates_writer.Makefile_template_writer as Makefile_writer
from Utils.DORY_utils import loadtxt


class Parser_HW_to_C:
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    def __init__(self, graph, network_directory, HW_description, verbose_level, perf_layer, save_string, app_directory):
        self.HWgraph = graph
        self.HW_description = HW_description
        self.verbose_level = verbose_level
        self.perf_layer = perf_layer
        self.save_string_for_Makefile = save_string
        self.network_directory = network_directory
        self.app_directory = app_directory

    def adding_numbers_to_layers(self):
        for i, node in enumerate(self.HWgraph):
            node.name = node.name + str(i)

    def mapping_network_to_C_file(self):
        print("\nGenerating the .c file of the network.")
        Network_writer.print_template_network(
            self.HWgraph,
            self.HW_description,
            self.config_file,
            self.verbose_level,
            self.perf_layer,
            self.app_directory)

    def mapping_makefile(self):
        print("\nGenerating the Makefile.")
        Makefile_writer.print_template_Makefile(
            self.HWgraph,
            self.HW_description,
            self.save_string_for_Makefile,
            self.app_directory)

    def mapping_layers_to_C_files(self):
        print("\nTo be implemented in the target backend")

    def copy_backend_files(self, node):
        print("\nTo be implemented in the target backend")

    def copy_utils_files(self):
        print("\nCopying Utils.")
        utils_files_dir = os.path.join(os.path.dirname(__file__), '../Hardware-targets', self.HW_description["name"], 'Utils_files')
        for file in os.listdir(utils_files_dir):
            file_to_copy = os.path.join(utils_files_dir, file)
            if file_to_copy[-1] == 'c':
                os.system('cp "{}" {}/DORY_network/src'.format(file_to_copy, self.app_directory))
            elif file_to_copy[-1] == 'h': 
                os.system('cp "{}" {}/DORY_network/inc'.format(file_to_copy, self.app_directory))

    def create_hex_weights_files(self):
        print("\nGenerating .hex weight files.")

        for node in self.HWgraph:
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

            weights = bytearray()
            for const in constants:
                if const != 0:
                    weights += getattr(node, const)['value'].tobytes()

            if len(weights) == 0:
                continue

            if len(weights) % 4 != 0:
                weights += bytearray([0] * (4 - len(weights) % 4))

            filepath = os.path.join(self.app_directory, 'DORY_network', node.name + "_weights.hex")
            with open(filepath, 'wb') as file:
                file.write(weights)

    def create_hex_input(self):    
        print("\nGenerating .hex input file.")

        input_txt = os.path.join(self.network_directory, 'input.txt')
        try:
            x = loadtxt(input_txt, dtype=np.uint8)
        except FileNotFoundError:
            print(f"File input.txt doesn't exist. Exiting DORY...")
            sys.exit(-1)

        input_hex = os.path.join(self.app_directory, 'DORY_network', 'inputs.hex')
        x.tofile(input_hex)

    def full_graph_parsing(self):
        print("#####################################################")
        print("## DORY GENERAL PARSING FROM DORY HW IR TO C FILES ##")
        print("## FINAL RAPRESENTATION: COMPILABLE C PROJECT      ##")
        print("#####################################################")
        os.system('rm -rf {}'.format(self.app_directory))
        os.system('mkdir {}'.format(self.app_directory))
        os.system('mkdir {}/DORY_network'.format(self.app_directory))
        os.system('mkdir {}/DORY_network/inc'.format(self.app_directory))
        os.system('mkdir {}/DORY_network/src'.format(self.app_directory))
        self.adding_numbers_to_layers()
        self.mapping_network_to_C_file()
        self.mapping_makefile()
        self.mapping_layers_to_C_files()
        self.copy_utils_files()
        self.create_hex_weights_files()
        self.create_hex_input()

