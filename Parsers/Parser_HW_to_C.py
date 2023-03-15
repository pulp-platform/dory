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
import shutil
import numpy as np

# DORY modules
from Utils.Templates_writer.MiscTemplateWriter import MiscTemplateWriter
from Utils.DORY_utils import loadtxt


class Parser_HW_to_C:
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    def __init__(self, graph, conf, netdir, hw_desc, verbose_level, perf_layer, save_string, appdir):
        self.graph = graph
        self.conf = conf
        self.hw_desc = hw_desc
        self.verbose_level = verbose_level
        self.perf_layer = perf_layer
        self.save_string_for_Makefile = save_string
        self.netdir = netdir
        self.backendfiles = []
        self.backenddirs = []
        self.rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        self.targetdir = os.path.join(self.rootdir, 'Hardware-targets', self.hw_desc["name"])
        self.appdir = appdir
        self.srcdir = os.path.join(self.appdir, 'src')
        self.incdir = os.path.join(self.appdir, 'inc')
        self.hexdir = os.path.join(self.appdir, 'hex')
        shutil.rmtree(self.appdir, ignore_errors=True)
        os.makedirs(self.incdir)
        os.makedirs(self.srcdir)
        os.makedirs(self.hexdir)

    def adding_numbers_to_layers(self):
        for i, node in enumerate(self.graph):
            node.name = node.name + str(i)

    def map_misc(self):
        print("\nGenerating miscellaneous files.")
        templateWriter = MiscTemplateWriter(self.graph, self.hw_desc, self.conf, self.verbose_level, self.perf_layer)
        tmplfiles = ['tmpl_network.h', 'tmpl_network.c', 'tmpl_main.c', 'tmpl_Makefile']
        dest = [self.destdir(file) for file in tmplfiles]
        templateWriter.write(tmplfiles, dest)

    def map_layers(self):
        print("\nTo be implemented in the target backend")

    def destdir(self, file):
        _, ext = os.path.splitext(file)
        if ext == '.c':
            return self.srcdir
        elif ext == '.h':
            return self.incdir
        elif ext == '.hex':
            return self.hexdir
        elif ext == '':
            return self.appdir
        else:
            print(f"WARNING: Unsupported extension of file {file}")
            return None

    def copy_files(self, files):
        for file in files:
            destdir = self.destdir(file)
            if destdir is not None:
                shutil.copy(file, self.destdir(file))
            else:
                print(f"NOTE: Skipping copying of file {file}")

    def copy_backend(self):
        print("\nCopying Backend Kernels.")
        files = self.backendfiles
        for backenddir in self.backenddirs:
            for root, _, filenames in os.walk(backenddir):
                files += [os.path.join(root, filename) for filename in filenames]
        self.copy_files(files)

    def copy_utils(self):
        print("\nCopying Utils.")
        utils_dir = os.path.join(self.targetdir, 'Utils_files')
        utils_files = [os.path.join(utils_dir, file) for file in os.listdir(utils_dir)]
        self.copy_files(utils_files)

    def create_hex_weights(self):
        print("\nGenerating .hex weight files.")

        for node in self.graph:
            # Hack DepthwisePointwise
            if "DepthwisePointwise" in node.name:
                constants = [ ["weights0", "k0", "l0"], ["weights1", "k1", "l1"] ]
            else:
                constants = [ [0, 0, 0, 0] ]
                for name in node.constant_names:
                    if "weight" in name:
                        constants[0][0] = name
                    elif "bias" in name:
                        constants[0][1] = name
                    elif "k" == name:
                        constants[0][2] = name
                    elif "l" == name:
                        constants[0][3] = name

            weights = bytearray()
            for const_group in constants:
                for const in const_group:
                    if const != 0:
                        weights += getattr(node, const)['value'].tobytes()

            if len(weights) == 0:
                continue

            if len(weights) % 4 != 0:
                weights += bytearray([0] * (4 - len(weights) % 4))

            node.hex_weights_size = len(weights)

            filepath = os.path.join(self.hexdir, node.name + "_weights.hex")
            with open(filepath, 'wb') as file:
                file.write(weights)

    def create_hex_input(self):    
        print("\nGenerating .hex input file.")

        input_txt = os.path.join(self.netdir, 'input.txt')
        try:
            x = loadtxt(input_txt, dtype=np.uint8)
        except FileNotFoundError:
            print(f"File input.txt doesn't exist. Exiting DORY...")
            sys.exit(-1)

        input_hex = os.path.join(self.hexdir, 'inputs.hex')
        x.tofile(input_hex)

    def full_graph_parsing(self):
        print("#####################################################")
        print("## DORY GENERAL PARSING FROM DORY HW IR TO C FILES ##")
        print("## FINAL RAPRESENTATION: COMPILABLE C PROJECT      ##")
        print("#####################################################")
        self.adding_numbers_to_layers()
        self.create_hex_weights()
        self.create_hex_input()
        self.map_misc()
        self.map_layers()
        self.copy_backend()
        self.copy_utils()

