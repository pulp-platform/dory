# C_Parser.py
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

import os
import numpy as np
from collections import OrderedDict
from mako.template import Template

from dory.Parsers.HW_node import HW_node
from dory.Hardware_targets.PULP.Common import C_Parser_PULP
import dory.Utils.Templates_writer.writer_utils as utils

class C_Parser(C_Parser_PULP):

    def get_file_path(self):
        return "/".join(os.path.realpath(__file__).split("/")[:-1])

    # weights and input files are headers instead of .hex
    def create_hex_weights_files(self):
        print("\nGenerating .h weight files.")
        weights_vectors = []
        weights_dimensions = []
        prefix = self.HWgraph[0].prefix
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
            ww, ww_dim = utils.print_test_vector(weights, 'char'), weights.shape[0]
            weights_vectors.append(ww)
            weights_dimensions.append(ww_dim)
        tk = OrderedDict([])
        tk['weights_vectors'] = weights_vectors
        tk['weights_dimensions'] = weights_dimensions
        tk['DORY_HW_graph'] = self.HWgraph
        tk['sdk'] = node.HW_description["software development kit"]["name"]
        tk['prefix'] = prefix
        root = os.path.dirname(__file__)
        tmpl = Template(filename=os.path.join(root, "Templates/weights_h_template.h"))
        s = tmpl.render(**tk)
        save_string = os.path.join(self.inc_dir, prefix+'weights.h')
        with open(save_string, "w") as f:
            f.write(s)
        tmpl = Template(filename=os.path.join(root, "Templates/weights_definition_h_template.h"))
        s = tmpl.render(**tk)
        save_string = os.path.join(self.inc_dir, prefix+'weights_definition.h') 
        with open(save_string, "w") as f:
            f.write(s)

    def create_hex_input(self):
        print("\nGenerating .h input file.")
        prefix = self.HWgraph[0].prefix
        x_in_l = []
        for in_idx in range(self.n_inputs):
            infile = 'input.txt' if self.n_inputs == 1 else f'input_{in_idx}.txt'
            try:
                x_in = np.loadtxt(os.path.join(self.network_directory, infile), delimiter=',', dtype=np.uint8, usecols=[0])
            except FileNotFoundError:
                print(f"========= WARNING ==========\nInput file {os.path.join(self.network_directory, 'input.txt')} not found; generating random inputs!")
                x_in = np.random.randint(low=0, high=2*8,
                                         size=self.group * self.input_channels * self.input_dimensions[0] * self.input_dimensions[1],
                                         dtype=np.uint8)
            x_in_l.append(x_in.flatten())

        x_in = np.concatenate(x_in_l)
        in_node = self.HWgraph[0]
        in_bits = in_node.input_activation_bits
        if in_bits != 8:
            x_in = HW_node._compress(x_in, in_bits)


        temp = x_in
        input_values = utils.print_test_vector(temp.flatten(), 'char')
        tk = OrderedDict([])
        tk['input_values'] = input_values
        tk['dimension'] = len(x_in)
        tk['sdk'] = self.HW_description["software development kit"]["name"]
        tk['prefix'] = prefix
        root = os.path.dirname(__file__)
        tmpl = Template(filename=os.path.join(root, "Templates/input_h_template.h"))
        s = tmpl.render(**tk)
        save_string = os.path.join(self.inc_dir, prefix+'input.h') 
        with open(save_string, "w") as f:
            f.write(s)
