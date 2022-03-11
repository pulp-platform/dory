#
# template.py
# Alessio Burrello <alessio.burrello@unibo.it>
# Thorir Mar Ingolfsson <thoriri@iis.ee.ethz.ch>
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

import math
from mako.template import Template
import re
from collections import OrderedDict
import numpy as np
import sys
import os
import re
import writer_utils as utils

def print_template_network(
    file_list_w,
    PULP_Nodes_Graph,
    type_data,
    name,
    test = True,
    has_bias = True,
    verbose_level = 'None',
    performance_single_layer = 'Yes',
    check_layer = 0,
    act_compare = 0,
    act_size = [0, 0, 0],
    class_out = 0,
    l1_buffer = 35000,
    master_stack = 4096,
    slave_stack = 3072,
    l2_buffer_size = 400000,
    fc_frequency = 100000000,
    cl_frequency = 100000000,
    MACs = 1,
    platform = 'GAP8',
    sdk = 'gap_sdk',
    backend = 'MCU',
    dma_parallelization = '8-cores'
):
    # Generate the Network management c file.
    tk = OrderedDict([])
    if 'Check' in verbose_level:
        tk['verbose'] = True
    else:
        tk['verbose'] = False
    i_conv = []
    i = 0
    for ind, nodes in enumerate(PULP_Nodes_Graph[:-1]):
        if ('Gemm' in PULP_Nodes_Graph[ind + 1].name or 'Conv' in PULP_Nodes_Graph[ind + 1].name):
            i += 1
            i_conv.append(i)
        else:
            i_conv.append(i)
    weights_number = 0
    for nodes in PULP_Nodes_Graph:
        if 'Gemm' in nodes.name or 'Conv' in nodes.name or 'MatMul' in nodes.name:
            weights_number += 1
    tk['dma_parallelization'] = dma_parallelization
    tk['weights_number'] = weights_number
    tk['i_conv'] = i_conv
    tk['verbose_level'] = verbose_level
    tk['performance'] = performance_single_layer
    tk['l1_buffer'] = l1_buffer
    tk['master_stack'] = master_stack
    tk['slave_stack'] = slave_stack
    tk['l2_buffer_size'] = l2_buffer_size
    tk['MACs'] = MACs
    tk['files_list'] = utils.print_file_list(file_list_w)
    tk['test'] = test
    tk['check_layer'] = check_layer
    tk['act_size'] = act_size
    tk['nof_check'] = act_size[2]
    tk['h_out_check'] = act_size[0]
    tk['w_out_check'] = act_size[1]
    tk['class_out'] = class_out
    tk['platform'] = platform
    tk['fc_frequency'] = fc_frequency
    tk['cl_frequency'] = cl_frequency
    tk['sdk'] = sdk
    tk['act_compare'] = utils.print_test_vector(act_compare, 'char')
    list_h = []
    for i, _ in enumerate(name):
        list_h.append(name[i] + '.h')
    list_h = list(set(list_h))
    tk['list_h'] = list_h
    list_name = []
    for i, _ in enumerate(name):
        list_name.append(name[i])
    tk['func_name'] = list_name
    tk['has_bias'] = has_bias
    tk['type'] = type_data
    l = ""
    for k, v in tk.items():
        try:
            l += "// %s %d\n" % (k.ljust(30), v)
        except TypeError:
            try:
                l += "// %s %d\n" % (k.ljust(30), v[0])
            except TypeError:
                l += "// %s %s\n" % (k.ljust(30), v)
    root = '/'.join(os.getcwd().split('/')[:-1])
    tmpl = Template(filename=root + f"/Templates/{backend}/network_template.c")
    tk['PULP_Nodes_Graph'] = PULP_Nodes_Graph
    s = tmpl.render(verbose_log=l,**tk)
    save_string = './application/DORY_network/src/network.c'
    with open(save_string, "w") as f:
        f.write(s)
