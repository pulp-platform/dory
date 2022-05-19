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

from mako.template import Template
from collections import OrderedDict
import os
from . import writer_utils as utils


def print_template_network(
    graph,
    HW_description,
    verbose_level,
    perf_layer
):
    # Generate the Network management c file.
    tk = OrderedDict([])
    if 'Check' in verbose_level:
        tk['verbose'] = True
    else:
        tk['verbose'] = False
    i_conv = []
    i = 0
    for ind, nodes in enumerate(graph[:-1]):
        if ('FullyConnected' in graph[ind + 1].name or 'Conv' in graph[ind + 1].name):
            i += 1
            i_conv.append(i)
        else:
            i_conv.append(i)
    weights_number = 0
    for nodes in graph:
        if 'FullyConnected' in nodes.name or 'Conv' in nodes.name:
            weights_number += 1
    tk['weights_number'] = weights_number
    tk['i_conv'] = i_conv
    tk['verbose_level'] = verbose_level
    tk['performance'] = perf_layer
    tk['l1_buffer'] = HW_description["memory"]["L1"]["dimension"] - HW_description["HW specific parameters"]["accelerator core0 stack"] - 7 * HW_description["HW specific parameters"]["accelerator core1-7 stack"]
    tk['master_stack'] = HW_description["HW specific parameters"]["accelerator core0 stack"] 
    tk['slave_stack'] = HW_description["HW specific parameters"]["accelerator core1-7 stack"]
    tk['l2_buffer_size'] = HW_description["memory"]["L2"]["dimension"] - HW_description["HW specific parameters"]["code reserved space"] 
    MACs = 0
    file_list_w = []
    list_h = []
    list_name = []
    for i, node in enumerate(graph):
        MACs += node.MACs
        if "Conv" in node.name or "FullyConnected" in node.name:
            file_list_w.append(node.name+"_weights.hex")
        list_h.append(node.name+".h")
        list_name.append(node.name)
    tk['MACs'] = MACs
    tk['files_list'] = utils.print_file_list(file_list_w)
    tk['fc_frequency'] = HW_description["core frequency"]
    tk['cl_frequency'] = HW_description["accelerator frequency"]
    tk['sdk'] = HW_description["software development kit"]["name"]
    list_h = list(set(list_h))
    tk['list_h'] = list_h
    tk['func_name'] = list_name
    l = ""
    for k, v in tk.items():
        try:
            l += "// %s %d\n" % (k.ljust(30), v)
        except TypeError:
            try:
                l += "// %s %d\n" % (k.ljust(30), v[0])
            except TypeError:
                l += "// %s %s\n" % (k.ljust(30), v)
    tk['DORY_HW_graph'] = graph
    root = os.path.dirname(__file__)
    tmpl = Template(filename=os.path.join(root, "../../_03_Hardware-targets", HW_description["name"], "Templates/network_c_template.c"))
    s = tmpl.render(verbose_log=l, **tk)
    save_string = './application/DORY_network/src/network.c'
    with open(save_string, "w") as f:
        f.write(s)

    tmpl = Template(filename=os.path.join(root, "../../_03_Hardware-targets", HW_description["name"], "Templates/network_h_template.h"))
    s = tmpl.render(verbose_log=l, **tk)
    save_string = './application/DORY_network/inc/network.h'
    with open(save_string, "w") as f:
        f.write(s)

    tmpl = Template(filename=os.path.join(root, "../../_03_Hardware-targets", HW_description["name"], "Templates/main_template.c"))
    s = tmpl.render(verbose_log=l, **tk)
    save_string = './application/DORY_network/src/main.c'
    with open(save_string, "w") as f:
        f.write(s)