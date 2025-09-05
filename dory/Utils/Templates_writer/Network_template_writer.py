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
    config_file,
    verbose_level,
    perf_layer,
    tmpl_dir,
        app_directory,
        inc_dir_rel,
        src_dir_rel
):
    # Generate the Network management c file.
    tk = OrderedDict([])
    prefix = graph[0].prefix
    tk['prefix'] = prefix
    if 'Check' in verbose_level:
        tk['verbose'] = True
    else:
        tk['verbose'] = False
    weights_number = 0
    for nodes in graph:
        if 'FullyConnected' in nodes.name or 'Conv' in nodes.name:
            weights_number += 1
    tk['weights_number'] = weights_number
    tk['verbose_level'] = verbose_level
    tk['performance'] = perf_layer
    tk['l1_buffer'] = HW_description["memory"]["L1"]["dimension"] - HW_description["HW specific parameters"]["accelerator core0 stack"] - 7 * HW_description["HW specific parameters"]["accelerator core1-7 stack"]
    tk['master_stack'] = HW_description["HW specific parameters"]["accelerator core0 stack"] 
    tk['slave_stack'] = HW_description["HW specific parameters"]["accelerator core1-7 stack"]
    tk['l2_buffer_size'] = HW_description["memory"]["L2"]["dimension"] - config_file["code reserved space"] 
    MACs = 0
    file_list_w = []
    list_h = []
    list_name = []
    for i, node in enumerate(graph):
        MACs += node.MACs
        if "Conv" in node.name or "FullyConnected" in node.name:
            file_list_w.append(node.prefixed_name+"_weights.hex")
        list_h.append(node.prefixed_name+".h")
        list_name.append(node.prefixed_name)
    tk['MACs'] = MACs
    tk['files_list'] = utils.print_file_list(file_list_w)
    tk['fc_frequency'] = HW_description["core frequency"]
    tk['cl_frequency'] = HW_description["accelerator frequency"]
    if "peripheral frequency" in HW_description:
        tk['periph_frequency'] = HW_description["peripheral frequency"]
    else:
        tk['periph_frequency'] = None
    tk['sdk'] = HW_description["software development kit"]["name"]
    list_h = list(set(list_h))
    tk['list_h'] = list_h
    tk['func_name'] = list_name
    tk['n_inputs'] = graph[0].n_test_inputs
    l = ""
    for k, v in tk.items():
        try:
            l += "// %s %d\n" % (k.ljust(30), v)
        except TypeError:
            try:
                l += "// %s %d\n" % (k.ljust(30), v[0])
            except (TypeError, IndexError):
                l += "// %s %s\n" % (k.ljust(30), v)
    tk['DORY_HW_graph'] = graph
    root = os.path.realpath(os.path.dirname(__file__))
    tmpl = Template(filename=os.path.join(tmpl_dir, "network_c_template.c"))
    s = tmpl.render(verbose_log=l, **tk)
    save_string = os.path.join(app_directory, src_dir_rel, prefix + 'network.c')
    with open(save_string, "w") as f:
        f.write(s)

    tmpl = Template(filename=os.path.join(tmpl_dir, "network_h_template.h"))
    s = tmpl.render(verbose_log=l, **tk)
    save_string = os.path.join(app_directory, inc_dir_rel, prefix + 'network.h')
    with open(save_string, "w") as f:
        f.write(s)

    tmpl = Template(filename=os.path.join(tmpl_dir, "main_template.c"))
    s = tmpl.render(verbose_log=l, **tk)
    save_string = os.path.join(app_directory, src_dir_rel, prefix + 'main.c')
    with open(save_string, "w") as f:
        f.write(s)
