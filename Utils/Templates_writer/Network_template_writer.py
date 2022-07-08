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
    app_directory
):
    # Generate the Network management c file.

    tk = OrderedDict([])
    tk['verbose'] = 'Check' in verbose_level
    tk['weights_number'] = sum([1 for node in graph if node.has_weights()])
    tk['verbose_level'] = verbose_level
    tk['performance'] = perf_layer
    tk['l1_buffer'] = HW_description["memory"]["L1"]["dimension"] - HW_description["HW specific parameters"]["accelerator core0 stack"] - 7 * HW_description["HW specific parameters"]["accelerator core1-7 stack"]
    tk['master_stack'] = HW_description["HW specific parameters"]["accelerator core0 stack"] 
    tk['slave_stack'] = HW_description["HW specific parameters"]["accelerator core1-7 stack"]
    tk['l2_buffer_size'] = HW_description["memory"]["L2"]["dimension"] - config_file["code reserved space"] 
    tk['MACs'] = sum([node.MACs for node in graph])
    tk['files_list'] = utils.print_file_list([node.name + "_weights.hex" for node in graph if node.has_weights()])
    tk['fc_frequency'] = HW_description["core frequency"]
    tk['cl_frequency'] = HW_description["accelerator frequency"]
    tk['sdk'] = HW_description["software development kit"]["name"]
    tk['list_h'] = [node.name + '.h' for node in graph]
    tk['func_name'] = [node.name for node in graph]
    log = "".join([f'// {k:<30} {v}' for k, v in tk.items()])
    tk['DORY_HW_graph'] = graph

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    tmpl_dir = os.path.join(root, 'Hardware-targets', HW_description['name'], 'Templates')

    def write(tmpl_file):
        template = Template(filename=os.path.join(tmpl_dir, tmpl_file))
        rendered_template = template.render(verbose_log=log, **tk)
        _, ext = os.path.splitext(tmpl_file)
        out_filename = tmpl_file.split('_')[0]
        outfile = os.path.join(app_directory, 'DORY_network', 'src' if ext == '.c' else 'inc', out_filename + ext)
        with open(outfile, "w") as file:
            file.write(rendered_template)

    write('network_c_template.c')
    write('network_h_template.h')
    write('main_template.c')
