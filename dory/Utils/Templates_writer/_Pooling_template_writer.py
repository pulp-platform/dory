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

def print_template_layer_Pooling(node, layer_type, tmpl_dir, out_dir, double_buffering = 2):
    ks =      node.kernel_shape
    s =       node.strides
    p =       node.pads
    padding_top = p[0]; padding_left = p[1]; padding_bottom = p[2]; padding_right = p[3];
    name_layer = node.prefixed_name + '.h'
    conv_overlap1 = 2 * (ks[0] // 2) + ks[0] % 2 - 1 - (s[0] - 1)
    conv_overlap2 = 2 * (ks[1] // 2) + ks[1] % 2 - 1 - (s[1] - 1)
    tk = OrderedDict([])
    if (re.search('.0',name_layer)):
        try:
            int(re.search('.0',name_layer).group())
            tk['first_layer'] = 0
        except ValueError:
            tk['first_layer'] = 1
    else:
        tk['first_layer'] = 0
    tk['node'] = node
    tk['sdk'] = node.HW_description["software development kit"]["name"]
    tk['number_of_clusters'] = node.HW_description["HW specific parameters"]["clusters"] if "clusters" in node.HW_description["HW specific parameters"].keys() else 1
    tk['optional_type'] = layer_type
    tk['func_name'] = node.prefixed_name
    tk['flag_DW'] = 1 if node.group > 1 else 0
    tk['optional'] = node.op_type
    tk['FLAG_BATCHNORM'] = 1 if 'k' in node.constant_names else 0
    tk['has_bias'] = int(len([1 for name in node.constant_names if "bias" in name])>0)
    tk['FLAG_RELU'] = 1 if 'outshift' in node.constant_names else 0
    tk['type'] = f"{node.input_activation_type}8_t" if node.input_activation_type in ["int", "uint"] else "float"
    tk['conv_overlap1'] = conv_overlap1
    tk['conv_overlap2'] = conv_overlap2
    tk['padding_top'] = padding_top
    tk['padding_bottom'] = padding_bottom
    tk['padding_left'] = padding_left
    tk['padding_right'] = padding_right
    tk['stride'] = s[0]

    ################## NEED A REWRITING IN THIS TEMPLATE PART ######################
    #### VARIABLE CREATION FOR COMPATIBILITY WITH THE SECTION AFTER ################
    tk['nif'] = node.tiling_dimensions["L2"]["input_dimensions"][0]
    n_in       = node.tiling_dimensions["L2"]["input_dimensions"][0]
    h_in       = node.tiling_dimensions["L2"]["input_dimensions"][1]
    w_in       = node.tiling_dimensions["L2"]["input_dimensions"][2]
    tile_n_in  = node.tiling_dimensions["L1"]["input_dimensions"][0]
    tile_h_in  = node.tiling_dimensions["L1"]["input_dimensions"][1]
    tile_w_in  = node.tiling_dimensions["L1"]["input_dimensions"][2]

    n_out  = node.tiling_dimensions["L2"]["output_dimensions"][0]
    h_out      = node.tiling_dimensions["L2"]["output_dimensions"][1]
    w_out      = node.tiling_dimensions["L2"]["output_dimensions"][2]
    tile_n_out = node.tiling_dimensions["L1"]["output_dimensions"][0]
    tile_h_out = node.tiling_dimensions["L1"]["output_dimensions"][1]
    tile_w_out = node.tiling_dimensions["L1"]["output_dimensions"][2]
 
    fs1        = node.kernel_shape[0]
    fs2        = node.kernel_shape[1]
    ds_x       = node.input_activation_bits
    ds_y       = node.output_activation_bits
    dt_x       = node.input_activation_type
    dt_y       = node.output_activation_type

    tk['out_mul'] = node.outmul["value"] if 'outmul' in node.constant_names else 1
    tk['out_add'] = node.outadd["value"] if 'outadd' in node.constant_names else 0
    tk['out_shift'] = node.outshift["value"] if 'outshift' in node.constant_names else 0
    number_of_clusters = tk['number_of_clusters']

    tk["data_type_x"] = dt_x
    tk["data_type_y"] = dt_y
    ################################################################################

    tk['nof'] = n_out
    if node.HW_description['memory']['levels'] > 2:
        tk['factor'] = node.tiling_dimensions["L3"]["output_dimensions"][0] / n_out
    else:
        tk['factor'] = 1
    # x parameters
    tk['double_buffering'] = double_buffering
    tk['x_h'] = h_in
    tk['x_w'] = w_in
    tk['x_data_size_byte'] = node.input_activation_bits
    tk['x_tile_size_nif'] = tile_n_in
    tk['x_tile_size_h'] = tile_h_in
    tk['x_tile_size_w'] = tile_w_in
    tk['x_tile_size_byte'] = int(math.ceil(ds_x * tile_n_in * tile_h_in * tile_w_in / 8.0))
    tk['x_tile_size_nif_byte'] = int(math.ceil(tile_n_in * ds_x / 8.0))
    tk['x_stride_w_byte'] = int(math.ceil(w_in * n_in * ds_x / 8.0))
    tk['x_stride_c_byte'] = int(math.ceil(n_in * ds_x / 8.0))
    # y parameters
    tk['y_h'] = h_out
    tk['y_w'] = w_out
    tk['y_data_size_byte'] = ds_y
    tk['y_tile_size_nof'] = tile_n_out if (n_out > tile_n_out) else n_out
    tk['y_tile_size_h'] = tile_h_out if (h_out > tile_h_out) > 0 else h_out
    tk['y_tile_size_w'] = tile_w_out if (w_out > tile_w_out) > 0 else w_out
    tk['y_tile_size_byte'] = int(math.ceil(tk['y_tile_size_nof'] * tk['y_tile_size_h'] * tk['y_tile_size_w'] * ds_y / 8.0))
    tk['y_stride_w_byte'] = int(math.ceil(w_out * n_out * tk['factor'] * ds_y / 8.0))
    tk['y_stride_c_byte'] = int(math.ceil(n_out * tk['factor'] * ds_y / 8.0))
    tk['y_tile_size_nof_byte'] = int(math.ceil(tile_n_out * ds_y / 8.0))

    tk['tile_dim_h'] = max(int(math.ceil(float(h_out) / float(tk['y_tile_size_h']))), 1)
    tk['tile_dim_w'] = max(int(math.ceil(float(w_out) / float(tk['y_tile_size_w']))), 1)
    tk['tile_dim_nof'] = max(int(math.ceil(float(n_out) / float(tk['y_tile_size_nof']))), 1)
    tk['tile_dim_nif'] = max(int(math.ceil(float(n_in) / float(tile_n_in))), 1)
    tk['tile_n_in_last'] = n_in % tile_n_in if n_in % tile_n_in > 0 else tile_n_in
    # W parameters
    tk['fs1'] = fs1
    tk['fs2'] = fs2
    # l2 parameters
    if n_in == tile_n_in and w_in == tile_w_in and h_in == tile_h_in:
        x_buffer_size = int(math.ceil(ds_x * tile_n_in * tile_h_in * tile_w_in / 8.0))
    else:
        x_buffer_size = tk['double_buffering'] * int(math.ceil(ds_x * tile_n_in * tile_h_in * tile_w_in / 8.0))
        if x_buffer_size % 16 != 0:
            x_buffer_size = x_buffer_size
    if (n_in == (tile_n_in * number_of_clusters) and w_in == tile_w_in and h_in == tile_h_in and n_out == (tile_n_out * number_of_clusters) and n_in > number_of_clusters) \
    or (n_in == tile_n_in and w_in == tile_w_in and h_in == tile_h_in and n_out == (tile_n_out * number_of_clusters)):
        y_buffer_size = int(math.ceil(ds_y * tk['y_tile_size_nof'] * tk['y_tile_size_h'] * tk['y_tile_size_w'] / 8.0))
    else:
        y_buffer_size = tk['double_buffering'] * int(math.ceil(ds_y * tk['y_tile_size_nof'] * tk['y_tile_size_h'] * tk['y_tile_size_w'] / 8.0))

    # l1 parameters
    tk['l1_x_offset'] = 0
    tk['l1_y_offset'] = x_buffer_size + 8
    # y last
    tk['y_tile_size_nof_last'] = n_out % tile_n_out if (n_out % tile_n_out) > 0 else tile_n_out
    tk['y_tile_size_h_last'] = h_out % tile_h_out if (h_out % tile_h_out) > 0 else tile_h_out
    tk['y_tile_size_w_last'] = w_out % tile_w_out if (w_out % tile_w_out) > 0 else tile_w_out
    tk['y_length_nof_byte_last'] = int(math.ceil(tk['y_tile_size_nof_last'] * ds_y / 8.0))

    # x last
    tk['x_tile_size_nif_last'] = n_in % tile_n_in if (n_in % tile_n_in) > 0 else tile_n_in
    tk['x_tile_size_nif_byte_last'] = int(math.ceil(tk['x_tile_size_nif_last'] * ds_x / 8.0))
    tk['x_tile_size_h_last'] = tk['y_tile_size_h_last'] * s[0] + ks[0] - s[0] - (padding_bottom - ((h_in + padding_bottom + padding_top) - (h_out* s[0] + ks[0] - s[0])))
    tk['x_tile_size_w_last'] = tk['y_tile_size_w_last'] * s[1] + ks[1] - s[1] - (padding_right - ((w_in + padding_left + padding_right) - (w_out* s[1] + ks[1] - s[1])))
    ## single tile execution
    if tk['x_tile_size_h_last'] > tk['x_tile_size_h']:
        tk['x_tile_size_h_last'] = tk['x_tile_size_h']
    if tk['x_tile_size_w_last'] > tk['x_tile_size_w']:
        tk['x_tile_size_w_last'] = tk['x_tile_size_w']

    l = ""
    for k, v in tk.items():
        try:
            l += "// %s %d\n" % (k.ljust(30), v)
        except TypeError:
            try:
                l += "// %s %d\n" % (k.ljust(30), v[0])
            except TypeError:
                l += "// %s %s\n" % (k.ljust(30), v)
    buffer_l1_all = x_buffer_size + y_buffer_size
    tk['buffer_l1_all'] = buffer_l1_all
    tk['conv1d'] = node.conv1d
    tk['dilations'] = node.dilations
    tmpl = Template(filename=os.path.join(tmpl_dir, "layer_L2_c_pooling_template.c"))
    s_c = tmpl.render(verbose_log=l, **tk)
    save_string = os.path.join(out_dir, 'src', name_layer.replace(".h", ".c"))
    with open(save_string, "w") as f:
        f.write(s_c)
    tmpl = Template(filename=os.path.join(tmpl_dir, "layer_L2_h_template.h"))
    s = tmpl.render(verbose_log=l, **tk)
    save_string = os.path.join(out_dir, 'inc', name_layer)
    with open(save_string, "w") as f:
        f.write(s)
    return s, s_c

