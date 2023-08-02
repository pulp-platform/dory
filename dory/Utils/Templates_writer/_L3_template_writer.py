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

def print_template_layer_L3(node, tmpl_dir, out_dir):

    tk = OrderedDict([])
    ################## NEED A REWRITING IN THIS TEMPLATE PART ######################
    #### VARIABLE CREATION FOR COMPATIBILITY WITH THE SECTION AFTER ################
    if "Fused" in node.name:
        ks =      node.node0.kernel_shape
        s =       node.node0.strides
        g =       node.node0.group
        p =       node.node0.pads
        h_intermediate_L2_recomputed = node.tiling_dimensions["L2"]["output_dimensions"][1] * node.node1.strides[0] + (node.node1.kernel_shape[0] - 1) - (node.node1.strides[0] - 1)
        h_in_L2_recomputed = h_intermediate_L2_recomputed * s[0] + (ks[0] - 1) - (s[0] - 1)
        h_intermediate_L2_recomputed = int(np.floor((node.tiling_dimensions["L2"]["input_dimensions"][1] - (ks[0] - 1) + (s[0] - 1)) / s[0]))
        h_out_L2_recomputed = int(np.floor((h_intermediate_L2_recomputed - (node.node1.kernel_shape[0] - 1) + (node.node1.strides[0] - 1)) / node.node1.strides[0]))
        tk['flag_DW_0'] = 1 if node.node0.group > 1 else 0
        tk['flag_DW_1'] = 1 if node.node1.group > 1 else 0
    else:
        ks =      node.kernel_shape
        s =       node.strides
        g =       node.group
        p =       node.pads
        h_in_L2_recomputed = node.tiling_dimensions["L2"]["output_dimensions"][1] * s[0] + (ks[0] - 1) - (s[0] - 1)
        h_out_L2_recomputed = int(np.floor((node.tiling_dimensions["L2"]["input_dimensions"][1] - (ks[0] - 1) + (s[0] - 1)) / s[0]))
        tk['flag_DW'] = 1 if g > 1 else 0
    padding_top = p[0]
    conv_overlap1 = 2 * (ks[0] // 2) + ks[0] % 2 - 1 - (s[0] - 1)
    conv_overlap2 = 2 * (ks[1] // 2) + ks[1] % 2 - 1 - (s[1] - 1)

    ds_x       = node.input_activation_bits
    ds_y       = node.output_activation_bits
    h_out      = node.tiling_dimensions["L3"]["output_dimensions"][1]

    n_in_L2    = node.tiling_dimensions["L2"]["input_dimensions"][0]
    if node.tiling_dimensions["L3"]["output_dimensions"][1] > node.tiling_dimensions["L2"]["output_dimensions"][1]:
        h_in_L2= h_in_L2_recomputed
    else:
        h_in_L2= node.tiling_dimensions["L2"]["input_dimensions"][1]
    w_in_L2    = node.tiling_dimensions["L2"]["input_dimensions"][2]

    if "Addition" not in node.name and "Pool" not in node.name:
        n_out_L2= node.tiling_dimensions["L2"]["weights_dimensions"][0]
    else:
        n_out_L2= node.tiling_dimensions["L2"]["output_dimensions"][0]
    if node.tiling_dimensions["L3"]["input_dimensions"][1] > node.tiling_dimensions["L2"]["input_dimensions"][1]:
        h_out_L2= h_out_L2_recomputed
    else:
        h_out_L2= node.tiling_dimensions["L2"]["output_dimensions"][1]
    w_out_L2    = node.tiling_dimensions["L2"]["output_dimensions"][2]

    ################################################################################

    tk['conv_overlap1'] = conv_overlap1
    tk['conv_overlap2'] = conv_overlap2
    tk['padding'] = padding_top
    if (node.L3_input):
        tk['input_L3'] = 1
        factor_h_in = int(h_out / h_out_L2) 
    else:
        tk['input_L3'] = 0
        factor_h_in = 1
    factor_h_out = int(h_out / node.tiling_dimensions["L2"]["output_dimensions"][1])
    if not isinstance(node.tiling_dimensions["L2"]["weights_dimensions"], type(None)):
        factor_ch_out = int(node.tiling_dimensions["L3"]["weights_dimensions"][0] / n_out_L2)
    else:
        factor_ch_out = 1
    tk['n_tile_W'] = factor_ch_out
    tk['n_tile_x'] = factor_h_in
    tk['n_tile_y'] = factor_h_out
    tk['verbose'] = False
    if tk['padding'] > 0:
        tk['func_name'] = [node.prefixed_name + "_L2", node.prefixed_name + "_L2_p_t", node.prefixed_name + "_L2_p_b"]
    else:
        tk['func_name'] = [node.prefixed_name + "_L2"]
    tk['func_name_L3'] = node.prefixed_name
    tk['BitIn'] = ds_x
    tk['y_data_size_byte'] = ds_y
    tk['x_data_size_byte'] = ds_x
    tk['w_out'] = w_out_L2
    tk['h_out'] = h_out_L2
    tk['n_out'] = n_out_L2
    tk['w_in'] = w_in_L2
    tk['h_in'] = h_in_L2
    tk['n_in'] = n_in_L2

    tk['has_bias'] = int(len([1 for name in node.constant_names if "bias" in name])>0)

    offset = 0
    tk['l3_offset_w'] = offset
    offset += node.tiling_dimensions["L3"]["weight_memory"]

    if tk['has_bias'] == 1:
        tk['l3_offset_b'] = offset
        offset += node.tiling_dimensions["L3"]["bias_memory"]

    if not isinstance(node.tiling_dimensions["L2"]["constants_memory"], type(None)):
        tk['l3_offset_k'] = offset
        offset += int(node.tiling_dimensions["L3"]["constants_memory"] / 2)

        tk['l3_offset_l'] = offset
        offset += int(node.tiling_dimensions["L3"]["constants_memory"] / 2)

    tk['weight_dim'] = int( node.tiling_dimensions["L2"]["weight_memory"] )
    if tk['has_bias'] == 1:
        tk['bias_dim'] = node.tiling_dimensions["L2"]["bias_memory"]
    else:
        tk['bias_dim'] = 0
    if not isinstance(node.tiling_dimensions["L2"]["constants_memory"], type(None)):
        tk['lambda_dim'] = int(node.tiling_dimensions["L2"]["constants_memory"] / 2)
        tk['k_dim'] = int(node.tiling_dimensions["L2"]["constants_memory"] / 2)
    else:
        tk['lambda_dim'] = 0
        tk['k_dim'] = 0
    tk['dim_out'] = int( n_out_L2 * w_out_L2 * h_out_L2 * node.output_activation_bits / 8 )
    tk['dim_in'] = int( n_in_L2 * w_in_L2 * h_in_L2 * node.input_activation_bits / 8 )

    tmpl = Template(filename=os.path.join(tmpl_dir, "layer_L3_c_template.c"))
    l = ""
    s = tmpl.render(verbose_log=l, **tk)
    save_string = os.path.join(out_dir, 'src', node.prefixed_name + '.c')
    with open(save_string, "w") as f:
        f.write(s)
    tmpl = Template(filename=os.path.join(tmpl_dir, "layer_L3_h_template.h"))
    l = ""
    s = tmpl.render(verbose_log=l, **tk)
    save_string = os.path.join(out_dir, 'inc', node.prefixed_name + '.h')
    with open(save_string, "w") as f:
        f.write(s)
