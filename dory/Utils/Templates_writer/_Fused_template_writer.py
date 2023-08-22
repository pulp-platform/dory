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

def print_template_layer_Fused(node, layer_type, tmpl_dir, out_dir, double_buffering = 2):
    #### Parameters for the fused layer 
    name_layer = node.prefixed_name + '.h'
    tk = OrderedDict([])
    if (re.search('.0',name_layer)):
        try:
            int(re.search('.0',name_layer).group())
            tk['first_layer'] = 0
        except ValueError:
            tk['first_layer'] = 1
    else:
        tk['first_layer'] = 0
    tk['double_buffering'] = double_buffering
    tk['sdk'] = node.HW_description["software development kit"]["name"]
    tk['number_of_clusters'] = node.HW_description["HW specific parameters"]["clusters"] if "clusters" in node.HW_description["HW specific parameters"].keys() else 1
    tk['optional_type'] = layer_type
    tk['func_name'] = node.prefixed_name
    tk['type'] = f"{node.input_activation_type}8_t" if node.input_activation_type in ["int", "uint"] else "float"
    if node.HW_description['memory']['levels'] > 2:
        tk['factor'] = node.tiling_dimensions["L3"]["output_dimensions"][0] / node.tiling_dimensions["L2"]["weights_dimensions"][0]
    else:
        tk['factor'] = 1

    #####################################
    ######### NODE 0 PARAMETERS #########
    #####################################
    ks0            = node.node0.kernel_shape
    s0             = node.node0.strides
    tk['flag_DW_0'] = 1 if node.node0.group > 1 else 0
    inp_dim0       = node.tiling_dimensions["L2"]["input_dimensions"][1:]
    out_dim0       = [int(np.floor((node.tiling_dimensions["L2"]["input_dimensions"][1] - (ks0[0] - 1) + (s0[0] - 1)) / s0[0])), int(np.floor((node.tiling_dimensions["L2"]["input_dimensions"][2] - (ks0[1] - 1) + (s0[1] - 1)) / s0[1]))]
    in_ch0         = node.tiling_dimensions["L2"]["input_dimensions"][0]
    if tk['flag_DW_0'] == 1:
        out_ch0         = in_ch0
    elif tk['flag_DW_1'] == 1:
        out_ch0         = node.tiling_dimensions["L2"]["output_dimensions"][0]
    else:
        out_ch0         = node.output_channels
    g0             = node.node0.group
    p0             = node.node0.pads
    padding_top0   = p0[0]; padding_left0 = p0[1]; padding_bottom0 = p0[2]; padding_right0 = p0[3]
    conv_overlaph0 = 2 * (ks0[0] // 2) + ks0[0] % 2 - 1 - (s0[0] - 1)
    conv_overlapw0 = 2 * (ks0[1] // 2) + ks0[1] % 2 - 1 - (s0[1] - 1)
    ds_x0          = node.node0.input_activation_bits
    ds_y0          = node.node0.output_activation_bits
    ds_act0        = node.node0.constant_bits
    ds_W0          = node.node0.weight_bits
    ds_bias0       = node.node0.bias_bits
    dt_x0          = node.node0.input_activation_type
    dt_y0          = node.node0.output_activation_type
    dt_W0          = node.node0.weight_type
    tk['FLAG_BATCHNORM_0'] = 1 if 'k' in node.node0.constant_names else 0
    tk['has_bias_0']       = int(len([1 for name in node.node0.constant_names if "bias" in name])>0)
    tk['FLAG_RELU_0']      = 1 if 'outshift' in node.node0.constant_names else 0
    tk['conv_overlaph_0']  = conv_overlaph0
    tk['conv_overlapw_0']  = conv_overlapw0
    tk['padding_top_0']    = padding_top0
    tk['padding_bottom_0'] = padding_bottom0
    tk['padding_left_0']   = padding_left0
    tk['padding_right_0']  = padding_right0
    tk['stride_0']         = s0[0]
    if tk['flag_DW_0'] == 0:
        tk['g_0'] = 1
        tk['l2_input_channels_0'] = node.tiling_dimensions["L2"]["input_dimensions"][0]
    else:
        tk['g_0'] = node.tiling_dimensions["L2"]["input_dimensions"][0]
        tk['l2_input_channels_0'] = 1 
    tk['fsx0'] = ks0[0]
    tk['fsy0'] = ks0[1]
    tk['W_data_bit_0'] = ds_W0
    tk['b_data_bit_0'] = ds_bias0
    tk['x_data_bit_0'] = ds_x0
    tk['y_data_bit_0'] = ds_y0
    tk['activation_data_bit_0'] = ds_act0
    tk['W_data_type_0'] = dt_W0
    tk['x_data_type_0'] = dt_x0
    tk['y_data_type_0'] = dt_y0
    tk['l2_b_size_byte_0'] = 0
    if tk['has_bias_0'] == 1:
        tk['l2_off_bias_0'] = int(math.ceil(out_ch0 * in_ch0 * ks0[0] * ks0[1] * ds_W0 / 8.0 ))
        tk['l2_b_size_byte_0'] = int(out_ch0)* int(ds_bias0 / 8.0)
    if tk['FLAG_BATCHNORM_0'] == 1:
        tk['l2_off_k_0'] = int(math.ceil(out_ch0 * in_ch0 * ks0[0] * ks0[1] * ds_W0 / 8.0 + tk['l2_b_size_byte_0']))
        tk['l2_off_lambda_0'] = int(math.ceil((out_ch0 * in_ch0 * ks0[0] * ks0[1] * ds_W0 + out_ch0 * ds_act0) / 8.0 + tk['l2_b_size_byte_0']))
    tk['l2_output_channels_0'] = out_ch0
    tk['l2_x_h_0'] = inp_dim0[0]
    tk['l2_x_w_0'] = inp_dim0[1]
    tk['l2_y_h_0'] = out_dim0[0]
    tk['l2_y_w_0'] = out_dim0[1]

    tk['flag_DW_0'] = 1 if node.node0.group > 1 else 0
    inp_dim0       = node.tiling_dimensions["L2"]["input_dimensions"][1:]
    out_dim0       = [int(np.floor((node.tiling_dimensions["L2"]["input_dimensions"][1] - (ks0[0] - 1) + (s0[0] - 1)) / s0[0])), int(np.floor((node.tiling_dimensions["L2"]["input_dimensions"][2] - (ks0[1] - 1) + (s0[1] - 1)) / s0[1]))]
    in_ch0         = node.tiling_dimensions["L2"]["input_dimensions"][0]
    if tk['flag_DW_0'] == 1:
        out_ch0         = in_ch0
    elif tk['flag_DW_1'] == 1:
        out_ch0         = node.tiling_dimensions["L2"]["output_dimensions"][0]
    else:
        out_ch0         = node.output_channels
    #####################################
    ######### NODE 1 PARAMETERS #########
    #####################################
    ks1            = node.node1.kernel_shape
    s1             = node.node1.strides
    inp_dim1       = out_dim0
    out_dim1       = node.tiling_dimensions["L2"]["output_dimensions"][1:]
    in_ch1         = out_ch0
    out_ch1        = node.tiling_dimensions["L2"]["weights_dimensions"][0]
    p1             = node.node1.pads
    padding_top1   = p1[0]; padding_left1 = p1[1]; padding_bottom1 = p1[2]; padding_right1 = p1[3]
    conv_overlaph1 = 2 * (ks1[0] // 2) + ks1[0] % 2 - 1 - (s1[0] - 1)
    conv_overlapw1 = 2 * (ks1[1] // 2) + ks1[1] % 2 - 1 - (s1[1] - 1)
    ds_x1          = node.node1.input_activation_bits
    ds_y1          = node.node1.output_activation_bits
    ds_act1        = node.node1.constant_bits
    ds_W1          = node.node1.weight_bits
    ds_bias1       = node.node1.bias_bits
    dt_x1          = node.node1.input_activation_type
    dt_y1          = node.node1.output_activation_type
    dt_W1          = node.node1.weight_type
    tk['flag_DW_1']        = 1 if node.node1.group > 1 else 0
    tk['FLAG_BATCHNORM_1'] = 1 if 'k' in node.node1.constant_names else 0
    tk['has_bias_1']       = int(len([1 for name in node.node1.constant_names if "bias" in name])>0)
    tk['FLAG_RELU_1']      = 1 if 'outshift' in node.node1.constant_names else 0
    tk['conv_overlaph_1']  = conv_overlaph1
    tk['conv_overlapw_1']  = conv_overlapw1
    tk['padding_top_1']    = padding_top1
    tk['padding_bottom_1'] = padding_bottom1
    tk['padding_left_1']   = padding_left1
    tk['padding_right_1']  = padding_right1
    tk['stride_1']         = s1[0]
    if tk['flag_DW_1'] == 0:
        tk['g_1'] = 1
        tk['l2_input_channels_1'] = node.tiling_dimensions["L2"]["input_dimensions"][0]
    else:
        tk['g_1'] = node.tiling_dimensions["L2"]["output_dimensions"][0]
        tk['l2_input_channels_1'] = 1  
    tk['fsx1'] = ks1[0]
    tk['fsy1'] = ks1[1]
    tk['W_data_bit_1'] = ds_W1
    tk['b_data_bit_1'] = ds_bias1
    tk['x_data_bit_1'] = ds_x1
    tk['y_data_bit_1'] = ds_y1
    tk['activation_data_bit_1'] = ds_act1
    tk['W_data_type_1'] = dt_W1
    tk['x_data_type_1'] = dt_x1
    tk['y_data_type_1'] = dt_y1
    tk['activation_data_bit_1'] = ds_act1
    tk['l2_b_size_byte_1'] = 0
    if tk['has_bias_1'] == 1:
        tk['l2_off_bias_1'] = int(math.ceil(out_ch1 * in_ch1 * ks1[0] * ks1[1] * ds_W1 / 8.0 ))
        tk['l2_b_size_byte_1'] = int(out_ch1)* int(ds_bias1 / 8.0)
    if tk['FLAG_BATCHNORM_1'] == 1:
        tk['l2_off_k_1'] = int(math.ceil(out_ch1 * in_ch1 * ks1[0] * ks1[1] * ds_W1 / 8.0 + tk['l2_b_size_byte_1']))
        tk['l2_off_lambda_1'] = int(math.ceil((out_ch1 * in_ch1 * ks1[0] * ks1[1] * ds_W1 + out_ch1 * ds_act1) / 8.0 + tk['l2_b_size_byte_1']))
    tk['l2_output_channels_1'] = out_ch1
    tk['l2_x_h_1'] = inp_dim1[0]
    tk['l2_x_w_1'] = inp_dim1[1]
    tk['l2_y_h_1'] = out_dim1[0]
    tk['l2_y_w_1'] = out_dim1[1]

    ###############################
    ######## L1 Parameters ########
    ###############################
    # y parameters
    tk['l1_output_channels'] = min(node.tiling_dimensions["L1"]["output_dimensions"][0], tk['l2_output_channels_1'])
    tk['l1_output_channels_last'] = tk['l2_output_channels_1'] % tk['l1_output_channels'] if (tk['l2_output_channels_1'] % tk['l1_output_channels']) > 0 else tk['l1_output_channels']
    tk['l1_y_h'] = min(node.tiling_dimensions["L1"]["output_dimensions"][1], tk['l2_y_h_1'])
    tk['l1_y_h_last'] = tk['l2_y_h_1'] % tk['l1_y_h'] if (tk['l2_y_h_1'] % tk['l1_y_h']) > 0 else tk['l1_y_h']
    tk['l1_y_w'] = min(node.tiling_dimensions["L1"]["output_dimensions"][2], tk['l2_y_w_1'])
    tk['l1_y_w_last'] = tk['l2_y_w_1'] % tk['l1_y_w'] if (tk['l2_y_w_1'] % tk['l1_y_w']) > 0 else tk['l1_y_w']
    tk['l1_y_stride_w'] = int(math.ceil(tk['l2_y_w_1'] * tk['l2_output_channels_1'] * tk['factor'] * ds_y1 / 8.0))
    tk['l1_y_stride_channels'] = int(math.ceil(tk['l2_output_channels_1'] * tk['factor'] * ds_y1 / 8.0))

    # x parameters
    tk['l1_input_channels'] = node.tiling_dimensions["L1"]["input_dimensions"][0]
    tk['l1_input_channels_last'] = tk['l2_input_channels_0']*tk['g_0'] % tk['l1_input_channels'] if (tk['l2_input_channels_0']*tk['g_0'] % tk['l1_input_channels']) > 0 else tk['l1_input_channels']
    tk['l1_x_h'] = node.tiling_dimensions["L1"]["input_dimensions"][1]
    l1_x_h_last_intermediate = tk['l1_y_h_last'] * s1[0] + ks1[0] - s1[0] - (padding_bottom1 - ((inp_dim1[0] + padding_bottom1 + padding_top1) - (out_dim1[0] * s1[0] + ks1[0] - s1[0])))
    tk['l1_x_h_last'] = min(l1_x_h_last_intermediate * s0[0] + ks0[0] - s0[0] - (padding_bottom0 - ((inp_dim0[0] + padding_bottom0 + padding_top0) - (out_dim0[0] * s0[0] + ks0[0] - s0[0]))), tk['l1_x_h'])
    tk['l1_x_w'] = node.tiling_dimensions["L1"]["input_dimensions"][2]
    l1_x_w_last_intermediate = tk['l1_y_h_last'] * s1[1] + ks1[1] - s1[1] - (padding_right1 - ((inp_dim1[1] + padding_left1 + padding_right1) - (out_dim1[1] * s1[1] + ks1[1] - s1[1])))
    tk['l1_x_w_last'] = min(l1_x_w_last_intermediate * s0[1] + ks0[1] - s0[1] - (padding_right0 - ((inp_dim0[1] + padding_left0 + padding_right0) - (out_dim0[1]* s0[1] + ks0[1] - s0[1]))), tk['l1_x_w'])
    tk['l1_x_stride_w'] = int(math.ceil(tk['l2_x_w_0'] * tk['l2_input_channels_0']*tk['g_0'] * ds_x0 / 8.0))
    tk['l1_x_stride_channels'] = int(math.ceil(tk['l2_input_channels_0']*tk['g_0'] * ds_x0 / 8.0))

    # W parameters node0
    tk['l1_W_input_channels_0'] = tk['l2_input_channels_0']
    tk['l1_W_input_channels_last_0'] = tk['l2_input_channels_0']
    tk['l1_W_output_channels_0'] = tk['l1_output_channels'] if tk['flag_DW_1'] == 1 else tk['l2_output_channels_0']
    tk['l1_W_output_channels_last_0'] = tk['l1_output_channels_last'] if tk['flag_DW_1'] == 1 else tk['l2_output_channels_0']
    tk['l1_W_stride_nof_byte_0'] = int(math.ceil(tk['l1_W_input_channels_0'] * ks0[0] * ks0[1] * ds_W0 / 8.0))     
    tk['l1_W_stride_hw_byte_0'] = int(math.ceil(tk['l1_W_input_channels_0'] * ds_W0 / 8.0))
    tk['l1_k_size_0'] = 0
    tk['l1_lambda_size_0'] = 0
    if tk['FLAG_BATCHNORM_0'] == 1:
        tk['l1_k_size_0'] = int(math.ceil(tk['l1_W_output_channels_0'] * ds_act0 / 8.0))
        tk['l1_lambda_size_0'] = int(math.ceil(tk['l1_W_output_channels_0'] * ds_act0 / 8.0))
    if tk['has_bias_0'] == 1:
        tk['l1_bias_size_0'] = tk['l1_W_output_channels_0'] * int(ds_bias0 / 8.0)
    # W parameters node1
    tk['l1_W_input_channels_1'] = tk['l1_output_channels'] if tk['flag_DW_1'] == 1 else tk['l2_output_channels_0']
    tk['l1_W_input_channels_last_1'] = tk['l1_output_channels_last'] if tk['flag_DW_1'] == 1 else tk['l2_output_channels_0']
    tk['l1_W_output_channels_1'] = tk['l1_output_channels']
    tk['l1_W_output_channels_last_1'] = tk['l1_output_channels_last']
    tk['l1_W_stride_nof_byte_1'] = int(math.ceil(tk['l1_W_input_channels_1'] * ks1[0] * ks1[1] * ds_W1 / 8.0))     
    tk['l1_W_stride_hw_byte_1'] = int(math.ceil(tk['l1_W_input_channels_1'] * ds_W1 / 8.0))
    tk['l1_k_size_1'] = 0
    tk['l1_lambda_size_1'] = 0
    if tk['FLAG_BATCHNORM_1'] == 1:
        tk['l1_k_size_1'] = int(math.ceil(tk['l1_W_output_channels_1'] * ds_act1 / 8.0))
        tk['l1_lambda_size_1'] = int(math.ceil(tk['l1_W_output_channels_1'] * ds_act1 / 8.0))
    if tk['has_bias_1'] == 1:
        tk['l1_bias_size_1'] = tk['l1_W_output_channels_1'] * int(ds_bias1 / 8.0)

    # l1 general parameters
    tk['tile_dim_h'] = max(int(math.ceil(float(tk['l2_y_h_1']) / float(tk['l1_y_h']))), 1)
    tk['tile_dim_w'] = max(int(math.ceil(float(tk['l2_y_w_1']) / float(tk['l1_y_w']))), 1)
    tk['tile_dim_nof'] = max(int(math.ceil(float(tk['l2_output_channels_1']) / float(tk['l1_output_channels']))), 1)
    tk['tile_dim_nif'] = max(int(math.ceil(float(tk['l2_input_channels_0']*tk['g_0']) / float(tk['l1_input_channels']))), 1)
    x_buffer_size = (tk['double_buffering'] if (tk['tile_dim_h'] * tk['tile_dim_w'] * tk['tile_dim_nif']) == 1 else 1) * int(math.ceil(ds_x0 * tk['l1_input_channels'] * tk['l1_x_h'] * tk['l1_x_w'] / 8.0))
    y_buffer_size = (tk['double_buffering'] if (tk['tile_dim_h'] * tk['tile_dim_w'] * tk['tile_dim_nif'] * tk['tile_dim_nof']) == 1 else 1) * int(math.ceil(ds_y1 * tk['l1_output_channels'] * tk['l1_y_h'] * tk['l1_y_w'] / 8.0))
    W_buffer_size_0 = (tk['double_buffering'] if (tk['tile_dim_nif'] * tk['tile_dim_nof']) == 1 else 1) * int(math.ceil(ds_W0 * tk['l1_W_output_channels_0'] * tk['l1_W_input_channels_0'] * ks0[0] * ks0[1] / 8.0))
    W_buffer_size_1 = (tk['double_buffering'] if (tk['tile_dim_nif'] * tk['tile_dim_nof']) == 1 else 1) * int(math.ceil(ds_W0 * tk['l1_W_output_channels_1'] * tk['l1_W_input_channels_1'] * ks1[0] * ks1[1] / 8.0))
    k_buffer_size_0 = 0
    lambd_buffer_size_0 = 0
    bias_size_0 = 0
    k_buffer_size_1 = 0
    lambd_buffer_size_1 = 0
    bias_size_1 = 0
    if tk['FLAG_BATCHNORM_0'] == 1:
        k_buffer_size_0 = int(tk['l2_output_channels_0'] * ds_act0 / 8.0)
        lambd_buffer_size_0 = int(tk['l2_output_channels_0'] * ds_act0 / 8.0)
    if tk['FLAG_BATCHNORM_1'] == 1:
        k_buffer_size_1 = int(tk['l2_output_channels_1'] * ds_act1 / 8.0)
        lambd_buffer_size_1 = int(tk['l2_output_channels_1'] * ds_act1 / 8.0)
    if tk['has_bias_0'] == 1:
        bias_size_0 = tk['l2_output_channels_0'] * int(ds_bias0 / 8.0)
    if tk['has_bias_1'] == 1:
        bias_size_1 = tk['l2_output_channels_1'] * int(ds_bias1 / 8.0)

    tk['im2col_dim_0'] = (8 * (ks0[0] * (tk['l1_x_h'] + padding_bottom0 + padding_top0) + ks0[0])) * int( 8 / min(ds_x0, ds_y0, ds_W0))
    tk['im2col_dim_1'] = (8 * (ks1[0] * (tk['l1_x_h'] + padding_bottom1 + padding_top1) + ks1[0])) * int( 8 / min(ds_x1, ds_y1, ds_W1))
    tk['buffer_l1_all'] = x_buffer_size + y_buffer_size + W_buffer_size_0 + W_buffer_size_1 + k_buffer_size_0 + lambd_buffer_size_0 + bias_size_0 + k_buffer_size_1 + lambd_buffer_size_1 + bias_size_1 + 40

    tk['l1_x_offset'] = 0
    tk['l1_y_offset'] = x_buffer_size + 8
    tk['l1_W_offset_0'] = tk['l1_y_offset'] + y_buffer_size + 8
    if tk['FLAG_BATCHNORM_0'] == 1:
        tk['l1_k_offset_0'] = tk['l1_W_offset_0'] + W_buffer_size_0 + 8
        tk['l1_lambda_offset_0'] = tk['l1_k_offset_0'] + k_buffer_size_0 + 8
    if tk['has_bias_0'] == 1:
        tk['l1_b_offset_0'] = tk['l1_W_offset_0'] + W_buffer_size_0 + 8 + k_buffer_size_0 + 8 + lambd_buffer_size_0  + 8
    else:
        tk['l1_b_offset_0'] = tk['l1_lambda_offset_0'] + W_buffer_size_0 + 8
    tk['l1_W_offset_1'] = tk['l1_y_offset'] + W_buffer_size_0 + 8 + k_buffer_size_0 + 8 + lambd_buffer_size_0  + 8 + bias_size_0  + 8
    if tk['FLAG_BATCHNORM_1'] == 1:
        tk['l1_k_offset_1'] = tk['l1_W_offset_1'] + W_buffer_size_1 + 8
        tk['l1_lambda_offset_1'] = tk['l1_k_offset_0'] + k_buffer_size_1 + 8
    if tk['has_bias_1'] == 1:
        tk['l1_b_offset_1'] = tk['l1_W_offset_1'] + W_buffer_size_1 + 8 + k_buffer_size_1 + 8 + lambd_buffer_size_1  + 8

    l = ""
    for k, v in tk.items():
        try:
            l += "// %s %d\n" % (k.ljust(30), v)
        except TypeError:
            try:
                l += "// %s %d\n" % (k.ljust(30), v[0])
            except TypeError:
                l += "// %s %s\n" % (k.ljust(30), v)

    tmpl = Template(filename=os.path.join(tmpl_dir, "layer_L2_c_fused_template.c"))
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

