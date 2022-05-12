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

##CHANGED ALL H TO W
def print_template_layer_1D(x, y_gold, W,
                         n_in, w_in,
                         n_out, w_out,
                         tile_n_in, tile_w_in, tile_w_out,
                         tile_n_out,
                         ds_x, ds_y, ds_W, ds_act, type_data,
                         fs1, padding_left, padding_right, 
                         stride, dilation,
                         relu, BN,
                         out_mul, out_shift,
                         name_layer='layer',
                         ultra_verbose=True,
                         test=False,
                         test_location='L3',
                         has_bias=True,
                         conv_order='CHW',
                         optional='conv',
                         l1_buffer=44000,
                         platform='GAP8',
                         chip='GAP8v2',
                         optional_type='8bit',
                         backend = 'MCU',
                         layer_type = 'normal'):
    # Generate the Layer management c file.
    if w_out * stride + fs1 - 1 - stride + 1 > w_in:
        if (w_out * stride + fs1 - 1 - stride + 1 - w_in) == padding_left:
            padding_r = 0
            padding_l = padding_left
        else:
            padding_r = padding_right
            padding_l = padding_left
    # add padding from "regular" tile where necessary
    tile_w_in = tile_w_in if w_in > tile_w_in else tile_w_in
    name = re.sub(r'\W', '', name_layer).replace("hex", "").replace(".", "").replace("_weights", "")
    name_layer = name + '.h'
    conv_overlap1 = 2 * (((fs1-1)*dilation+1) // 2) + ((fs1-1)*dilation+1) % 2 - 1 - (stride - 1)
    tk = OrderedDict([])
    tk['optional_type'] = optional_type
    tk['func_name'] = name
    tk['layer_type'] = layer_type
    tk['optional'] = optional
    tk['FLAG_BATCHNORM'] = BN
    tk['has_bias'] = has_bias
    tk['FLAG_RELU'] = relu
    tk['test_location'] = test_location
    tk['platform'] = platform
    tk['chip'] = chip
    tk['dilation'] = dilation
    tk['type'] = type_data
    tk['nof'] = n_out
    tk['nif'] = n_in
    tk['conv_overlap1'] = conv_overlap1
    tk['padding_left'] = padding_left
    tk['padding_right'] = padding_right
    tk['stride'] = stride
    # x parameters
    tk['x_h'] = w_in
    tk['x_data_size_byte'] = ds_x
    tk['x_tile_size_nif'] = tile_n_in
    tk['x_tile_size_h'] = tile_w_in
    tk['x_tile_size_byte'] = int(math.ceil(ds_x * tile_n_in * tile_w_in / 8.0))
    tk['x_tile_size_nif_byte'] = int(math.ceil(tile_n_in * ds_x / 8.0))
    tk['x_stride_c_byte'] = int(math.ceil(n_in * ds_x / 8.0))
    # y parameters
    tk['y_h'] = w_out
    tk['y_data_size_byte'] = ds_y
    tk['act_dim_bit'] = ds_act
    tk['y_tile_size_nof'] = tile_n_out
    tk['y_tile_size_h'] = tile_w_out
    tk['y_tile_size_byte'] = int(math.ceil(tile_n_out * tile_w_out * ds_y / 8.0))
    tk['y_stride_c_byte'] = int(math.ceil(n_out * ds_y / 8.0))
    tk['y_tile_size_nof_byte'] = int(math.ceil(tile_n_out * ds_y / 8.0))

    tk['tile_dim_h'] = max(int(math.ceil(float(w_out) / float(tile_w_out))), 1)
    tk['tile_dim_nof'] = max(int(math.ceil(float(n_out) / float(tile_n_out))), 1)
    tk['tile_dim_nif'] = max(int(math.ceil(float(n_in) / float(tile_n_in))), 1)
    # W parameters
    tk['fs1'] = fs1
    tk['W_data_size_byte'] = ds_W
    tk['W_tile_size_nof'] = tile_n_out 
    tk['b_size_byte'] = int(math.ceil(n_out * ds_W / 8.0))
    tk['W_tile_size_nif'] = tile_n_in
    tk['W_tile_size_byte'] = int(math.ceil(tile_n_out * tk['W_tile_size_nif'] * fs1 * ds_W / 8.0))
    tk['W_stride_nof_byte'] = int(math.ceil(tk['nif'] * fs1 * ds_W / 8.0))
    tk['W_stride_hw_byte'] = int(math.ceil(tk['nif'] * ds_W / 8.0))
    tk['W_tile_nif_byte'] = int(math.ceil(tk['W_tile_size_nif'] * ds_W / 8.0))
    # l2 parameters
    if tk['FLAG_BATCHNORM'] == 1:
        tk['l2_off_k'] = int(math.ceil(tk['nof'] * tk['nif'] * fs1  * ds_W / 8.0))
        tk['l2_off_lambda'] = int(math.ceil((tk['nof'] * tk['nif'] * fs1 * ds_W + tk['nof'] * ds_act) / 8.0))
    if has_bias == 1:
        tk['l2_off_bias'] = 0
    if n_in == tile_n_in and w_in == tile_w_in:
        x_buffer_size = int(math.ceil(ds_x * tile_n_in * tile_w_in / 8.0))
    else:
        x_buffer_size = 2 * int(math.ceil(ds_x * tile_n_in * tile_w_in / 8.0))
    if n_in == tile_n_in and w_in == tile_w_in and n_out == tile_n_out:
        y_buffer_size = int(math.ceil(ds_y * tile_n_out * tile_w_out / 8.0))
        W_buffer_size = int(math.ceil(ds_W * tile_n_out * tile_n_in * fs1 / 8.0))
    else:
        y_buffer_size = 2 * int(math.ceil(ds_y * tile_n_out * tile_w_out / 8.0))
        W_buffer_size = 2 * int(math.ceil(ds_W * tile_n_out * tile_n_in * fs1 / 8.0))
    if tk['FLAG_BATCHNORM'] == 1:
        k_buffer_size = int(n_out * ds_act / 8.0)
        lambd_buffer_size = int(n_out * ds_act / 8.0)
    else:
        k_buffer_size = 0
        lambd_buffer_size = 0
    if conv_order == 'PULP-NN-MAX' or conv_order == 'PULP-NN-ADD':
        W_buffer_size = 0
    # l1 parameters
    tk['l1_x_offset'] = 0
    tk['l1_y_offset'] = x_buffer_size + 4
    if conv_order == 'PULP-NN-ADD':
        tk['l1_x2_offset'] = x_buffer_size + 4 + y_buffer_size + 4
    if conv_order != 'PULP-NN-MAX' and conv_order != 'PULP-NN-ADD':
        tk['l1_W_offset'] = x_buffer_size + 4 + y_buffer_size + 4
        if tk['FLAG_BATCHNORM'] == 1:
            tk['l1_k_offset'] = x_buffer_size + 4 + y_buffer_size + 4 + W_buffer_size + 4
            tk['l1_lambda_offset'] = x_buffer_size + 4 + y_buffer_size + 4 + W_buffer_size + 4 + k_buffer_size + 4
        if has_bias == 1:
            tk['l1_b_offset'] = x_buffer_size + 4 + y_buffer_size + 4 + W_buffer_size + 4 + k_buffer_size + 4 + lambd_buffer_size + 4

    if conv_order != 'PULP-NN-MAX':
        if tk['FLAG_BATCHNORM'] == 1:
            tk['k_size_byte'] = k_buffer_size
            tk['lambda_size_byte'] = k_buffer_size
            tk['k_tile_size_byte'] = int(math.ceil(tile_n_out * ds_act / 8.0))
            tk['lambda_tile_size_byte'] = int(math.ceil(tile_n_out * ds_act / 8.0))
        if has_bias == 1:
            tk['bias_tile_size_byte'] = tile_n_out
            tk['b_size_byte'] = int(n_out)

    # x last
    tk['x_tile_size_nif_last'] = n_in % tile_n_in if (n_in % tile_n_in) > 0 else tile_n_in
    tk['x_tile_size_nif_byte_last'] = int(math.ceil(tk['x_tile_size_nif_last'] * ds_x / 8.0))
    if tk['tile_dim_h'] == 1:
        tk['x_tile_size_h_last'] = tk['x_tile_size_h']
    elif tk['tile_dim_h'] == 2:
        tk['x_tile_size_h_last'] = w_in - tile_w_in + tk['conv_overlap1'] + padding_left
    elif tk['tile_dim_h'] == 3:
        tk['x_tile_size_h_last'] = w_in - tile_w_in - (tile_w_in - tk['conv_overlap1'] - padding_left) + tk['conv_overlap1']
    else:
        tk['x_tile_size_h_last'] = w_in - tile_w_in - (tile_w_in - tk['conv_overlap1'] - padding_left) - (tk['tile_dim_h'] - 3) * (tile_w_in - tk['conv_overlap1']) + tk['conv_overlap1']
    if tk['x_tile_size_h_last'] > tk['x_tile_size_h']:
        tk['x_tile_size_h_last'] = tk['x_tile_size_h']

    # W last
    if conv_order != 'PULP-NN-MAX' and conv_order != 'PULP-NN-ADD':
        tk['W_tile_size_nof_last'] = n_out % tile_n_out if (n_out % tile_n_out) > 0 else tile_n_out
        tk['W_tile_size_nif_last'] = tk['W_tile_size_nif']
        tk['W_tile_size_nif_byte_last'] = int(math.ceil(tk['W_tile_size_nif_last'] * ds_W / 8.0))
    # y last

    tk['y_tile_size_nof_last'] = n_out % tile_n_out if (n_out % tile_n_out) > 0 else tile_n_out
    tk['y_tile_size_h_last'] = w_out % tile_w_out if (w_out % tile_w_out) > 0 else tile_w_out

    #tk['y_tile_size_nof_last'] = n_in
    #tk['y_tile_size_h_last'] = w_in
    tk['y_length_nof_byte_last'] = int(math.ceil(tk['y_tile_size_nof_last'] * ds_y / 8.0))
    l = ""
    for k, v in tk.items():
        try:
            l += "// %s %d\n" % (k.ljust(30), v)
        except TypeError:
            try:
                l += "// %s %d\n" % (k.ljust(30), v[0])
            except TypeError:
                l += "// %s %s\n" % (k.ljust(30), v)
    if conv_order == 'PULP-NN':
        buffer_l1_all = W_buffer_size + x_buffer_size + y_buffer_size + k_buffer_size + lambd_buffer_size + 40
    elif conv_order == 'PULP-NN-ADD':
        buffer_l1_all = x_buffer_size * 2 + y_buffer_size + k_buffer_size + lambd_buffer_size + 40
    elif conv_order == 'PULP-NN-MAX':
        buffer_l1_all = x_buffer_size + y_buffer_size + k_buffer_size + lambd_buffer_size + 40
    tk['buffer_l1_all'] = buffer_l1_all
    l2_dim_input = (n_in) * tk['x_h']
    l2_dim_output = (tk['nof']) * tk['y_h']
    l2_dim_weights = tk['nof'] * tk['nif'] * tk['fs1']
    l2_dim_k = k_buffer_size
    l2_dim_lambda = lambd_buffer_size
    if conv_order == 'PULP-NN':
        root = '/'.join(os.getcwd().split('/')[:-1])
        tmpl = Template(filename=root + f"/Templates/{backend}/layer_templates/layer_template_conv_1D.c")
    s = tmpl.render(TEST=test,VERBOSE=False,ULTRA_VERBOSE=ultra_verbose,PULP_TEST=True,verbose_log=l,**tk)
    if 'L2' in test_location:
        save_string = './application/DORY_network/src/' + name_layer.replace("h", "c")
    elif 'L3' in test_location:
        save_string = './application/DORY_network/src/' + name_layer.replace("h", "c")
    with open(save_string, "w") as f:
        f.write(s)
    root = '/'.join(os.getcwd().split('/')[:-1])
    tmpl = Template(filename=root + f"/Templates/{backend}/layer_templates/layer_template_h.h")
    s = tmpl.render(
        TEST=test,
        VERBOSE=False,
        ULTRA_VERBOSE=ultra_verbose,
        PULP_TEST=True,
        verbose_log=l,
        **tk)
    if 'L2' in test_location:
        save_string = './application/DORY_network/inc/' + name_layer
    elif 'L3' in test_location:
        save_string = './application/DORY_network/inc/' + name_layer
    with open(save_string, "w") as f:
        f.write(s)
    if 'L2' in test_location:
        tk['out_mul'] = out_mul
        tk['out_shift'] = out_shift
        tk['l1_buffer'] = l1_buffer
        tk['activation_size_out'] = int(math.ceil(l2_dim_output * ds_y / 8.0))
        tk['activation_size_in'] = int(math.ceil(l2_dim_input * ds_x / 8.0))
        tk['x_content'] = print_test_vector(x, 'char')
        tk['y_expected_content'] = print_test_vector(y_gold, 'char')
        tk['check_sum'] = sum(y_gold)
        tk['W_content'] = print_test_vector(W, 'char')
        tk['buffer_l1_all'] = buffer_l1_all
        tk['l2_dim_weights'] = int(math.ceil((l2_dim_weights) * ds_W / 8.0) + (l2_dim_k + l2_dim_lambda))
        tk['h_out'] = tk['y_h']
        tk['ultra_test'] = True
        root = '/'.join(os.getcwd().split('/')[:-1])
        tmpl = Template(filename=root+f"/Templates/{backend}/test_templateL2.c")
        s = tmpl.render(
            TEST=test,
            VERBOSE=False,
            ULTRA_VERBOSE=ultra_verbose,
            PULP_TEST=True,
            verbose_log=l,
            **tk)
        save_string = './application/DORY_network/src/main.c'
        with open(save_string, "w") as f:
            f.write(s)
        tk['build_layers'] = os.listdir('./application/DORY_network/src/') 
        tk['platform'] = 'GAP8'
        tmpl = Template(filename=root+f"/Templates/{backend}/Makefile_template_L2")
        s = tmpl.render(**tk)
        save_string = './application/Makefile'
        with open(save_string, "w") as f:
            f.write(s)
    return l2_dim_input, l2_dim_output, l2_dim_weights, l2_dim_k, l2_dim_lambda, tk['nof'], buffer_l1_all, n_out, w_out
