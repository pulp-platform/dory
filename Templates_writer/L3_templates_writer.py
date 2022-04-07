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

def print_pool_template_layer_L3(X, W, Y, fs1, fs2, padding, stride,
                            factor_ch_out,
                            factor_h_out, 
                            factor_h_in,
                            name,
                            out_dim1,
                            in_dim1,
                            in_dim_full,
                            w_out,
                            h_out,
                            n_out,
                            w_in,
                            h_in,
                            n_in,
                            full_net,
                            platform,
                            data_type_x,
                            data_type_y,
                            test_location,
                            buffer_l1_all,
                                 input_L3,
                                 backend
                            ):
    # generation of L3 layers. The layers are generated with this infrustructure if an L3 tiling is demanded.
    tk = OrderedDict([])
    conv_overlap1 = 2 * (fs1 // 2) + fs1 % 2 - 1 - (stride - 1)
    conv_overlap2 = 2 * (fs2 // 2) + fs2 % 2 - 1 - (stride - 1)
    tk['conv_overlap1'] = conv_overlap1
    tk['conv_overlap2'] = conv_overlap2
    tk['padding'] = padding
    tk['input_L3'] = input_L3
    tk['n_tile_W'] = int(factor_ch_out)
    tk['n_tile_x'] = int(factor_h_in)
    tk['n_tile_y'] = int(factor_h_out)
    tk['verbose'] = False
    tk['func_name'] = name
    tk['func_name_L3'] = name[0] + 'L3'
    tk['act_out_dim_partial'] = int(out_dim1)
    tk['w_out'] = w_out
    tk['h_out'] = h_out
    tk['n_out'] = n_out
    tk['w_in'] = w_in
    tk['h_in'] = h_in
    tk['n_in'] = n_in
    tk['dim_out'] = out_dim1
    tk['dim_in'] = in_dim1
    tk['platform'] = platform
    tk['y_data_size_byte'] = data_type_y
    tk['x_data_size_byte'] = data_type_x
    tk['BitIn'] = data_type_x
    tk['BitOut'] = data_type_y
    tk['weight_dim'] = 0
    tk['k_dim'] = 0
    tk['lambda_dim'] = 0
    root = '/'.join(os.getcwd().split('/')[:-1])
    tmpl = Template(filename=root + f"/Templates/{backend}/layer_templates/layer_template_L3.c", strict_undefined=True)
    l = ""
    s = tmpl.render(verbose_log=l,**tk)
    #
    save_string = './application/DORY_network/src/' + tk['func_name_L3'] + '.c'
    with open(save_string, "w") as f: f.write(s)
    tmpl = Template(filename=root + f"/Templates/{backend}/layer_templates/layer_template_L3-h.h")
    s = tmpl.render(verbose_log=l, **tk)
    if full_net == 1:
        save_string = './application/DORY_network/inc/' + \
            tk['func_name_L3'] + '.h'
    else:
        save_string = './applicationL3/DORY_network/inc/' + \
            tk['func_name_L3'] + '.h'
    with open(save_string, "w") as f:
        f.write(s)
    if 'partial' in test_location:
        string_layer = "inputs.hex"
        save_s = './application/DORY_network/' + string_layer
        with open(save_s, 'wb') as f:
            for i in X.astype('uint8').flatten():
                f.write(bytes((i,)))
        tk['x_content'] = print_test_vector(X, 'char')
        if tk['n_tile_W'] == 1:
            tk['W_content'] = print_test_vector(W, 'char')
            tk['weight_dim'] = W.shape[0]
        tk['check_sum'] = sum(Y)
        tk['activation_size_out'] = out_dim1
        tk['activation_size_in'] = in_dim1
        tk['activation_size_in_full'] = in_dim_full
        tk['func_nameL3'] = tk['func_name_L3']
        tk['file'] = name[0][5:] + '_weights.hex'
        tk['buffer_l1_all'] = buffer_l1_all
        tmpl = Template(filename=root + f"/Templates/{backend}/test_templateL3.c")
        s = tmpl.render(**tk)
        save_string = './application/DORY_network/src/main.c'
        with open(save_string, "w") as f: f.write(s)

def print_template_layer_L3(X, W, Y, fs1, fs2, padding, stride,
                            BitIn, BitW, BitOut,
                            factor_ch_out,
                            factor_h_out, 
                            factor_h_in,
                            name,
                            out_dim1,
                            in_dim1,
                            in_dim_full,
                            weight_dim1,
                            lambda_dim,
                            k_dim,
                            w_out,
                            h_out,
                            n_out,
                            w_in,
                            h_in,
                            n_in,
                            full_net,
                            platform,
                            data_type_x,
                            data_type_y,
                            test_location,
                            out_mul, out_shift,
                            buffer_l1_all,
                            input_L3,
                            backend
                            ):
    # generation of L3 layers. The layers are generated with this infrustructure if an L3 tiling is demanded.
    tk = OrderedDict([])
    conv_overlap1 = 2 * (fs1 // 2) + fs1 % 2 - 1 - (stride - 1)
    conv_overlap2 = 2 * (fs2 // 2) + fs2 % 2 - 1 - (stride - 1)
    tk['conv_overlap1'] = conv_overlap1
    tk['conv_overlap2'] = conv_overlap2
    tk['BitIn'] = BitIn
    tk['BitW'] = BitW
    tk['BitOut'] = BitOut
    tk['padding'] = padding
    tk['input_L3'] = input_L3
    tk['n_tile_W'] = int(factor_ch_out)
    tk['n_tile_x'] = int(factor_h_in)
    tk['n_tile_y'] = int(factor_h_out)
    tk['verbose'] = False
    tk['func_name'] = name
    tk['func_name_L3'] = name[0] + 'L3'
    tk['act_out_dim_partial'] = int(out_dim1)
    tk['weight_dim'] = int(weight_dim1)
    tk['lambda_dim'] = lambda_dim
    tk['k_dim'] = k_dim
    tk['w_out'] = w_out
    tk['h_out'] = h_out
    tk['n_out'] = n_out
    tk['w_in'] = w_in
    tk['h_in'] = h_in
    tk['n_in'] = n_in
    tk['dim_out'] = out_dim1
    tk['dim_in'] = in_dim1
    tk['platform'] = platform
    tk['y_data_size_byte'] = data_type_y
    tk['x_data_size_byte'] = data_type_x

    root = '/'.join(os.getcwd().split('/')[:-1])
    tmpl = Template(filename=root + f"/Templates/{backend}/layer_templates/layer_template_L3.c")
    l = ""
    s = tmpl.render(verbose_log=l,**tk)
    #
    save_string = './application/DORY_network/src/' + tk['func_name_L3'] + '.c'
    with open(save_string, "w") as f: f.write(s)

    tmpl = Template(filename=root + f"/Templates/{backend}/layer_templates/layer_template_L3-h.h")
    s = tmpl.render(verbose_log=l, **tk)
    if full_net == 1:
        save_string = './application/DORY_network/inc/' + \
            tk['func_name_L3'] + '.h'
    else:
        save_string = './applicationL3/DORY_network/inc/' + \
            tk['func_name_L3'] + '.h'
    with open(save_string, "w") as f:
        f.write(s)
    if 'partial' in test_location:
        string_layer = "inputs.hex"
        save_s = './application/DORY_network/' + string_layer
        with open(save_s, 'wb') as f:
            for i in X.astype('uint8').flatten():
                f.write(bytes((i,)))
        tk['x_content'] = print_test_vector(X, 'char')
        if tk['n_tile_W'] == 1:
            tk['W_content'] = print_test_vector(W, 'char')
            tk['weight_dim'] = W.shape[0]
        tk['check_sum'] = sum(Y)
        tk['activation_size_out'] = out_dim1
        tk['activation_size_in'] = in_dim1
        tk['activation_size_in_full'] = in_dim_full
        tk['out_mul'] = out_mul
        tk['out_shift'] = out_shift
        tk['func_nameL3'] = tk['func_name_L3']
        tk['file'] = name[0][5:] + '_weights.hex'
        tk['buffer_l1_all'] = buffer_l1_all
        tmpl = Template(filename=root + f"/Templates/{backend}/test_templateL3.c")
        s = tmpl.render(**tk)
        save_string = './application/DORY_network/src/main.c'
        with open(save_string, "w") as f: f.write(s)
