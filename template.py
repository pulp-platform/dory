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

def print_file_list(x):
    # This function is used to generate a string with all input files.
    s = repr(x).replace("[", "").replace("]", "").replace("'", '"')
    return s


def print_template_Makefile(file_list_w, platform, sdk, backend):
    # Generate the Makefile, including all files to upload on the hyperflash
    tk = OrderedDict([])
    tk['build_layers'] = os.listdir('./application/DORY_network/src/')
    tk['layers_w'] = file_list_w
    tk['platform'] = 'GAP8'
    tk['sdk'] = sdk
    root = '/'.join(os.getcwd().split('/')[:-1])
    tmpl = Template(filename=root + f"/templates_{backend}/Makefile_template")
    s = tmpl.render(**tk)
    if backend == 'MCU':
        save_string = './application/Makefile'
    else:
        save_string = './application/CMakeLists.txt'
    with open(save_string, "w") as f:
        f.write(s)

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
        tmpl = Template(filename=root + f"/templates_{backend}/layer_templates/layer_template_conv_1D.c")
    s = tmpl.render(TEST=test,VERBOSE=False,ULTRA_VERBOSE=ultra_verbose,PULP_TEST=True,verbose_log=l,**tk)
    if 'L2' in test_location:
        save_string = './application/DORY_network/src/' + name_layer.replace("h", "c")
    elif 'L3' in test_location:
        save_string = './application/DORY_network/src/' + name_layer.replace("h", "c")
    with open(save_string, "w") as f:
        f.write(s)
    root = '/'.join(os.getcwd().split('/')[:-1])
    tmpl = Template(filename=root + f"/templates_{backend}/layer_templates/layer_template_h.h")
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
        tmpl = Template(filename=root+f"/templates_{backend}/test_templateL2.c")
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
        tmpl = Template(filename=root+f"/templates_{backend}/Makefile_template_L2")
        s = tmpl.render(**tk)
        save_string = './application/Makefile'
        with open(save_string, "w") as f:
            f.write(s)
    return l2_dim_input, l2_dim_output, l2_dim_weights, l2_dim_k, l2_dim_lambda, tk['nof'], buffer_l1_all, n_out, w_out

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
    tk['files_list'] = print_file_list(file_list_w)
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
    tk['act_compare'] = print_test_vector(act_compare, 'char')
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
    tmpl = Template(filename=root + f"/templates_{backend}/network_template.c")
    tk['PULP_Nodes_Graph'] = PULP_Nodes_Graph
    s = tmpl.render(verbose_log=l,**tk)
    save_string = './application/DORY_network/src/network.c'
    with open(save_string, "w") as f:
        f.write(s)

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
                            input_L3
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
    root = '/'.join(os.getcwd().split('/')[:-1])
    tmpl = Template(filename=root + f"/templates_{backend}/layer_templates/layer_template_L3.c")
    l = ""
    s = tmpl.render(verbose_log=l,**tk)
    #
    save_string = './application/DORY_network/src/' + tk['func_name_L3'] + '.c'
    with open(save_string, "w") as f: f.write(s)

    tmpl = Template(filename=root + f"/templates_{backend}/layer_templates/layer_template_L3-h.h")
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
        tmpl = Template(filename=root + f"/templates_{backend}/test_templateL3.c")
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
    tmpl = Template(filename=root + f"/templates_{backend}/layer_templates/layer_template_L3.c")
    l = ""
    s = tmpl.render(verbose_log=l,**tk)
    #
    save_string = './application/DORY_network/src/' + tk['func_name_L3'] + '.c'
    with open(save_string, "w") as f: f.write(s)

    tmpl = Template(filename=root + f"/templates_{backend}/layer_templates/layer_template_L3-h.h")
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
        tmpl = Template(filename=root + f"/templates_{backend}/test_templateL3.c")
        s = tmpl.render(**tk)
        save_string = './application/DORY_network/src/main.c'
        with open(save_string, "w") as f: f.write(s)

def print_template_layer(x, y_gold, W,
                         n_in, h_in, w_in,
                         n_out,h_out, w_out,
                         tile_n_in, tile_h_in, tile_w_in, tile_h_out, tile_w_out,
                         tile_n_out,
                         ds_x, ds_y, ds_W, ds_act, type_data,
                         fs1, fs2, padding_top, padding_bottom, padding_left, padding_right, stride,
                         relu, BN, DW,
                         out_mul, out_mul2, out_shift, factor_ch_out, factor_h_out, factor_h_in,
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
                         L3_tiling = 0,
                         sdk = 'gap_sdk',
                         backend = 'MCU',
                         number_of_clusters = 1,
                         dma_parallelization = '8-cores'
                         ):
    # Generate the Layer management c file.
    if type_data == 'float':
        ds_x = 32
        ds_y = 32
        ds_W = 32
    if h_out * stride + fs1 - 1 - stride + 1 > h_in:
        if (h_out * stride + fs1 - 1 - stride + 1 - h_in) == padding_top:
            padding_b = 0
            padding_t = padding_top
        else:
            padding_b = padding_bottom
            padding_t = padding_top
    if w_out * stride + fs2 - 1 - stride + 1 > w_in:
        if (w_out * stride + fs2 - 1 - stride + 1 - w_in) == padding_left:
            padding_r = 0
            padding_l = padding_left
        else:
            padding_r = padding_right
            padding_l = padding_left
    # add padding from "regular" tile where necessary
    tile_h_in = tile_h_in if h_in > tile_h_in else tile_h_in
    tile_w_in = tile_w_in if w_in > tile_w_in else tile_w_in
    if w_in > tile_w_in:
        tile_w_out = int((tile_w_in - (fs2 - 1) + (stride - 1)) / stride)
    else:
        tile_w_out = int((tile_w_in + (padding_left + padding_right) - (fs2 - 1) + (stride - 1)) / stride)
    name = re.sub(r'\W', '', name_layer).replace("hex", "").replace(".", "").replace("_weights", "")
    name_layer = name + '.h'
    conv_overlap1 = 2 * (fs1 // 2) + fs1 % 2 - 1 - (stride - 1)
    conv_overlap2 = 2 * (fs2 // 2) + fs2 % 2 - 1 - (stride - 1)
    tk = OrderedDict([])
    if (re.search('.0',name_layer)):
        try:
            int(re.search('.0',name_layer).group())
            tk['first_layer'] = 0
        except ValueError:
            tk['first_layer'] = 1
    else:
        tk['first_layer'] = 0
    tk['sdk'] = sdk
    tk['number_of_clusters'] = number_of_clusters
    tk['dma_parallelization'] = dma_parallelization
    tk['optional_type'] = optional_type
    tk['func_name'] = name
    tk['flag_DW'] = DW
    tk['optional'] = optional
    tk['FLAG_BATCHNORM'] = BN
    tk['has_bias'] = has_bias
    tk['FLAG_RELU'] = relu
    tk['test_location'] = test_location
    tk['platform'] = platform
    if DW == 1:
        tk['chip'] = 'GAPv2'
    else:
        tk['chip'] = chip
    tk['type'] = type_data
    tk['nof'] = n_out
    tk['factor'] = factor_ch_out
    if DW == 0:
        tk['g'] = 1
    else:
        tk['g'] = n_in
    if DW == 0:
        tk['nif'] = n_in
    else:
        tk['nif'] = 1
    tk['conv_overlap1'] = conv_overlap1
    tk['conv_overlap2'] = conv_overlap2
    tk['padding_top'] = padding_top
    tk['padding_bottom'] = padding_bottom
    tk['padding_left'] = padding_left
    tk['padding_right'] = padding_right
    tk['stride'] = stride
    # x parameters
    tk['x_h'] = h_in
    tk['x_w'] = w_in
    tk['x_data_size_byte'] = ds_x
    tk['x_tile_size_nif'] = tile_n_in
    tk['x_tile_size_h'] = tile_h_in
    tk['x_tile_size_w'] = tile_w_in
    tk['x_tile_size_byte'] = int(math.ceil(ds_x * tile_n_in * tile_h_in * tile_w_in / 8.0))
    if backend == 'Occamy':
        tk['x_tile_size_byte'] = int(math.ceil(ds_x * tile_n_in * (tile_h_in + padding_top + padding_bottom) * (tile_w_in + padding_left + padding_right) / 8.0))
        tk['x_tile_size_byte'] = tk['x_tile_size_byte'] + (tk['x_tile_size_byte'] % 8)
    if type_data == 'float':
        tk['x_tile_size_byte'] += 16 ##### FIX TO CHECK #######
    tk['x_tile_size_nif_byte'] = int(math.ceil(tile_n_in * ds_x / 8.0))
    tk['x_stride_w_byte'] = int(math.ceil(w_in * n_in * ds_x / 8.0))
    tk['x_stride_c_byte'] = int(math.ceil(n_in * ds_x / 8.0))
    # y parameters
    tk['y_h'] = h_out
    tk['y_w'] = w_out
    tk['y_data_size_byte'] = ds_y
    tk['act_dim_bit'] = ds_act
    tk['y_tile_size_nof'] = tile_n_out if (n_out > tile_n_out) else n_out
    tk['y_tile_size_h'] = tile_h_out if (h_out > tile_h_out) > 0 else h_out
    tk['y_tile_size_w'] = tile_w_out if (w_out > tile_w_out) > 0 else w_out
    tk['y_tile_size_byte'] = int(math.ceil(tk['y_tile_size_nof'] * tk['y_tile_size_h'] * tk['y_tile_size_w'] * ds_y / 8.0))
    if backend == 'Occamy':
    	tk['y_tile_size_byte'] = tk['y_tile_size_byte'] + (tk['y_tile_size_byte'] % 8)
    tk['y_stride_w_byte'] = int(math.ceil(w_out * n_out * factor_ch_out * ds_y / 8.0))
    tk['y_stride_c_byte'] = int(math.ceil(n_out * factor_ch_out * ds_y / 8.0))
    tk['y_tile_size_nof_byte'] = int(math.ceil(tile_n_out * ds_y / 8.0))

    tk['tile_dim_h'] = max(int(math.ceil(float(h_out) / float(tk['y_tile_size_h']))), 1)
    tk['tile_dim_w'] = max(int(math.ceil(float(w_out) / float(tk['y_tile_size_w']))), 1)
    tk['tile_dim_nof'] = max(int(math.ceil(float(n_out) / float(tk['y_tile_size_nof']))), 1)
    tk['tile_dim_nif'] = max(int(math.ceil(float(n_in) / float(tile_n_in))), 1)
    tk['tile_n_in_last'] = n_in % tile_n_in if n_in % tile_n_in > 0 else tile_n_in
    # W parameters
    tk['fs1'] = fs1
    tk['fs2'] = fs2
    tk['W_data_size_byte'] = ds_W
    tk['W_tile_size_nof'] = tile_n_out 
    if tk['has_bias'] == 1:
        tk['b_size_byte'] = int(math.ceil(n_out * ds_W / 8.0))
    else:
        tk['b_size_byte'] = 0

    if DW == 0:
        tk['W_tile_size_nif'] = tile_n_in * tk['tile_dim_nif']
        tk['W_tile_size_nif_last'] = tk['tile_n_in_last'] * tk['tile_dim_nif']
    else:
        tk['W_tile_size_nif'] = 1
        tk['W_tile_size_nif_last'] = 1
    tk['W_tile_size_byte'] = int(math.ceil(tile_n_out * tk['W_tile_size_nif'] * fs1 * fs2 * ds_W / 8.0))
    if backend == 'Occamy':
    	tk['W_tile_size_byte'] = tk['W_tile_size_byte'] + (tk['W_tile_size_byte'] % 8)
    if DW == 0:
        tk['W_stride_nof_byte'] = int(math.ceil(tk['nif'] * fs1 * fs2 * ds_W / 8.0))
    else:
        tk['W_stride_nof_byte'] = int(math.ceil(tk['nif'] * fs1 * fs2 * ds_W / 8.0))        
    tk['W_stride_hw_byte'] = int(math.ceil(tk['nif'] * ds_W / 8.0))
    tk['W_tile_nif_byte'] = int(math.ceil(tk['W_tile_size_nif'] * ds_W / 8.0))
    tk['W_tile_nif_byte_last'] = int(math.ceil(tk['W_tile_size_nif_last'] * ds_W / 8.0))
    # l2 parameters
    if tk['FLAG_BATCHNORM'] == 1:
        tk['l2_off_k'] = int(
            math.ceil(tk['nof'] * tk['nif'] * fs1 * fs2 * ds_W / 8.0 + tk['b_size_byte']))
        tk['l2_off_lambda'] = int(
            math.ceil((tk['nof'] * tk['nif'] * fs1 * fs2 * ds_W + tk['nof'] * ds_act) / 8.0 + tk['b_size_byte']))
    if has_bias == 1:
        tk['l2_off_bias'] = int(math.ceil(tk['nof'] * tk['nif'] * fs1 * fs2 * ds_W / 8.0 ))
    if n_in == tile_n_in and w_in == tile_w_in and h_in == tile_h_in:
        x_buffer_size = int(math.ceil(ds_x * tile_n_in * tile_h_in * tile_w_in / 8.0))
    else:
        x_buffer_size = 2 * int(math.ceil(ds_x * tile_n_in * tile_h_in * tile_w_in / 8.0))
        if x_buffer_size % 16 != 0:
            x_buffer_size = x_buffer_size
    if backend == 'Occamy':
        if n_in == tile_n_in and w_in == tile_w_in and h_in == tile_h_in:
            x_buffer_size = int(math.ceil(ds_x * tile_n_in * (tile_h_in + padding_top + padding_bottom) * (tile_w_in + padding_left + padding_right) / 8.0))
            x_buffer_size = x_buffer_size + (x_buffer_size % 8)
        else:
            x_buffer_size = 2 * int(math.ceil(ds_x * tile_n_in * (tile_h_in + padding_top + padding_bottom) * (tile_w_in + padding_left + padding_right) / 8.0))
            x_buffer_size = x_buffer_size + (x_buffer_size % 16)
    if n_in == (tile_n_in * number_of_clusters) and w_in == tile_w_in and h_in == tile_h_in and n_out == (tile_n_out * number_of_clusters):
        y_buffer_size = int(math.ceil(ds_y * tk['y_tile_size_nof'] * tk['y_tile_size_h'] * tk['y_tile_size_w'] / 8.0))
        if backend == 'Occamy':
            y_buffer_size = y_buffer_size + (y_buffer_size % 8)
        if DW == 0:
            W_buffer_size = int(math.ceil(ds_W * tk['y_tile_size_nof']  * tk['W_tile_size_nif'] * fs1 * fs2 / 8.0))
            if backend == 'Occamy':
                W_buffer_size = W_buffer_size + (W_buffer_size % 8)
        else:
            W_buffer_size = int(math.ceil(ds_W * tk['y_tile_size_nof']  * 1 * fs1 * fs2 / 8.0))
            if backend == 'Occamy':
                W_buffer_size = W_buffer_size + (W_buffer_size % 8)
    else:
        y_buffer_size = 2 * int(math.ceil(ds_y * tk['y_tile_size_nof'] * tk['y_tile_size_h'] * tk['y_tile_size_w'] / 8.0))
        if backend == 'Occamy':
            y_buffer_size = y_buffer_size + (y_buffer_size % 16)
        if DW == 0:
            W_buffer_size = 2 * int(math.ceil(ds_W * tk['y_tile_size_nof'] * tk['W_tile_size_nif'] * fs1 * fs2 / 8.0))
            if backend == 'Occamy':
                W_buffer_size = W_buffer_size + (W_buffer_size % 16)
        else:
            W_buffer_size = 2 * int(math.ceil(ds_W * tk['y_tile_size_nof'] * 1 * fs1 * fs2 / 8.0))
            if backend == 'Occamy':
                W_buffer_size = W_buffer_size + (W_buffer_size % 16)
    if tk['FLAG_BATCHNORM'] == 1:
        k_buffer_size = int(n_out * ds_act / 8.0)
        lambd_buffer_size = int(n_out * ds_act / 8.0)
    else:
        k_buffer_size = 0
        lambd_buffer_size = 0

    tk['k_tile_size_byte'] = 0
    tk['lambda_tile_size_byte'] = 0
    tk['k_size_byte'] = 0
    tk['lambda_size_byte'] = 0
    if conv_order != 'PULP-NN-MAX':
        if tk['FLAG_BATCHNORM'] == 1:
            tk['k_size_byte'] = k_buffer_size
            tk['lambda_size_byte'] = k_buffer_size
            tk['k_tile_size_byte_transfer'] = int(math.ceil(tile_n_out * ds_act / 8.0))
            if backend == 'Occamy':
                tk['k_tile_size_byte_transfer'] = tk['k_tile_size_byte_transfer'] + (tk['k_tile_size_byte_transfer'] % 8)
            tk['lambda_tile_size_byte_transfer'] = int(math.ceil(tile_n_out * ds_act / 8.0))
            if backend == 'Occamy':
                tk['lambda_tile_size_byte_transfer'] = tk['lambda_tile_size_byte_transfer'] + (tk['lambda_tile_size_byte_transfer'] % 8)
            if n_in == tile_n_in and w_in == tile_w_in and h_in == tile_h_in and n_out == tile_n_out:
                tk['k_tile_size_byte'] = int(math.ceil(tile_n_out * ds_act / 8.0))
                tk['lambda_tile_size_byte'] = int(math.ceil(tile_n_out * ds_act / 8.0))
            else:
                tk['k_tile_size_byte'] = int(math.ceil(tile_n_out * ds_act / 8.0 * 2))
                tk['lambda_tile_size_byte'] = int(math.ceil(tile_n_out * ds_act / 8.0 * 2))
        if has_bias == 1:
            tk['bias_tile_size_byte'] = tile_n_out
            tk['b_size_byte'] = int(n_out)
        else:
            tk['bias_tile_size_byte'] = 0
            tk['b_size_byte'] = 0
    if conv_order == 'PULP-NN-MAX' or conv_order == 'PULP-NN-ADD':
        W_buffer_size = 0
    # l1 parameters
    
    if type_data == 'float':
        x_buffer_size += 16 ##### FIX TO CHECK #######
    tk['l1_x_offset'] = 0
    tk['l1_y_offset'] = x_buffer_size + 8
    if conv_order == 'PULP-NN-ADD':
        tk['l1_x2_offset'] = x_buffer_size + 8 + y_buffer_size + 8
    if conv_order != 'PULP-NN-MAX' and conv_order != 'PULP-NN-ADD':
        tk['l1_W_offset'] = x_buffer_size + 8 + y_buffer_size + 8
        if tk['FLAG_BATCHNORM'] == 1:
            tk['l1_k_offset'] = x_buffer_size + 8 + y_buffer_size + 8 + W_buffer_size + 8
            tk['l1_lambda_offset'] = x_buffer_size + 8 + y_buffer_size + 8 + W_buffer_size + 8 + tk['k_tile_size_byte'] + 8
        if has_bias == 1:
            tk['l1_b_offset'] = x_buffer_size + 8 + y_buffer_size + 8 + W_buffer_size + 8 + tk['k_tile_size_byte'] + 8 + tk['lambda_tile_size_byte']  + 8


    # x last
    tk['x_tile_size_nif_last'] = n_in % tile_n_in if (n_in % tile_n_in) > 0 else tile_n_in
    tk['x_tile_size_nif_byte_last'] = int(math.ceil(tk['x_tile_size_nif_last'] * ds_x / 8.0))
    if tk['tile_dim_h'] == 1:
        tk['x_tile_size_h_last'] = tk['x_tile_size_h']
    elif tk['tile_dim_h'] == 2:
        tk['x_tile_size_h_last'] = h_in - tile_h_in + tk['conv_overlap1'] + padding_top
    elif tk['tile_dim_h'] == 3:
        tk['x_tile_size_h_last'] = h_in - tile_h_in - (tile_h_in - tk['conv_overlap1'] - padding_top) + tk['conv_overlap1']
    else:
        tk['x_tile_size_h_last'] = h_in - tile_h_in - (tile_h_in - tk['conv_overlap1'] - padding_top) - (tk['tile_dim_h'] - 3) * (tile_h_in - tk['conv_overlap1']) + tk['conv_overlap1']
    if tk['tile_dim_w'] == 1:
        tk['x_tile_size_w_last'] = tk['x_tile_size_w']
    elif tk['tile_dim_w'] == 2:
        tk['x_tile_size_w_last'] = w_in - tile_w_in + tk['conv_overlap2'] + padding_left
    elif tk['tile_dim_w'] == 3:
        tk['x_tile_size_w_last'] = w_in - tile_w_in - (tile_w_in - tk['conv_overlap2'] - padding_left) + tk['conv_overlap2']
    else:
        tk['x_tile_size_w_last'] = w_in - tile_w_in - (tile_w_in - tk['conv_overlap2'] - padding_left) - (tk['tile_dim_w'] - 3) * (tile_w_in - tk['conv_overlap2']) + tk['conv_overlap2']
    if tk['x_tile_size_h_last'] > tk['x_tile_size_h']:
        tk['x_tile_size_h_last'] = tk['x_tile_size_h']
    if tk['x_tile_size_w_last'] > tk['x_tile_size_w']:
        tk['x_tile_size_w_last'] = tk['x_tile_size_w']

    # W last
    if conv_order != 'PULP-NN-MAX' and conv_order != 'PULP-NN-ADD':
        tk['W_tile_size_nof_last'] = n_out % tile_n_out if (n_out % tile_n_out) > 0 else tile_n_out
        tk['W_tile_size_nif_last'] = tk['W_tile_size_nif']
        tk['W_tile_size_nif_byte_last'] = int(math.ceil(tk['W_tile_size_nif_last'] * ds_W / 8.0))
    # y last
    tk['y_tile_size_nof_last'] = n_out % tile_n_out if (n_out % tile_n_out) > 0 else tile_n_out
    tk['y_tile_size_h_last'] = h_out % tile_h_out if (h_out % tile_h_out) > 0 else tile_h_out
    tk['y_tile_size_w_last'] = w_out % tile_w_out if (w_out % tile_w_out) > 0 else tile_w_out
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
        buffer_l1_all = W_buffer_size + x_buffer_size + y_buffer_size + tk['k_tile_size_byte'] + tk['lambda_tile_size_byte'] + 40 + tk['b_size_byte']
        tk['im2col_dim'] = (8 * (fs1 * (tile_h_in + padding_bottom + padding_top) + fs1)) * int( 8 / min(ds_x, ds_y, ds_W))
    elif conv_order == 'PULP-NN-ADD':
        buffer_l1_all = x_buffer_size * 2 + y_buffer_size + tk['k_tile_size_byte'] + tk['lambda_tile_size_byte'] + 40 + tk['b_size_byte']
    elif conv_order == 'PULP-NN-MAX':
        buffer_l1_all = x_buffer_size + y_buffer_size + tk['k_tile_size_byte'] + tk['lambda_tile_size_byte'] + 40 + tk['b_size_byte']
    tk['buffer_l1_all'] = buffer_l1_all
    l2_dim_input = (n_in) * tk['x_h'] * tk['x_w']
    l2_dim_output = (tk['nof']) * tk['y_h'] * tk['y_w']
    if DW == 0:
        l2_dim_weights = int(tk['nof'] * tk['nif'] * tk['fs1'] * tk['fs2'] * ds_W / 8.0)
    else:
        l2_dim_weights = int(tk['nof'] * 1 * tk['fs1'] * tk['fs2'] * ds_W / 8.0)
    l2_dim_k = k_buffer_size
    l2_dim_lambda = lambd_buffer_size
    root = '/'.join(os.getcwd().split('/')[:-1])
    if conv_order == 'PULP-NN':
        tmpl = Template(filename=root+f"/templates_{backend}/layer_templates/layer_template.c")
    elif conv_order == 'PULP-NN-MAX':
        if(optional_type == '1D_Conv'):
            tmpl = Template(filename=root+f"/templates_{backend}/layer_templates/pooling_layer_1D_template.c")
        else:
            tmpl = Template(filename=root+f"/templates_{backend}/layer_templates/pooling_layer_template.c")
    elif conv_order == 'PULP-NN-ADD':
        if(optional_type == '1D_Conv'):
            tmpl = Template(filename=root+f"/templates_{backend}/layer_templates/add_layer_1D_template.c")
        else:
            tmpl = Template(filename=root+f"/templates_{backend}/layer_templates/add_layer_template.c")
    s = tmpl.render(TEST=test,VERBOSE=False,ULTRA_VERBOSE=ultra_verbose,PULP_TEST=True,verbose_log=l,**tk)
    if 'L2' in test_location:
        save_string = './application/DORY_network/src/' + name_layer.replace("h", "c")
    elif 'L3' in test_location:
        save_string = './application/DORY_network/src/' + name_layer.replace("h", "c")
    with open(save_string, "w") as f:
        f.write(s)
    tmpl = Template(filename=root+f"/templates_{backend}/layer_templates/layer_template_h.h")
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
    if 'L2' in test_location and L3_tiling == 0:
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
        tk['l2_dim_weights'] = int(l2_dim_weights + (l2_dim_k + l2_dim_lambda))
        tk['w_out'] = tk['y_w']
        tk['h_out'] = tk['y_h']
        tk['ultra_test'] = True
        root = '/'.join(os.getcwd().split('/')[:-1])
        tmpl = Template(filename=root+f"/templates_{backend}/test_templateL2.c")
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
        tmpl = Template(filename=root+f"/templates_{backend}/Makefile_template_L2")
        s = tmpl.render(**tk)
        save_string = './application/Makefile'
        with open(save_string, "w") as f:
            f.write(s)
    return l2_dim_input, l2_dim_output, l2_dim_weights, l2_dim_k, l2_dim_lambda, tk['b_size_byte'], buffer_l1_all, n_out, w_out, h_out


def print_test_vector(x, type_data):
    # Print the test vector in the c file.
    if type_data == 'char':
        try:
            np.set_printoptions(
                threshold=sys.maxsize,
                formatter={'int': lambda x: hex(np.uint8(x)) if (
                    x < 0) else hex(np.uint8(x)), }
            )
        except TypeError:
            np.set_printoptions(threshold=sys.maxsize)
        s = repr(x.flatten()).replace("array([", "").replace("]", "").replace("[", "").replace(")", "").replace(",\n      dtype=int8)", "").replace(", dtype=uint8", "").replace(",\n      dtype=uint8)", "").replace(",\n      dtype=uint8", "").replace(",\n      dtype=int8", "").replace(", dtype=int8", "").replace(", dtype=int8)", "").replace(", dtype=int8)", "").replace(", dtype=uint8)", "")

    elif type_data == 'int16_t':
        try:
            np.set_printoptions(
                threshold=sys.maxsize,
                formatter={'int': lambda x: hex(np.uint16(x)) if (
                    x < 0) else hex(np.int16(x)), }
            )
        except TypeError:
            np.set_printoptions(threshold=sys.maxsize)
        s = repr(x.flatten()).replace("array([", "").replace("]", "").replace("[", "").replace(",\n      dtype=int16)", "").replace(
            ", dtype=int16)", "").replace(", dtype=int16)", "").replace(", dtype=uint16)", "").replace(")", "")

    else:
        try:
            np.set_printoptions(
                threshold=sys.maxsize,
                formatter={'int': lambda x: hex(np.uint32(x)) if (
                    x < 0) else hex(np.int32(x)), }
            )
        except TypeError:
            np.set_printoptions(threshold=sys.maxsize)
        s = repr(x.flatten()).replace("array([", "").replace("]", "").replace("[", "").replace(
            ",\n      dtype=int32)", "").replace(", dtype=int32)", "").replace(", dtype=int32)", "").replace(", dtype=uint32)", "")
    return s
