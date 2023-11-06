#!/bin/bash
# network_generate.py
# Alessio Burrello <alessio.burrello@unibo.it>
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
import json
import os
import importlib
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import sys

from dory.Parsers.DORY_node import DORY_node
from dory.Parsers.Layer_node import Layer_node


def borders(bits, signed):
    low = -(2 ** (bits-1)) if signed else 0
    high = 2 ** (bits-1) - 1 if signed else 2 ** bits - 1
    return low, high


def mean(bits, signed):
    return 0 if signed else 2**(bits-1)


def std(bits):
    return 2**(bits-1)


def create_dory_node(params, i):
    node = DORY_node()
    node.branch_out = 0
    node.branch_in = 0
    node.branch_last = 0
    node.branch_change = 0

    name = 'BNRelu' if params['batchnorm'] else 'Relu'
    node.name = name
    node.op_type = name
    node.layout = 'CHW'
    node.bias_bits = 32

    # constant -> bn and relu
    node.constant_type = 'int'
    node.constant_bits = params['BNRelu_bits']
    node.constant_names = []
    node.input_activation_type = params['input_type']
    node.input_activation_bits = params['intermediate_bits']
    node.output_activation_type = params['output_type']
    node.output_activation_bits = params['output_bits']
    node.weight_type = 'int'
    node.weight_bits = None
    node.min, node.max = borders(node.output_activation_bits, node.output_activation_type == 'int')

    # Ids of previous nodes, node can have multiple input nodes
    node.number_of_input_nodes = 1
    node.input_indexes = [str(i)]
    node.output_index = str(i+1)
    # Constants: weights, bias, k, lambda
    node.number_of_input_constants = 4

    return node


def calculate_output_dimensions(node):
    if node.name == 'FullyConnected':
        return [1,1]
    h = (node.input_dimensions[0] + node.pads[0] + node.pads[1] - node.kernel_shape[0]) / node.strides[0] + 1
    w = (node.input_dimensions[1] + node.pads[2] + node.pads[3] - node.kernel_shape[1]) / node.strides[1] + 1
    return [int(h), int(w)]


def create_layer_node(params, i):
    node = Layer_node()
    node.name = params['layer_type']
    node.op_type = params['operation_type']  # TODO might be redundant
    node.pads = params['padding']
    node.group = params['group']
    node.strides = params['stride']
    node.kernel_shape = params['kernel_shape']
    node.input_dimensions = params['input_dimensions']
    node.output_dimensions = calculate_output_dimensions(node)
    node.input_channels = params['input_channels']
    node.output_channels = params['output_channels']
    node.output_activation_type = params['output_type']
    node.output_activation_bits = params['intermediate_bits']
    node.input_activation_type = params['input_type']
    node.input_activation_bits = params['input_bits']
    node.constant_names = []
    node.constant_type = 'int'
    node.constants_memory = None
    node.constant_bits = params['BNRelu_bits']
    node.weight_type = 'int'
    node.weight_bits = params['weight_bits']
    node.bias_bits = 32
    node.weight_memory = None
    node.MACs = node.output_dimensions[0] * node.output_dimensions[1] * node.output_channels \
                * node.kernel_shape[1] * node.kernel_shape[0] * node.input_channels
    node.n_test_inputs = 1

    # Ids of previous nodes, node can have multiple input nodes
    node.number_of_input_nodes = 1
    node.input_indexes = [str(i)]  # '0' is the network input
    node.output_index = str(i+1)
    # Constants: weights
    node.number_of_input_constants = 1
    return node


def clip(x, bits, signed=False):
    low, high = borders(bits, signed)
    x[x > high] = high
    x[x < low] = low
    return x


def calculate_shift(x, bits, signed):
    """
    Calculate shift

    This function calculates the shift in a way that it maximizes the number of values
    that are in between min and max after shifting. It looks only at positive values since
    all the negative ones are going to be clipped to 0.
    Calculates the maximum distance from the mean and shifts to fit it into standard deviation.
    """
    x = x[x > 0]
    shift = 0
    if x.numel() > 0:
        dist = torch.abs(x - mean(bits, signed))
        ratio = dist.max().item() / std(bits)
        if ratio != 0:
            shift = round(np.log2(ratio))
    return shift


def batchnorm(x, scale, bias):
    return scale * x + bias


def calculate_batchnorm_params(x, output_bits, normalization_bits, bias_bits, signed):
    """
    Calculate batchnorm

    Calculate Batch-Normalization parameters scale and bias such that we maximize the number
    of values that fall into range [0, 2**output_bits - 1].
    Shifts the mean towards the center of the range and changes the standard deviation so that
    most of the values fall into the range.
    """
    x = x.type(torch.float)

    desired_mean = mean(output_bits, signed)
    desired_std = std(output_bits)

    # Calculate mean and std for each output channel
    m = x.mean(dim=(-2, -1), keepdim=True)
    s = x.std(dim=(-2, -1), keepdim=True)

    scale = torch.empty_like(s)
    scale[s.isnan()] = 1
    scale[torch.logical_not(s.isnan())] = desired_std / s[torch.logical_not(s.isnan())]
    scale = scale.round()
    scale = clip(scale, normalization_bits)
    scale[scale == 0] = 1

    bias = scale * (desired_mean - m)
    bias = bias.round()
    bias = clip(bias, bias_bits, signed=True)

    return scale.type(torch.int64), bias.type(torch.int64)


def create_input(node):
    low, high = borders(node.input_activation_bits, node.input_activation_type == 'int')
    size = (1, node.input_channels, node.input_dimensions[0], node.input_dimensions[1])
    return torch.randint(low=low, high=high, size=size)


def create_weight(node):
    low, high = borders(node.weight_bits, signed=True)
    size = (node.output_channels, node.input_channels // node.group, node.kernel_shape[0], node.kernel_shape[1])
    return torch.randint(low=low, high=high, size=size)


def create_layer(i_layer, layer_node, dory_node, network_dir, hardware_target, input=None, weight=None, batchnorm_params=None):

    def save(a, filename):
        np.savetxt(os.path.join(network_dir, filename), a.permute(0, 2, 3, 1).flatten(), delimiter=',', fmt='%d')

    x = input if input is not None else create_input(layer_node)

    save(x, 'input.txt')

    dory_padding = layer_node.pads  # Top, Left, Bottom, Right
    torch_padding = (dory_padding[1], dory_padding[3], dory_padding[0], dory_padding[2])  # Left, Right, Top, Bottom
    x = F.pad(x, torch_padding)

    w = weight if weight is not None else create_weight(layer_node)

    layer_node.constant_names.append('weights')
    layer_node.weights = {
        'value': w.numpy(),
        'layout': 'CoutCinK'
    }

    y = F.conv2d(input=x, weight=w, stride=layer_node.strides, groups=layer_node.group)

    if layer_node.output_activation_bits == 64:
        y_type = torch.int64
    elif layer_node.output_activation_bits == 32:
        y_type = torch.int32
    else:
        print("Unsupported output activation bitwidth")
        sys.exit(-1)

    y = y.type(y_type)

    save(y, f'inter_layer{i_layer}.txt')

    y_signed = layer_node.output_activation_type == 'int'

    if  dory_node:
        if 'BN' in dory_node.op_type:
            if batchnorm_params is not None:
                k, l = batchnorm_params
            else:
                k, l = calculate_batchnorm_params(y, dory_node.output_activation_bits, dory_node.constant_bits, dory_node.bias_bits, y_signed)
            dory_node.constant_names.append('k')
            dory_node.k = {'value': k.type(torch.float).numpy(), 'layout': ''}
            dory_node.constant_names.append('l')
            dory_node.l = {'value': l.type(torch.float).numpy(), 'layout': ''}
            y = batchnorm(y, k, l)
        else:
            dory_node.constant_names.append('outmul')
            dory_node.outmul = {
            'value': 1,
            'layout': ''
            }

        dory_node.constant_names.append('outshift')
        dory_node.outshift = {
            'value': calculate_shift(y, dory_node.output_activation_bits, y_signed),
            'layout': ''
        }
        y = y >> dory_node.outshift['value']
        y = clip(y, dory_node.output_activation_bits, y_signed)
    else:
        layer_node.constant_names.append('outmul')
        layer_node.outmul = {
        'value': 1,
        'layout': ''
        }
        layer_node.constant_names.append('outshift')
        layer_node.outshift = {
            'value': 0,
        'layout': ''
    }

    y = y.type(y_type)

    save(y, f'out_layer{i_layer}.txt')

    return y


def create_graph(params, network_dir, hardware_target):
    layer_node = create_layer_node(params, 0)
    dory_node = create_dory_node(params, 1)

    with torch.no_grad():
        create_layer(0, layer_node, dory_node, network_dir, hardware_target)

    return [layer_node, dory_node]


def layer_generate(
        json_configuration_file,
        json_configuration_file_root,
        network_dir,
        hardware_target,
        verbose_level='Check_all',
        perf_layer='No',
        optional='auto',
        app_dir='./application'
):
    torch.manual_seed(0)
    DORY_Graph = create_graph(json_configuration_file, network_dir, hardware_target)

    # Including and running the transformation from DORY IR to DORY HW IR
    onnx_manager = importlib.import_module(f'dory.Hardware_targets.{hardware_target}.HW_Parser')
    DORY_to_DORY_HW = onnx_manager.onnx_manager
    DORY_Graph = DORY_to_DORY_HW(DORY_Graph, json_configuration_file, json_configuration_file_root).full_graph_parsing()

    # Deployment of the model on the target architecture
    onnx_manager = importlib.import_module(f'dory.Hardware_targets.{hardware_target}.C_Parser')
    DORY_HW_to_C = onnx_manager.C_Parser
    DORY_Graph = DORY_HW_to_C(DORY_Graph, json_configuration_file, json_configuration_file_root,
                              verbose_level, perf_layer, optional, app_dir).full_graph_parsing()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('hardware_target', type=str, choices=["PULP.PULP_gvsoc","PULP.GAP8", "PULP.GAP9", "PULP.GAP9_NE16", "Occamy", "Diana"],
                        help='Hardware platform for which the code is optimized')
    parser.add_argument('--config_file', default='config_files/config_single_layer.json', type=str,
                        help='Path to the JSON file that specifies the ONNX file of the network and other information. Default: config_files/config_single_layer.json')
    parser.add_argument('--app_dir', default='./application',
                        help='Path to the generated application. Default: ./application')
    parser.add_argument('--perf_layer', default='Yes', help='Yes: MAC/cycles per layer. No: No perf per layer.')
    parser.add_argument('--verbose_level', default='Check_all+Perf_final',
                        help="None: No_printf.\nPerf_final: only total performance\nCheck_all+Perf_final: all check + final performances \nLast+Perf_final: all check + final performances \nExtract the parameters from the onnx model")
    parser.add_argument('--optional', default='8bit',
                        help='auto (based on layer precision, 8bits or mixed-sw), 8bit, mixed-hw, mixed-sw')
    args = parser.parse_args()

    json_configuration_file_root = os.path.dirname(args.config_file)
    with open(args.config_file, 'r') as f:
        json_configuration_file = json.load(f)

    network_dir = os.path.join(json_configuration_file_root, os.path.dirname(json_configuration_file['onnx_file']))
    os.makedirs(network_dir, exist_ok=True)

    torch.manual_seed(0)

    layer_generate(json_configuration_file, json_configuration_file_root, network_dir,
                   args.hardware_target, args.verbose_level, args.perf_layer, args.optional, args.app_dir)
