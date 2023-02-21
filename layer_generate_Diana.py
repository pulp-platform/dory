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
import copy

sys.path.append('..')
from dory.Parsers.DORY_node import DORY_node
from dory.Parsers.Layer_node import Layer_node


def borders(bits, signed):
    low = -(2 ** (bits-1)) if signed else 0
    high = 2 ** (bits-1) - 1
    return low, high


def mean(bits, signed):
    return 0 if signed else 2**(bits-1)


def std(bits):
    return 2**(bits-1)


def create_dory_node(params, index, index_out):
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
    node.input_indexes = [str(index)]  # '0' is the network input
    node.output_index = str(index_out)
    # Constants: weights, bias, k, lambda
    node.number_of_input_constants = 4

    return node


def calculate_output_dimensions(input_dimensions, kernel_shape, stride, padding):
    h = np.floor((input_dimensions[0] + padding[0] + padding[2] - kernel_shape[0]) / stride[0] + 1)
    w = np.floor((input_dimensions[1] + padding[1] + padding[3] - kernel_shape[1]) / stride[1] + 1)
    return [int(h), int(w)]


def create_layer_conv(params, index, index_out):
    node = Layer_node()
    node.name = params['layer_type']
    node.op_type = params['operation_type']  # TODO might be redundant
    node.pads = params['padding']
    node.group = params['group']
    node.strides = params['stride']
    node.kernel_shape = params['kernel_shape']
    node.input_dimensions = params['input_dimensions']
    node.output_dimensions = calculate_output_dimensions(node.input_dimensions, node.kernel_shape, node.strides, node.pads)
    node.input_channels = params['input_channels']
    node.output_channels = params['output_channels']
    node.output_activation_type = params['output_type']
    node.output_activation_bits = params['intermediate_bits']
    node.input_activation_type = params['input_type']
    node.input_activation_bits = params['input_bits']
    node.constant_names = []
    node.constant_type = 'int'
    node.constants_memory = None
    node.constant_bits = None
    node.weight_type = 'int'
    node.weight_bits = params['weight_bits']
    node.bias_bits = params['bias_bits']
    node.branch_in = params['branch_in']
    node.branch_out = params['branch_out']
    node.branch_change = params['branch_change']
    node.weight_memory = None
    node.MACs = node.output_dimensions[0] * node.output_dimensions[1] * node.output_channels \
                * node.kernel_shape[1] * node.kernel_shape[0] * node.input_channels

    # Ids of previous nodes, node can have multiple input nodes
    node.number_of_input_nodes = 1
    node.input_indexes = [str(index)]  # '0' is the network input
    node.output_index = str(index_out)
    # Constants: weights
    node.number_of_input_constants = 1
    if node.group > 1:
        if not (node.group == node.output_channels == node.input_channels):
            print(" Depthwise convolution with input channels != output channels != groups")
            os._exit(0)
    return node

def create_layer_add(params, index_1, index_2, index_out):
    node = Layer_node()
    node.name = params['layer_type']
    node.op_type = params['operation_type']  # TODO might be redundant
    node.pads = params['padding']
    node.group = params['group']
    node.strides = params['stride']
    node.kernel_shape = params['kernel_shape']
    node.input_dimensions = params['input_dimensions']
    node.output_dimensions = params['input_dimensions']
    node.input_channels = params['input_channels']
    node.output_channels = params['output_channels']
    node.output_activation_type = params['output_type']
    node.output_activation_bits = 8
    node.input_activation_type = params['input_type']
    node.input_activation_bits = params['input_bits']
    node.constant_names = []
    node.constant_names.append('inmul1')
    node.inmul1 = {
        'value': 1,
        'layout': ''
    }
    node.constant_names.append('inadd1')
    node.inadd1 = {
        'value': 0,
        'layout': ''
    }
    node.constant_names.append('inshift1')
    node.inshift1 = {
        'value': 0,
        'layout': ''
    }
    node.constant_names.append('inmul2')
    node.inmul2 = {
        'value': 1,
        'layout': ''
    }
    node.constant_names.append('inadd2')
    node.inadd2 = {
        'value': 0,
        'layout': ''
    }
    node.constant_names.append('inshift2')
    node.inshift2 = {
        'value': 0,
        'layout': ''
    }
    node.constant_names.append('outmul')
    node.outmul = {
        'value': 1,
        'layout': ''
    }
    node.constant_names.append('outadd')
    node.outadd = {
        'value': 0,
        'layout': ''
    }
    node.constant_names.append('outshift')
    node.outshift = {
        'value': 0,
        'layout': ''
    }
    node.constant_type = 'int'
    node.constants_memory = None
    node.constant_bits = None
    node.weight_type = 'int'
    node.weight_bits = params['weight_bits']
    node.bias_bits = params['bias_bits']
    node.branch_in = params['branch_in']
    node.branch_out = params['branch_out']
    node.branch_change = params['branch_change']
    node.weight_memory = None
    node.MACs = node.output_dimensions[0] * node.output_dimensions[1] * node.output_channels \
                * node.kernel_shape[1] * node.kernel_shape[0] * node.input_channels

    # Ids of previous nodes, node can have multiple input nodes
    node.number_of_input_nodes = 1
    node.input_indexes = [str(index_1), str(index_2)]  # '0' is the network input
    node.output_index = str(index_out)
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
    all the negative ones are going to bi clipped to 0.
    Signed: Tries to get the standard deviation to be equal to range / 2
    Unsigned: Tries to shift the mean of positive values towards the middle of the range [0, 2**bits - 1]
    """
    x = x.type(torch.float)
    if signed:
        s = x.std()
        ratio = 1 if s.isnan() or s.isinf() or s < 1 else s.item() / std(bits)
    else:
        m = x[x > 0].mean().item()
        ratio = m / mean(bits, signed)
    shift = round(np.log2(ratio))
    shift = 0 if shift < 0 else shift
    return shift


def batchnorm(x, scale, bias):
    return scale * x + bias


def calculate_batchnorm_params(x, output_bits, constant_bits, signed):
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
    scale = clip(scale, constant_bits)
    scale[scale == 0] = 1

    bias = scale * (desired_mean - m)
    bias = bias.round()
    bias = clip(bias, constant_bits, signed=True)

    return scale.type(torch.int64), bias.type(torch.int64)

def create_input(node):
    low, high = borders(node.input_activation_bits, node.input_activation_type == 'int')
    size = (1, node.input_channels, node.input_dimensions[0], node.input_dimensions[1])
    # return torch.randint(low=low, high=high, size=size)
    return torch.randint(low=1, high=2, size=size)

def create_weight(node):
    low, high = borders(node.weight_bits, signed=True)
    if node.weight_bits == 2:
        low, high = -1, 2
    size = (node.output_channels, node.input_channels // node.group, node.kernel_shape[0], node.kernel_shape[1])
    if node.weight_bits == 2:
        increasing_factor = 2
        vec_weights = torch.tensor([])
        for i in np.arange(node.output_channels-1,-1,-1):
            ones = torch.ones(min(i * increasing_factor + 1, node.input_channels // node.group * node.kernel_shape[0] * node.kernel_shape[1]), 1)
            zeros = torch.zeros((node.input_channels // node.group * node.kernel_shape[0] * node.kernel_shape[1] ) - min(i * increasing_factor + 1, node.input_channels // node.group * node.kernel_shape[0] * node.kernel_shape[1]), 1)
            column = torch.cat((ones, zeros), 0)
            vec_weights = torch.cat((vec_weights, column), 1)
        vec_weights = vec_weights.transpose(0, 1)
        vec_weights = vec_weights.reshape(size).long()
        return vec_weights
    else:
        return torch.randint(low=low, high=high, size=size)
        # return torch.randint(low=2, high=50, size=size)

def create_bias(node):
    low, high = borders(node.bias_bits, signed=True)
    size = (node.output_channels,1)
    return torch.randint(low=low, high=high, size=size).flatten()
    # return torch.randint(low=0, high=1, size=size).flatten()
    ### STILL NEED TO FIX BIAS FOR DW
    # return torch.randint(low=0x01020304, high=0x01020305, size=size).flatten()

def create_conv(i_layer, layer_node, dory_node, network_dir, input=None, weight=None, batchnorm_params=None):
    x = input if input is not None else create_input(layer_node)
    x_save = x.flatten()
    if input is None:
        np.savetxt(os.path.join(network_dir, 'input.txt'), x_save, delimiter=',', fmt='%d')

    w = weight if weight is not None else create_weight(layer_node)
    layer_node.constant_names.append('weights')
    layer_node.weights = {
        'value': w.numpy(),
        'layout': 'CoutCinK'
    }
    b = create_bias(layer_node)
    layer_node.constant_names.append('bias')
    layer_node.bias = {
        'value': b.numpy(),
        'layout': ''
    }
    y = F.conv2d(input=x, weight=w, bias=b, stride=layer_node.strides, padding=layer_node.pads[:2], groups=layer_node.group)
    y_type = torch.int32
    y = y.type(y_type)
    y_signed = layer_node.output_activation_type == 'int'

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
    y_save = copy.deepcopy(y.flatten())
    y_save = y_save.reshape(int(y_save.shape[0]/4), 4)
    y_save1 = copy.deepcopy(y_save)
    y_save[:,0] = y_save1[:,3] 
    y_save[:,1] = y_save1[:,2] 
    y_save[:,2] = y_save1[:,1] 
    y_save[:,3] = y_save1[:,0] 
    y_save = y_save.flatten().numpy()
    np.savetxt(os.path.join(network_dir, f'out_layer{i_layer}.txt'), y_save, delimiter=',', fmt='%d')
    return y


def create_add(i_layer, layer_node, network_dir, input1, input2):

    y = input1 + input2
    y_type = torch.int32
    y = y.type(y_type)
    y_signed = layer_node.output_activation_type == 'int'


    y = clip(y, layer_node.output_activation_bits, y_signed)

    y_save = copy.deepcopy(y.flatten())
    y_save = y_save.reshape(int(y_save.shape[0]/4), 4)
    y_save1 = copy.deepcopy(y_save)
    y_save[:,0] = y_save1[:,3] 
    y_save[:,1] = y_save1[:,2] 
    y_save[:,2] = y_save1[:,1] 
    y_save[:,3] = y_save1[:,0] 
    y_save = y_save.flatten().numpy()
    np.savetxt(os.path.join(network_dir, f'out_layer{i_layer}.txt'), y_save, delimiter=',', fmt='%d')
    return y



def create_graph(params, network_dir,number_of_nodes):
    layers = []
    index_layer = 0
    for index in np.arange(number_of_nodes):
        if params[index]['layer_type'] in ["Convolution", "FullyConnected"]:
            increment = index_layer + 1
            if index > 0:
                if params[index-1]['branch_change'] == 1:
                    increment = index_branch+1
            layer_node  = create_layer_conv(params[index], index_layer, increment)
            if index > 0:
                if params[index-1]['branch_change'] == 1:
                    index_layer = index_branch
            index_layer += 1
            dory_node   = create_dory_node(params[index], index_layer, index_layer + 1)
            index_layer += 1
            with torch.no_grad():
                if index == 0:
                    y = create_conv(index, layer_node, dory_node, network_dir)
                else:
                    y = create_conv(index, layer_node, dory_node, network_dir, input = y.type(torch.long))
            if params[index]['branch_out'] == 1:
                y_branch = y
                index_branch = index_layer
            if params[index]['branch_change'] == 1:
                y_new = y_branch
                y_branch = y
                y = y_new
                index_new = index_branch
                index_branch = index_layer
                index_layer = index_new
            layers.append(layer_node)
            layers.append(dory_node)
        elif params[index]['layer_type'] == "Addition":
            layer_node  = create_layer_add(params[index], index_branch, index_layer, index_layer + 1)
            index_layer += 1
            if params[index]['branch_out'] == 1:
                index_branch = index_layer
            with torch.no_grad():
                y = create_add(index, layer_node, network_dir, input1 = y.type(torch.long), input2 = y_branch.type(torch.long))
            if params[index]['branch_out'] == 1:
                y_branch = y
            layers.append(layer_node)

    return layers
    # return [layer_node, dory_node]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('hardware_target', type=str, choices=["Diana.Diana_SoC", "Diana.Diana_TVM"],
                        help='Hardware platform for which the code is optimized')
    parser.add_argument('layer_type', type=str, choices=["analog", "digital_conv", "digital_FC"],
                        help='Layer to be deployed. Used to choose the config file')
    parser.add_argument('--config_file', default='dory/dory_examples/config_files/config_single_layer.json', type=str,
                        help='Path to the JSON file that specifies the ONNX file of the network and other information. Default: config_files/config_single_layer.json')
    parser.add_argument('--app_dir', default='./application',
                        help='Path to the generated application. Default: ./application')
    parser.add_argument('--perf_layer', default='Yes', help='Yes: MAC/cycles per layer. No: No perf per layer.')
    parser.add_argument('--verbose_level', default='None',
                        help="None: No_printf.\nPerf_final: only total performance\nCheck_all+Perf_final: all check + final performances \nLast+Perf_final: all check + final performances \nExtract the parameters from the onnx model")
    parser.add_argument('--backend', default='MCU', help='MCU or Occamy')
    parser.add_argument('--optional', default='auto',
                        help='auto (based on layer precision, 8bits or mixed-sw), 8bit, mixed-hw, mixed-sw')
    args = parser.parse_args()

    number_of_nodes = 1
    json_configuration_file = []
    if number_of_nodes > 1:
        for i in np.arange(number_of_nodes):
            json_configuration_file_root = os.path.dirname((str(i)+'.').join((args.config_file).split('.')))
            with open((str(i)+'.').join((args.config_file).split('.')), 'r') as f:
                json_configuration_file.append(json.load(f)) 
    elif number_of_nodes == 1:
        if args.layer_type == "analog":
            json_configuration_file_root = os.path.dirname(('_analog.').join((args.config_file).split('.')))
            with open(('_analog.').join((args.config_file).split('.')), 'r') as f:
                json_configuration_file.append(json.load(f)) 
        elif args.layer_type == "digital_conv":
            json_configuration_file_root = os.path.dirname(('_digital_conv.').join((args.config_file).split('.')))
            with open(('_digital_conv.').join((args.config_file).split('.')), 'r') as f:
                json_configuration_file.append(json.load(f)) 
        elif args.layer_type == "digital_FC":
            json_configuration_file_root = os.path.dirname(('_digital_FC.').join((args.config_file).split('.')))
            with open(('_digital_FC.').join((args.config_file).split('.')), 'r') as f:
                json_configuration_file.append(json.load(f)) 

    network_dir = os.path.join(json_configuration_file_root, os.path.dirname(json_configuration_file[0]['onnx_file']))
    os.makedirs(network_dir, exist_ok=True)

    torch.manual_seed(0)

    DORY_Graph = create_graph(json_configuration_file, network_dir, number_of_nodes)
    # Including and running the transformation from DORY IR to DORY HW IR
    onnx_manager = importlib.import_module(f'dory.Hardware_targets.{args.hardware_target}.HW_Parser')
    DORY_to_DORY_HW = onnx_manager.onnx_manager
    DORY_Graph = DORY_to_DORY_HW(DORY_Graph, json_configuration_file[0], json_configuration_file_root).full_graph_parsing()

    # Deployment of the model on the target architecture
    onnx_manager = importlib.import_module(f'dory.Hardware_targets.{args.hardware_target}.C_Parser')
    DORY_HW_to_C = onnx_manager.C_Parser
    DORY_Graph = DORY_HW_to_C(DORY_Graph, json_configuration_file[0], json_configuration_file_root,
                              args.verbose_level, args.perf_layer, args.optional, args.app_dir).full_graph_parsing()
